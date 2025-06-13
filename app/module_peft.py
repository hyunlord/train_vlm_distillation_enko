import os
import math
from loguru import logger

import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim import SGD, Adam, AdamW

from peft import get_peft_model, LoraConfig, TaskType

from transformers import (
    AutoModel,
    AutoFeatureExtractor,
    AutoTokenizer,
    AutoProcessor
)


class EnKoDistillationModule(pl.LightningModule):
    def __init__(
         self,
        teacher_model_name: str,
        student_model_name: str,
        optimizer: str = "adamw",
        learning_rate: float = 5e-4,
        weight_decay: float = 1e-4,
        loss_type: str = 'mse',

        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: list = None
    ):
        super().__init__()
        self.save_hyperparameters()
        logger.debug(f"hyperparameters: {self.hparams}")

        # init model
        self.teacher, self.student = self.init_model(self.hparams.teacher_model_name,
                                                     self.hparams.student_model_name)
        self.print_trainable_parameters()

        if 'mse' in self.hparams.loss_type:
            self.mse = torch.nn.MSELoss()
        elif 'cosine' in self.hparams.loss_type:
            self.cosine_loss = torch.nn.CosineEmbeddingLoss()

    def init_model(self, teacher_model_name: str, student_model_name: str):
        teacher = AutoModel.from_pretrained(teacher_model_name, torch_dtype=torch.bfloat16)
        for param in teacher.parameters():
            param.requires_grad = False
        teacher.eval()

        student = AutoModel.from_pretrained(student_model_name, torch_dtype=torch.bfloat16)
        for param in student.vision_model.parameters():
            param.requires_grad = False

        if self.hparams.use_lora:
            if self.hparams.lora_target_modules is None:
                self.hparams.lora_target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
            lora_config = LoraConfig(
                r=self.hparams.lora_r,
                lora_alpha=self.hparams.lora_alpha,
                target_modules=self.hparams.lora_target_modules,
                lora_dropout=self.hparams.lora_dropout,
                bias="none",
            )
            student = get_peft_model(student, lora_config)
        return teacher, student

    def print_trainable_parameters(self):
        trainable_params = 0
        all_param = 0
        for _, param in self.student.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || "
            f"trainable%: {100 * trainable_params / all_param:.2f}"
        )

    def step(self, batch):
        student_ko_batch, student_en_batch, teacher_en_batch = batch
        student_ko_emb = self.student.get_text_features(**student_ko_batch)
        student_en_emb = self.student.get_text_features(**student_en_batch)
        teacher_en_emb = self.teacher.get_text_features(**teacher_en_batch)

        if self.hparams.loss_type == 'mse':
            st_loss = self.mse(student_ko_emb, teacher_en_emb)
            en_loss = self.mse(student_en_emb, teacher_en_emb)
        elif self.hparams.loss_type == 'cosine':
            target = torch.ones(student_ko_emb.size(0), device=self.device)
            st_loss = self.cosine_loss(student_ko_emb, teacher_en_emb, target)
            en_loss = self.cosine_loss(student_en_emb, teacher_en_emb, target)
        elif self.hparams.loss_type == 'norm-mse':
            student_ko_emb_norm = F.normalize(student_ko_emb, p=2, dim=1)
            student_en_emb_norm = F.normalize(student_en_emb, p=2, dim=1)
            teacher_en_emb_norm = F.normalize(teacher_en_emb, p=2, dim=1)
            st_loss = self.mse(student_ko_emb_norm, teacher_en_emb_norm)
            en_loss = self.mse(student_en_emb_norm, teacher_en_emb_norm)
        loss = st_loss + en_loss
        loss_dict = {
            "loss": loss,
            "loss_st": st_loss,
            "loss_en": en_loss,
        }
        return loss_dict

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log_dict(
            {
                "train/loss": loss["loss"],
                "train/loss_st": loss["loss_st"],
                "train/loss_en": loss["loss_en"],
            },
            on_step=True,
            on_epoch=True,
        )
        return loss["loss"]

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log_dict(
            {
                "val/loss": loss["loss"],
                "val/loss_st": loss["loss_st"],
                "val/loss_en": loss["loss_en"],
            },
            on_epoch=True,
            on_step=True,
        )
        return loss["loss"]

    def create_optimizer(self, name: str):
        name = name.lower()
        if name == "adam":
            return Adam
        elif name == "adamw":
            return AdamW
        elif name == "sgd":
            return SGD

    def configure_optimizers(self):
        opt_class = self.create_optimizer(self.hparams.optimizer)
        optimizer = opt_class(
            self.student.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        if self.trainer and self.trainer.datamodule:
            num_devices = max(1, self.trainer.num_devices)
            effective_batch_size = self.trainer.datamodule.batch_size * num_devices
            steps_per_epoch = math.ceil(len(self.trainer.datamodule.train_dataloader().dataset) / effective_batch_size)
            total_training_steps = steps_per_epoch * self.trainer.max_epochs

            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                self.hparams.learning_rate,
                total_steps=total_training_steps,
            )
            scheduler_config = {"scheduler": scheduler, "interval": "step"}
            return [optimizer], [scheduler_config]
        return optimizer

    def on_train_epoch_end(self):
        self.save(f"save/test_siglip2base_siglip2base_{self.hparams.loss_type}_epoch-{self.current_epoch}")

    def save(self, save_dir: str = "save/my_model"):
        self.student.save_pretrained(save_dir)
        processor = AutoProcessor.from_pretrained(self.hparams.student_model_name, use_fast=True)
        processor.save_pretrained(save_dir)

        chat_template_path = os.path.join(save_dir, "chat_template.jinja")
        if os.path.exists(chat_template_path):
            os.remove(chat_template_path)
            print(f"불필요한 '{chat_template_path}' 파일을 삭제했습니다.")

