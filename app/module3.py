import math
import inspect

import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim import SGD, Adam, AdamW

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
        use_auth_token: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        # init model
        self.teacher_model_name = teacher_model_name
        self.student_model_name = student_model_name
        self.use_auth_token = use_auth_token
        self.teacher, self.student = self.init_model(teacher_model_name, student_model_name)

        self.mse = torch.nn.MSELoss()
        self.cosine_loss = torch.nn.CosineEmbeddingLoss()

        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def init_model(self, teacher_model_name: str, student_model_name: str):
        teacher = AutoModel.from_pretrained(teacher_model_name)
        student = AutoModel.from_pretrained(student_model_name)
        for param in teacher.parameters():
            param.requires_grad = False
        teacher.eval()
        return teacher, student

    def step(self, batch):
        student_ko_batch, student_en_batch, teacher_en_batch = batch
        student_ko_emb = self.student.get_text_features(**student_ko_batch)
        student_en_emb = self.student.get_text_features(**student_en_batch)
        teacher_en_emb = self.teacher.get_text_features(**teacher_en_batch)
        '''
        target = torch.ones(student_ko_emb.size(0), device=self.device)
        st_loss = self.cosine_loss(student_ko_emb, teacher_en_emb, target)
        en_loss = self.cosine_loss(student_en_emb, teacher_en_emb, target)

        student_ko_emb_norm = F.normalize(student_ko_emb, p=2, dim=1)
        student_en_emb_norm = F.normalize(student_en_emb, p=2, dim=1)
        teacher_en_emb_norm = F.normalize(teacher_en_emb, p=2, dim=1)
        st_loss = self.mse(student_ko_emb_norm, teacher_en_emb_norm)
        en_loss = self.mse(student_en_emb_norm, teacher_en_emb_norm)
        '''
        st_loss = self.mse(student_ko_emb, teacher_en_emb)
        en_loss = self.mse(student_en_emb, teacher_en_emb)

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
        params = list(self.student.text_model.named_parameters())

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in params if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in params if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        opt_class = self.create_optimizer(self.optimizer)
        signiture = inspect.signature(opt_class)
        opt_kwargs = {}
        if "capturable" in signiture.parameters:
            opt_kwargs["capturable"] = True
        if "weight_decouple" in signiture.parameters:
            opt_kwargs["weight_decouple"] = True
        if "decouple_decay" in signiture.parameters:
            opt_kwargs["decouple_decay"] = True

        optimizer = opt_class(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            **opt_kwargs,
        )

        num_devices = max(1, self.trainer.num_devices)
        effective_batch_size = self.trainer.datamodule.batch_size * num_devices
        steps_per_epoch = math.ceil(len(self.trainer.datamodule.train_dataloader().dataset) / effective_batch_size)
        total_training_steps = steps_per_epoch * self.trainer.max_epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            self.learning_rate,
            total_steps=total_training_steps,
        )

        scheduler_config = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [scheduler_config]

    def on_train_epoch_end(self):
        self.save(f"save/test_siglip2base_siglip2base_{self.hparams.loss_type}_epoch-{self.current_epoch}")

    def save(self, save_dir: str = "save/my_model"):
        self.student.save_pretrained(save_dir)
        processor = AutoProcessor.from_pretrained(self.student_model_name)
        processor.save_pretrained(save_dir)

