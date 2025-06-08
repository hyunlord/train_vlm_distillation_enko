import math
import inspect
from itertools import chain

import torch
import pytorch_lightning as pl
from torch.optim import SGD, Adam, AdamW
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoFeatureExtractor, AutoProcessor
from transformers import AutoImageProcessor, SiglipProcessor

from .combine_model_config import CombinedModelConfig, CombinedModel


class EnKoDistillationModule(pl.LightningModule):
    def __init__(self,
                 teacher_model_name: str,
                 student_model_name: str,
                 model_type: str = "siglipe2",
                 optimizer: str = "adamw",
                 learning_rate: float = 5e-4,
                 weight_decay: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.teacher_model_name = teacher_model_name
        self.student_model_name = student_model_name

        teacher_vision_config = AutoConfig.from_pretrained(self.teacher_model_name).vision_config
        student_text_config = AutoConfig.from_pretrained(self.student_model_name)
        self.combined_config = CombinedModelConfig(
            teacher_model_name_or_path=self.teacher_model_name,
            student_model_name_or_path=self.student_model_name,
            vision_projection_dim=teacher_vision_config.hidden_size,
            text_projection_dim=student_text_config.hidden_size
        )
        self.combined_model = CombinedModel(config=self.combined_config)

        self.mse = torch.nn.MSELoss()
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.teacher_model = AutoModel.from_pretrained(self.teacher_model_name)
        for param in self.teacher_text_model.parameters():
            param.requires_grad = False
        self.teacher_text_model.eval()

    def step(self, batch):
        student_ko_batch, student_en_batch, teacher_en_batch = batch
        student_ko_emb = self.combined_model.get_text_features(**student_ko_batch)
        student_en_emb = self.combined_model.get_text_features(**student_en_batch)
        teacher_en_emb = self.teacher_model.get_text_features(**teacher_en_batch)

        st_loss = self.mse(student_ko_emb, teacher_en_emb)
        en_loss = self.mse(student_en_emb, teacher_en_emb)
        loss = st_loss + en_loss

        loss_dict = {
            "loss": loss,
            "loss_st": st_loss,
            "loss_en": en_loss
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
        self.log(
            {
                "val/loss": loss["loss"],
                "val/loss_st": loss["loss_st"],
                "val/loss_en": loss["loss_en"]
            }, on_epoch=True
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
        params = list(
            chain(
                self.combined_model.text_model.named_parameters(),
                self.combined_model.text_projection.named_parameters(),
            )
        )
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in params if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay
            },
            {
                "params": [p for n, p in params if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            }
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
            **opt_kwargs
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

    def save(self, save_dir: str = "save/my_model"):
        self.combined_model.save_pretrained(save_dir)

        tokenizer = AutoTokenizer.from_pretrained(self.student_model_name)
        image_processor = AutoImageProcessor.from_pretrained(self.teacher_model_name)
        processor = SiglipProcessor(image_processor=image_processor, tokenizer=tokenizer)
        processor.save_pretrained(save_dir)
