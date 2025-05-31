import inspect
from itertools import chain
from typing import Literal

import pytorch_lightning as pl
import torch
from loguru import logger
from transformers import AutoFeatureExtractor, AutoTokenizer, AutoModel, VisionTextDualEncoderModel, AutoProcessor

from .util import create_optimizer


class KoSiglipModule(pl.LightningModule):
    def __init__(self, teacher_model_name: str, student_model_name: str, optimizer: str="adamw", learning_rate: float=5e-4, weight_decay: float=1e-4):
        super().__iniit__()
        self.save_hyperparameters()

        self.teacher_model_name = teacher_model_name
        self.student_model_name = student_model_name
        self.teacher, self.student = self.init_model(teacher_model_name, student_model_name)

        self.mse = torch.nn.MSELoss()
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def init_model(self, teacher_model_name: str, student_model_name: str):
        teacher = AutoModel.from_pretrained(teacher_model_name)
        student = AutoModel.from_pretrained(student_model_name)
        return teacher, student

    def configure_optimizers(self):
        params = list(self.student.text_model.named_parameters())

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in params if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay
            },
            {
                "params": [p for n, p in params if not any(nd in n for nd in no_decay)],
                "weight_decay": 0
            }
        ]

        opt_class = create_optimizer(self.optimizer)
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

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            self.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches
        )
        scheduler_config = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [scheduler_config]

    def training_step(self, batch, batch_idx):
        ko_batch, en_ko_batch, en_en_batch = batch

        ko_emb = self.student.text_model(**ko_batch)[1]
        en_ko_emb = self.student.text_model(**en_ko_batch)[1]
        en_en_emb = self.teacher.text_model(**en_en_batch)[1]

        ko_en_loss = self.mse(ko_emb, en_en_emb)
        en_en_loss = self.mse(en_ko_emb, en_en_emb)
        loss = ko_en_loss + en_en_loss

        loss_dict = {
            "loss": loss,
            "loss_ko": ko_en_loss,
            "loss_en": en_en_loss
        }
        return loss_dict

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)

        self.log({
            "val/loss": loss["loss"],
            "val/loss_ko": loss["loss_ko"],
            "val/loss_en": loss["loss_en"]
        }, on_epoch=True)
        return loss["loss"]

    def save(self, save_dir: str="save/model"):
        self.student.save_pretrained(save_dir)
        processor = AutoProcessor.from_pretrained(self.student_model_name)
        processor.save_pretrained(save_dir)
