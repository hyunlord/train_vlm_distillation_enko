import inspect
from itertools import chain
from loguru import logger

import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoFeatureExtractor, AutoTokenizer, AutoModel, VisionTextDualEncoderModel, AutoProcessor

from .util import create_optimizer


class EnKoDistillationModule(pl.LightningModule):
    def __init__(self, teacher_model_name: str, student_model_name: str, optimizer: str = "adamw", learning_rate: float = 5e-4, weight_decay: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.teacher_model_name = teacher_model_name
        self.student_model_name = student_model_name
        self.teacher_vision_model, self.teacher_text_model, self.student_text_model = self.init_model(teacher_model_name, student_model_name)

        self.vision_projection_dim = self.teacher_vision_model.config.hidden_size
        self.text_projection_dim = self.student_text_model.config.hidden_size
        self.text_projection = nn.Linear(self.text_projection_dim, self.vision_projection_dim)

        self.mse = torch.nn.MSELoss()
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def init_model(self, teacher_model_name: str, student_model_name: str):
        teacher_model = AutoModel.from_pretrained(teacher_model_name)
        for param in teacher_model.parameters():
            param.requires_grad = False
        teacher_model.eval()

        teacher_vision_model = teacher_model.vision_model
        teacher_text_model = teacher_model.text_model
        student_text_model = AutoModel.from_pretrained(student_model_name)
        return teacher_vision_model, teacher_text_model, student_text_model

    def configure_optimizers(self):
        params = list(
            chain(
                self.student_text_model.named_parameters(),
                self.text_projection.named_parameters(),
            )
        )

        #no_decay = ["bias", "LayerNorm.weight"]
        no_decay = []
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

        opt_class = create_optimizer(self.optimizer)
        optimizer = opt_class(
            optimizer_grouped_parameters,
            lr=self.learning_rate
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
            anneal_strategy='cos',
            div_factor=25,
            final_div_factor=1e4
        )
        scheduler_config = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [scheduler_config]

    def step(self, batch):
        student_ko_batch, student_en_batch, teacher_en_batch = batch

        student_ko_emb = self.text_projection(self.student_text_model(**student_ko_batch)[1])
        student_en_emb = self.text_projection(self.student_text_model(**student_en_batch)[1])
        teacher_en_emb = self.teacher_text_model(**teacher_en_batch)[1]

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

    def on_epoch_end(self):
        epoch_save_dir = f"save/sigilp2_sroberta_epoch_{self.current_epoch}"
        self.save(epoch_save_dir)
        logger.info(f"model saved at: {epoch_save_dir}")

    def save(self, save_dir: str):
        self.student.save_pretrained(save_dir)
