import inspect
from loguru import logger

import torch
import pytorch_lightning as pl
from transformers import AutoFeatureExtractor, AutoTokenizer, AutoModel, VisionTextDualEncoderModel, AutoProcessor

from .util import create_optimizer


class EnKoDistillationModule(pl.LightningModule):
    def __init__(self, teacher_model_name: str, student_model_name: str, optimizer: str = "adamw", learning_rate: float = 5e-4, weight_decay: float = 1e-4):
        super().__init__()
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
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False

        student = VisionTextDualEncoderModel.from_vision_text_pretrained(teacher_model_name, student_model_name)
        student.logit_scale = teacher.logit_scale
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
                "params": [p for n, p in params if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
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

    def step(self, batch):
        student_ko_batch, student_en_batch, teacher_en_batch = batch

        student_ko_emb = self.student.text_model(**student_ko_batch)[1]
        student_en_emb = self.student.text_model(**student_en_batch)[1]
        teacher_en_emb = self.teacher.text_model(**teacher_en_batch)[1]

        s_t_loss = self.mse(student_ko_emb, teacher_en_emb)
        en_loss = self.mse(student_en_emb, teacher_en_emb)
        loss = s_t_loss + en_loss

        loss_dict = {
            "loss": loss,
            "loss_st": s_t_loss,
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
        epoch_save_dir = f"save/sigilp2_sroberta_no-train-teacher_epoch_{self.current_epoch}"
        self.save(epoch_save_dir)
        logger.info(f"model saved at: {epoch_save_dir}")

    def save(self, save_dir: str = "save/model"):
        self.student.save_pretrained(save_dir)
