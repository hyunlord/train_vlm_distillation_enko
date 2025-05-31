from enum import Enum
from typing import Optional
from typer import Option, Typer

from loguru import logger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, RichProgressBar

from app.dataset import KoSiglipDataModule
from app.module import KoSiglipModule

cmd = Typer()


class ModelTypeEnum(str, Enum):
    CLIP = "clip"
    SIGLIP = "siglip"


@cmd.command()
def train(
        teacher_model_name: str = Option(
            "google/siglip-so400m-patch14-384",
            "-t", "--teacher",
            help="name of teacher model", rich_help_panel="model"
        ),
        student_model_name: str = Option(
            "google/siglip-so400m-patch14-384",
            "-s", "--student",
            help="name of student model", rich_help_panel="model"
        ),
        optimizer: str = Option(
            "adamw",
            "-o", "--optimizer",
            help="optimizer name", rich_help_panel="model"
        ),
        learning_rate: float = Option(
            5e-4,
            "-lr", "--learning-rate",
            help="learning_rate",  rich_help_panel="model"
        ),
        weight_decay: float = Option(
            1e-4,
            "-wd4", "--weight-decay",
            help="weight decay", rich_help_panel="model"
        ),
        batch_size: int = Option(
            16,
            "-b", "--batch-size",
            min=1, help="batch size", rich_help_panel="model"
        ),
        num_workers: int = Option(
            0, min=0,
            help="accumulate grad batches", rich_help_panel="train"
        ),
        max_epochs: int = Option(
            3,
            help="max_epochs", rich_help_panel="train"
        ),
        save_path: str = Option(
            "save/model",
            help="save path of trained model", rich_help_panel="train"
        ),
        log_every_n_steps: int = Option(
            100,
            help="log every n steps", rich_help_panel="train"
        ),
        seed: Optional[int] = Option(
            None,
            help="seed", rich_help_panel="train"
        )
):
    logger.debug("loading dataset")
    datamodule = KoSiglipDataModule(
        teacher_model_name,
        student_model_name,
        batch_size=batch_size,
        num_workers=num_workers
    )

    module = KoSiglipModule(
        teacher_model_name,
        student_model_name,
        optimizer=optimizer,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )

    checkpoints = ModelCheckpoint(monitor="train/loss_epoch", save_last=True)
    callbacks = [checkpoints, RichProgressBar(), LearningRateMonitor()]
    if seed is not None:
        pl.seed_everything(seed)
        logger.debug(f"set seed: {seed}")

    logger.debug("set trainer")
    trainer = pl.Trainer(
        accelerator="auto",
        precision='16-mixed',
        fast_dev_run=True,
        max_epochs=max_epochs,
        callbacks=callbacks,
        log_every_n_steps=log_every_n_steps,
    )

    logger.debug("start training")
    trainer.fit(module, datamodule=datamodule)
    logger.debug("training finished")

    module.save(save_path)
    logger.info(f"model saved at: {save_path}")


if __name__ == "__main__":
    cmd()
