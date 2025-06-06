from datetime import datetime
from enum import Enum
from typing import Optional

import pytorch_lightning as pl
from loguru import logger
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.tuner import Tuner
from pytorch_lightning.strategies import DDPStrategy
from typer import Option, Typer

from app.dataset2 import KoCLIPDataModule
from app.module2 import KoCLIPModule

cmd = Typer(pretty_exceptions_show_locals=False)


class ModelTypeEnum(str, Enum):
    CLIP = "clip"
    DUAL_ENCODER = "dual_encoder"


@cmd.command()
def train(
    teacher_model_name: str = Option(
        "openai/clip-vit-base-patch32",
        "-t",
        "--teacher",
        help="name of teacher model",
        rich_help_panel="model",
    ),
    student_model_name: str = Option(
        "jhgan/ko-sroberta-multitask",
        "-s",
        "--student",
        help="name of student model",
        rich_help_panel="model",
    ),
    use_auth_token: bool = Option(
        False, help="use auth token", rich_help_panel="model"
    ),
    model_type: ModelTypeEnum = Option(
        "clip",
        "-m",
        "--model-type",
        help="model type",
        rich_help_panel="model",
        case_sensitive=False,
    ),
    optimizer: str = Option(
        "adamw", "-o", "--optimizer", help="optimizer name", rich_help_panel="model"
    ),
    learning_rate: float = Option(
        5e-4,
        "--lr",
        help="learning rate",
        rich_help_panel="model",
    ),
    weight_decay: float = Option(
        1e-4, "-wd", "--weight-decay", help="weight decay", rich_help_panel="model"
    ),
    batch_size: int = Option(
        32, "-b", "--batch-size", min=1, help="batch size", rich_help_panel="model"
    ),
    num_workers: int = Option(
        0, min=0, help="num workers of dataloader", rich_help_panel="train"
    ),
    accumulate_grad_batches: Optional[int] = Option(
        None, min=1, help="accumulate grad batches", rich_help_panel="train"
    ),
    gradient_clip_val: Optional[float] = Option(
        None, min=0.0, help="gradient clip value", rich_help_panel="train"
    ),
    auto_scale_batch_size: bool = Option(
        False,
        help="auto find batch size, ignore batch_size option",
        rich_help_panel="train",
    ),
    max_epochs: int = Option(5, help="max epochs", rich_help_panel="train"),
    steps_per_epoch: Optional[int] = Option(
        None, min=1, help="steps per epoch", rich_help_panel="train"
    ),
    fast_dev_run: bool = Option(False, help="do test run", rich_help_panel="train"),
    save_path: str = Option(
        "save/my_model", help="save path of trained model", rich_help_panel="train"
    ),
    log_every_n_steps: int = Option(
        100, help="log every n steps", rich_help_panel="train"
    ),
    resume_from_checkpoint: Optional[str] = Option(
        None,
        help="Path/URL of the checkpoint from which training is resumed",
        rich_help_panel="train",
    ),
    wandb_name: Optional[str] = Option(
        None, help="wandb project name", rich_help_panel="train"
    ),
    seed: Optional[int] = Option(None, help="seed", rich_help_panel="train"),
):
    logger.debug("loading dataset")
    datamodule = KoCLIPDataModule(
        teacher_model_name,
        student_model_name,
        batch_size=batch_size,
        num_workers=num_workers,
        use_auth_token=use_auth_token,
    )
    logger.debug("loading model")
    module = KoCLIPModule(
        teacher_model_name,
        student_model_name,
        optimizer=optimizer,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        use_auth_token=use_auth_token,
    )

    checkpoints = ModelCheckpoint(monitor="train/loss_epoch", save_last=True)
    callbacks = [checkpoints, RichProgressBar(), LearningRateMonitor()]

    if seed is not None:
        pl.seed_everything(seed)
        logger.debug(f"set seed: {seed}")

    limit_train_batches = steps_per_epoch if steps_per_epoch else 1.0
    if isinstance(limit_train_batches, int) and accumulate_grad_batches is not None:
        limit_train_batches *= accumulate_grad_batches

    logger.debug("set trainer")
    trainer = pl.Trainer(
        accelerator="auto",
        precision=16 if "bnb" not in optimizer else 32,
        max_epochs=max_epochs,
        callbacks=callbacks,
        log_every_n_steps=log_every_n_steps,
        strategy=DDPStrategy(find_unused_parameters=True)
    )
    logger.debug("start training")
    trainer.fit(module, datamodule=datamodule)
    logger.debug("training finished")

    module.save(save_path)
    logger.info(f"model saved at: {save_path}")


if __name__ == "__main__":
    cmd()
