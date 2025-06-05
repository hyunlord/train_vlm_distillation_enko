from typing import Optional
from typer import Option, Typer

from loguru import logger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, RichProgressBar
from pytorch_lightning.strategies import DDPStrategy

from app.dataset import EnKoDistillationDataModule
from app.module import EnKoDistillationModule
from app.siglip_enko_distillation_module import SiglipEnKoDistillationModule

cmd = Typer()


@cmd.command()
def train(
        teacher_model_name: str = Option(
            "/hanmail/.cache/gitlfs/siglip2-base-patch16-224",
            "-t", "--teacher",
            help="name of teacher model", rich_help_panel="model"
        ),
        student_model_name: str = Option(
            "jhgan/ko-sroberta-multitask",
            "-s", "--student",
            help="name of student model", rich_help_panel="model"
        ),
        optimizer: str = Option(
            "adamw",
            "-o", "--optimizer",
            help="optimizer name", rich_help_panel="model"
        ),
        learning_rate: float = Option(
            1e-5,
            "-lr", "--learning-rate",
            help="learning_rate",  rich_help_panel="model"
        ),
        weight_decay: float = Option(
            1e-4,
            "-wd4", "--weight-decay",
            help="weight decay", rich_help_panel="model"
        ),
        batch_size: int = Option(
            32,
            "-b", "--batch-size",
            min=1, help="batch size", rich_help_panel="model"
        ),
        num_workers: int = Option(
            0, min=0,
            help="num_workers", rich_help_panel="train"
        ),
        max_epochs: int = Option(
            5,
            help="max_epochs", rich_help_panel="train"
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
    datamodule = EnKoDistillationDataModule(
        teacher_model_name,
        student_model_name,
        batch_size=batch_size,
        num_workers=num_workers
    )

    module = SiglipEnKoDistillationModule(
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
        precision=32,
        max_epochs=max_epochs,
        callbacks=callbacks,
        log_every_n_steps=log_every_n_steps,
        strategy=DDPStrategy(find_unused_parameters=True)
    )

    logger.debug("start training")
    trainer.fit(module, datamodule=datamodule)
    logger.debug("training finished")


if __name__ == "__main__":
    cmd()
