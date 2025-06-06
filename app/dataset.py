from collections.abc import Sequence
from typing import TYPE_CHECKING

import pytorch_lightning as pl
from datasets import Dataset as HFDataset
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast

if TYPE_CHECKING:
    from transformers import BatchEncoding


class EnKoDistillationDataset(Dataset):
    def __init__(self, ds: HFDataset, teacher_tokenizer: PreTrainedTokenizerFast, student_tokenizer: PreTrainedTokenizerFast):
        self.ds = ds
        self.teacher_tokenizer = teacher_tokenizer
        self.student_tokenizer = student_tokenizer

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int) -> tuple["BatchEncoding", "BatchEncoding", "BatchEncoding"]:
        ko: str = self.ds[idx]["ko"]
        en: str = self.ds[idx]["en"]
        return ko, en, en


class EnKoDistillationDataCollator:
    def __init__(self, teacher_tokenizer: PreTrainedTokenizerFast, student_tokenizer: PreTrainedTokenizerFast):
        self.teacher_tokenizer = teacher_tokenizer
        self.student_tokenizer = student_tokenizer
        
    def __call__(self, features: Sequence[tuple["BatchEncoding", "BatchEncoding", "BatchEncoding"]]):
        student_ko_texts, student_en_texts, teacher_en_texts = zip(*features)
        student_ko_batch = self.student_tokenizer(list(student_ko_texts), padding="max_length", truncation=True, max_length=64, return_tensors="pt")
        student_en_batch = self.student_tokenizer(list(student_en_texts), padding="max_length", truncation=True, max_length=64, return_tensors="pt")
        teacher_en_batch = self.teacher_tokenizer(list(teacher_en_texts), padding="max_length", truncation=True, max_length=64, return_tensors="pt")
        return student_ko_batch, student_en_batch, teacher_en_batch


class EnKoDistillationDataModule(pl.LightningDataModule):
    def __init__(self, teacher_tokenizer_name: str, student_tokenizer_name: str, batch_size: int = 32, num_workers: int = 8):
        super().__init__()
        self.teacher_tokenizer_name = teacher_tokenizer_name
        self.student_tokenizer_name = student_tokenizer_name
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # "traintogpb/aihub-koen-translation-integrated-large-10m"
        load_dataset("hyunlord/aihub_ko-en_parallel_corpus_collection", split="train+validation")
        
    def setup(self, stage=None):
        ds: HFDataset = load_dataset("hyunlord/aihub_ko-en_parallel_corpus_collection", split="train+validation")
        teacher_tokenizer = AutoTokenizer.from_pretrained(self.teacher_tokenizer_name)
        student_tokenizer = AutoTokenizer.from_pretrained(self.student_tokenizer_name)
        self.data_collator = EnKoDistillationDataCollator(teacher_tokenizer, student_tokenizer)
        self.train_dataset = EnKoDistillationDataset(ds, teacher_tokenizer, student_tokenizer)
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True
        )
