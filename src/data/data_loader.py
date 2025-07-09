# src/data/data_loader.py
"""Data loading utilities for compression experiments."""

from datasets import load_dataset
from typing import List, Dict, Optional, Tuple
import random
import torch
from torch.utils.data import Dataset, DataLoader


class CompressionDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "text": text,
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
        }


class DataLoader:
    def __init__(
        self, dataset_name: str = "wikitext", dataset_config: str = "wikitext-2-raw-v1"
    ):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config

    def load_data(
        self,
        train_size: int = 10000,
        test_size: int = 1000,
        min_length: int = 50,
        max_length: int = 512,
    ) -> Tuple[List[str], List[str]]:
        dataset = load_dataset(self.dataset_name, self.dataset_config)
        train_texts = self._prepare_texts(
            dataset["train"], train_size, min_length, max_length
        )
        test_texts = self._prepare_texts(
            dataset["test"], test_size, min_length, max_length
        )
        return train_texts, test_texts

    def _prepare_texts(
        self, dataset_split, size: int, min_length: int, max_length: int
    ) -> List[str]:
        texts = []
        for item in dataset_split:
            text = item["text"] if isinstance(item, dict) else str(item)
            text = text.strip()
            if len(text) < min_length:
                continue
            if len(text) > max_length:
                text = text[:max_length]
            texts.append(text)
            if len(texts) >= size:
                break
        return texts

    def load_custom_data(
        self, file_path: str, train_ratio: float = 0.9
    ) -> Tuple[List[str], List[str]]:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        texts = [line.strip() for line in lines if line.strip()]
        random.shuffle(texts)
        split_idx = int(len(texts) * train_ratio)
        train_texts = texts[:split_idx]
        test_texts = texts[split_idx:]
        return train_texts, test_texts

    def create_data_loaders(
        self,
        train_texts: List[str],
        test_texts: List[str],
        tokenizer,
        batch_size: int = 16,
        max_length: int = 512,
    ) -> Tuple[DataLoader, DataLoader]:
        train_dataset = CompressionDataset(train_texts, tokenizer, max_length)
        test_dataset = CompressionDataset(test_texts, tokenizer, max_length)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        return train_loader, test_loader

    def get_diverse_test_samples(
        self, test_texts: List[str], num_samples: int = 100
    ) -> List[str]:
        if len(test_texts) <= num_samples:
            return test_texts
        sorted_texts = sorted(test_texts, key=len)
        indices = [int(i * len(sorted_texts) / num_samples) for i in range(num_samples)]
        diverse_samples = [sorted_texts[i] for i in indices]
        return diverse_samples
