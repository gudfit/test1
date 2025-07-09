# src/models/base_compressor.py
"""Base class for LLM-based compression methods."""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np
import inspect

class BaseCompressor(ABC):
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @abstractmethod
    def compress(self, text: str, **kwargs) -> Dict[str, any]:
        pass

    @abstractmethod
    def decompress(self, compressed_data: Dict[str, any]) -> str:
        pass

    def calculate_compression_ratio(self, original_text: str, compressed_data: Dict[str, any]) -> float:
        original_size = len(original_text.encode("utf-8")) * 8
        compressed_size = self._calculate_compressed_size(compressed_data)
        return original_size / compressed_size if compressed_size > 0 else float("inf")

    @abstractmethod
    def _calculate_compressed_size(self, compressed_data: Dict[str, any]) -> int:
        pass

    def fine_tune(self, texts: List[str], epochs: int = 3, learning_rate: float | str = 5e-5, batch_size: int = 16, warmup_steps: int = 500):
        from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
        from torch.utils.data import Dataset

        class TextDataset(Dataset):
            def __init__(self, texts, tokenizer, max_length=512):
                self.encodings = []
                for text in texts:
                    encoding = tokenizer(text, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
                    self.encodings.append({k: v.squeeze() for k, v in encoding.items()})
            def __len__(self):
                return len(self.encodings)
            def __getitem__(self, idx):
                return self.encodings[idx]

        learning_rate = float(learning_rate)
        dataset = TextDataset(texts, self.tokenizer)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15)

        base_kwargs = dict(
            output_dir=f"./fine_tuned_{self.model_name}",
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=100,
            save_steps=1000,
            load_best_model_at_end=False,
            fp16=torch.cuda.is_available(),
        )
        extra_kwargs = {"evaluation_strategy": "no", "save_strategy": "steps"}
        sig = inspect.signature(TrainingArguments.__init__).parameters
        base_kwargs.update({k: v for k, v in extra_kwargs.items() if k in sig})
        training_args = TrainingArguments(**base_kwargs)

        trainer = Trainer(model=self.model, args=training_args, data_collator=data_collator, train_dataset=dataset)
        trainer.train()
        self.model = trainer.model

    def save_model(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_model(self, path: str):
        self.model = AutoModelForMaskedLM.from_pretrained(path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(path)

