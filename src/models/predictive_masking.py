# src/models/predictive_masking.py
"""Predictive Masking compression implementation."""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from .base_compressor import BaseCompressor
import random


class PredictiveMaskingCompressor(BaseCompressor):
    """Compression via predictive masking of tokens."""

    def __init__(self, model_name: str, device: str = "cuda"):
        super().__init__(model_name, device)
        self.mask_token = self.tokenizer.mask_token
        self.mask_token_id = self.tokenizer.mask_token_id

    def compress(
        self, text: str, masking_probability: float = 0.5, top_k: Optional[int] = None
    ) -> Dict[str, any]:
        encoding = self.tokenizer(
            text, truncation=True, padding=False, return_tensors="pt", max_length=512
        )
        input_ids = encoding["input_ids"][0]
        attention_mask = encoding["attention_mask"][0]
        masked_ids = input_ids.clone()
        mask_positions: List[int] = []
        special_tokens = {
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id,
        }
        for i, tok in enumerate(input_ids):
            if (
                tok.item() not in special_tokens
                and random.random() < masking_probability
            ):
                masked_ids[i] = self.mask_token_id
                mask_positions.append(i)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids.unsqueeze(0).to(self.device),
                attention_mask=attention_mask.unsqueeze(0).to(self.device),
            )
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1)
            token_probs = probs[torch.arange(probs.size(0)), input_ids]
        token_probs_np = token_probs.cpu().numpy()
        sorted_idx = np.argsort(-token_probs_np)
        if top_k is not None:
            sorted_idx = sorted_idx[:top_k]

        return {
            "masked_ids": masked_ids.cpu().numpy(),
            "mask_positions": np.array(mask_positions),
            "original_length": len(input_ids),
            "masking_probability": masking_probability,
            "attention_mask": attention_mask.cpu().numpy(),
            "token_probabilities": token_probs_np,
            "most_predictable_positions": sorted_idx,
            "most_predictable_scores": token_probs_np[sorted_idx],
        }

    def decompress(self, compressed_data: Dict[str, any]) -> str:
        masked_ids = (
            torch.tensor(compressed_data["masked_ids"]).unsqueeze(0).to(self.device)
        )
        attention_mask = (
            torch.tensor(compressed_data["attention_mask"]).unsqueeze(0).to(self.device)
        )
        mask_positions = compressed_data["mask_positions"]
        with torch.no_grad():
            outputs = self.model(input_ids=masked_ids, attention_mask=attention_mask)
            predictions = outputs.logits
        reconstructed_ids = masked_ids.clone()

        for pos in mask_positions:
            predicted_id = predictions[0, pos].argmax().item()
            reconstructed_ids[0, pos] = predicted_id
        reconstructed_text = self.tokenizer.decode(
            reconstructed_ids[0], skip_special_tokens=True
        )

        return reconstructed_text

    def _calculate_compressed_size(self, compressed_data: Dict[str, any]) -> int:
        masked_ids = compressed_data["masked_ids"]
        mask_positions = compressed_data["mask_positions"]
        original_length = compressed_data["original_length"]
        num_unmasked = original_length - len(mask_positions)
        unmasked_bits = num_unmasked * 32
        if len(mask_positions) > 0:
            position_bits = int(np.ceil(np.log2(original_length))) * len(mask_positions)
        else:
            position_bits = 0

        total_bits = unmasked_bits + position_bits

        return total_bits

    def adaptive_masking(self, text: str, target_ratio: float = 2.0) -> Dict[str, any]:
        low, high = 0.0, 1.0
        best_prob = 0.5
        best_ratio = 0.0

        for _ in range(10):
            mid = (low + high) / 2
            compressed = self.compress(text, masking_probability=mid)
            ratio = self.calculate_compression_ratio(text, compressed)
            if abs(ratio - target_ratio) < abs(best_ratio - target_ratio):
                best_prob = mid
                best_ratio = ratio
            if ratio < target_ratio:
                low = mid
            else:
                high = mid
        return self.compress(text, masking_probability=best_prob)

    def confidence_based_masking(
        self, text: str, confidence_threshold: float = 0.9
    ) -> Dict[str, any]:
        encoding = self.tokenizer(
            text, truncation=True, padding=False, return_tensors="pt", max_length=512
        )
        input_ids = encoding["input_ids"][0]
        attention_mask = encoding["attention_mask"][0]
        masked_ids = input_ids.clone()
        mask_positions = []
        for i in range(len(input_ids)):
            if input_ids[i].item() in {
                self.tokenizer.cls_token_id,
                self.tokenizer.sep_token_id,
            }:
                continue
            temp_masked = input_ids.clone()
            temp_masked[i] = self.mask_token_id
            with torch.no_grad():
                outputs = self.model(
                    input_ids=temp_masked.unsqueeze(0).to(self.device),
                    attention_mask=attention_mask.unsqueeze(0).to(self.device),
                )
                probs = torch.softmax(outputs.logits[0, i], dim=-1)
                confidence = probs[input_ids[i]].item()
            if confidence >= confidence_threshold:
                masked_ids[i] = self.mask_token_id
                mask_positions.append(i)
        compressed_data = {
            "masked_ids": masked_ids.cpu().numpy(),
            "mask_positions": np.array(mask_positions),
            "original_length": len(input_ids),
            "masking_probability": len(mask_positions) / len(input_ids),
            "attention_mask": attention_mask.cpu().numpy(),
            "confidence_threshold": confidence_threshold,
        }

        return compressed_data
