# src/models/latent_space_quantization.py
"""Latent Space Quantization compression implementation."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from .base_compressor import BaseCompressor


class QuantizationDecoder(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, vocab_size),
        )

    def forward(self, hidden_states):
        return self.decoder(hidden_states)


class LatentSpaceQuantizationCompressor(BaseCompressor):
    def __init__(
        self, model_name: str, device: str = "cuda", quantization_bits: int = 8
    ):
        super().__init__(model_name, device)
        self.quantization_bits = quantization_bits
        self.decoder = None
        self._init_decoder()

    def _init_decoder(self):
        config = self.model.config
        self.decoder = QuantizationDecoder(
            hidden_size=config.hidden_size, vocab_size=config.vocab_size
        ).to(self.device)

    def compress(
        self, text: str, quantization_bits: Optional[int] = None
    ) -> Dict[str, any]:
        if quantization_bits is None:
            quantization_bits = self.quantization_bits
        encoding = self.tokenizer(
            text, truncation=True, padding=False, return_tensors="pt", max_length=512
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        with torch.no_grad():
            outputs = (
                self.model.bert(input_ids=input_ids, attention_mask=attention_mask)
                if hasattr(self.model, "bert")
                else (
                    self.model.roberta(
                        input_ids=input_ids, attention_mask=attention_mask
                    )
                    if hasattr(self.model, "roberta")
                    else self.model.distilbert(
                        input_ids=input_ids, attention_mask=attention_mask
                    )
                )
            )
            hidden_states = outputs.last_hidden_state

        H = hidden_states.cpu().numpy()
        min_val = H.min()
        max_val = H.max()
        scale_factor = (max_val - min_val) / (2**quantization_bits - 1)
        H_normalized = (H - min_val) / scale_factor
        H_quantized = np.round(H_normalized).astype(
            np.uint8 if quantization_bits == 8 else np.uint16
        )

        compressed_data = {
            "quantized_states": H_quantized,
            "scale_factor": scale_factor,
            "min_val": min_val,
            "shape": H.shape,
            "attention_mask": attention_mask.cpu().numpy(),
            "quantization_bits": quantization_bits,
        }

        return compressed_data

    def decompress(self, compressed_data: Dict[str, any]) -> str:
        H_quantized = compressed_data["quantized_states"]
        scale_factor = compressed_data["scale_factor"]
        min_val = compressed_data["min_val"]
        attention_mask = torch.tensor(compressed_data["attention_mask"]).to(self.device)
        H_dequantized = H_quantized.astype(np.float32) * scale_factor + min_val
        H_dequantized = torch.tensor(H_dequantized).to(self.device)
        with torch.no_grad():
            logits = self.decoder(H_dequantized)
        predicted_ids = logits.argmax(dim=-1)
        predicted_ids = predicted_ids * attention_mask
        reconstructed_text = self.tokenizer.decode(
            predicted_ids[0], skip_special_tokens=True
        )

        return reconstructed_text

    def _calculate_compressed_size(self, compressed_data: Dict[str, any]) -> int:
        quantized_states = compressed_data["quantized_states"]
        quantization_bits = compressed_data["quantization_bits"]
        num_elements = quantized_states.size
        state_bits = num_elements * quantization_bits
        metadata_bits = 2 * 32
        return state_bits + metadata_bits

    def calculate_model_size(self) -> dict[str, float]:
        """
        Return the approximate disk footprint (FP‑32 weights) and parameter count
        for the encoder *and* the small decoder we train on top.
        """
        param_count = (
            sum(p.numel() for p in self.model.parameters()) +
            sum(p.numel() for p in self.decoder.parameters())
        )
        # 4 bytes per fp32 parameter  → MB
        disk_size_mb = (param_count * 4) / (1024 ** 2)

        # add extra tables / codebooks here if you store any
        extras_mb = 0.0
        return {
            "disk_size_mb": disk_size_mb + extras_mb,
            "param_count":  param_count,
        }

    def train_decoder(
        self,
        texts: List[str],
        epochs: int = 10,
        batch_size: int = 16,
        learning_rate: float = 1e-3,
    ):
        optimizer = torch.optim.Adam(self.decoder.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        self.decoder.train()

        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]

                encoding = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    max_length=512,
                )

                input_ids = encoding["input_ids"].to(self.device)
                attention_mask = encoding["attention_mask"].to(self.device)

                with torch.no_grad():
                    outputs = (
                        self.model.bert(
                            input_ids=input_ids, attention_mask=attention_mask
                        )
                        if hasattr(self.model, "bert")
                        else (
                            self.model.roberta(
                                input_ids=input_ids, attention_mask=attention_mask
                            )
                            if hasattr(self.model, "roberta")
                            else self.model.distilbert(
                                input_ids=input_ids, attention_mask=attention_mask
                            )
                        )
                    )
                    hidden_states = outputs.last_hidden_state

                logits = self.decoder(hidden_states)
                loss = criterion(logits.view(-1, logits.size(-1)), input_ids.view(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        self.decoder.eval()

    def vector_quantization(self, text: str, num_clusters: int = 256) -> Dict[str, any]:
        from sklearn.cluster import KMeans

        encoding = self.tokenizer(
            text, truncation=True, return_tensors="pt", max_length=512
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = (
                self.model.bert(input_ids=input_ids, attention_mask=attention_mask)
                if hasattr(self.model, "bert")
                else (
                    self.model.roberta(
                        input_ids=input_ids, attention_mask=attention_mask
                    )
                    if hasattr(self.model, "roberta")
                    else self.model.distilbert(
                        input_ids=input_ids, attention_mask=attention_mask
                    )
                )
            )
            hidden_states = outputs.last_hidden_state[0].cpu().numpy()
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_indices = kmeans.fit_predict(hidden_states)

        compressed_data = {
            "cluster_indices": cluster_indices,
            "cluster_centers": kmeans.cluster_centers_,
            "shape": hidden_states.shape,
            "attention_mask": attention_mask.cpu().numpy(),
            "num_clusters": num_clusters,
        }
        return compressed_data
