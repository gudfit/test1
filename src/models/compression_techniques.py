# src/models/compression_techniques.py 
"""Model compression techniques: Pruning and Quantization."""

import time
from typing import Dict, List, Any

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from transformers import AutoModelForMaskedLM, AutoTokenizer
import bitsandbytes as bnb


class ModelCompressor:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.original_model = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def load_original_model(self) -> AutoModelForMaskedLM:
        if self.original_model is None:
            self.original_model = AutoModelForMaskedLM.from_pretrained(self.model_name).to(self.device)
        return self.original_model

    def apply_magnitude_pruning(self, model: nn.Module, sparsity: float) -> nn.Module:
        pruned_model = AutoModelForMaskedLM.from_pretrained(self.model_name).to(self.device)
        pruned_model.load_state_dict(model.state_dict())
        parameters = [(m, "weight") for m in pruned_model.modules() if isinstance(m, (nn.Linear, nn.Embedding))]
        prune.global_unstructured(parameters, prune.L1Unstructured, amount=sparsity)
        for m, n in parameters:
            prune.remove(m, n)
        return pruned_model

    def apply_structured_pruning(self, model: nn.Module, sparsity: float) -> nn.Module:
        pruned_model = AutoModelForMaskedLM.from_pretrained(self.model_name).to(self.device)
        pruned_model.load_state_dict(model.state_dict())
        for module in pruned_model.modules():
            if isinstance(module, nn.Linear):
                prune.ln_structured(module, "weight", amount=sparsity, n=2, dim=0)
                prune.remove(module, "weight")
        return pruned_model

    def quantize_model(self, model: nn.Module, bits: int = 8) -> Dict[str, Any]:
        if bits == 16:
            q_model = model.half().to(self.device)
            return self._quant_stats(model, q_model, bits)
        if bits == 8:
            q_model = bnb.nn.utils.convert_linear_layers(model.cpu(), dtype=torch.int8)
            return self._quant_stats(model, q_model, bits)
        if bits == 4:
            q_model = bnb.nn.utils.convert_linear_layers(model.cpu(), dtype=torch.float16, quant_type="nf4")
            return self._quant_stats(model, q_model, bits)
        raise ValueError("bits must be 4, 8, or 16")

    def create_compression_variants(self, pruning_levels: List[float] | None = None, quantization_bits: List[int] | None = None) -> Dict[str, Any]:
        if pruning_levels is None:
            pruning_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        if quantization_bits is None:
            quantization_bits = [16, 8, 4]
        variants: Dict[str, Any] = {}
        original = self.load_original_model()
        variants["original"] = {"model": original, "type": "original", "compression_ratio": 1.0, "size_mb": self._model_size(original)}
        for s in pruning_levels:
            p = self.apply_magnitude_pruning(original, s)
            variants[f"pruned_{s}"] = {"model": p, "type": "pruned", "sparsity": s, "compression_ratio": 1 / (1 - s), "size_mb": self._model_size(p)}
        for b in quantization_bits:
            q = self.quantize_model(original, b)
            variants[f"quantized_{b}bit"] = {"model": q["model"], "type": "quantized", "bits": b, "compression_ratio": q["compression_ratio"], "size_mb": q["quantized_size_mb"]}
        for s in [0.3, 0.5]:
            p = self.apply_magnitude_pruning(original, s)
            for b in [8, 4]:
                q = self.quantize_model(p, b)
                variants[f"pruned_{s}_quantized_{b}bit"] = {"model": q["model"], "type": "combined", "sparsity": s, "bits": b, "compression_ratio": q["compression_ratio"] / (1 - s), "size_mb": q["quantized_size_mb"]}
        return variants

    def measure_inference_performance(self, model: nn.Module, test_texts: List[str], batch_size: int = 8) -> Dict[str, float]:
        model.eval()
        inputs = self.tokenizer(test_texts[:batch_size], padding=True, truncation=True, max_length=128, return_tensors="pt").to(next(model.parameters()).device)
        with torch.no_grad():
            for _ in range(5):
                _ = model(**inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        runs = 50
        with torch.no_grad():
            for _ in range(runs):
                _ = model(**inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.time()
        latency = (end - start) / runs * 1000
        throughput = batch_size / ((end - start) / runs)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            _ = model(**inputs)
            mem = torch.cuda.max_memory_allocated() / 1048576
        else:
            mem = self._model_size(model)
        return {"avg_latency_ms": latency, "throughput_samples_per_sec": throughput, "memory_footprint_mb": mem, "batch_size": batch_size}

    def count_nonzero_parameters(self, model: nn.Module) -> Dict[str, int]:
        total = sum(p.numel() for p in model.parameters())
        nonzero = sum(torch.count_nonzero(p).item() for p in model.parameters())
        return {"total_parameters": total, "nonzero_parameters": nonzero, "sparsity": 1 - nonzero / total, "compression_ratio": total / nonzero if nonzero else float("inf")}

    def _model_size(self, model: nn.Module) -> float:
        return sum(p.numel() * p.element_size() for p in model.parameters()) / 1048576

    def _quant_stats(self, orig: nn.Module, q: nn.Module, bits: int) -> Dict[str, Any]:
        o = self._model_size(orig)
        q_size = self._model_size(q)
        return {"model": q.to(self.device), "bits": bits, "original_size_mb": o, "quantized_size_mb": q_size, "compression_ratio": o / q_size}

    def _calculate_model_size(self, model: nn.Module) -> float:
        return self._model_size(model)

