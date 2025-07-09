# src/models/__init__.py
"""Compression model implementations."""

from .base_compressor import BaseCompressor
from .predictive_masking import PredictiveMaskingCompressor
from .latent_space_quantization import LatentSpaceQuantizationCompressor

__all__ = [
    "BaseCompressor",
    "PredictiveMaskingCompressor",
    "LatentSpaceQuantizationCompressor",
]
