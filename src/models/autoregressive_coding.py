# src/models/autoregressive_coding.py
"""Autoregressive Coding for lossless LLM-based compression."""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import math


class AutoregressiveCoding:
    def __init__(self, model_name: str = "gpt2", device: str = "cuda"):
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def calculate_compressed_size(self, text: str, batch_size: int = 1) -> Dict[str, float]:
        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=1024
        ).to(self.device)
        
        input_ids = encoding['input_ids']
        total_nll_bits = 0.0
        num_tokens = 0
        
        with torch.no_grad():
            for i in range(0, input_ids.shape[1] - 1, batch_size):
                batch_start = i
                batch_end = min(i + batch_size, input_ids.shape[1] - 1)
                input_batch = input_ids[:, :batch_end]
                target_batch = input_ids[:, batch_start + 1:batch_end + 1]
                outputs = self.model(input_batch)
                logits = outputs.logits[:, batch_start:batch_end]
                log_probs = torch.log_softmax(logits, dim=-1)
                target_log_probs = log_probs.gather(
                    dim=-1,
                    index=target_batch.unsqueeze(-1)
                ).squeeze(-1)
                nll_bits = -target_log_probs / math.log(2)
                total_nll_bits += nll_bits.sum().item()
                num_tokens += target_batch.numel()
                
        original_size_bits = len(text.encode('utf-8')) * 8
        compressed_size_bits = total_nll_bits
        bits_per_token = (total_nll_bits / num_tokens) if num_tokens > 0 else 0.0
        bits_per_char = (total_nll_bits / len(text)) if len(text) > 0 else 0.0
        
        compression_ratio = original_size_bits / compressed_size_bits if compressed_size_bits > 0 else float('inf')
        
        return {
            'original_size_bits': original_size_bits,
            'compressed_size_bits': compressed_size_bits,
            'compression_ratio': compression_ratio,
            'bits_per_token': bits_per_token,
            'bits_per_character': bits_per_char,
            'num_tokens': num_tokens,
            'num_characters': len(text),
            'perplexity': 2 ** bits_per_token 
        }
    
    def calculate_cross_entropy(self, texts: List[str]) -> float:
        total_bits = 0.0
        total_tokens = 0
        
        for text in tqdm(texts, desc="Calculating cross-entropy"):
            metrics = self.calculate_compressed_size(text)
            total_bits += metrics['compressed_size_bits']
            total_tokens += metrics['num_tokens']
            
        return total_bits / total_tokens if total_tokens > 0 else 0.0
    
    def compare_with_actual_compression(self, text: str) -> Dict[str, float]:
        import gzip
        import bz2
        import lzma
        ar_metrics = self.calculate_compressed_size(text)
        text_bytes = text.encode('utf-8')
        gzip_compressed = gzip.compress(text_bytes, compresslevel=9)
        bz2_compressed = bz2.compress(text_bytes, compresslevel=9)
        lzma_compressed = lzma.compress(text_bytes, preset=9)
        original_bits = len(text_bytes) * 8
        gzip_bits = len(gzip_compressed) * 8
        bz2_bits = len(bz2_compressed) * 8
        lzma_bits = len(lzma_compressed) * 8
        
        return {
            'original_size_bits': original_bits,
            'ar_theoretical_bits': ar_metrics['compressed_size_bits'],
            'ar_bits_per_char': ar_metrics['bits_per_character'],
            'gzip_bits': gzip_bits,
            'gzip_bits_per_char': gzip_bits / len(text),
            'bz2_bits': bz2_bits,
            'bz2_bits_per_char': bz2_bits / len(text),
            'lzma_bits': lzma_bits,
            'lzma_bits_per_char': lzma_bits / len(text),
            'ar_vs_gzip_ratio': ar_metrics['compressed_size_bits'] / gzip_bits,
            'ar_vs_bz2_ratio': ar_metrics['compressed_size_bits'] / bz2_bits,
            'ar_vs_lzma_ratio': ar_metrics['compressed_size_bits'] / lzma_bits
        }
    
    def estimate_model_size(self) -> Dict[str, float]:
        total_params = 0
        total_bytes = 0
        
        for param in self.model.parameters():
            total_params += param.numel()
            # Assuming float32 (4 bytes per parameter)
            total_bytes += param.numel() * 4
            
        return {
            'total_parameters': total_params,
            'model_size_mb': total_bytes / (1024 * 1024),
            'model_size_bits': total_bytes * 8
        }
    
    def theoretical_compression_analysis(self, texts: List[str], 
                                       sample_size: int = 100) -> Dict[str, any]:
        # Sample texts if needed
        if len(texts) > sample_size:
            import random
            texts = random.sample(texts, sample_size)
            
        results = {
            'num_texts': len(texts),
            'total_characters': 0,
            'total_tokens': 0,
            'total_original_bits': 0,
            'total_compressed_bits': 0,
            'per_text_metrics': []
        }
        
        for text in tqdm(texts, desc="Analyzing compression"):
            metrics = self.calculate_compressed_size(text)
            results['per_text_metrics'].append(metrics)
            
            results['total_characters'] += metrics['num_characters']
            results['total_tokens'] += metrics['num_tokens']
            results['total_original_bits'] += metrics['original_size_bits']
            results['total_compressed_bits'] += metrics['compressed_size_bits']
            
        # Calculate averages
        results['avg_bits_per_character'] = results['total_compressed_bits'] / results['total_characters']
        results['avg_bits_per_token'] = results['total_compressed_bits'] / results['total_tokens']
        results['avg_compression_ratio'] = results['total_original_bits'] / results['total_compressed_bits']
        results['avg_perplexity'] = 2 ** results['avg_bits_per_token']
        
        # Add model size
        results['model_info'] = self.estimate_model_size()
        
        return results
