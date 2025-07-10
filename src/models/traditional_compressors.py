# src/models/traditional_compressors.py
"""Traditional compression algorithms wrapper for benchmarking."""

import gzip
import bz2
import lzma
import zstandard as zstd
import time
from typing import Dict, List, Tuple, Optional
import numpy as np


class TraditionalCompressors:
    def __init__(self):
        self.compressors = {
            'gzip': {
                'compress': lambda data: gzip.compress(data, compresslevel=9),
                'decompress': gzip.decompress,
                'description': 'Gzip (LZ77-based)'
            },
            'bzip2': {
                'compress': lambda data: bz2.compress(data, compresslevel=9),
                'decompress': bz2.decompress,
                'description': 'Bzip2 (Burrows-Wheeler transform)'
            },
            'lzma': {
                'compress': lambda data: lzma.compress(data, preset=9),
                'decompress': lzma.decompress,
                'description': 'LZMA (Lempel-Ziv-Markov chain)'
            },
            'zstd': {
                'compress': lambda data: zstd.compress(data, 22),  # Max compression level
                'decompress': zstd.decompress,
                'description': 'Zstandard (modern LZ-variant)'
            }
        }
        
    def compress_text(self, text: str, algorithm: str) -> Dict[str, any]:
        if algorithm not in self.compressors:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        text_bytes = text.encode('utf-8')
        original_size = len(text_bytes)
        start_time = time.time()
        compressed_data = self.compressors[algorithm]['compress'](text_bytes)
        compression_time = time.time() - start_time
        compressed_size = len(compressed_data)
        compression_ratio = original_size / compressed_size if compressed_size > 0 else float('inf')
        bits_per_character = (compressed_size * 8) / len(text) if len(text) > 0 else 0
        
        return {
            'algorithm': algorithm,
            'original_size_bytes': original_size,
            'compressed_size_bytes': compressed_size,
            'compression_ratio': compression_ratio,
            'bits_per_character': bits_per_character,
            'compression_time_seconds': compression_time,
            'compressed_data': compressed_data
        }
    
    def decompress_text(self, compressed_data: bytes, algorithm: str) -> Tuple[str, float]:
        if algorithm not in self.compressors:
            raise ValueError(f"Unknown algorithm: {algorithm}")
            
        start_time = time.time()
        decompressed_bytes = self.compressors[algorithm]['decompress'](compressed_data)
        decompression_time = time.time() - start_time
        
        decompressed_text = decompressed_bytes.decode('utf-8')
        
        return decompressed_text, decompression_time
    
    def benchmark_all_algorithms(self, texts: List[str]) -> Dict[str, Dict]:
        results = {}
        
        for algorithm in self.compressors:
            print(f"Benchmarking {algorithm}...")
            
            total_original_size = 0
            total_compressed_size = 0
            total_compression_time = 0
            total_decompression_time = 0
            bits_per_char_list = []
            
            for text in texts:
                compress_result = self.compress_text(text, algorithm)
                total_original_size += compress_result['original_size_bytes']
                total_compressed_size += compress_result['compressed_size_bytes']
                total_compression_time += compress_result['compression_time_seconds']
                bits_per_char_list.append(compress_result['bits_per_character'])
                decompressed, decomp_time = self.decompress_text(
                    compress_result['compressed_data'], algorithm
                )
                total_decompression_time += decomp_time
                assert decompressed == text, f"{algorithm} decompression failed!"
            results[algorithm] = {
                'description': self.compressors[algorithm]['description'],
                'total_original_size_mb': total_original_size / (1024 * 1024),
                'total_compressed_size_mb': total_compressed_size / (1024 * 1024),
                'avg_compression_ratio': total_original_size / total_compressed_size,
                'avg_bits_per_character': np.mean(bits_per_char_list),
                'std_bits_per_character': np.std(bits_per_char_list),
                'total_compression_time': total_compression_time,
                'total_decompression_time': total_decompression_time,
                'avg_compression_speed_mbps': (total_original_size / (1024 * 1024)) / total_compression_time,
                'avg_decompression_speed_mbps': (total_original_size / (1024 * 1024)) / total_decompression_time,
                'is_lossless': True,
                'system_size_mb': 0.001  # Traditional compressors are tiny (~1KB)
            }
            
        return results
    
    def compare_algorithms(self, text: str) -> Dict[str, Dict]:
        results = {}
        for algorithm in self.compressors:
            compress_result = self.compress_text(text, algorithm)
            del compress_result['compressed_data']
            
            results[algorithm] = compress_result
        best_algorithm = min(results.keys(), 
                           key=lambda x: results[x]['compressed_size_bytes'])
        for algorithm in results:
            results[algorithm]['vs_best_ratio'] = (
                results[algorithm]['compressed_size_bytes'] / 
                results[best_algorithm]['compressed_size_bytes']
            )
            
        return results
    
    def get_algorithm_info(self) -> Dict[str, str]:
        return {
            algo: info['description'] 
            for algo, info in self.compressors.items()
        }
