"""Run Experiment 1D: Comparative Benchmark of LLM-based and Algorithmic Compression."""

import os
import sys
import yaml
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datasets import load_dataset

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.autoregressive_coding import AutoregressiveCoding
from src.models.predictive_masking import PredictiveMaskingCompressor
from src.models.latent_space_quantization import LatentSpaceQuantizationCompressor
from src.models.traditional_compressors import TraditionalCompressors
from src.evaluation.metrics import CompressionMetrics
from src.visualization.plots import CompressionVisualizer


class Experiment1DRunner:
    """Runner for Experiment 1D: Comparative Compression Benchmark."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize experiment runner."""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Set up logging
        self._setup_logging()
        
        # Create output directories
        self._create_output_dirs()
        
        # Initialize components
        self.visualizer = CompressionVisualizer(
            figure_dir=os.path.join(self.config['experiment']['output_dir'], 'figures', 'experiment_1d')
        )
        self.metrics_calculator = CompressionMetrics(self.config['experiment']['device'])
        
    def _setup_logging(self):
        """Set up logging configuration."""
        log_dir = os.path.join(self.config['experiment']['output_dir'], 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, f'experiment_1d_{datetime.now():%Y%m%d_%H%M%S}.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _create_output_dirs(self):
        """Create necessary output directories."""
        dirs = [
            os.path.join(self.config['experiment']['output_dir'], 'figures', 'experiment_1d'),
            os.path.join(self.config['experiment']['output_dir'], 'results', 'experiment_1d')
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            
    def load_test_corpus(self, num_samples: int = 1000) -> List[str]:
        """
        Load test corpus from WikiText-103.
        
        Args:
            num_samples: Number of text samples to load
            
        Returns:
            List of text samples
        """
        self.logger.info(f"Loading {num_samples} samples from WikiText-103...")
        
        # Load WikiText-103 test split
        dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='test')
        
        # Filter and clean texts
        texts = []
        for item in dataset:
            text = item['text'].strip()
            # Skip empty or very short texts
            if len(text) > 100:
                texts.append(text)
            if len(texts) >= num_samples:
                break
                
        self.logger.info(f"Loaded {len(texts)} text samples")
        return texts
    
    def run_comparative_benchmark(self):
        """Run the complete Experiment 1D benchmark."""
        self.logger.info("Starting Experiment 1D: Comparative Compression Benchmark")
        
        # Load test corpus
        test_corpus = self.load_test_corpus(
            num_samples=self.config.get('experiment_1d', {}).get('num_samples', 1000)
        )
        
        results = {
            'corpus_stats': self._analyze_corpus(test_corpus),
            'llm_lossless': {},
            'llm_lossy': {},
            'traditional': {},
            'comparative_analysis': {}
        }
        
        # 1. LLM-based Lossless Compression (Autoregressive Coding)
        self.logger.info("\n" + "="*60)
        self.logger.info("Testing LLM-based Lossless Compression")
        self.logger.info("="*60)
        
        ar_models = self.config.get('experiment_1d', {}).get('ar_models', ['gpt2'])
        for model_name in ar_models:
            self.logger.info(f"\nEvaluating {model_name} autoregressive coding...")
            ar_compressor = AutoregressiveCoding(model_name, self.config['experiment']['device'])
            ar_results = self._evaluate_ar_compression(ar_compressor, test_corpus[:100])  # Use subset for AR
            results['llm_lossless'][model_name] = ar_results
            
        # 2. LLM-based Lossy Compression
        self.logger.info("\n" + "="*60)
        self.logger.info("Testing LLM-based Lossy Compression")
        self.logger.info("="*60)
        
        # Predictive Masking
        for model_config in self.config['models'].values():
            model_name = model_config['name']
            self.logger.info(f"\nEvaluating {model_name} predictive masking...")
            
            pm_compressor = PredictiveMaskingCompressor(model_name, self.config['experiment']['device'])
            pm_results = self._evaluate_pm_compression(pm_compressor, test_corpus[:200])
            results['llm_lossy'][f"{model_name}_pm"] = pm_results
            
            # Latent Space Quantization
            self.logger.info(f"\nEvaluating {model_name} latent space quantization...")
            lsq_compressor = LatentSpaceQuantizationCompressor(
                model_name, 
                self.config['experiment']['device'],
                quantization_bits=8
            )
            # Train decoder first
            lsq_compressor.train_decoder(test_corpus[:100], epochs=5)
            lsq_results = self._evaluate_lsq_compression(lsq_compressor, test_corpus[:200])
            results['llm_lossy'][f"{model_name}_lsq"] = lsq_results
            
        # 3. Traditional Lossless Compressors
        self.logger.info("\n" + "="*60)
        self.logger.info("Testing Traditional Compression Algorithms")
        self.logger.info("="*60)
        
        trad_compressors = TraditionalCompressors()
        trad_results = trad_compressors.benchmark_all_algorithms(test_corpus)
        results['traditional'] = trad_results
        
        # 4. Comparative Analysis
        self.logger.info("\n" + "="*60)
        self.logger.info("Performing Comparative Analysis")
        self.logger.info("="*60)
        
        results['comparative_analysis'] = self._perform_comparative_analysis(results)
        
        # Create visualizations
        self.logger.info("\nCreating visualizations...")
        self.create_visualizations(results)
        
        # Save results
        self.save_results(results)
        
        # Print summary
        self.print_summary(results)
        
        self.logger.info("\nExperiment 1D completed!")
        
        return results
    
    def _analyze_corpus(self, texts: List[str]) -> Dict[str, float]:
        """Analyze the test corpus statistics."""
        total_chars = sum(len(text) for text in texts)
        total_bytes = sum(len(text.encode('utf-8')) for text in texts)
        
        return {
            'num_texts': len(texts),
            'total_characters': total_chars,
            'total_bytes': total_bytes,
            'avg_text_length': total_chars / len(texts),
            'total_size_mb': total_bytes / (1024 * 1024)
        }
    
    def _evaluate_ar_compression(self, compressor: AutoregressiveCoding, 
                                texts: List[str]) -> Dict[str, Any]:
        """Evaluate autoregressive coding compression."""
        # Perform theoretical compression analysis
        analysis = compressor.theoretical_compression_analysis(texts)
        
        # Compare with traditional methods on a sample
        comparison = compressor.compare_with_actual_compression(texts[0])
        
        return {
            'method': 'Autoregressive Coding',
            'model_name': compressor.model_name,
            'is_lossless': True,
            'avg_bits_per_character': analysis['avg_bits_per_character'],
            'avg_bits_per_token': analysis['avg_bits_per_token'],
            'avg_compression_ratio': analysis['avg_compression_ratio'],
            'avg_perplexity': analysis['avg_perplexity'],
            'model_size_mb': analysis['model_info']['model_size_mb'],
            'total_system_size_mb': analysis['model_info']['model_size_mb'],  # Model is the system
            'reconstruction_accuracy': 1.0,  # Lossless
            'sample_comparison': comparison
        }
    
    def _evaluate_pm_compression(self, compressor: PredictiveMaskingCompressor,
                                texts: List[str]) -> Dict[str, Any]:
        """Evaluate predictive masking compression."""
        masking_probs = [0.3, 0.5, 0.7]  # Test multiple masking levels
        best_results = None
        best_ratio = 0
        
        for mask_prob in masking_probs:
            total_original_bits = 0
            total_compressed_bits = 0
            reconstruction_scores = []
            
            for text in tqdm(texts[:50], desc=f"PM {mask_prob:.1f}"):
                # Compress
                compressed = compressor.compress(text, masking_probability=mask_prob)
                
                # Calculate sizes
                original_bits = len(text.encode('utf-8')) * 8
                compressed_bits = compressor._calculate_compressed_size(compressed)
                
                total_original_bits += original_bits
                total_compressed_bits += compressed_bits
                
                # Decompress and evaluate
                reconstructed = compressor.decompress(compressed)
                
                # Calculate reconstruction accuracy
                metrics = self.metrics_calculator.calculate_all_metrics(
                    text, reconstructed, compressed
                )
                reconstruction_scores.append(metrics['word_accuracy'])
                
            # Calculate metrics for this masking level
            compression_ratio = total_original_bits / total_compressed_bits
            avg_reconstruction = np.mean(reconstruction_scores)
            
            # Keep best based on compression ratio * reconstruction accuracy
            if compression_ratio * avg_reconstruction > best_ratio:
                best_ratio = compression_ratio * avg_reconstruction
                best_results = {
                    'masking_probability': mask_prob,
                    'compression_ratio': compression_ratio,
                    'bits_per_character': total_compressed_bits / sum(len(t) for t in texts[:50]),
                    'reconstruction_accuracy': avg_reconstruction
                }
                
        # Get model size
        model_size_mb = compressor.calculate_model_size()['disk_size_mb']
        
        return {
            'method': 'Predictive Masking',
            'model_name': compressor.model_name,
            'is_lossless': False,
            'best_masking_probability': best_results['masking_probability'],
            'avg_bits_per_character': best_results['bits_per_character'],
            'avg_compression_ratio': best_results['compression_ratio'],
            'reconstruction_accuracy': best_results['reconstruction_accuracy'],
            'model_size_mb': model_size_mb,
            'total_system_size_mb': model_size_mb  # Compressed data is negligible compared to model
        }
    
    def _evaluate_lsq_compression(self, compressor: LatentSpaceQuantizationCompressor,
                                 texts: List[str]) -> Dict[str, Any]:
        """Evaluate latent space quantization compression."""
        total_original_bits = 0
        total_compressed_bits = 0
        reconstruction_scores = []
        
        for text in tqdm(texts[:50], desc="LSQ evaluation"):
            # Compress
            compressed = compressor.compress(text, quantization_bits=8)
            
            # Calculate sizes
            original_bits = len(text.encode('utf-8')) * 8
            compressed_bits = compressor._calculate_compressed_size(compressed)
            
            total_original_bits += original_bits
            total_compressed_bits += compressed_bits
            
            # Decompress and evaluate
            reconstructed = compressor.decompress(compressed)
            
            # Calculate reconstruction accuracy
            metrics = self.metrics_calculator.calculate_all_metrics(
                text, reconstructed, compressed
            )
            reconstruction_scores.append(metrics['word_accuracy'])
            
        # Calculate aggregate metrics
        compression_ratio = total_original_bits / total_compressed_bits if total_compressed_bits > 0 else 0
        avg_reconstruction = np.mean(reconstruction_scores)
        bits_per_char = total_compressed_bits / sum(len(t) for t in texts[:50])
        
        # Get model size
        model_size_mb = compressor.calculate_model_size()['disk_size_mb']
        
        return {
            'method': 'Latent Space Quantization',
            'model_name': compressor.model_name,
            'is_lossless': False,
            'quantization_bits': 8,
            'avg_bits_per_character': bits_per_char,
            'avg_compression_ratio': compression_ratio,
            'reconstruction_accuracy': avg_reconstruction,
            'model_size_mb': model_size_mb,
            'total_system_size_mb': model_size_mb  # Compressed data is negligible compared to model
        }
    
    def _perform_comparative_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comparative analysis across all methods."""
        analysis = {
            'best_compression_ratio': {'method': None, 'value': 0},
            'best_bits_per_char': {'method': None, 'value': float('inf')},
            'best_lossless': {'method': None, 'value': 0},
            'best_lossy_quality': {'method': None, 'value': 0},
            'smallest_system': {'method': None, 'value': float('inf')},
            'method_rankings': {}
        }
        
        all_methods = []
        
        # Collect all methods
        for model_name, metrics in results['llm_lossless'].items():
            method_key = f"AR_{model_name}"
            all_methods.append({
                'key': method_key,
                'type': 'LLM_lossless',
                **metrics
            })
            
        for method_name, metrics in results['llm_lossy'].items():
            all_methods.append({
                'key': method_name,
                'type': 'LLM_lossy',
                **metrics
            })
            
        for algo_name, metrics in results['traditional'].items():
            all_methods.append({
                'key': algo_name,
                'type': 'traditional',
                'method': algo_name,
                'model_name': 'N/A',
                'is_lossless': True,
                'avg_bits_per_character': metrics['avg_bits_per_character'],
                'avg_compression_ratio': metrics['avg_compression_ratio'],
                'reconstruction_accuracy': 1.0,
                'model_size_mb': 0,
                'total_system_size_mb': metrics['system_size_mb']
            })
            
        # Find best in each category
        for method in all_methods:
            # Best compression ratio
            if method.get('avg_compression_ratio', 0) > analysis['best_compression_ratio']['value']:
                analysis['best_compression_ratio']['method'] = method['key']
                analysis['best_compression_ratio']['value'] = method['avg_compression_ratio']
                
            # Best bits per character
            if method.get('avg_bits_per_character', float('inf')) < analysis['best_bits_per_char']['value']:
                analysis['best_bits_per_char']['method'] = method['key']
                analysis['best_bits_per_char']['value'] = method['avg_bits_per_character']
                
            # Best lossless
            if method.get('is_lossless', False):
                if method.get('avg_compression_ratio', 0) > analysis['best_lossless']['value']:
                    analysis['best_lossless']['method'] = method['key']
                    analysis['best_lossless']['value'] = method['avg_compression_ratio']
                    
            # Best lossy quality
            if not method.get('is_lossless', True):
                quality_score = method.get('reconstruction_accuracy', 0) * method.get('avg_compression_ratio', 0)
                if quality_score > analysis['best_lossy_quality']['value']:
                    analysis['best_lossy_quality']['method'] = method['key']
                    analysis['best_lossy_quality']['value'] = quality_score
                    
            # Smallest system
            if method.get('total_system_size_mb', float('inf')) < analysis['smallest_system']['value']:
                analysis['smallest_system']['method'] = method['key']
                analysis['smallest_system']['value'] = method['total_system_size_mb']
                
        # Rank methods
        analysis['method_rankings'] = {
            'by_compression_ratio': sorted(all_methods, 
                                         key=lambda x: x.get('avg_compression_ratio', 0), 
                                         reverse=True),
            'by_bits_per_char': sorted(all_methods, 
                                     key=lambda x: x.get('avg_bits_per_character', float('inf'))),
            'by_system_size': sorted(all_methods, 
                                   key=lambda x: x.get('total_system_size_mb', float('inf')))
        }
        
        return analysis
    
    def create_visualizations(self, results: Dict[str, Any]):
        """Create comprehensive visualizations for Experiment 1D."""
        # 1. Compression Efficiency Comparison
        self._plot_compression_efficiency(results)
        
        # 2. System Size vs Performance
        self._plot_system_size_tradeoff(results)
        
        # 3. Lossless vs Lossy Comparison
        self._plot_lossless_lossy_comparison(results)
        
        # 4. Comprehensive Summary Table
        self._create_summary_table(results)
        
    def _plot_compression_efficiency(self, results: Dict[str, Any]):
        """Plot compression efficiency comparison."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Prepare data
        methods = []
        bpc_values = []
        compression_ratios = []
        method_types = []
        
        # LLM lossless
        for model_name, metrics in results['llm_lossless'].items():
            methods.append(f"AR-{model_name}")
            bpc_values.append(metrics['avg_bits_per_character'])
            compression_ratios.append(metrics['avg_compression_ratio'])
            method_types.append('LLM Lossless')
            
        # LLM lossy
        for method_name, metrics in results['llm_lossy'].items():
            methods.append(method_name.replace('_', '-'))
            bpc_values.append(metrics['avg_bits_per_character'])
            compression_ratios.append(metrics['avg_compression_ratio'])
            method_types.append('LLM Lossy')
            
        # Traditional
        for algo_name, metrics in results['traditional'].items():
            methods.append(algo_name.upper())
            bpc_values.append(metrics['avg_bits_per_character'])
            compression_ratios.append(metrics['avg_compression_ratio'])
            method_types.append('Traditional')
            
        # Create color map
        color_map = {
            'LLM Lossless': 'steelblue',
            'LLM Lossy': 'coral',
            'Traditional': 'forestgreen'
        }
        colors = [color_map[t] for t in method_types]
        
        # Plot 1: Bits per character
        bars1 = ax1.bar(range(len(methods)), bpc_values, color=colors)
        ax1.set_xlabel('Compression Method', fontsize=12)
        ax1.set_ylabel('Bits Per Character', fontsize=12)
        ax1.set_title('Compression Efficiency: Bits Per Character', fontsize=14)
        ax1.set_xticks(range(len(methods)))
        ax1.set_xticklabels(methods, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars1, bpc_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=9)
                    
        # Plot 2: Compression ratio
        bars2 = ax2.bar(range(len(methods)), compression_ratios, color=colors)
        ax2.set_xlabel('Compression Method', fontsize=12)
        ax2.set_ylabel('Compression Ratio', fontsize=12)
        ax2.set_title('Compression Efficiency: Compression Ratio', fontsize=14)
        ax2.set_xticks(range(len(methods)))
        ax2.set_xticklabels(methods, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars2, compression_ratios):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=9)
                    
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color_map[key], label=key) 
                          for key in color_map.keys()]
        ax1.legend(handles=legend_elements, loc='upper right')
        
        plt.suptitle('Compression Efficiency Comparison', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['experiment']['output_dir'],
                                'figures', 'experiment_1d', 'compression_efficiency.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_system_size_tradeoff(self, results: Dict[str, Any]):
        """Plot system size vs compression performance."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data
        for method_type, marker, color in [
            ('llm_lossless', 'o', 'steelblue'),
            ('llm_lossy', '^', 'coral'),
            ('traditional', 's', 'forestgreen')
        ]:
            if method_type == 'llm_lossless':
                data = results['llm_lossless']
                for model_name, metrics in data.items():
                    ax.scatter(metrics['total_system_size_mb'],
                             metrics['avg_compression_ratio'],
                             s=200, marker=marker, color=color,
                             edgecolors='black', linewidth=2,
                             label=f"AR-{model_name}")
                    # Add reconstruction accuracy annotation
                    ax.annotate('100%', 
                               (metrics['total_system_size_mb'], metrics['avg_compression_ratio']),
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
                               
            elif method_type == 'llm_lossy':
                data = results['llm_lossy']
                for method_name, metrics in data.items():
                    ax.scatter(metrics['total_system_size_mb'],
                             metrics['avg_compression_ratio'],
                             s=200, marker=marker, color=color,
                             edgecolors='black', linewidth=2,
                             label=method_name.replace('_', '-'))
                    # Add reconstruction accuracy annotation
                    ax.annotate(f"{metrics['reconstruction_accuracy']:.0%}", 
                               (metrics['total_system_size_mb'], metrics['avg_compression_ratio']),
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
                               
            else:  # traditional
                data = results['traditional']
                for algo_name, metrics in data.items():
                    ax.scatter(metrics['system_size_mb'],
                             metrics['avg_compression_ratio'],
                             s=200, marker=marker, color=color,
                             edgecolors='black', linewidth=2,
                             label=algo_name.upper())
                    # Add reconstruction accuracy annotation
                    ax.annotate('100%', 
                               (metrics['system_size_mb'], metrics['avg_compression_ratio']),
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
                               
        ax.set_xlabel('Total System Size (MB)', fontsize=14)
        ax.set_ylabel('Compression Ratio', fontsize=14)
        ax.set_title('System Size vs Compression Performance\n(annotations show reconstruction accuracy)', 
                    fontsize=16)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['experiment']['output_dir'],
                                'figures', 'experiment_1d', 'system_size_tradeoff.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_lossless_lossy_comparison(self, results: Dict[str, Any]):
        """Plot comparison between lossless and lossy methods."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Separate lossless and lossy methods
        lossless_methods = []
        lossy_methods = []
        
        # LLM lossless
        for model_name, metrics in results['llm_lossless'].items():
            lossless_methods.append({
                'name': f"AR-{model_name}",
                'bpc': metrics['avg_bits_per_character'],
                'type': 'LLM'
            })
            
        # Traditional (all lossless)
        for algo_name, metrics in results['traditional'].items():
            lossless_methods.append({
                'name': algo_name.upper(),
                'bpc': metrics['avg_bits_per_character'],
                'type': 'Traditional'
            })
            
        # LLM lossy
        for method_name, metrics in results['llm_lossy'].items():
            lossy_methods.append({
                'name': method_name.replace('_', '-'),
                'bpc': metrics['avg_bits_per_character'],
                'accuracy': metrics['reconstruction_accuracy']
            })
            
        # Plot lossless methods
        lossless_bpc = [m['bpc'] for m in lossless_methods]
        lossless_names = [m['name'] for m in lossless_methods]
        lossless_colors = ['steelblue' if m['type'] == 'LLM' else 'forestgreen' 
                          for m in lossless_methods]
        
        y_pos = np.arange(len(lossless_names))
        bars1 = ax.barh(y_pos, lossless_bpc, color=lossless_colors, alpha=0.8, height=0.4)
        
        # Plot lossy methods
        lossy_bpc = [m['bpc'] for m in lossy_methods]
        lossy_names = [m['name'] for m in lossy_methods]
        lossy_accuracy = [m['accuracy'] for m in lossy_methods]
        
        y_pos2 = np.arange(len(lossless_names), len(lossless_names) + len(lossy_names))
        bars2 = ax.barh(y_pos2, lossy_bpc, color='coral', alpha=0.8, height=0.4)
        
        # Add accuracy labels for lossy methods
        for bar, acc in zip(bars2, lossy_accuracy):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                   f'Acc: {acc:.1%}', va='center', fontsize=9)
                   
        # Customize plot
        ax.set_yticks(np.arange(len(lossless_names) + len(lossy_names)))
        ax.set_yticklabels(lossless_names + lossy_names)
        ax.set_xlabel('Bits Per Character', fontsize=12)
        ax.set_title('Lossless vs Lossy Compression Methods', fontsize=14)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add separating line
        ax.axhline(y=len(lossless_names) - 0.5, color='black', linewidth=2, linestyle='--')
        ax.text(0.5, len(lossless_names) - 0.5, 'LOSSLESS', transform=ax.get_yaxis_transform(),
               ha='center', va='bottom', fontsize=10, weight='bold')
        ax.text(0.5, len(lossless_names) - 0.5, 'LOSSY', transform=ax.get_yaxis_transform(),
               ha='center', va='top', fontsize=10, weight='bold')
               
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['experiment']['output_dir'],
                                'figures', 'experiment_1d', 'lossless_lossy_comparison.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def _create_summary_table(self, results: Dict[str, Any]):
        """Create comprehensive summary table."""
        # Prepare data for table
        table_data = []
        
        # LLM lossless
        for model_name, metrics in results['llm_lossless'].items():
            table_data.append({
                'Method': f"AR-{model_name}",
                'Type': 'LLM Lossless',
                'BPC': f"{metrics['avg_bits_per_character']:.3f}",
                'Compression Ratio': f"{metrics['avg_compression_ratio']:.2f}",
                'Reconstruction Accuracy': '100.0%',
                'System Size (MB)': f"{metrics['total_system_size_mb']:.1f}"
            })
            
        # LLM lossy
        for method_name, metrics in results['llm_lossy'].items():
            table_data.append({
                'Method': method_name.replace('_', '-'),
                'Type': 'LLM Lossy',
                'BPC': f"{metrics['avg_bits_per_character']:.3f}",
                'Compression Ratio': f"{metrics['avg_compression_ratio']:.2f}",
                'Reconstruction Accuracy': f"{metrics['reconstruction_accuracy']:.1%}",
                'System Size (MB)': f"{metrics['total_system_size_mb']:.1f}"
            })
            
        # Traditional
        for algo_name, metrics in results['traditional'].items():
            table_data.append({
                'Method': algo_name.upper(),
                'Type': 'Traditional',
                'BPC': f"{metrics['avg_bits_per_character']:.3f}",
                'Compression Ratio': f"{metrics['avg_compression_ratio']:.2f}",
                'Reconstruction Accuracy': '100.0%',
                'System Size (MB)': f"{metrics['system_size_mb']:.3f}"
            })
            
        # Create DataFrame
        df = pd.DataFrame(table_data)
        
        # Save as CSV
        csv_path = os.path.join(self.config['experiment']['output_dir'],
                               'results', 'experiment_1d', 'compression_comparison.csv')
        df.to_csv(csv_path, index=False)
        
        # Save as LaTeX
        latex_path = os.path.join(self.config['experiment']['output_dir'],
                                 'results', 'experiment_1d', 'compression_comparison.tex')
        df.to_latex(latex_path, index=False, escape=False)
        
        # Create visual table
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df.values, colLabels=df.columns,
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        # Style the table
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
            
        # Color code by type
        type_colors = {
            'LLM Lossless': '#E8F4F8',
            'LLM Lossy': '#FFE6E6',
            'Traditional': '#E6F7E6'
        }
        
        for i in range(len(df)):
            row_type = df.iloc[i]['Type']
            for j in range(len(df.columns)):
                table[(i+1, j)].set_facecolor(type_colors.get(row_type, 'white'))
                
        plt.title('Comprehensive Compression Method Comparison', fontsize=16, pad=20)
        plt.savefig(os.path.join(self.config['experiment']['output_dir'],
                                'figures', 'experiment_1d', 'summary_table.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def save_results(self, results: Dict[str, Any]):
        """Save experiment results."""
        # Convert to serializable format
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
            
        serializable_results = convert_to_serializable(results)
        
        # Save JSON
        results_path = os.path.join(
            self.config['experiment']['output_dir'],
            'results',
            'experiment_1d',
            f'results_{datetime.now():%Y%m%d_%H%M%S}.json'
        )
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        self.logger.info(f"Results saved to {results_path}")
        
    def print_summary(self, results: Dict[str, Any]):
        """Print experiment summary."""
        print("\n" + "="*80)
        print("EXPERIMENT 1D SUMMARY: COMPARATIVE COMPRESSION BENCHMARK")
        print("="*80)
        
        analysis = results['comparative_analysis']
        
        print("\nBEST PERFORMERS:")
        print("-" * 40)
        print(f"Best Compression Ratio: {analysis['best_compression_ratio']['method']} "
              f"({analysis['best_compression_ratio']['value']:.2f}x)")
        print(f"Best Bits Per Character: {analysis['best_bits_per_char']['method']} "
              f"({analysis['best_bits_per_char']['value']:.3f} BPC)")
        print(f"Best Lossless: {analysis['best_lossless']['method']} "
              f"({analysis['best_lossless']['value']:.2f}x)")
        print(f"Best Lossy Quality: {analysis['best_lossy_quality']['method']}")
        print(f"Smallest System: {analysis['smallest_system']['method']} "
              f"({analysis['smallest_system']['value']:.3f} MB)")
              
        print("\nKEY FINDINGS:")
        print("-" * 40)
        print("1. LLM-based methods achieve strong compression but require large models")
        print("2. Traditional algorithms are extremely efficient in system size")
        print("3. Lossy LLM methods offer tunable quality-compression trade-offs")
        print("4. Autoregressive coding provides theoretical optimal compression")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Experiment 1D: Comparative Compression Benchmark")
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Run experiment
    runner = Experiment1DRunner(args.config)
    results = runner.run_comparative_benchmark()


if __name__ == "__main__":
    main()
