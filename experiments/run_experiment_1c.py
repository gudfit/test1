# vim experiments/run_experiment_1c.py 
"""Run Experiment 1C: Effects of Pruning and Quantization on Knowledge Compression."""

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

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.compression_techniques import ModelCompressor
from src.models.knowledge_compressor import KnowledgeCompressor
from src.evaluation.crumpled_paper_metrics import CrumpledPaperMetrics
from src.evaluation.glue_evaluator import GLUEEvaluator
from src.data.lama_probe_loader import LAMAProbeLoader
from src.visualization.plots import CompressionVisualizer


class Experiment1CRunner:
    """Runner for Experiment 1C: Model Compression Analysis."""

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
            figure_dir=os.path.join(self.config['experiment']['output_dir'], 'figures', 'experiment_1c')
        )

    def _setup_logging(self):
        """Set up logging configuration."""
        log_dir = os.path.join(self.config['experiment']['output_dir'], 'logs')
        os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, f'experiment_1c_{datetime.now():%Y%m%d_%H%M%S}.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _create_output_dirs(self):
        """Create necessary output directories."""
        dirs = [
            os.path.join(self.config['experiment']['output_dir'], 'figures', 'experiment_1c'),
            os.path.join(self.config['experiment']['output_dir'], 'results', 'experiment_1c')
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)

    def run_compression_analysis(self):
        """Run the complete Experiment 1C analysis."""
        self.logger.info("Starting Experiment 1C: Model Compression Analysis")

        results = {}

        # Configuration for compression levels
        pruning_levels = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]

        # Run for each model
        for model_config in self.config['models'].values():
            model_name = model_config['name']
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Analyzing compression for {model_name}")
            self.logger.info(f"{'='*60}")

            # Initialize compressor
            compressor = ModelCompressor(model_name, self.config['experiment']['device'])

            # Create pruning variants only
            self.logger.info("Creating pruned model variants...")
            variants = self._create_pruning_variants(compressor, pruning_levels)

            # Run three-phase evaluation
            model_results = {
                'variants': {},
                'phase1_structural_fidelity': {},
                'phase2_downstream_performance': {},
                'phase3_computational_performance': {}
            }

            # Evaluate each variant
            for variant_name, variant_info in variants.items():
                self.logger.info(f"\nEvaluating variant: {variant_name}")

                # Phase 1: Structural Fidelity (Crumpled Paper)
                self.logger.info("Phase 1: Measuring structural fidelity...")
                structural_results = self.evaluate_structural_fidelity(
                    variant_info['model'], model_name
                )
                model_results['phase1_structural_fidelity'][variant_name] = structural_results

                # Phase 2: Downstream Task Performance (GLUE)
                self.logger.info("Phase 2: Evaluating downstream performance...")
                downstream_results = self.evaluate_downstream_performance(
                    variant_info['model'], model_name
                )
                model_results['phase2_downstream_performance'][variant_name] = downstream_results

                # Phase 3: Computational Performance
                self.logger.info("Phase 3: Measuring computational performance...")
                computational_results = self.evaluate_computational_performance(
                    variant_info['model'], compressor
                )
                model_results['phase3_computational_performance'][variant_name] = computational_results

                # Store variant info
                model_results['variants'][variant_name] = {
                    'sparsity': variant_info.get('sparsity', 0.0),
                    'size_mb': variant_info['size_mb'],
                    'compression_ratio': variant_info['compression_ratio']
                }

            results[model_name] = model_results

        # Create visualizations
        self.logger.info("\nCreating visualizations...")
        self.create_visualizations(results)

        # Save results
        self.save_results(results)

        # Print summary
        self.print_summary(results)

        self.logger.info("\nExperiment 1C completed!")

        return results

    def _create_pruning_variants(self, compressor: ModelCompressor,
                               pruning_levels: List[float]) -> Dict[str, Any]:
        """Create pruned model variants."""
        variants = {}
        original_model = compressor.load_original_model()

        for sparsity in pruning_levels:
            if sparsity == 0.0:
                # Original model
                variants['original'] = {
                    'model': original_model,
                    'type': 'original',
                    'sparsity': 0.0,
                    'compression_ratio': 1.0,
                    'size_mb': compressor._calculate_model_size(original_model)
                }
            else:
                # Pruned model
                self.logger.info(f"Creating pruned variant with {sparsity:.0%} sparsity...")
                pruned = compressor.apply_magnitude_pruning(original_model, sparsity)

                # Count actual sparsity
                param_stats = compressor.count_nonzero_parameters(pruned)

                variants[f'pruned_{sparsity}'] = {
                    'model': pruned,
                    'type': 'pruned',
                    'sparsity': sparsity,
                    'actual_sparsity': param_stats['sparsity'],
                    'compression_ratio': param_stats['compression_ratio'],
                    'size_mb': compressor._calculate_model_size(pruned)
                }

        return variants

    def evaluate_structural_fidelity(self, model, model_name: str) -> Dict[str, float]:
        """Phase 1: Evaluate structural fidelity using Crumpled Paper metrics."""
        # Initialize metrics calculator
        crumpled_metrics = CrumpledPaperMetrics()

        # Load test texts
        from datasets import load_dataset
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test[:100]')
        test_texts = [item['text'] for item in dataset if len(item['text'].strip()) > 50][:20]

        # Use predictive masking to test reconstruction quality
        from src.models.predictive_masking import PredictiveMaskingCompressor

        # Create a temporary compressor with the pruned model
        temp_compressor = PredictiveMaskingCompressor(model_name)
        temp_compressor.model = model  # Replace with pruned model

        all_metrics = []

        for text in test_texts:
            try:
                # Compress and decompress with moderate masking
                compressed = temp_compressor.compress(text, masking_probability=0.3)
                reconstructed = temp_compressor.decompress(compressed)

                # Calculate crumpled paper metrics
                metrics = crumpled_metrics.calculate_crease_metrics(text, reconstructed)
                all_metrics.append(metrics)
            except Exception as e:
                self.logger.warning(f"Error in structural fidelity evaluation: {e}")

        # Aggregate results
        if all_metrics:
            aggregated = {
                'mean_tcm': np.mean([m['total_crease_magnitude'] for m in all_metrics]),
                'std_tcm': np.std([m['total_crease_magnitude'] for m in all_metrics]),
                'mean_pcm': np.mean([m['peak_crease_magnitude'] for m in all_metrics]),
                'std_pcm': np.std([m['peak_crease_magnitude'] for m in all_metrics]),
                'mean_crease_density': np.mean([m['crease_density'] for m in all_metrics])
            }
        else:
            aggregated = {
                'mean_tcm': float('inf'),
                'std_tcm': 0.0,
                'mean_pcm': float('inf'),
                'std_pcm': 0.0,
                'mean_crease_density': 1.0
            }

        return aggregated

    def evaluate_downstream_performance(self, model, model_name: str) -> Dict[str, float]:
        """Phase 2: Evaluate downstream task performance on GLUE."""
        # Initialize GLUE evaluator
        glue_evaluator = GLUEEvaluator(model_name, self.config['experiment']['device'])

        # Evaluate on selected GLUE tasks
        tasks = ['sst2', 'mrpc', 'rte']
        train_samples = 1000  # Small dataset for quick evaluation
        eval_samples = 200
        epochs = 50

        results = glue_evaluator.evaluate_on_tasks(
            model,
            tasks=tasks,
            train_samples=train_samples,
            eval_samples=eval_samples,
            epochs=epochs
        )

        return results

    def evaluate_computational_performance(self, model, compressor: ModelCompressor) -> Dict[str, float]:
        """Phase 3: Evaluate computational performance."""
        # Load actual WikiText samples for benchmarking
        from datasets import load_dataset
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test[:100]')

        # Get text samples that are reasonable length
        test_texts = []
        for item in dataset:
            text = item['text'].strip()
            if len(text) > 50 and len(text) < 500:  # Reasonable length texts
                test_texts.append(text)
            if len(test_texts) >= 16:  # Get enough for 2 batches
                break

        # If we don't have enough samples, load more
        if len(test_texts) < 8:
            self.logger.warning("Not enough test texts found, loading more samples...")
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test[:500]')
            test_texts = [item['text'].strip() for item in dataset
                         if 50 < len(item['text'].strip()) < 500][:16]

        # Measure inference performance
        perf_metrics = compressor.measure_inference_performance(
            model, test_texts, batch_size=8
        )

        # Add parameter count statistics
        param_stats = compressor.count_nonzero_parameters(model)
        perf_metrics.update(param_stats)

        return perf_metrics

    def create_visualizations(self, results: Dict[str, Any]):
        """Create comprehensive visualizations for Experiment 1C."""
        # 1. Pruning vs Performance Trade-off Curves
        self._plot_pruning_tradeoff_curves(results)

        # 2. Structural Fidelity Heatmap
        self._plot_structural_fidelity_heatmap(results)

        # 3. Computational Performance Comparison
        self._plot_computational_performance(results)

        # 4. Combined Summary Plot
        self._plot_combined_summary(results)

    def _plot_pruning_tradeoff_curves(self, results: Dict[str, Any]):
        """Plot trade-off curves between pruning and performance metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        for model_name, model_results in results.items():
            sparsity_levels = []
            tcm_values = []
            pcm_values = []
            glue_scores = []
            inference_times = []

            for variant_name, variant_info in model_results['variants'].items():
                sparsity = variant_info['sparsity']
                sparsity_levels.append(sparsity)

                # Get metrics
                structural = model_results['phase1_structural_fidelity'][variant_name]
                tcm_values.append(structural['mean_tcm'])
                pcm_values.append(structural['mean_pcm'])

                downstream = model_results['phase2_downstream_performance'][variant_name]
                glue_scores.append(downstream['average'])

                computational = model_results['phase3_computational_performance'][variant_name]
                inference_times.append(computational['avg_latency_ms'])

            # Sort by sparsity
            sorted_indices = np.argsort(sparsity_levels)
            sparsity_levels = np.array(sparsity_levels)[sorted_indices]
            tcm_values = np.array(tcm_values)[sorted_indices]
            pcm_values = np.array(pcm_values)[sorted_indices]
            glue_scores = np.array(glue_scores)[sorted_indices]
            inference_times = np.array(inference_times)[sorted_indices]

            # Plot TCM vs Sparsity
            axes[0].plot(sparsity_levels * 100, tcm_values, 'o-', label=model_name, linewidth=2)
            axes[0].set_xlabel('Sparsity (%)')
            axes[0].set_ylabel('Total Crease Magnitude (TCM)')
            axes[0].set_title('Structural Degradation vs Pruning')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Plot PCM vs Sparsity
            axes[1].plot(sparsity_levels * 100, pcm_values, 's-', label=model_name, linewidth=2)
            axes[1].set_xlabel('Sparsity (%)')
            axes[1].set_ylabel('Peak Crease Magnitude (PCM)')
            axes[1].set_title('Worst-Case Degradation vs Pruning')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            # Plot GLUE Score vs Sparsity
            axes[2].plot(sparsity_levels * 100, glue_scores, '^-', label=model_name, linewidth=2)
            axes[2].set_xlabel('Sparsity (%)')
            axes[2].set_ylabel('Average GLUE Score')
            axes[2].set_title('Downstream Performance vs Pruning')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

            # Plot Inference Time vs Sparsity
            axes[3].plot(sparsity_levels * 100, inference_times, 'd-', label=model_name, linewidth=2)
            axes[3].set_xlabel('Sparsity (%)')
            axes[3].set_ylabel('Inference Latency (ms)')
            axes[3].set_title('Computational Efficiency vs Pruning')
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)

        plt.suptitle('Pruning Trade-off Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['experiment']['output_dir'],
                                'figures', 'experiment_1c', 'pruning_tradeoff_curves.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_structural_fidelity_heatmap(self, results: Dict[str, Any]):
        """Plot heatmap of structural fidelity metrics."""
        # Prepare data for heatmap
        models = list(results.keys())
        sparsity_levels = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]

        tcm_matrix = np.zeros((len(models), len(sparsity_levels)))
        pcm_matrix = np.zeros((len(models), len(sparsity_levels)))

        for i, model in enumerate(models):
            for j, sparsity in enumerate(sparsity_levels):
                variant_name = 'original' if sparsity == 0.0 else f'pruned_{sparsity}'
                if variant_name in results[model]['phase1_structural_fidelity']:
                    metrics = results[model]['phase1_structural_fidelity'][variant_name]
                    tcm_matrix[i, j] = metrics['mean_tcm']
                    pcm_matrix[i, j] = metrics['mean_pcm']

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # TCM heatmap
        sns.heatmap(tcm_matrix,
                   xticklabels=[f'{s:.0%}' for s in sparsity_levels],
                   yticklabels=[m.split('-')[0].upper() for m in models],
                   annot=True, fmt='.1f', cmap='YlOrRd',
                   ax=ax1, cbar_kws={'label': 'TCM'})
        ax1.set_xlabel('Sparsity Level')
        ax1.set_ylabel('Model')
        ax1.set_title('Total Crease Magnitude (TCM) Heatmap')

        # PCM heatmap
        sns.heatmap(pcm_matrix,
                   xticklabels=[f'{s:.0%}' for s in sparsity_levels],
                   yticklabels=[m.split('-')[0].upper() for m in models],
                   annot=True, fmt='.1f', cmap='YlOrRd',
                   ax=ax2, cbar_kws={'label': 'PCM'})
        ax2.set_xlabel('Sparsity Level')
        ax2.set_ylabel('Model')
        ax2.set_title('Peak Crease Magnitude (PCM) Heatmap')

        plt.suptitle('Structural Fidelity Degradation Across Models and Sparsity Levels', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['experiment']['output_dir'],
                                'figures', 'experiment_1c', 'structural_fidelity_heatmap.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_computational_performance(self, results: Dict[str, Any]):
        """Plot computational performance metrics."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for model_name, model_results in results.items():
            sparsity_levels = []
            memory_footprints = []
            throughputs = []
            actual_compressions = []

            for variant_name, variant_info in model_results['variants'].items():
                sparsity = variant_info['sparsity']
                sparsity_levels.append(sparsity)

                computational = model_results['phase3_computational_performance'][variant_name]
                memory_footprints.append(computational['memory_footprint_mb'])
                throughputs.append(computational['throughput_samples_per_sec'])
                actual_compressions.append(computational.get('compression_ratio', 1.0))

            # Sort by sparsity
            sorted_indices = np.argsort(sparsity_levels)
            sparsity_levels = np.array(sparsity_levels)[sorted_indices]
            memory_footprints = np.array(memory_footprints)[sorted_indices]
            throughputs = np.array(throughputs)[sorted_indices]
            actual_compressions = np.array(actual_compressions)[sorted_indices]

            # Memory footprint
            axes[0].plot(sparsity_levels * 100, memory_footprints, 'o-',
                        label=model_name, linewidth=2, markersize=8)
            axes[0].set_xlabel('Sparsity (%)')
            axes[0].set_ylabel('Memory Footprint (MB)')
            axes[0].set_title('Memory Usage vs Pruning')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Throughput
            axes[1].plot(sparsity_levels * 100, throughputs, 's-',
                        label=model_name, linewidth=2, markersize=8)
            axes[1].set_xlabel('Sparsity (%)')
            axes[1].set_ylabel('Throughput (samples/sec)')
            axes[1].set_title('Inference Throughput vs Pruning')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            # Actual compression ratio
            axes[2].plot(sparsity_levels * 100, actual_compressions, '^-',
                        label=model_name, linewidth=2, markersize=8)
            axes[2].set_xlabel('Sparsity (%)')
            axes[2].set_ylabel('Compression Ratio')
            axes[2].set_title('Actual Compression vs Pruning')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

        plt.suptitle('Computational Performance Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['experiment']['output_dir'],
                                'figures', 'experiment_1c', 'computational_performance.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_combined_summary(self, results: Dict[str, Any]):
        """Create a combined summary plot showing the sweet spot analysis."""
        fig, ax = plt.subplots(figsize=(12, 8))

        for model_name, model_results in results.items():
            sparsity_levels = []
            combined_scores = []

            for variant_name, variant_info in model_results['variants'].items():
                sparsity = variant_info['sparsity']
                sparsity_levels.append(sparsity)

                # Calculate combined score (normalized)
                structural = model_results['phase1_structural_fidelity'][variant_name]
                downstream = model_results['phase2_downstream_performance'][variant_name]
                computational = model_results['phase3_computational_performance'][variant_name]

                # Normalize metrics (higher is better)
                tcm_score = 1.0 / (1.0 + structural['mean_tcm'])  # Lower TCM is better
                glue_score = downstream['average']  # Already 0-1
                efficiency_score = computational['throughput_samples_per_sec'] / 100  # Normalize

                # Combined score (weighted average)
                combined = 0.4 * glue_score + 0.3 * tcm_score + 0.3 * efficiency_score
                combined_scores.append(combined)

            # Sort and plot
            sorted_indices = np.argsort(sparsity_levels)
            sparsity_levels = np.array(sparsity_levels)[sorted_indices]
            combined_scores = np.array(combined_scores)[sorted_indices]

            ax.plot(sparsity_levels * 100, combined_scores, 'o-',
                   label=model_name, linewidth=3, markersize=10)

            # Mark the sweet spot
            sweet_spot_idx = np.argmax(combined_scores)
            ax.scatter(sparsity_levels[sweet_spot_idx] * 100,
                      combined_scores[sweet_spot_idx],
                      s=200, marker='*', edgecolors='black', linewidth=2)

        ax.set_xlabel('Sparsity (%)', fontsize=14)
        ax.set_ylabel('Combined Performance Score', fontsize=14)
        ax.set_title('Sweet Spot Analysis: Optimal Pruning Level', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        # Add annotation
        ax.text(0.02, 0.02,
               'Combined Score = 0.4×GLUE + 0.3×(1/TCM) + 0.3×Efficiency',
               transform=ax.transAxes, fontsize=10, style='italic',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(os.path.join(self.config['experiment']['output_dir'],
                                'figures', 'experiment_1c', 'sweet_spot_analysis.png'),
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
            elif hasattr(obj, '__dict__'):
                return str(obj)
            return obj

        serializable_results = convert_to_serializable(results)

        # Save JSON
        results_path = os.path.join(
            self.config['experiment']['output_dir'],
            'results',
            'experiment_1c',
            f'results_{datetime.now():%Y%m%d_%H%M%S}.json'
        )

        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        self.logger.info(f"Results saved to {results_path}")

        # Create summary CSV
        summary_data = []
        for model_name, model_results in results.items():
            for variant_name, variant_info in model_results['variants'].items():
                row = {
                    'model': model_name,
                    'variant': variant_name,
                    'sparsity': variant_info['sparsity'],
                    'size_mb': variant_info['size_mb'],
                    'compression_ratio': variant_info['compression_ratio']
                }

                # Add metrics
                structural = model_results['phase1_structural_fidelity'][variant_name]
                row.update({
                    'tcm': structural['mean_tcm'],
                    'pcm': structural['mean_pcm']
                })

                downstream = model_results['phase2_downstream_performance'][variant_name]
                row.update({
                    'glue_average': downstream['average'],
                    'sst2': downstream.get('sst2', 0),
                    'mrpc': downstream.get('mrpc', 0),
                    'rte': downstream.get('rte', 0)
                })

                computational = model_results['phase3_computational_performance'][variant_name]
                row.update({
                    'latency_ms': computational['avg_latency_ms'],
                    'throughput': computational['throughput_samples_per_sec'],
                    'memory_mb': computational['memory_footprint_mb']
                })

                summary_data.append(row)

        df_summary = pd.DataFrame(summary_data)
        csv_path = results_path.replace('.json', '_summary.csv')
        df_summary.to_csv(csv_path, index=False)
        self.logger.info(f"Summary CSV saved to {csv_path}")

    def print_summary(self, results: Dict[str, Any]):
        """Print experiment summary."""
        print("\n" + "="*80)
        print("EXPERIMENT 1C SUMMARY: PRUNING EFFECTS ON KNOWLEDGE COMPRESSION")
        print("="*80)

        for model_name, model_results in results.items():
            print(f"\n{model_name.upper()}:")
            print("-" * 40)

            # Find sweet spot
            best_combined_score = -float('inf')
            best_sparsity = 0.0

            for variant_name, variant_info in model_results['variants'].items():
                sparsity = variant_info['sparsity']

                # Calculate combined score
                structural = model_results['phase1_structural_fidelity'][variant_name]
                downstream = model_results['phase2_downstream_performance'][variant_name]
                computational = model_results['phase3_computational_performance'][variant_name]

                tcm_score = 1.0 / (1.0 + structural['mean_tcm'])
                glue_score = downstream['average']
                efficiency_score = computational['throughput_samples_per_sec'] / 100

                combined = 0.4 * glue_score + 0.3 * tcm_score + 0.3 * efficiency_score

                if combined > best_combined_score:
                    best_combined_score = combined
                    best_sparsity = sparsity

                if sparsity in [0.0, 0.3, 0.5, 0.7, 0.9]:
                    print(f"\n  Sparsity: {sparsity:.0%}")
                    print(f"    - TCM: {structural['mean_tcm']:.2f}")
                    print(f"    - GLUE Average: {downstream['average']:.3f}")
                    print(f"    - Latency: {computational['avg_latency_ms']:.1f} ms")
                    print(f"    - Compression Ratio: {variant_info['compression_ratio']:.2f}x")

            print(f"\n  OPTIMAL SPARSITY (Sweet Spot): {best_sparsity:.0%}")
            print(f"  Combined Score: {best_combined_score:.3f}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run Experiment 1C: Model Compression Analysis")
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')

    args = parser.parse_args()

    # Run experiment
    runner = Experiment1CRunner(args.config)
    results = runner.run_compression_analysis()


if __name__ == "__main__":
    main()

