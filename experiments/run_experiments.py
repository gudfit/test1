# experiments/run_experiments.py
"""Main script to run all compression experiments."""

import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import time
from datetime import datetime
import logging
from typing import Dict, List, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.predictive_masking import PredictiveMaskingCompressor
from src.models.latent_space_quantization import LatentSpaceQuantizationCompressor
from src.data.data_loader import DataLoader
from src.evaluation.metrics import CompressionMetrics
from src.visualization.plots import CompressionVisualizer


class CompressionExperimentRunner:
    """Run compression experiments across models and parameters."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize experiment runner."""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Set up logging
        self._setup_logging()

        # Set random seeds
        self._set_seeds(self.config['experiment']['seed'])

        # Initialize components
        self.data_loader = DataLoader(
            self.config['data']['dataset_name'],
            self.config['data']['dataset_config']
        )
        self.metrics_calculator = CompressionMetrics(self.config['experiment']['device'])
        self.visualizer = CompressionVisualizer(
            style=self.config['visualization']['style'],
            figure_dir=os.path.join(self.config['experiment']['output_dir'], 'figures')
        )

        # Create output directories
        self._create_output_dirs()

    def _setup_logging(self):
        """Set up logging configuration."""
        log_dir = os.path.join(self.config['experiment']['output_dir'], 'logs')
        os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, f'experiment_{datetime.now():%Y%m%d_%H%M%S}.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _create_output_dirs(self):
        """Create necessary output directories."""
        dirs = [
            self.config['experiment']['output_dir'],
            os.path.join(self.config['experiment']['output_dir'], 'figures'),
            os.path.join(self.config['experiment']['output_dir'], 'models'),
            os.path.join(self.config['experiment']['output_dir'], 'logs'),
            os.path.join(self.config['experiment']['output_dir'], 'results')
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)

    def run_all_experiments(self):
        """Run experiments for all models and configurations."""
        self.logger.info("Starting compression experiments...")

        # Load data
        self.logger.info("Loading data...")
        train_texts, test_texts = self.data_loader.load_data(
            train_size=self.config['data']['train_size'],
            test_size=self.config['data']['test_size'],
            max_length=self.config['data']['max_length']
        )

        # Get diverse test samples
        test_samples = self.data_loader.get_diverse_test_samples(test_texts, num_samples=100)

        # Results storage
        all_results = {}

        # Run experiments for each model
        for model_config in self.config['models'].values():
            model_name = model_config['name']
            self.logger.info(f"\nRunning experiments for {model_name}...")

            # Run predictive masking experiments
            pm_results = self.run_predictive_masking_experiments(
                model_name, train_texts, test_samples
            )
            all_results[f"{model_name}_predictive_masking"] = pm_results

            # Run latent space quantization experiments
            lsq_results = self.run_latent_space_quantization_experiments(
                model_name, train_texts, test_samples
            )
            all_results[f"{model_name}_lsq"] = lsq_results

        # Save results
        self.save_results(all_results)

        # Create visualizations
        self.logger.info("Creating visualizations...")
        self.create_all_visualizations(all_results)

        self.logger.info("Experiments completed!")

    def run_predictive_masking_experiments(self, model_name: str, train_texts: List[str],
                                         test_texts: List[str]) -> Dict:
        """Run predictive masking experiments for a model."""
        results = {}

        # Initialize compressor
        compressor = PredictiveMaskingCompressor(
            model_name=model_name,
            device=self.config['experiment']['device']
        )

        # Fine-tune if enabled
        if self.config['training']['epochs'] > 0:
            self.logger.info(f"Fine-tuning {model_name}...")
            compressor.fine_tune(
                texts=train_texts[:1000],  # Use subset for faster fine-tuning
                epochs=self.config['training']['epochs'],
                learning_rate=self.config['training']['learning_rate'],
                batch_size=self.config['compression']['batch_size']
            )

            # Save fine-tuned model
            if self.config['experiment']['save_models']:
                model_path = os.path.join(
                    self.config['experiment']['output_dir'],
                    'models',
                    f"{model_name}_predictive_masking"
                )
                compressor.save_model(model_path)

        # Test different masking probabilities
        for masking_prob in self.config['compression']['masking_probabilities']:
            self.logger.info(f"Testing masking probability: {masking_prob}")

            prob_results = {
                'compression_ratio': [],
                'word_accuracy': [],
                'character_accuracy': [],
                'semantic_similarity': [],
                'rouge1_fmeasure': [],
                'rouge2_fmeasure': [],
                'rougeL_fmeasure': [],
                'bert_score_f1': [],
                'bits_per_character': []
            }

            # Run multiple times for averaging
            for run in range(self.config['experiment']['num_runs']):
                run_metrics = []

                # Process test samples
                for text in tqdm(test_texts, desc=f"Run {run+1}/{self.config['experiment']['num_runs']}"):
                    try:
                        # Compress
                        compressed = compressor.compress(text, masking_probability=masking_prob)

                        # Decompress
                        reconstructed = compressor.decompress(compressed)

                        # Calculate metrics
                        metrics = self.metrics_calculator.calculate_all_metrics(
                            text, reconstructed, compressed
                        )
                        run_metrics.append(metrics)

                    except Exception as e:
                        self.logger.error(f"Error processing text: {e}")
                        continue

                # Average metrics for this run
                if run_metrics:
                    for key in prob_results:
                        values = [m.get(key, 0) for m in run_metrics]
                        prob_results[key].append(np.mean(values))

            # Average across runs
            results[masking_prob] = {
                key: np.mean(values) for key, values in prob_results.items()
            }

            # Add example reconstructions
            if self.config['evaluation']['save_reconstructions']:
                examples = []
                for i in range(min(3, len(test_texts))):
                    compressed = compressor.compress(test_texts[i], masking_probability=masking_prob)
                    reconstructed = compressor.decompress(compressed)
                    examples.append({
                        'original': test_texts[i],
                        'reconstructed': reconstructed,
                        'model': model_name,
                        'masking_prob': masking_prob
                    })
                results[masking_prob]['examples'] = examples

        return results

    def run_latent_space_quantization_experiments(self, model_name: str, train_texts: List[str],
                                                test_texts: List[str]) -> Dict:
        """Run latent space quantization experiments for a model."""
        results = {}

        # Initialize compressor
        compressor = LatentSpaceQuantizationCompressor(
            model_name=model_name,
            device=self.config['experiment']['device'],
            quantization_bits=self.config['compression']['quantization_bits']
        )

        # Train decoder
        self.logger.info(f"Training decoder for {model_name}...")
        compressor.train_decoder(
            texts=train_texts[:500],  # Use subset
            epochs=5,
            batch_size=self.config['compression']['batch_size']
        )

        # Test different quantization levels (simulated through masking probabilities for consistency)
        for idx, masking_prob in enumerate(self.config['compression']['masking_probabilities']):
            # Map masking probability to quantization bits (inverse relationship)
            quantization_bits = int(16 - (masking_prob * 14))  # 16 bits to 2 bits
            self.logger.info(f"Testing {quantization_bits}-bit quantization")

            prob_results = {
                'compression_ratio': [],
                'word_accuracy': [],
                'character_accuracy': [],
                'semantic_similarity': [],
                'rouge1_fmeasure': [],
                'rouge2_fmeasure': [],
                'rougeL_fmeasure': [],
                'bert_score_f1': [],
                'bits_per_character': []
            }

            # Process test samples
            for text in tqdm(test_texts, desc=f"LSQ {quantization_bits}-bit"):
                try:
                    # Compress
                    compressed = compressor.compress(text, quantization_bits=quantization_bits)

                    # Decompress
                    reconstructed = compressor.decompress(compressed)

                    # Calculate metrics
                    metrics = self.metrics_calculator.calculate_all_metrics(
                        text, reconstructed, compressed
                    )

                    for key in prob_results:
                        prob_results[key].append(metrics.get(key, 0))

                except Exception as e:
                    self.logger.error(f"Error processing text: {e}")
                    continue

            # Average metrics
            results[masking_prob] = {
                key: np.mean(values) for key, values in prob_results.items()
            }
            results[masking_prob]['quantization_bits'] = quantization_bits

        return results

    def save_results(self, results: Dict):
        """Save experimental results."""
        # Save as JSON
        results_path = os.path.join(
            self.config['experiment']['output_dir'],
            'results',
            f'results_{datetime.now():%Y%m%d_%H%M%S}.json'
        )

        # Convert numpy values to Python types for JSON serialization
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

        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        self.logger.info(f"Results saved to {results_path}")

        # Also save as CSV for easy analysis
        csv_data = []
        for model_method, model_results in results.items():
            for prob, metrics in model_results.items():
                if isinstance(prob, float):
                    row = {
                        'model_method': model_method,
                        'masking_probability': prob,
                        **{k: v for k, v in metrics.items() if not isinstance(v, list)}
                    }
                    csv_data.append(row)

        df = pd.DataFrame(csv_data)
        csv_path = results_path.replace('.json', '.csv')
        df.to_csv(csv_path, index=False)
        self.logger.info(f"CSV results saved to {csv_path}")

    def create_all_visualizations(self, results: Dict):
        """Create all visualizations from results."""
        # Separate results by method
        pm_results = {k.replace('_predictive_masking', ''): v
                     for k, v in results.items() if 'predictive_masking' in k}
        lsq_results = {k.replace('_lsq', ''): v
                      for k, v in results.items() if 'lsq' in k}

        # Create visualizations for predictive masking
        if pm_results:
            self.visualizer.create_summary_report(pm_results, "predictive_masking")

            # Create radar charts for different masking probabilities
            for prob in [0.3, 0.5, 0.7]:
                self.visualizer.plot_model_comparison_radar(pm_results, prob, f"pm_radar_{prob}")

        # Create visualizations for LSQ
        if lsq_results:
            self.visualizer.create_summary_report(lsq_results, "latent_space_quantization")

        # Create combined comparison
        self.visualizer.plot_compression_vs_accuracy(results, "all_methods_comparison")

        self.logger.info("All visualizations created!")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run LLM compression experiments")
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--models', nargs='+',
                       help='Specific models to run (default: all)')
    parser.add_argument('--method', choices=['predictive_masking', 'lsq', 'both'],
                       default='both', help='Compression method to use')
    parser.add_argument('--experiment', choices=['1a', '1b', '1c', '1d', 'all'],
                       default='1a', help='Which experiment to run')
    parser.add_argument('--probe-source', type=str, default='mixed',
                       choices=['wikitext', 'lama', 'mixed'],
                       help='Source for factual probes (for experiment 1b)')

    args = parser.parse_args()

    # Run experiments based on selection
    if args.experiment in ['1a', 'all']:
        print("Running Experiment 1A: Text Compression...")
        runner = CompressionExperimentRunner(args.config)
        runner.run_all_experiments()

    if args.experiment in ['1b', 'all']:
        print("\nRunning Experiment 1B: Knowledge Compression...")
        from run_experiment_1b import Experiment1BRunner
        runner_1b = Experiment1BRunner(args.config)
        runner_1b.run_knowledge_compression_analysis(probe_source=args.probe_source)

    if args.experiment in ['1c', 'all']:
        print("\nRunning Experiment 1C: Pruning Effects...")
        from run_experiment_1c import Experiment1CRunner
        runner_1c = Experiment1CRunner(args.config)
        runner_1c.run_compression_analysis()

    if args.experiment in ['1d', 'all']:
        print("\nRunning Experiment 1D: Comparative Benchmark...")
        from run_experiment_1d import Experiment1DRunner
        runner_1d = Experiment1DRunner(args.config)
        runner_1d.run_comparative_benchmark()


if __name__ == "__main__":
    main()
