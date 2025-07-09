# experiments/run_experiments.py
"""Main script to run all compression experiments."""

import argparse
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.predictive_masking import PredictiveMaskingCompressor
from src.models.latent_space_quantization import LatentSpaceQuantizationCompressor
from src.data.data_loader import DataLoader
from src.evaluation.metrics import CompressionMetrics
from src.visualization.plots import CompressionVisualizer


class CompressionExperimentRunner:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self._setup_logging()
        self._set_seeds(self.config["experiment"]["seed"])
        self.data_loader = DataLoader(
            self.config["data"]["dataset_name"], self.config["data"]["dataset_config"]
        )
        self.metrics_calculator = CompressionMetrics(
            self.config["experiment"]["device"]
        )
        self.visualizer = CompressionVisualizer(
            style=self.config["visualization"]["style"],
            figure_dir=os.path.join(self.config["experiment"]["output_dir"], "figures"),
        )
        self._create_output_dirs()

    def _setup_logging(self):
        log_dir = os.path.join(self.config["experiment"]["output_dir"], "logs")
        os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(
                    os.path.join(
                        log_dir, f"experiment_{datetime.now():%Y%m%d_%H%M%S}.log"
                    )
                ),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def _set_seeds(self, seed: int):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _create_output_dirs(self):
        dirs = [
            self.config["experiment"]["output_dir"],
            os.path.join(self.config["experiment"]["output_dir"], "figures"),
            os.path.join(self.config["experiment"]["output_dir"], "models"),
            os.path.join(self.config["experiment"]["output_dir"], "logs"),
            os.path.join(self.config["experiment"]["output_dir"], "results"),
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)

    def run_all_experiments(self):
        self.logger.info("Starting compression experiments...")
        self.logger.info("Loading data...")
        train_texts, test_texts = self.data_loader.load_data(
            train_size=self.config["data"]["train_size"],
            test_size=self.config["data"]["test_size"],
            max_length=self.config["data"]["max_length"],
        )
        test_samples = self.data_loader.get_diverse_test_samples(
            test_texts, num_samples=100
        )
        all_results = {}
        for model_config in self.config["models"].values():
            model_name = model_config["name"]
            self.logger.info(f"\nRunning experiments for {model_name}...")
            pm_results = self.run_predictive_masking_experiments(
                model_name, train_texts, test_samples
            )
            all_results[f"{model_name}_predictive_masking"] = pm_results
            lsq_results = self.run_latent_space_quantization_experiments(
                model_name, train_texts, test_samples
            )
            all_results[f"{model_name}_lsq"] = lsq_results
        self.save_results(all_results)
        self.logger.info("Creating visualizations...")
        self.create_all_visualizations(all_results)
        self.logger.info("Experiments completed!")

    def run_predictive_masking_experiments(
        self, model_name: str, train_texts: List[str], test_texts: List[str]
    ) -> Dict:
        results = {}
        compressor = PredictiveMaskingCompressor(
            model_name=model_name, device=self.config["experiment"]["device"]
        )
        if self.config["training"]["epochs"] > 0:
            self.logger.info(f"Fine-tuning {model_name}...")
            compressor.fine_tune(
                texts=train_texts[:1000],
                epochs=self.config["training"]["epochs"],
                learning_rate=self.config["training"]["learning_rate"],
                batch_size=self.config["compression"]["batch_size"],
            )
            if self.config["experiment"]["save_models"]:
                model_path = os.path.join(
                    self.config["experiment"]["output_dir"],
                    "models",
                    f"{model_name}_predictive_masking",
                )
                compressor.save_model(model_path)

        for masking_prob in self.config["compression"]["masking_probabilities"]:
            self.logger.info(f"Testing masking probability: {masking_prob}")
            prob_results = {
                "compression_ratio": [],
                "word_accuracy": [],
                "character_accuracy": [],
                "semantic_similarity": [],
                "rouge1_fmeasure": [],
                "rouge2_fmeasure": [],
                "rougeL_fmeasure": [],
                "bert_score_f1": [],
                "bits_per_character": [],
            }

            for run in range(self.config["experiment"]["num_runs"]):
                run_metrics = []
                for text in tqdm(
                    test_texts,
                    desc=f"Run {run+1}/{self.config['experiment']['num_runs']}",
                ):
                    try:
                        compressed = compressor.compress(
                            text, masking_probability=masking_prob
                        )
                        reconstructed = compressor.decompress(compressed)
                        metrics = self.metrics_calculator.calculate_all_metrics(
                            text, reconstructed, compressed
                        )
                        run_metrics.append(metrics)
                    except Exception as e:
                        self.logger.error(f"Error processing text: {e}")
                        continue
                if run_metrics:
                    for key in prob_results:
                        values = [m.get(key, 0) for m in run_metrics]
                        prob_results[key].append(np.mean(values))

            results[masking_prob] = {
                key: np.mean(values) for key, values in prob_results.items()
            }
            if self.config["evaluation"]["save_reconstructions"]:
                examples = []
                for i in range(min(3, len(test_texts))):
                    compressed = compressor.compress(
                        test_texts[i], masking_probability=masking_prob
                    )
                    reconstructed = compressor.decompress(compressed)
                    examples.append(
                        {
                            "original": test_texts[i],
                            "reconstructed": reconstructed,
                            "model": model_name,
                            "masking_prob": masking_prob,
                        }
                    )
                results[masking_prob]["examples"] = examples

        return results

    def run_latent_space_quantization_experiments(
        self, model_name: str, train_texts: List[str], test_texts: List[str]
    ) -> Dict:
        results = {}
        compressor = LatentSpaceQuantizationCompressor(
            model_name=model_name,
            device=self.config["experiment"]["device"],
            quantization_bits=self.config["compression"]["quantization_bits"],
        )
        self.logger.info(f"Training decoder for {model_name}...")
        compressor.train_decoder(
            texts=train_texts[:500],
            epochs=5,
            batch_size=self.config["compression"]["batch_size"],
        )
        for idx, masking_prob in enumerate(
            self.config["compression"]["masking_probabilities"]
        ):
            quantization_bits = int(16 - (masking_prob * 14))
            self.logger.info(f"Testing {quantization_bits}-bit quantization")

            prob_results = {
                "compression_ratio": [],
                "word_accuracy": [],
                "character_accuracy": [],
                "semantic_similarity": [],
                "rouge1_fmeasure": [],
                "rouge2_fmeasure": [],
                "rougeL_fmeasure": [],
                "bert_score_f1": [],
                "bits_per_character": [],
            }
            for text in tqdm(test_texts, desc=f"LSQ {quantization_bits}-bit"):
                try:
                    compressed = compressor.compress(
                        text, quantization_bits=quantization_bits
                    )
                    reconstructed = compressor.decompress(compressed)
                    metrics = self.metrics_calculator.calculate_all_metrics(
                        text, reconstructed, compressed
                    )
                    for key in prob_results:
                        prob_results[key].append(metrics.get(key, 0))

                except Exception as e:
                    self.logger.error(f"Error processing text: {e}")
                    continue
            results[masking_prob] = {
                key: np.mean(values) for key, values in prob_results.items()
            }
            results[masking_prob]["quantization_bits"] = quantization_bits

        return results

    def save_results(self, results: Dict):
        results_path = os.path.join(
            self.config["experiment"]["output_dir"],
            "results",
            f"results_{datetime.now():%Y%m%d_%H%M%S}.json",
        )

        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, dict):
                return {
                    key: convert_to_serializable(value) for key, value in obj.items()
                }
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj

        serializable_results = convert_to_serializable(results)

        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        self.logger.info(f"Results saved to {results_path}")
        csv_data = []
        for model_method, model_results in results.items():
            for prob, metrics in model_results.items():
                if isinstance(prob, float):
                    row = {
                        "model_method": model_method,
                        "masking_probability": prob,
                        **{k: v for k, v in metrics.items() if not isinstance(v, list)},
                    }
                    csv_data.append(row)

        df = pd.DataFrame(csv_data)
        csv_path = results_path.replace(".json", ".csv")
        df.to_csv(csv_path, index=False)
        self.logger.info(f"CSV results saved to {csv_path}")

    def create_all_visualizations(self, results: Dict):
        pm_results = {
            k.replace("_predictive_masking", ""): v
            for k, v in results.items()
            if "predictive_masking" in k
        }
        lsq_results = {
            k.replace("_lsq", ""): v for k, v in results.items() if "lsq" in k
        }
        if pm_results:
            self.visualizer.create_summary_report(pm_results, "predictive_masking")
            for prob in [0.3, 0.5, 0.7]:
                self.visualizer.plot_model_comparison_radar(
                    pm_results, prob, f"pm_radar_{prob}"
                )
        if lsq_results:
            self.visualizer.create_summary_report(
                lsq_results, "latent_space_quantization"
            )

        self.visualizer.plot_compression_vs_accuracy(results, "all_methods_comparison")

        self.logger.info("All visualizations created!")


def main():
    parser = argparse.ArgumentParser(description="Run LLM compression experiments")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--models", nargs="+", help="Specific models to run (default: all)"
    )
    parser.add_argument(
        "--method",
        choices=["predictive_masking", "lsq", "both"],
        default="both",
        help="Compression method to use",
    )
    parser.add_argument(
        "--experiment",
        choices=["1a", "1b", "both"],
        default="1a",
        help="Which experiment to run",
    )
    parser.add_argument(
        "--probe-source",
        type=str,
        default="mixed",
        choices=["wikitext", "lama", "mixed"],
        help="Source for factual probes (for experiment 1b)",
    )

    args = parser.parse_args()
    if args.experiment in ["1a", "both"]:
        print("Running Experiment 1A: Text Compression...")
        runner = CompressionExperimentRunner(args.config)
        runner.run_all_experiments()

    if args.experiment in ["1b", "both"]:
        print("\nRunning Experiment 1B: Knowledge Compression...")
        from run_experiment_1b import Experiment1BRunner

        runner_1b = Experiment1BRunner(args.config)
        runner_1b.run_knowledge_compression_analysis(probe_source=args.probe_source)


if __name__ == "__main__":
    main()
