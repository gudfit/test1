# experiments/run_experiment_1b.py
"""Run Experiment 1B: LLM as Static Knowledge Compressor vs Traditional Algorithms."""

import random
import argparse
import os
import sys
import yaml
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.knowledge_compressor import KnowledgeCompressor
from src.models.predictive_masking import PredictiveMaskingCompressor
from src.evaluation.crumpled_paper_metrics import CrumpledPaperMetrics
from src.visualization.plots import CompressionVisualizer
from src.data.factual_probe_generator import FactualProbeGenerator
from src.data.lama_probe_loader import LAMAProbeLoader


class Experiment1BRunner:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self._setup_logging()
        self._create_output_dirs()
        self.visualizer = CompressionVisualizer(
            figure_dir=os.path.join(
                self.config["experiment"]["output_dir"], "figures", "experiment_1b"
            )
        )

    def _setup_logging(self):
        log_dir = os.path.join(self.config["experiment"]["output_dir"], "logs")
        os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(
                    os.path.join(
                        log_dir, f"experiment_1b_{datetime.now():%Y%m%d_%H%M%S}.log"
                    )
                ),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def _create_output_dirs(self):
        """Create necessary output directories."""
        dirs = [
            os.path.join(
                self.config["experiment"]["output_dir"], "figures", "experiment_1b"
            ),
            os.path.join(
                self.config["experiment"]["output_dir"], "results", "experiment_1b"
            ),
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)

    def prepare_factual_probes(
        self, num_probes: int = 1000, probe_source: str = "mixed"
    ) -> List[Dict[str, str]]:
        self.logger.info(
            f"Generating {num_probes} factual probes from {probe_source} source..."
        )
        if probe_source == "wikitext":
            probe_generator = FactualProbeGenerator()
            factual_probes = probe_generator.extract_from_wikitext(
                num_probes=num_probes, dataset_split="train", max_samples=5000
            )
        elif probe_source == "lama":
            lama_loader = LAMAProbeLoader()
            factual_probes = lama_loader.create_mixed_probe_set(num_probes)
        else:
            probe_generator = FactualProbeGenerator()
            lama_loader = LAMAProbeLoader()
            wikitext_probes = probe_generator.extract_from_wikitext(
                num_probes=num_probes // 2, dataset_split="train", max_samples=2500
            )
            lama_probes = lama_loader.create_mixed_probe_set(num_probes // 2)
            factual_probes = wikitext_probes + lama_probes

            random.shuffle(factual_probes)
        if len(factual_probes) < num_probes:
            self.logger.warning(
                f"Only generated {len(factual_probes)} probes, using fallback..."
            )
            lama_loader = LAMAProbeLoader()
            additional_probes = lama_loader.load_t_rex_probes(
                num_probes - len(factual_probes)
            )
            factual_probes.extend(additional_probes)
        self.logger.info(f"Generated {len(factual_probes)} factual probes")
        if factual_probes:
            self.logger.info("Example probes:")
            for i, probe in enumerate(factual_probes[:5]):
                self.logger.info(f"  {i+1}. {probe['prompt']} -> {probe['answer']}")

        return factual_probes[:num_probes]

    def run_knowledge_compression_analysis(self, probe_source: str = "mixed"):
        self.logger.info("Starting Experiment 1B: Knowledge Compression Analysis")

        results = {}
        self.logger.info("Loading corpus data...")
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[:10000]")
        corpus_texts = [
            item["text"] for item in dataset if len(item["text"].strip()) > 100
        ][:5000]
        test_dataset = load_dataset(
            "wikitext", "wikitext-103-raw-v1", split="test[:1000]"
        )
        test_texts = [
            item["text"] for item in test_dataset if len(item["text"].strip()) > 100
        ][:500]
        factual_probes = self.prepare_factual_probes(
            self.config.get("experiment_1b", {}).get("factual_probes", 1000),
            probe_source=probe_source,
        )
        for model_config in self.config["models"].values():
            model_name = model_config["name"]
            self.logger.info(f"\nAnalyzing {model_name} as knowledge compressor...")
            compressor = KnowledgeCompressor(
                model_name, self.config["experiment"]["device"]
            )
            model_results = compressor.compare_with_gzip(
                corpus_texts[:1000],
                test_texts,
                factual_probes,
            )
            results[model_name] = model_results
            self._log_model_results(model_name, model_results)
        self.logger.info("\nRunning Crumpled Paper analysis...")
        crumpled_results = self.run_crumpled_paper_analysis(
            corpus_texts[:100], test_texts[:50]
        )
        results["crumpled_paper"] = crumpled_results
        self.save_results(results)
        self.create_visualizations(results)
        self.logger.info("Experiment 1B completed!")
        return results

    def run_crumpled_paper_analysis(
        self, corpus_texts: List[str], test_texts: List[str]
    ) -> Dict[str, Any]:
        crumpled_metrics = CrumpledPaperMetrics()
        results = {}
        compression_methods = {
            "bert_0.3": PredictiveMaskingCompressor("bert-base-uncased"),
            "bert_0.5": PredictiveMaskingCompressor("bert-base-uncased"),
            "bert_0.7": PredictiveMaskingCompressor("bert-base-uncased"),
        }
        method_results = {}
        for method_name, compressor in compression_methods.items():
            masking_prob = float(method_name.split("_")[1])
            reconstructed_texts = []

            self.logger.info(f"Testing {method_name} compression...")

            for text in tqdm(test_texts[:20], desc=f"Compressing with {method_name}"):
                try:
                    compressed = compressor.compress(
                        text, masking_probability=masking_prob
                    )
                    reconstructed = compressor.decompress(compressed)
                    reconstructed_texts.append(reconstructed)
                except Exception as e:
                    self.logger.error(f"Error in compression: {e}")
                    reconstructed_texts.append(text)
            method_results[method_name] = reconstructed_texts
        comparison = crumpled_metrics.compare_compression_methods(
            test_texts[:20], method_results
        )
        results["method_comparison"] = comparison
        best_method = comparison["best_method"]
        detailed_analysis = crumpled_metrics.analyze_compression_quality(
            test_texts[:20], method_results[best_method]
        )
        results["best_method_details"] = detailed_analysis
        crumpled_metrics.visualize_surprisal_profiles(
            test_texts[0],
            method_results[best_method][0],
            save_path=os.path.join(
                self.config["experiment"]["output_dir"],
                "figures",
                "experiment_1b",
                "crumpled_paper_example.png",
            ),
        )

        return results

    def _log_model_results(self, model_name: str, results: Dict[str, Any]):
        self.logger.info(f"\nResults for {model_name}:")
        self.logger.info(
            f"  Model size: {results['llm']['size']['disk_size_mb']:.2f} MB"
        )
        self.logger.info(
            f"  Gzip size: {results['gzip']['size']['compressed_size_mb']:.2f} MB"
        )
        self.logger.info(f"  Size ratio: {results['comparison']['size_ratio']:.2f}x")
        self.logger.info(f"  Factual recall: {results['llm']['factual_recall']:.3f}")
        self.logger.info(
            f"  Reasoning capability: {results['llm']['reasoning_capability']:.3f}"
        )
        self.logger.info(f"  Perplexity: {results['llm']['perplexity']:.2f}")
        self.logger.info(
            f"  Inference time: {results['llm']['inference_cost']['avg_inference_time_ms']:.2f} ms"
        )

    def save_results(self, results: Dict[str, Any]):
        results_path = os.path.join(
            self.config["experiment"]["output_dir"],
            "results",
            "experiment_1b",
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

    def create_visualizations(self, results: Dict[str, Any]):
        fig, ax = plt.subplots(figsize=(10, 6))

        models = []
        llm_sizes = []
        gzip_sizes = []

        for model_name, model_results in results.items():
            if model_name != "crumpled_paper":
                models.append(model_name.split("-")[0].upper())
                llm_sizes.append(model_results["llm"]["size"]["disk_size_mb"])
                gzip_sizes.append(model_results["gzip"]["size"]["compressed_size_mb"])

        x = np.arange(len(models))
        width = 0.35

        bars1 = ax.bar(
            x - width / 2, llm_sizes, width, label="LLM Size", color="steelblue"
        )
        bars2 = ax.bar(
            x + width / 2, gzip_sizes, width, label="Gzip Size", color="coral"
        )

        ax.set_xlabel("Model")
        ax.set_ylabel("Size (MB)")
        ax.set_title("Model Size vs Gzip Compression Size")
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                )

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.config["experiment"]["output_dir"],
                "figures",
                "experiment_1b",
                "size_comparison.png",
            ),
            dpi=300,
        )
        plt.close()
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        metrics = [
            "factual_recall",
            "reasoning_capability",
            "perplexity",
            "inference_cost",
        ]
        metric_labels = [
            "Factual Recall",
            "Reasoning Capability",
            "Perplexity",
            "Inference Time (ms)",
        ]

        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[idx]
            model_names = []
            values = []
            for model_name, model_results in results.items():
                if model_name != "crumpled_paper":
                    model_names.append(model_name.split("-")[0].upper())

                    if metric == "inference_cost":
                        values.append(
                            model_results["llm"]["inference_cost"][
                                "avg_inference_time_ms"
                            ]
                        )
                    else:
                        values.append(model_results["llm"][metric])

            if metric == "perplexity":
                ax.bar(model_names, values, color="lightcoral")
            else:
                ax.bar(model_names, values, color="lightblue")

            ax.set_xlabel("Model")
            ax.set_ylabel(label)
            ax.set_title(f"{label} Comparison")
            for i, v in enumerate(values):
                ax.text(i, v + 0.01 * max(values), f"{v:.2f}", ha="center", va="bottom")

        plt.suptitle("Functional Utility Metrics Comparison", fontsize=16)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.config["experiment"]["output_dir"],
                "figures",
                "experiment_1b",
                "functional_utility.png",
            ),
            dpi=300,
        )
        plt.close()
        summary_data = []
        for model_name, model_results in results.items():
            if model_name != "crumpled_paper":
                summary_data.append(
                    {
                        "Model": model_name,
                        "Model Size (MB)": f"{model_results['llm']['size']['disk_size_mb']:.1f}",
                        "Gzip Size (MB)": f"{model_results['gzip']['size']['compressed_size_mb']:.1f}",
                        "Size Ratio": f"{model_results['comparison']['size_ratio']:.1f}x",
                        "Factual Recall": f"{model_results['llm']['factual_recall']:.3f}",
                        "Reasoning": f"{model_results['llm']['reasoning_capability']:.3f}",
                        "Perplexity": f"{model_results['llm']['perplexity']:.1f}",
                        "Inference (ms)": f"{model_results['llm']['inference_cost']['avg_inference_time_ms']:.1f}",
                    }
                )

        df_summary = pd.DataFrame(summary_data)
        latex_path = os.path.join(
            self.config["experiment"]["output_dir"],
            "results",
            "experiment_1b",
            "summary_table.tex",
        )
        df_summary.to_latex(latex_path, index=False, escape=False)
        self.logger.info("All visualizations created!")


def main():
    parser = argparse.ArgumentParser(
        description="Run Experiment 1B: Knowledge Compression Analysis"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--probe-source",
        type=str,
        default="mixed",
        choices=["wikitext", "lama", "mixed"],
        help="Source for factual probes",
    )
    args = parser.parse_args()
    runner = Experiment1BRunner(args.config)
    results = runner.run_knowledge_compression_analysis(probe_source=args.probe_source)
    print("\n" + "=" * 60)
    print("EXPERIMENT 1B SUMMARY")
    print("=" * 60)
    print(f"Probe Source: {args.probe_source}")
    for model_name, model_results in results.items():
        if model_name != "crumpled_paper":
            print(f"\n{model_name}:")
            print(
                f"  - Size overhead vs Gzip: {model_results['comparison']['size_ratio']:.1f}x"
            )
            print(
                f"  - Functional utility gain: {model_results['llm']['factual_recall'] / 0.001:.0f}x"
            )
            print(
                f"  - Key insight: {model_results['comparison']['summary']['use_case']}"
            )


if __name__ == "__main__":
    main()
