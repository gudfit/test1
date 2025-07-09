# src/evaluation/crumpled_paper_metrics.py
"""Crumpled Paper Model for quantifying fidelity degradation in text compression."""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import seaborn as sns


class CrumpledPaperMetrics:
    def __init__(self, oracle_model_name: str = "gpt2-large", device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.oracle_tokenizer = AutoTokenizer.from_pretrained(oracle_model_name)
        self.oracle_model = AutoModelForCausalLM.from_pretrained(oracle_model_name).to(
            self.device
        )
        self.oracle_model.eval()
        if self.oracle_tokenizer.pad_token is None:
            self.oracle_tokenizer.pad_token = self.oracle_tokenizer.eos_token

    def calculate_surprisal_profile(self, text: str) -> np.ndarray:
        inputs = self.oracle_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=1024
        ).to(self.device)

        input_ids = inputs["input_ids"]
        surprisal_values = []
        with torch.no_grad():
            outputs = self.oracle_model(input_ids)
            logits = outputs.logits
            for i in range(1, input_ids.shape[1]):
                log_probs = torch.log_softmax(logits[0, i - 1], dim=-1)
                actual_token_id = input_ids[0, i]
                log_prob = log_probs[actual_token_id].item()
                surprisal = -log_prob / np.log(2)
                surprisal_values.append(surprisal)

        return np.array(surprisal_values)

    def calculate_crease_metrics(
        self, original_text: str, reconstructed_text: str
    ) -> Dict[str, float]:
        original_surprisal = self.calculate_surprisal_profile(original_text)
        reconstructed_surprisal = self.calculate_surprisal_profile(reconstructed_text)
        min_length = min(len(original_surprisal), len(reconstructed_surprisal))
        original_surprisal = original_surprisal[:min_length]
        reconstructed_surprisal = reconstructed_surprisal[:min_length]
        surprisal_diff = np.abs(reconstructed_surprisal - original_surprisal)
        tcm = np.sum(surprisal_diff)
        pcm = np.max(surprisal_diff) if len(surprisal_diff) > 0 else 0.0
        mean_crease = np.mean(surprisal_diff) if len(surprisal_diff) > 0 else 0.0
        std_crease = np.std(surprisal_diff) if len(surprisal_diff) > 0 else 0.0
        if std_crease > 0:
            significant_creases = np.sum(
                surprisal_diff > (mean_crease + 2 * std_crease)
            )
        else:
            significant_creases = 0

        return {
            "total_crease_magnitude": tcm,
            "peak_crease_magnitude": pcm,
            "mean_crease": mean_crease,
            "std_crease": std_crease,
            "significant_creases": int(significant_creases),
            "crease_density": (
                significant_creases / len(surprisal_diff)
                if len(surprisal_diff) > 0
                else 0.0
            ),
            "profile_length": len(surprisal_diff),
        }

    def analyze_compression_quality(
        self, original_texts: List[str], reconstructed_texts: List[str]
    ) -> Dict[str, Any]:
        all_metrics = []
        for orig, recon in zip(original_texts, reconstructed_texts):
            metrics = self.calculate_crease_metrics(orig, recon)
            all_metrics.append(metrics)
        aggregated = {
            "mean_tcm": np.mean([m["total_crease_magnitude"] for m in all_metrics]),
            "std_tcm": np.std([m["total_crease_magnitude"] for m in all_metrics]),
            "mean_pcm": np.mean([m["peak_crease_magnitude"] for m in all_metrics]),
            "std_pcm": np.std([m["peak_crease_magnitude"] for m in all_metrics]),
            "mean_crease_density": np.mean([m["crease_density"] for m in all_metrics]),
            "total_samples": len(all_metrics),
            "individual_metrics": all_metrics,
        }
        aggregated["quality_assessment"] = self._assess_quality(aggregated)

        return aggregated

    def _assess_quality(self, metrics: Dict[str, float]) -> str:
        mean_tcm = metrics["mean_tcm"]
        mean_pcm = metrics["mean_pcm"]

        if mean_tcm < 10 and mean_pcm < 5:
            return "Excellent - Near-perfect reconstruction"
        elif mean_tcm < 50 and mean_pcm < 10:
            return "Good - Minor imperfections"
        elif mean_tcm < 100 and mean_pcm < 20:
            return "Acceptable - Noticeable but manageable degradation"
        elif mean_tcm < 200 and mean_pcm < 30:
            return "Poor - Significant structural damage"
        else:
            return "Failed - Severe degradation"

    def visualize_surprisal_profiles(
        self,
        original_text: str,
        reconstructed_text: str,
        save_path: Optional[str] = None,
    ):
        original_surprisal = self.calculate_surprisal_profile(original_text)
        reconstructed_surprisal = self.calculate_surprisal_profile(reconstructed_text)
        min_length = min(len(original_surprisal), len(reconstructed_surprisal))
        original_surprisal = original_surprisal[:min_length]
        reconstructed_surprisal = reconstructed_surprisal[:min_length]
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        axes[0].plot(original_surprisal, "b-", alpha=0.7, label="Original")
        axes[0].set_ylabel("Surprisal (bits)")
        axes[0].set_title("Original Text Surprisal Profile")
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(reconstructed_surprisal, "r-", alpha=0.7, label="Reconstructed")
        axes[1].set_ylabel("Surprisal (bits)")
        axes[1].set_title("Reconstructed Text Surprisal Profile")
        axes[1].grid(True, alpha=0.3)
        surprisal_diff = np.abs(reconstructed_surprisal - original_surprisal)
        axes[2].bar(
            range(len(surprisal_diff)), surprisal_diff, color="orange", alpha=0.7
        )
        axes[2].set_xlabel("Token Position")
        axes[2].set_ylabel("Crease Magnitude")
        axes[2].set_title("Crease Profile (Absolute Surprisal Difference)")
        axes[2].grid(True, alpha=0.3)
        tcm = np.sum(surprisal_diff)
        pcm = np.max(surprisal_diff)
        axes[2].text(
            0.02,
            0.95,
            f"TCM: {tcm:.2f}\nPCM: {pcm:.2f}",
            transform=axes[2].transAxes,
            bbox=dict(boxstyle="round", facecolor="wheat"),
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()

        plt.close()

    def create_heatmap_visualization(
        self,
        compression_results: Dict[str, List[Dict]],
        save_path: Optional[str] = None,
    ):
        settings = list(compression_results.keys())
        metrics_names = ["TCM", "PCM", "Mean Crease", "Crease Density"]
        data = []
        for setting in settings:
            metrics = compression_results[setting]
            row = [
                np.mean([m["total_crease_magnitude"] for m in metrics]),
                np.mean([m["peak_crease_magnitude"] for m in metrics]),
                np.mean([m["mean_crease"] for m in metrics]),
                np.mean([m["crease_density"] for m in metrics]) * 100,
            ]
            data.append(row)

        data = np.array(data).T
        plt.figure(figsize=(10, 6))
        normalized_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            row_min = data[i].min()
            row_max = data[i].max()
            if row_max > row_min:
                normalized_data[i] = (data[i] - row_min) / (row_max - row_min)
            else:
                normalized_data[i] = 0.5

        sns.heatmap(
            normalized_data,
            xticklabels=settings,
            yticklabels=metrics_names,
            annot=data,
            fmt=".2f",
            cmap="RdYlBu_r",
            cbar_kws={"label": "Normalized Value"},
        )

        plt.title("Crumpled Paper Metrics Across Compression Settings")
        plt.xlabel("Compression Setting")
        plt.ylabel("Metric")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()

        plt.close()

    def compare_compression_methods(
        self, original_texts: List[str], method_results: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        comparison = {}
        for method_name, reconstructed_texts in method_results.items():
            analysis = self.analyze_compression_quality(
                original_texts, reconstructed_texts
            )

            comparison[method_name] = {
                "mean_tcm": analysis["mean_tcm"],
                "std_tcm": analysis["std_tcm"],
                "mean_pcm": analysis["mean_pcm"],
                "std_pcm": analysis["std_pcm"],
                "crease_density": analysis["mean_crease_density"],
                "quality": analysis["quality_assessment"],
            }

        methods_ranked = sorted(
            comparison.keys(), key=lambda x: comparison[x]["mean_tcm"]
        )
        return {
            "method_metrics": comparison,
            "ranking": methods_ranked,
            "best_method": methods_ranked[0],
            "worst_method": methods_ranked[-1],
        }
