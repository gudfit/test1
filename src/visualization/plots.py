# src/visualization/plots.py
"""Visualization utilities for compression experiments."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class CompressionVisualizer:
    def __init__(
        self,
        style: str = "seaborn-v0_8-darkgrid",
        figure_dir: str = "./results/figures",
    ):
        plt.style.use(style)
        self.figure_dir = figure_dir
        import os

        os.makedirs(figure_dir, exist_ok=True)

    def plot_compression_vs_accuracy(
        self, results: Dict[str, Dict], save_name: str = "compression_vs_accuracy"
    ):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        metrics_to_plot = [
            ("compression_ratio", "Compression Ratio"),
            ("word_accuracy", "Word Accuracy"),
            ("semantic_similarity", "Semantic Similarity"),
            ("rouge1_fmeasure", "ROUGE-1 F1 Score"),
        ]
        for idx, (metric_key, metric_label) in enumerate(metrics_to_plot):
            ax = axes[idx]
            for model_name, model_results in results.items():
                masking_probs = []
                metric_values = []
                for prob, metrics in model_results.items():
                    if isinstance(prob, float):
                        masking_probs.append(prob)
                        metric_values.append(metrics.get(metric_key, 0))
                sorted_data = sorted(zip(masking_probs, metric_values))
                masking_probs, metric_values = zip(*sorted_data)
                ax.plot(
                    masking_probs,
                    metric_values,
                    marker="o",
                    label=model_name,
                    linewidth=2,
                )

            ax.set_xlabel("Masking Probability", fontsize=12)
            ax.set_ylabel(metric_label, fontsize=12)
            ax.set_title(f"{metric_label} vs Masking Probability", fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle("Compression Performance Metrics", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{self.figure_dir}/{save_name}.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_interactive_comparison(
        self, results: Dict[str, Dict], save_name: str = "interactive_comparison"
    ):
        data_frames = []
        for model_name, model_results in results.items():
            for prob, metrics in model_results.items():
                if isinstance(prob, float):
                    data_frames.append(
                        {
                            "Model": model_name,
                            "Masking Probability": prob,
                            "Compression Ratio": metrics.get("compression_ratio", 0),
                            "Word Accuracy": metrics.get("word_accuracy", 0),
                            "Semantic Similarity": metrics.get(
                                "semantic_similarity", 0
                            ),
                            "ROUGE-1 F1": metrics.get("rouge1_fmeasure", 0),
                            "BERT Score F1": metrics.get("bert_score_f1", 0),
                        }
                    )

        df = pd.DataFrame(data_frames)
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Compression Ratio",
                "Word Accuracy",
                "Semantic Similarity",
                "ROUGE-1 F1",
            ),
        )

        metrics = [
            "Compression Ratio",
            "Word Accuracy",
            "Semantic Similarity",
            "ROUGE-1 F1",
        ]
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

        for metric, (row, col) in zip(metrics, positions):
            for model in df["Model"].unique():
                model_data = df[df["Model"] == model].sort_values("Masking Probability")

                fig.add_trace(
                    go.Scatter(
                        x=model_data["Masking Probability"],
                        y=model_data[metric],
                        mode="lines+markers",
                        name=f"{model}",
                        showlegend=(row == 1 and col == 1),
                        line=dict(width=2),
                        marker=dict(size=8),
                    ),
                    row=row,
                    col=col,
                )

        fig.update_xaxes(title_text="Masking Probability")
        fig.update_layout(
            title_text="Interactive Compression Performance Comparison",
            height=800,
            hovermode="x unified",
        )

        fig.write_html(f"{self.figure_dir}/{save_name}.html")

    def plot_heatmap_comparison(
        self,
        results: Dict[str, Dict],
        metric: str = "semantic_similarity",
        save_name: str = "heatmap_comparison",
    ):
        models = list(results.keys())
        masking_probs = sorted(
            [p for p in results[models[0]].keys() if isinstance(p, float)]
        )

        matrix = np.zeros((len(models), len(masking_probs)))

        for i, model in enumerate(models):
            for j, prob in enumerate(masking_probs):
                matrix[i, j] = results[model][prob].get(metric, 0)
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            matrix,
            xticklabels=[f"{p:.1f}" for p in masking_probs],
            yticklabels=models,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            cbar_kws={"label": metric.replace("_", " ").title()},
        )

        plt.xlabel("Masking Probability", fontsize=12)
        plt.ylabel("Model", fontsize=12)
        plt.title(f'{metric.replace("_", " ").title()} Heatmap', fontsize=14)

        plt.tight_layout()
        plt.savefig(f"{self.figure_dir}/{save_name}.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_bits_per_token_analysis(
        self, results: Dict[str, Dict], save_name: str = "bits_per_token"
    ):
        plt.figure(figsize=(12, 8))
        for model_name, model_results in results.items():
            masking_probs = []
            bpt_values = []
            for prob, metrics in model_results.items():
                if isinstance(prob, float):
                    masking_probs.append(prob)
                    bpt_values.append(metrics.get("bits_per_character", 0) * 4)
            sorted_data = sorted(zip(masking_probs, bpt_values))
            masking_probs, bpt_values = zip(*sorted_data)

            plt.plot(
                masking_probs, bpt_values, marker="o", label=model_name, linewidth=2
            )

        plt.xlabel("Masking Probability", fontsize=12)
        plt.ylabel("Bits Per Token", fontsize=12)
        plt.title("Compression Efficiency: Bits Per Token", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(
            y=32, color="red", linestyle="--", alpha=0.5, label="Uncompressed (32 bits)"
        )

        plt.tight_layout()
        plt.savefig(f"{self.figure_dir}/{save_name}.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_model_comparison_radar(
        self,
        results: Dict[str, Dict],
        masking_prob: float = 0.5,
        save_name: str = "radar_comparison",
    ):
        metrics = [
            "word_accuracy",
            "semantic_similarity",
            "rouge1_fmeasure",
            "bert_score_f1",
            "compression_ratio",
        ]
        metric_labels = [
            "Word\nAccuracy",
            "Semantic\nSimilarity",
            "ROUGE-1\nF1",
            "BERT Score\nF1",
            "Compression\nRatio",
        ]

        fig = go.Figure()

        for model_name, model_results in results.items():
            if masking_prob in model_results:
                values = []
                for metric in metrics:
                    value = model_results[masking_prob].get(metric, 0)
                    if metric == "compression_ratio":
                        value = min(value / 10, 1.0)
                    values.append(value)

                fig.add_trace(
                    go.Scatterpolar(
                        r=values, theta=metric_labels, fill="toself", name=model_name
                    )
                )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title=f"Model Comparison at {masking_prob:.0%} Masking Probability",
        )

        fig.write_html(f"{self.figure_dir}/{save_name}.html")

    def plot_reconstruction_examples(
        self, examples: List[Dict[str, str]], save_name: str = "reconstruction_examples"
    ):
        fig, axes = plt.subplots(len(examples), 1, figsize=(14, 4 * len(examples)))
        if len(examples) == 1:
            axes = [axes]

        for idx, (ax, example) in enumerate(zip(axes, examples)):
            original = example["original"]
            reconstructed = example["reconstructed"]

            original_words = original.split()
            reconstructed_words = reconstructed.split()

            # Create colored text visualization
            ax.text(0.02, 0.9, "Original:", transform=ax.transAxes, fontweight="bold")
            ax.text(0.02, 0.7, original, transform=ax.transAxes, wrap=True, fontsize=10)

            ax.text(
                0.02, 0.5, "Reconstructed:", transform=ax.transAxes, fontweight="bold"
            )
            ax.text(
                0.02, 0.3, reconstructed, transform=ax.transAxes, wrap=True, fontsize=10
            )

            ax.text(
                0.02,
                0.1,
                f"Model: {example['model']} | Masking: {example['masking_prob']:.0%}",
                transform=ax.transAxes,
                fontsize=9,
                style="italic",
            )

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")

        plt.suptitle("Reconstruction Examples", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{self.figure_dir}/{save_name}.png", dpi=300, bbox_inches="tight")
        plt.close()

    def create_summary_report(
        self, results: Dict[str, Dict], save_name: str = "summary_report"
    ):
        self.plot_compression_vs_accuracy(results, f"{save_name}_accuracy")
        self.plot_interactive_comparison(results, f"{save_name}_interactive")
        self.plot_heatmap_comparison(
            results, "semantic_similarity", f"{save_name}_heatmap_semantic"
        )
        self.plot_heatmap_comparison(
            results, "compression_ratio", f"{save_name}_heatmap_compression"
        )
        self.plot_bits_per_token_analysis(results, f"{save_name}_bits_per_token")
        self._create_summary_table(results, f"{save_name}_table")

    def _create_summary_table(self, results: Dict[str, Dict], save_name: str):
        summary_data = []
        for model_name, model_results in results.items():
            for prob, metrics in model_results.items():
                if isinstance(prob, float):
                    summary_data.append(
                        {
                            "Model": model_name,
                            "Masking Prob": prob,
                            "Compression Ratio": f"{metrics.get('compression_ratio', 0):.2f}",
                            "Word Accuracy": f"{metrics.get('word_accuracy', 0):.3f}",
                            "Semantic Similarity": f"{metrics.get('semantic_similarity', 0):.3f}",
                            "ROUGE-1 F1": f"{metrics.get('rouge1_fmeasure', 0):.3f}",
                            "BERT Score F1": f"{metrics.get('bert_score_f1', 0):.3f}",
                        }
                    )

        df = pd.DataFrame(summary_data)
        fig, ax = plt.subplots(figsize=(14, len(summary_data) * 0.5 + 2))
        ax.axis("tight")
        ax.axis("off")
        table = ax.table(
            cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor("#40466e")
            table[(0, i)].set_text_props(weight="bold", color="white")

        plt.title("Compression Experiment Results Summary", fontsize=16, pad=20)
        plt.savefig(f"{self.figure_dir}/{save_name}.png", dpi=300, bbox_inches="tight")
        plt.close()
