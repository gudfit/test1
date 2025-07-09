"""Evaluation metrics for compression quality."""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import Levenshtein


class CompressionMetrics:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")
        self.semantic_model = AutoModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        ).to(device)
        self.semantic_tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def calculate_all_metrics(
        self,
        original_text: str,
        reconstructed_text: str,
        compressed_data: Dict[str, any],
    ) -> Dict[str, float]:
        metrics = {}
        metrics["exact_match"] = float(
            original_text.strip() == reconstructed_text.strip()
        )
        metrics["character_accuracy"] = self.character_accuracy(
            original_text, reconstructed_text
        )
        metrics["word_accuracy"] = self.word_accuracy(original_text, reconstructed_text)
        metrics["levenshtein_distance"] = self.levenshtein_distance(
            original_text, reconstructed_text
        )
        rouge_scores = self.rouge_scores(original_text, reconstructed_text)
        metrics.update(rouge_scores)
        metrics["bleu_score"] = self.bleu_score(original_text, reconstructed_text)
        bert_scores = self.bert_score(original_text, reconstructed_text)
        metrics.update(bert_scores)
        metrics["semantic_similarity"] = self.semantic_similarity(
            original_text, reconstructed_text
        )
        metrics["compression_ratio"] = self.compression_ratio(
            original_text, compressed_data
        )
        metrics["bits_per_character"] = self.bits_per_character(
            original_text, compressed_data
        )
        return metrics

    def character_accuracy(self, original: str, reconstructed: str) -> float:
        if len(original) == 0:
            return 1.0 if len(reconstructed) == 0 else 0.0

        matches = sum(1 for o, r in zip(original, reconstructed) if o == r)
        return matches / max(len(original), len(reconstructed))

    def word_accuracy(self, original: str, reconstructed: str) -> float:
        original_words = original.split()
        reconstructed_words = reconstructed.split()
        if len(original_words) == 0:
            return 1.0 if len(reconstructed_words) == 0 else 0.0
        matches = sum(1 for o, r in zip(original_words, reconstructed_words) if o == r)
        return matches / max(len(original_words), len(reconstructed_words))

    def levenshtein_distance(self, original: str, reconstructed: str) -> float:
        if len(original) == 0 and len(reconstructed) == 0:
            return 0.0
        distance = Levenshtein.distance(original, reconstructed)
        return distance / max(len(original), len(reconstructed))

    def rouge_scores(self, original: str, reconstructed: str) -> Dict[str, float]:
        scores = self.rouge_scorer.score(original, reconstructed)

        return {
            "rouge1_precision": scores["rouge1"].precision,
            "rouge1_recall": scores["rouge1"].recall,
            "rouge1_fmeasure": scores["rouge1"].fmeasure,
            "rouge2_precision": scores["rouge2"].precision,
            "rouge2_recall": scores["rouge2"].recall,
            "rouge2_fmeasure": scores["rouge2"].fmeasure,
            "rougeL_precision": scores["rougeL"].precision,
            "rougeL_recall": scores["rougeL"].recall,
            "rougeL_fmeasure": scores["rougeL"].fmeasure,
        }

    def bleu_score(self, original: str, reconstructed: str) -> float:
        original_tokens = nltk.word_tokenize(original.lower())
        reconstructed_tokens = nltk.word_tokenize(reconstructed.lower())
        if len(original_tokens) == 0:
            return 1.0 if len(reconstructed_tokens) == 0 else 0.0
        smoothie = SmoothingFunction().method4
        score = sentence_bleu(
            [original_tokens], reconstructed_tokens, smoothing_function=smoothie
        )

        return score

    def bert_score(self, original: str, reconstructed: str) -> Dict[str, float]:
        P, R, F1 = bert_score(
            [reconstructed], [original], lang="en", device=self.device, verbose=False
        )

        return {
            "bert_score_precision": P.mean().item(),
            "bert_score_recall": R.mean().item(),
            "bert_score_f1": F1.mean().item(),
        }

    def semantic_similarity(self, original: str, reconstructed: str) -> float:
        with torch.no_grad():
            orig_encoding = self.semantic_tokenizer(
                original,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(self.device)

            orig_outputs = self.semantic_model(**orig_encoding)
            orig_embedding = self._mean_pooling(
                orig_outputs, orig_encoding["attention_mask"]
            )

            # Reconstructed text
            recon_encoding = self.semantic_tokenizer(
                reconstructed,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(self.device)

            recon_outputs = self.semantic_model(**recon_encoding)
            recon_embedding = self._mean_pooling(
                recon_outputs, recon_encoding["attention_mask"]
            )
        similarity = cosine_similarity(
            orig_embedding.cpu().numpy(), recon_embedding.cpu().numpy()
        )[0, 0]

        return float(similarity)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def compression_ratio(
        self, original_text: str, compressed_data: Dict[str, any]
    ) -> float:
        original_size = len(original_text.encode("utf-8")) * 8

        if "masked_ids" in compressed_data:
            mask_positions = compressed_data["mask_positions"]
            original_length = compressed_data["original_length"]
            compressed_size = (original_length - len(mask_positions)) * 32
            compressed_size += len(mask_positions) * np.ceil(np.log2(original_length))
        elif "quantized_states" in compressed_data:
            quantized_states = compressed_data["quantized_states"]
            quantization_bits = compressed_data["quantization_bits"]
            compressed_size = quantized_states.size * quantization_bits + 64
        else:
            compressed_size = original_size
        return original_size / compressed_size if compressed_size > 0 else float("inf")

    def bits_per_character(
        self, original_text: str, compressed_data: Dict[str, any]
    ) -> float:
        if len(original_text) == 0:
            return 0.0

        if "masked_ids" in compressed_data:
            mask_positions = compressed_data["mask_positions"]
            original_length = compressed_data["original_length"]
            compressed_bits = (original_length - len(mask_positions)) * 32
            compressed_bits += len(mask_positions) * np.ceil(np.log2(original_length))
        elif "quantized_states" in compressed_data:
            quantized_states = compressed_data["quantized_states"]
            quantization_bits = compressed_data["quantization_bits"]
            compressed_bits = quantized_states.size * quantization_bits + 64
        else:
            compressed_bits = len(original_text) * 8

        return compressed_bits / len(original_text)

    def perplexity(self, model, tokenizer, text: str) -> float:
        encoding = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = encoding["input_ids"].to(self.device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss).item()

        return perplexity
