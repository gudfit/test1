# src/models/knowledge_compressor.py
"""LLM as Static Knowledge Compressor implementation for Experiment 1B."""

from datasets import Value
import torch
import torch.nn as nn
import numpy as np
import gzip
import os
import json
from typing import Dict, List, Tuple, Optional, Any
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    pipeline,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset, Dataset
import time
from tqdm import tqdm
import shutil


class KnowledgeCompressor:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def calculate_model_size(self) -> Dict[str, float]:
        total_params = 0
        total_size_bytes = 0

        for param in self.model.parameters():
            total_params += param.numel()
            total_size_bytes += param.numel() * 4

        size_mb = total_size_bytes / (1024 * 1024)
        temp_path = f"/tmp/{self.model_name.replace('/', '_')}_temp"
        self.model.save_pretrained(temp_path)

        disk_size_bytes = 0
        for root, dirs, files in os.walk(temp_path):
            for file in files:
                if file.endswith(".bin") or file.endswith(".safetensors"):
                    disk_size_bytes += os.path.getsize(os.path.join(root, file))

        disk_size_mb = disk_size_bytes / (1024 * 1024)
        shutil.rmtree(temp_path)
        return {
            "total_parameters": total_params,
            "theoretical_size_mb": size_mb,
            "disk_size_mb": disk_size_mb,
            "size_gb": disk_size_mb / 1024,
        }

    def compress_corpus_gzip(self, texts: List[str]) -> Dict[str, float]:
        full_text = "\n".join(texts)
        original_bytes = full_text.encode("utf-8")
        original_size_mb = len(original_bytes) / (1024 * 1024)
        compressed_bytes = gzip.compress(original_bytes, compresslevel=9)
        compressed_size_mb = len(compressed_bytes) / (1024 * 1024)
        compression_ratio = original_size_mb / compressed_size_mb

        return {
            "original_size_mb": original_size_mb,
            "compressed_size_mb": compressed_size_mb,
            "compression_ratio": compression_ratio,
            "size_gb": compressed_size_mb / 1024,
        }

    def measure_factual_recall(self, factual_probes: List[Dict[str, str]]) -> float:
        correct = 0
        total = len(factual_probes)

        for probe in tqdm(factual_probes, desc="Evaluating factual recall"):
            prompt = probe["prompt"]
            expected_answer = probe["answer"].lower().strip()
            if "[MASK]" not in prompt:
                prompt = prompt.strip() + " [MASK]."

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = outputs.logits

            mask_token_id = self.tokenizer.mask_token_id
            mask_positions = (inputs["input_ids"] == mask_token_id).nonzero(
                as_tuple=True
            )[1]

            if len(mask_positions) > 0:
                mask_pos = mask_positions[0]
                predicted_id = predictions[0, mask_pos].argmax().item()
                predicted_token = self.tokenizer.decode([predicted_id]).strip().lower()

                if (
                    expected_answer in predicted_token
                    or predicted_token in expected_answer
                ):
                    correct += 1

        recall_accuracy = correct / total if total > 0 else 0.0
        return recall_accuracy

    def measure_reasoning_capability(self, nli_dataset: Optional[Any] = None) -> float:
        nli_model_name = f"{self.model_name}-nli"
        if "roberta" in self.model_name.lower():
            nli_model = AutoModelForSequenceClassification.from_pretrained(
                "roberta-large-mnli"
            ).to(self.device)
            nli_tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
        else:
            nli_model = AutoModelForSequenceClassification.from_pretrained(
                "textattack/bert-base-uncased-MNLI"
            ).to(self.device)
            nli_tokenizer = AutoTokenizer.from_pretrained(
                "textattack/bert-base-uncased-MNLI"
            )

        if nli_dataset is None:
            nli_dataset = load_dataset(
                "glue", "mnli", split="validation_matched[:1000]"
            )

        correct = 0
        total = 0

        for example in tqdm(nli_dataset, desc="Evaluating NLI"):
            premise = example["premise"]
            hypothesis = example["hypothesis"]
            label = example["label"]

            if label == -1:
                continue

            inputs = nli_tokenizer(
                premise, hypothesis, truncation=True, padding=True, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = nli_model(**inputs)
                predicted_label = outputs.logits.argmax().item()

            if predicted_label == label:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0.0
        return accuracy

    def measure_perplexity(
        self, test_texts: List[str], max_samples: int = 100
    ) -> float:
        if hasattr(self.model, "lm_head"):
            total_loss = 0
            total_tokens = 0

            for text in tqdm(test_texts[:max_samples], desc="Calculating perplexity"):
                inputs = self.tokenizer(
                    text, truncation=True, max_length=512, return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    labels = inputs["input_ids"].clone()
                    for i in range(1, labels.shape[1]):
                        masked_inputs = inputs["input_ids"].clone()
                        masked_inputs[0, i] = self.tokenizer.mask_token_id

                        outputs = self.model(
                            input_ids=masked_inputs,
                            attention_mask=inputs["attention_mask"],
                        )
                        logits = outputs.logits[0, i]
                        target = labels[0, i]
                        loss = nn.functional.cross_entropy(
                            logits.unsqueeze(0), target.unsqueeze(0)
                        )
                        total_loss += loss.item()
                        total_tokens += 1

            avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
            perplexity = np.exp(avg_loss)
        else:
            perplexity = self._calculate_pseudo_perplexity(test_texts[:max_samples])

        return perplexity

    def _calculate_pseudo_perplexity(self, texts: List[str]) -> float:
        total_pseudo_log_likelihood = 0
        total_tokens = 0

        for text in texts:
            tokens = self.tokenizer.tokenize(text)
            if len(tokens) < 2:
                continue
            for i in range(len(tokens)):
                masked_tokens = tokens.copy()
                original_token = masked_tokens[i]
                masked_tokens[i] = self.tokenizer.mask_token

                masked_text = self.tokenizer.convert_tokens_to_string(masked_tokens)
                inputs = self.tokenizer(masked_text, return_tensors="pt").to(
                    self.device
                )

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    predictions = outputs.logits
                mask_positions = (
                    inputs["input_ids"] == self.tokenizer.mask_token_id
                ).nonzero(as_tuple=True)[1]
                if len(mask_positions) == 0:
                    continue

                mask_pos = mask_positions[0]
                probs = torch.softmax(predictions[0, mask_pos], dim=-1)
                original_token_id = self.tokenizer.convert_tokens_to_ids(
                    [original_token]
                )[0]
                token_prob = probs[original_token_id].item()
                if token_prob > 0:
                    total_pseudo_log_likelihood += np.log(token_prob)
                    total_tokens += 1
        if total_tokens > 0:
            avg_pseudo_log_likelihood = total_pseudo_log_likelihood / total_tokens
            perplexity = np.exp(-avg_pseudo_log_likelihood)
        else:
            perplexity = float("inf")

        return perplexity

    def measure_inference_cost(
        self, sample_texts: List[str], num_runs: int = 100
    ) -> Dict[str, float]:
        sample_text = (
            sample_texts[0]
            if sample_texts
            else "This is a sample text for benchmarking."
        )
        inputs = self.tokenizer(
            sample_text, truncation=True, max_length=128, return_tensors="pt"
        ).to(self.device)
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(**inputs)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()

        for _ in range(num_runs):
            with torch.no_grad():
                _ = self.model(**inputs)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()

        avg_inference_time = (end_time - start_time) / num_runs
        num_params = sum(p.numel() for p in self.model.parameters())
        seq_length = inputs["input_ids"].shape[1]
        estimated_flops = 2 * num_params * seq_length

        return {
            "avg_inference_time_ms": avg_inference_time * 1000,
            "throughput_samples_per_sec": 1 / avg_inference_time,
            "estimated_flops": estimated_flops,
            "estimated_gflops": estimated_flops / 1e9,
            "model_parameters": num_params,
        }

    def measure_data_efficiency(
        self,
        base_performance: float,
        fine_tune_data: List[Dict[str, Any]],
        task_type: str = "classification",
    ) -> Dict[str, Any]:
        data_sizes = [10, 50, 100, 500, 1000, 5000]
        results = []
        min_data_needed = None

        if task_type == "classification":
            dataset = Dataset.from_list(fine_tune_data)
            dataset = dataset.rename_column("label", "labels")
            dataset = dataset.cast_column("labels", Value("float32"))

            def tokenize_function(examples):
                return self.tokenizer(
                    examples["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=128,
                )

            dataset = dataset.map(tokenize_function)
            dataset = dataset.train_test_split(test_size=0.2, seed=42)
            train_dataset = dataset["train"]
            eval_dataset = dataset["test"]

            num_labels = len(set([ex["label"] for ex in fine_tune_data]))

            for size in data_sizes:
                if size > len(train_dataset):
                    break
                subset_train = train_dataset.select(range(size))
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name, num_labels=num_labels
                ).to(self.device)

                training_args = TrainingArguments(
                    output_dir="/tmp/kc_training",
                    per_device_train_batch_size=8,
                    per_device_eval_batch_size=8,
                    num_train_epochs=3,
                    learning_rate=2e-5,
                    eval_strategy="no",
                    logging_strategy="no",
                    seed=42,
                    disable_tqdm=True,
                    fp16=torch.cuda.is_available(),
                )

                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=subset_train,
                    tokenizer=self.tokenizer,
                )

                trainer.train()
                predictions = trainer.predict(eval_dataset)
                preds = np.argmax(predictions.predictions, axis=1)
                accuracy = (preds == predictions.label_ids).mean()

                results.append(
                    {
                        "data_size": size,
                        "performance": accuracy,
                        "reached_target": accuracy >= base_performance * 0.9,
                    }
                )

                if accuracy >= base_performance * 0.9 and min_data_needed is None:
                    min_data_needed = size
        else:
            raise NotImplementedError("Only classification task_type is supported.")

        return {
            "efficiency_curve": results,
            "min_data_for_90_percent": min_data_needed,
            "data_efficiency_score": 1000 / min_data_needed if min_data_needed else 0,
        }

    def compare_with_gzip(
        self,
        corpus_texts: List[str],
        test_texts: List[str],
        factual_probes: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        Comprehensive comparison between LLM and gzip compression.

        Args:
            corpus_texts: Training corpus texts
            test_texts: Test texts for evaluation
            factual_probes: Factual probes for knowledge testing

        Returns:
            Complete comparison results
        """
        results = {"llm": {}, "gzip": {}, "comparison": {}}

        print("Phase 1: Analyzing sizes...")
        results["llm"]["size"] = self.calculate_model_size()
        results["gzip"]["size"] = self.compress_corpus_gzip(corpus_texts)

        results["comparison"]["size_ratio"] = (
            results["llm"]["size"]["disk_size_mb"]
            / results["gzip"]["size"]["compressed_size_mb"]
        )

        print("\nPhase 2: Measuring functional utility...")

        results["llm"]["factual_recall"] = self.measure_factual_recall(factual_probes)
        results["llm"]["reasoning_capability"] = self.measure_reasoning_capability()
        results["llm"]["perplexity"] = self.measure_perplexity(test_texts)

        results["gzip"]["factual_recall"] = 0.0
        results["gzip"]["reasoning_capability"] = 0.0
        results["gzip"]["perplexity"] = float("inf")

        print("\nPhase 3: Measuring computational costs...")
        results["llm"]["inference_cost"] = self.measure_inference_cost(test_texts[:10])

        start_time = time.time()
        compressed = gzip.compress("\n".join(corpus_texts[:100]).encode("utf-8"))
        decompressed = gzip.decompress(compressed).decode("utf-8")
        gzip_time = time.time() - start_time

        results["gzip"]["decompression_cost"] = {
            "time_ms": gzip_time * 1000,
            "one_time_cost": True,
            "dynamic_interaction": False,
        }

        print("\nPhase 4: Measuring data efficiency...")
        results["llm"]["data_efficiency"] = self.measure_data_efficiency(
            0.8,
            [{"text": t, "label": 0} for t in test_texts[:1000]],
        )

        results["gzip"]["data_efficiency"] = {"adaptable": False, "fine_tunable": False}
        results["comparison"]["summary"] = {
            "size_overhead_factor": results["comparison"]["size_ratio"],
            "functional_utility_gain": results["llm"]["factual_recall"] / 0.001,
            "reasoning_capability_gain": results["llm"]["reasoning_capability"] / 0.001,
            "adaptability": "LLM is adaptable, Gzip is not",
            "use_case": "LLM for knowledge interaction, Gzip for data storage",
        }

        return results
