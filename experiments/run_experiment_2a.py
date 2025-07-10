# FILE: experiments/run_experiment_2a.py
"""
Run Experiment 2A: An Empirical Instantiation of Storage and Retrieval Loss.

This experiment aims to provide a concrete, empirical instantiation of the 
theoretical framework for storage and retrieval loss decomposition. It evaluates 
various Llama-2 models (base, quantized, pruned) to measure how information 
degrades due to storage constraints (model size, precision) and retrieval 
constraints (inference-time context).
"""

import os
import sys
import yaml
import json
import logging
import time
import argparse
from datetime import datetime
from typing import Dict, List, Any, Tuple
import shutil

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from scipy.optimize import curve_fit
import torch.nn.utils.prune as prune

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.metrics import CompressionMetrics


class Experiment2ARunner:
    """Runner for Experiment 2A: Storage vs. Retrieval Loss."""

    def __init__(self, config_path: str, hf_token: str):
        """Initialize experiment runner."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['experiment_2a']
        self.hf_token = hf_token
        self._setup_logging()
        self._create_output_dirs()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics_calculator = CompressionMetrics(device=self.device)
        self.logger.info(f"Using device: {self.device}")

    def _setup_logging(self):
        """Set up logging configuration."""
        log_dir = os.path.join(self.config['output_dir'], 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'experiment_2a_{datetime.now():%Y%m%d_%H%M%S}.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)

    def _create_output_dirs(self):
        """Create necessary output directories."""
        self.results_dir = os.path.join(self.config['output_dir'], 'results')
        self.figures_dir = os.path.join(self.config['output_dir'], 'figures')
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

    def _get_model_storage_cost(self, model) -> float:
        # --- CHANGE START: Use fast in-memory footprint calculation ---
        """Calculate the model's memory footprint in MB."""
        self.logger.info("Calculating model memory footprint...")
        # get_memory_footprint() returns bytes
        mem_footprint_bytes = model.get_memory_footprint()
        return mem_footprint_bytes / (1024 * 1024)
        # --- CHANGE END ---

    def load_models_under_test(self) -> Dict[str, Any]:
        """Loads all models specified in the config, including quantized versions."""
        models_data = {}
        for model_config in self.config['models']:
            name = model_config['name']
            path = model_config['path']
            quant_type = model_config.get('quantization', 'none')
            self.logger.info(f"Loading model: {name} ({path}) with quantization: {quant_type}")
            
            model_args = {'token': self.hf_token}
            
            if quant_type == "4bit":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
                model_args['quantization_config'] = bnb_config
            elif quant_type == "half":
                self.logger.info("Loading model in half-precision (float16).")
                model_args['torch_dtype'] = torch.float16
            # --- CHANGE START: Make loading more robust ---
            elif quant_type != "none":
                 self.logger.warning(f"Quantization type '{quant_type}' not directly supported by this script. Attempting to load in half-precision.")
                 model_args['torch_dtype'] = torch.float16
            # --- CHANGE END ---

            try:
                tokenizer = AutoTokenizer.from_pretrained(path, token=self.hf_token)
                model = AutoModelForCausalLM.from_pretrained(
                    path, 
                    device_map="auto", 
                    **model_args
                )
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                storage_cost_mb = self._get_model_storage_cost(model)
                
                models_data[name] = {
                    'model': model,
                    'tokenizer': tokenizer,
                    'storage_cost_mb': storage_cost_mb,
                    'param_count': model.num_parameters()
                }
                self.logger.info(f"Successfully loaded {name}. Storage cost: {storage_cost_mb:.2f} MB")
            except Exception as e:
                self.logger.error(f"Failed to load model {name}: {e}")
                # Continue to the next model instead of crashing
                continue
        return models_data

    def load_dataset(self) -> List[str]:
        """Loads and prepares the WikiText-2 validation set."""
        self.logger.info("Loading WikiText-2 validation set...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        sentences = [
            s.strip() for s in dataset['text'] 
            if 50 < len(s.strip()) < 500 and not s.strip().startswith("=")
        ]
        num_samples = self.config['num_test_samples']
        self.logger.info(f"Loaded {len(sentences)} sentences. Using {num_samples} for evaluation.")
        return sentences[:num_samples]

    def run_sentence_completion(self, model_info: Dict, sentence: str, theta: int, eval_mode: str) -> Tuple[bool, float, str]:
        """Runs a single sentence completion task and evaluates the result."""
        model, tokenizer = model_info['model'], model_info['tokenizer']

        prompt_tokens = tokenizer.encode(sentence, add_special_tokens=False)[:theta]
        prompt_text = tokenizer.decode(prompt_tokens)

        inputs = tokenizer(prompt_text, return_tensors="pt").to(self.device)

        if self.device.type == 'cuda': torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad():
            # --- CHANGE START: Be explicit about deterministic generation ---
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=len(tokenizer.encode(sentence)) + 10,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False, # Explicitly use greedy decoding
                temperature=None,
                top_p=None,
            )
            # --- CHANGE END ---

        if self.device.type == 'cuda': torch.cuda.synchronize()
        latency_ms = (time.time() - start_time) * 1000

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

        is_success = False
        if eval_mode == 'verbatim':
            if generated_text.startswith(sentence):
                is_success = True
        elif eval_mode == 'semantic':
            similarity = self.metrics_calculator.semantic_similarity(sentence, generated_text)
            is_success = similarity >= self.config['semantic_threshold']

        return is_success, latency_ms, generated_text

    def run_part_a_storage_degradation(self, models: Dict[str, Any], sentences: List[str], eval_mode: str) -> pd.DataFrame:
        """Implements Algorithm 1 to measure storage degradation."""
        self.logger.info(f"--- Part A: Quantifying Storage Degradation ({eval_mode} mode) ---")
        results = []
        theta_max = self.config['retrieval_budgets'][-1]
        
        for model_name, model_info in models.items():
            self.logger.info(f"Evaluating model: {model_name} with theta={theta_max}")
            successful_reconstructions = []
            latencies = []
            
            for sentence in tqdm(sentences, desc=f"Model {model_name}"):
                is_success, latency, _ = self.run_sentence_completion(model_info, sentence, theta_max, eval_mode)
                if is_success:
                    successful_reconstructions.append(sentence)
                latencies.append(latency)
            
            success_rate = len(successful_reconstructions) / len(sentences) if sentences else 0.0
            avg_latency = np.mean(latencies) if latencies else 0.0
            
            results.append({
                'model_name': model_name,
                'storage_cost_mb': model_info['storage_cost_mb'],
                'param_count': model_info['param_count'],
                'theta': theta_max,
                'eval_mode': eval_mode,
                'success_rate': success_rate,
                'avg_latency_ms': avg_latency,
                'successful_sentences': successful_reconstructions
            })
        
        return pd.DataFrame(results)

    def run_part_b_retrieval_degradation(self, model_info: Dict, part_a_results: pd.Series, eval_mode: str) -> pd.DataFrame:
        """Implements Algorithm 2 to measure retrieval degradation."""
        model_name = part_a_results['model_name']
        self.logger.info(f"--- Part B: Quantifying Retrieval Degradation for {model_name} ({eval_mode} mode) ---")
        results = []
        
        baseline_sentences = part_a_results['successful_sentences']
        if not baseline_sentences:
            self.logger.warning(f"No successful sentences for {model_name} in Part A {eval_mode} mode. Skipping Part B.")
            return pd.DataFrame()

        for theta in self.config['retrieval_budgets'][:-1]:
            self.logger.info(f"Evaluating with retrieval budget theta={theta}")
            success_count = 0
            latencies = []
            
            for sentence in tqdm(baseline_sentences, desc=f"Theta={theta}"):
                is_success, latency, _ = self.run_sentence_completion(model_info, sentence, theta, eval_mode)
                if is_success:
                    success_count += 1
                latencies.append(latency)
            
            success_rate_on_baseline = success_count / len(baseline_sentences)
            avg_latency = np.mean(latencies)
            retrieval_degradation = 1.0 - success_rate_on_baseline
            
            results.append({
                'model_name': model_name,
                'theta': theta,
                'eval_mode': eval_mode,
                'success_rate_on_baseline': success_rate_on_baseline,
                'retrieval_degradation': retrieval_degradation,
                'retrieval_cost_ms': avg_latency
            })
            
        return pd.DataFrame(results)

    def apply_pruning(self, model, sparsity: float) -> torch.nn.Module:
        """Applies global unstructured pruning to a model."""
        parameters_to_prune = []
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                parameters_to_prune.append((module, 'weight'))
        
        if not parameters_to_prune:
            self.logger.warning("No linear layers found to prune.")
            return model
            
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=sparsity,
        )
        for module, name in parameters_to_prune:
            prune.remove(module, name)
        return model
    
    def run_pruning_analysis(self, base_model_info: Dict, sentences: List[str], eval_mode: str) -> pd.DataFrame:
        import gc
        self.logger.info(f"--- Pruning Analysis ({eval_mode}) ---")

        baseline_path = base_model_info['model'].config._name_or_path
        base_tokenizer = base_model_info['tokenizer']
        del base_model_info['model']
        torch.cuda.empty_cache()
        gc.collect()

        results = []
        pruning_levels = self.config['pruning_levels']

        for sparsity in tqdm(pruning_levels, desc="Pruning Levels"):
            self.logger.info(f"Applying {sparsity*100:.0f}% sparsity")

            pruned_model = AutoModelForCausalLM.from_pretrained(
            baseline_path,
            device_map="cpu",
            token=self.hf_token,
                torch_dtype=torch.float16
            )

            if sparsity > 0:
                pruned_model = self.apply_pruning(pruned_model, sparsity)

            pruned_model.to(self.device)
            torch.cuda.empty_cache()

            storage_cost = self._get_model_storage_cost(pruned_model)
            pruned_model_info = {'model': pruned_model, 'tokenizer': base_tokenizer}

            success_count = 0
            for sentence in sentences:
                is_success, _, _ = self.run_sentence_completion(
                    pruned_model_info,
                    sentence,
                    self.config['retrieval_budgets'][-1],
                    eval_mode
                )
                if is_success:
                    success_count += 1

            results.append({
            'sparsity': sparsity,
            'storage_cost_mb': storage_cost,
            'success_rate': success_count / len(sentences),
            'eval_mode': eval_mode
            })

            del pruned_model
            torch.cuda.empty_cache()
            gc.collect()

        return pd.DataFrame(results)


    def run_pruning_analysi1s(self, base_model_info: Dict, sentences: List[str], eval_mode: str) -> pd.DataFrame:
        """Creates a continuous budget scale using pruning and evaluates storage degradation."""
        self.logger.info(f"--- Extra: Running Pruning Analysis ({eval_mode} mode) ---")
        base_model = base_model_info['model']
        base_tokenizer = base_model_info['tokenizer']
        results = []
        
        pruning_levels = self.config['pruning_levels']
        
        for sparsity in tqdm(pruning_levels, desc="Pruning Levels"):
            self.logger.info(f"Applying {sparsity*100:.0f}% sparsity...")
            
            pruned_model = AutoModelForCausalLM.from_pretrained(
                base_model.config._name_or_path,
                device_map="auto",
                token=self.hf_token,
                torch_dtype=torch.float16
            )
            pruned_model.load_state_dict(base_model.state_dict())
            
            if sparsity > 0:
                pruned_model = self.apply_pruning(pruned_model, sparsity)
                
            storage_cost = self._get_model_storage_cost(pruned_model)
            
            pruned_model_info = {'model': pruned_model, 'tokenizer': base_tokenizer}
            
            success_count = 0
            for sentence in sentences:
                is_success, _, _ = self.run_sentence_completion(pruned_model_info, sentence, self.config['retrieval_budgets'][-1], eval_mode)
                if is_success:
                    success_count += 1
            
            success_rate = success_count / len(sentences)
            
            results.append({
                'sparsity': sparsity,
                'storage_cost_mb': storage_cost,
                'success_rate': success_rate,
                'eval_mode': eval_mode
            })
            del pruned_model
            if self.device.type == 'cuda': torch.cuda.empty_cache()

        return pd.DataFrame(results)

    # ... (the rest of the script from the previous answer remains the same) ...
    def fit_scaling_law(self, storage_df: pd.DataFrame, retrieval_df: pd.DataFrame):
        """Fits performance data to the proposed scaling law."""
        self.logger.info("--- Fitting Scaling Law ---")
        
        # Law: Q(N, theta) = Q_max * (1 - a*N**-gamma) * (1 - b*theta**-delta)
        
        def storage_law(N, Q_max, a, gamma):
            return Q_max * (1 - a * N**-gamma)
            
        storage_df = storage_df.dropna(subset=['param_count', 'success_rate'])
        N = storage_df['param_count'].values
        Q_N = storage_df['success_rate'].values
        
        gamma_fit = None
        if len(N) > 2:
            try:
                popt_storage, _ = curve_fit(storage_law, N, Q_N, p0=[max(Q_N), 1e10, 0.5], bounds=([0, 0, 0], [1, np.inf, 2]), maxfev=5000)
                Q_max_fit, a_fit, gamma_fit = popt_storage
                self.logger.info(f"Fitted storage params: Q_max={Q_max_fit:.3f}, a={a_fit:.2e}, gamma={gamma_fit:.3f}")
            except RuntimeError as e:
                self.logger.error(f"Could not fit storage scaling law: {e}")
        else:
            self.logger.warning("Not enough data points to fit storage scaling law.")

        def retrieval_law(theta, C, b, delta):
            return C * (1 - b * theta**-delta)
            
        retrieval_df = retrieval_df.dropna(subset=['theta', 'success_rate_on_baseline'])
        theta_vals = retrieval_df['theta'].values
        Q_theta = retrieval_df['success_rate_on_baseline'].values
        
        theta_vals = np.append(theta_vals, self.config['retrieval_budgets'][-1])
        Q_theta = np.append(Q_theta, 1.0)
        
        delta_fit = None
        if len(theta_vals) > 2:
            try:
                popt_retrieval, _ = curve_fit(lambda t, b, d: retrieval_law(t, 1.0, b, d), theta_vals, Q_theta, p0=[0.5, 0.5], bounds=([0, 0], [np.inf, 2]))
                b_fit, delta_fit = popt_retrieval
                self.logger.info(f"Fitted retrieval params: b={b_fit:.3f}, delta={delta_fit:.3f}")
            except RuntimeError as e:
                self.logger.error(f"Could not fit retrieval scaling law: {e}")
        else:
            self.logger.warning("Not enough data points to fit retrieval scaling law.")

        return gamma_fit, delta_fit

    def create_visualizations(self, results: Dict):
        """Generates and saves all plots for the experiment."""
        self.logger.info("--- Creating Visualizations ---")

        # Plot 1: Storage Degradation
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=results['part_a_verbatim'], x='storage_cost_mb', y='success_rate', hue='model_name', s=150, ax=ax, style='eval_mode', markers={'verbatim': 'o'}, palette='viridis')
        sns.scatterplot(data=results['part_a_semantic'], x='storage_cost_mb', y='success_rate', hue='model_name', s=150, ax=ax, style='eval_mode', markers={'semantic': 'X'}, legend=False, palette='viridis')
        
        base_model_name = self.config['pruning_base_model']
        if not results['pruning_verbatim'].empty:
            sns.lineplot(data=results['pruning_verbatim'], x='storage_cost_mb', y='success_rate', ax=ax, color='blue', marker='.', label=f'{base_model_name} Pruning (Verbatim)')
        if not results['pruning_semantic'].empty:
            sns.lineplot(data=results['pruning_semantic'], x='storage_cost_mb', y='success_rate', ax=ax, color='green', marker='.', label=f'{base_model_name} Pruning (Semantic)')
        
        ax.set_title('Performance vs. Storage Cost (Storage Degradation)')
        ax.set_xlabel('Storage Cost (MB)')
        ax.set_ylabel('Success Rate (Reconstruction)')
        ax.grid(True)
        ax.legend(title='Model/Method')
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'storage_degradation.png'), dpi=300)
        plt.close()

        # Plot 2: Retrieval Degradation
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Check if there's any data to plot
        if not results['part_b_verbatim'].empty or not results['part_b_semantic'].empty:
            all_models = pd.concat([results['part_b_verbatim'], results['part_b_semantic']])['model_name'].unique()
            palette = sns.color_palette("tab10", n_colors=len(all_models))
            
            if not results['part_b_verbatim'].empty:
                sns.lineplot(data=results['part_b_verbatim'], x='retrieval_cost_ms', y='success_rate_on_baseline', hue='model_name', style='eval_mode', marker='o', ax=ax, palette=palette)
            if not results['part_b_semantic'].empty:
                sns.lineplot(data=results['part_b_semantic'], x='retrieval_cost_ms', y='success_rate_on_baseline', hue='model_name', style='eval_mode', marker='X', ax=ax, legend=False, palette=palette)

            ax.set_title('Performance vs. Retrieval Cost (Retrieval Degradation)')
            ax.set_xlabel('Retrieval Cost (Avg. Latency in ms)')
            ax.set_ylabel('Success Rate (on sentences known at theta_max)')
            ax.grid(True)
            ax.legend(title='Model (Theta values as points)')
        else:
            ax.text(0.5, 0.5, 'No data for Retrieval Degradation plot.\n(This is expected if verbatim/semantic success was 0% in Part A)', 
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_title('Retrieval Degradation')

        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, 'retrieval_degradation.png'), dpi=300)
        plt.close()


    def run(self):
        sentences = self.load_dataset()
        part_a_verbatim_df_list, part_a_semantic_df_list = [], []
        part_b_verbatim_df_list, part_b_semantic_df_list = [], []
        models = self.load_models_under_test()
        for model_name, model_info in models.items():
            a_ver = self.run_part_a_storage_degradation(
                {model_name: model_info}, sentences, "verbatim"
            )
            a_sem = self.run_part_a_storage_degradation(
                {model_name: model_info}, sentences, "semantic"
            )
            part_a_verbatim_df_list.append(a_ver)
            part_a_semantic_df_list.append(a_sem)
            b_ver = self.run_part_b_retrieval_degradation(
                model_info, a_ver.iloc[0], "verbatim"
            )
            b_sem = self.run_part_b_retrieval_degradation(
                model_info, a_sem.iloc[0], "semantic"
            )
            part_b_verbatim_df_list.append(b_ver)
            part_b_semantic_df_list.append(b_sem)
        part_a_verbatim_df = pd.concat(part_a_verbatim_df_list, ignore_index=True)
        part_a_semantic_df = pd.concat(part_a_semantic_df_list, ignore_index=True)
        part_b_verbatim_df = pd.concat(part_b_verbatim_df_list, ignore_index=True)
        part_b_semantic_df = pd.concat(part_b_semantic_df_list, ignore_index=True)
        for m in models.values():
            m["model"].to("cpu")
        torch.cuda.empty_cache()
        pruning_levels = self.config["pruning_levels"]
        if pruning_levels:
            pruning_base = self.config["pruning_base_model"]
            base_model_info = self.load_models_under_test()[pruning_base]
            pruning_verbatim_df = self.run_pruning_analysis(
                base_model_info, sentences, "verbatim"
            )
            pruning_semantic_df = self.run_pruning_analysis(
                base_model_info, sentences, "semantic"
            )
        else:
            self.logger.info("Skipping pruning analysis (pruning_levels is empty).")
            pruning_verbatim_df = pd.DataFrame()
            pruning_semantic_df = pd.DataFrame()
        gamma, delta = self.fit_scaling_law(part_a_semantic_df, part_b_semantic_df)
        results_dir = self.results_dir
        part_a_verbatim_df.to_csv(os.path.join(results_dir, "part_a_verbatim.csv"), index=False)
        part_a_semantic_df.to_csv(os.path.join(results_dir, "part_a_semantic.csv"), index=False)
        part_b_verbatim_df.to_csv(os.path.join(results_dir, "part_b_verbatim.csv"), index=False)
        part_b_semantic_df.to_csv(os.path.join(results_dir, "part_b_semantic.csv"), index=False)
        if not pruning_verbatim_df.empty:
            pruning_verbatim_df.to_csv(os.path.join(results_dir, "pruning_verbatim.csv"), index=False)
        if not pruning_semantic_df.empty:
            pruning_semantic_df.to_csv(os.path.join(results_dir, "pruning_semantic.csv"), index=False)

        self.create_visualizations({
        "part_a_verbatim":  part_a_verbatim_df,
        "part_a_semantic":  part_a_semantic_df,
        "part_b_verbatim":  part_b_verbatim_df,
        "part_b_semantic":  part_b_semantic_df,
        "pruning_verbatim": pruning_verbatim_df,
        "pruning_semantic": pruning_semantic_df,
        "scaling_exponents": {"gamma": gamma, "delta": delta}
        })

        self.logger.info(
            f"Experiment 2A finished successfully. "
            f"Scaling exponents: gamma={gamma}, delta={delta}"
        )


    def ru1n(self):
        """Main execution flow for the experiment."""
        models = self.load_models_under_test()
        sentences = self.load_dataset()
        
        part_a_verbatim_df = self.run_part_a_storage_degradation(models, sentences, 'verbatim')
        part_a_semantic_df = self.run_part_a_storage_degradation(models, sentences, 'semantic')
        
        part_b_verbatim_dfs, part_b_semantic_dfs = [], []
        for i, row in part_a_verbatim_df.iterrows():
            if row['model_name'] in models:
                model_info = models[row['model_name']]
                part_b_verbatim_dfs.append(self.run_part_b_retrieval_degradation(model_info, row, 'verbatim'))
                part_b_semantic_dfs.append(self.run_part_b_retrieval_degradation(model_info, part_a_semantic_df.iloc[i], 'semantic'))
        
        part_b_verbatim_df = pd.concat(part_b_verbatim_dfs, ignore_index=True) if part_b_verbatim_dfs else pd.DataFrame()
        part_b_semantic_df = pd.concat(part_b_semantic_dfs, ignore_index=True) if part_b_semantic_dfs else pd.DataFrame()

        pruning_model_name = self.config['pruning_base_model']
        if pruning_model_name in models:
            pruning_verbatim_df = self.run_pruning_analysis(models[pruning_model_name], sentences, 'verbatim')
            pruning_semantic_df = self.run_pruning_analysis(models[pruning_model_name], sentences, 'semantic')
        else:
            self.logger.error(f"Pruning base model '{pruning_model_name}' not found in loaded models. Skipping pruning analysis.")
            pruning_verbatim_df, pruning_semantic_df = pd.DataFrame(), pd.DataFrame()

        gamma, delta = self.fit_scaling_law(part_a_semantic_df, part_b_semantic_df)
        self.logger.info(f"Final Scaling Exponents: Capacity (gamma) = {gamma}, Inference (delta) = {delta}")
        
        all_results = {
            'part_a_verbatim': part_a_verbatim_df, 'part_a_semantic': part_a_semantic_df,
            'part_b_verbatim': part_b_verbatim_df, 'part_b_semantic': part_b_semantic_df,
            'pruning_verbatim': pruning_verbatim_df, 'pruning_semantic': pruning_semantic_df,
            'scaling_exponents': {'gamma': gamma, 'delta': delta}
        }
        
        for name, df in all_results.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                df_to_save = df.drop(columns=['successful_sentences'], errors='ignore')
                df_to_save.to_csv(os.path.join(self.results_dir, f"{name}.csv"), index=False)
        self.logger.info(f"All results saved to {self.results_dir}")
        
        self.create_visualizations(all_results)
        self.logger.info(f"All figures saved to {self.figures_dir}")
        self.logger.info("Experiment 2A finished successfully.")

def main():
    parser = argparse.ArgumentParser(description="Run Experiment 2A: Storage vs Retrieval Loss.")
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to the configuration file.')
    parser.add_argument('--hf_token', type=str, default=os.environ.get("HUGGING_FACE_TOKEN"), help='Hugging Face token for gated models.')
    
    args = parser.parse_args()
    
    if not args.hf_token:
        raise ValueError("Hugging Face token is required. Set the HUGGING_FACE_TOKEN environment variable or pass it with --hf_token.")

    runner = Experiment2ARunner(config_path=args.config, hf_token=args.hf_token)
    runner.run()

if __name__ == "__main__":
    main()
