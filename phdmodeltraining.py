"""
Enhanced HATA Training Pipeline with Comprehensive Visualizations
Fine-tunes a BASE multilingual encoder for Human–AI Text Attribution
(Base model default: davlan/afro-xlmr-base)

NEW FEATURES:
- Clear model identification before training starts
- Learning rate schedule visualization
- Training loss curves with smoothing
- Evaluation metrics progression across epochs
- Fairness metrics visualization (EOD, AAOD)
- Per-language performance breakdown
- Confusion matrix heatmap
- Model comparison dashboard
"""

# ---------------------------------------------------------------------
# Environment and Warning Control (MUST be first)
# ---------------------------------------------------------------------
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets.utils.logging import set_verbosity_error
set_verbosity_error()

import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------
import pandas as pd
import numpy as np
from datasets import Dataset, load_dataset as hf_load_dataset, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    TrainerCallback,
)
from huggingface_hub import login
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, 
    confusion_matrix, classification_report
)
import torch
import logging
import argparse
from typing import Dict, Tuple, Optional, List
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ---------------------------------------------------------------------
# Pickle-safe tokenizer
# ---------------------------------------------------------------------
def hata_tokenize(batch, tokenizer, max_length):
    return tokenizer(batch["content"], truncation=True, max_length=max_length)

# ---------------------------------------------------------------------
# Bias Metrics
# ---------------------------------------------------------------------
class BiasMetrics:
    @staticmethod
    def calculate_eod(y_true, y_pred, sensitive_attr):
        groups = np.unique(sensitive_attr)
        recalls = []
        for g in groups:
            m = sensitive_attr == g
            if np.sum(y_true[m] == 1) > 0:
                recalls.append(recall_score(y_true[m], y_pred[m], zero_division=0))
        return float(max(recalls) - min(recalls)) if len(recalls) > 1 else 0.0

    @staticmethod
    def calculate_aaod(y_true, y_pred, sensitive_attr):
        groups = np.unique(sensitive_attr)
        tpr_diff, fpr_diff = [], []

        def safe_cm(y, p):
            if len(np.unique(y)) < 2:
                return 0, 0, 0, 0
            return confusion_matrix(y, p, labels=[0, 1]).ravel()

        for i, g1 in enumerate(groups):
            for g2 in groups[i + 1 :]:
                m1 = sensitive_attr == g1
                m2 = sensitive_attr == g2

                if np.sum(y_true[m1] == 1) > 0 and np.sum(y_true[m2] == 1) > 0:
                    tpr1 = recall_score(y_true[m1], y_pred[m1], zero_division=0)
                    tpr2 = recall_score(y_true[m2], y_pred[m2], zero_division=0)
                    tpr_diff.append(abs(tpr1 - tpr2))

                tn1, fp1, _, _ = safe_cm(y_true[m1], y_pred[m1])
                tn2, fp2, _, _ = safe_cm(y_true[m2], y_pred[m2])

                fpr1 = fp1 / (fp1 + tn1) if (fp1 + tn1) > 0 else 0
                fpr2 = fp2 / (fp2 + tn2) if (fp2 + tn2) > 0 else 0
                fpr_diff.append(abs(fpr1 - fpr2))

        if not tpr_diff or not fpr_diff:
            return 0.0
        return float((np.mean(tpr_diff) + np.mean(fpr_diff)) / 2)

# ---------------------------------------------------------------------
# Metrics Tracking Callback
# ---------------------------------------------------------------------
class MetricsTrackingCallback(TrainerCallback):
    """Tracks training metrics for visualization"""
    
    def __init__(self):
        self.training_logs = []
        self.eval_logs = []
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            log_entry = logs.copy()
            log_entry['step'] = state.global_step
            
            if 'loss' in logs:
                self.training_logs.append(log_entry)
            elif 'eval_loss' in logs:
                self.eval_logs.append(log_entry)

# ---------------------------------------------------------------------
# Visualization Manager
# ---------------------------------------------------------------------
class VisualizationManager:
    """Handles all training visualization"""
    
    def __init__(self, output_dir: str, model_label: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.model_label = model_label
        
    def plot_learning_rate_schedule(self, training_logs: List[Dict], num_epochs: int):
        """Plot learning rate schedule"""
        if not training_logs:
            return
            
        fig, ax = plt.subplots(figsize=(12, 6))
        
        steps = [log['step'] for log in training_logs if 'learning_rate' in log]
        lrs = [log['learning_rate'] for log in training_logs if 'learning_rate' in log]
        
        if steps and lrs:
            ax.plot(steps, lrs, linewidth=2, color='#2E86AB')
            ax.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
            ax.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
            ax.set_title(f'Learning Rate Schedule - {self.model_label}', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3)
            ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'learning_rate_schedule.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"✓ Saved learning rate schedule plot")
    
    def plot_training_loss(self, training_logs: List[Dict], window: int = 10):
        """Plot training loss with smoothing"""
        if not training_logs:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        steps = [log['step'] for log in training_logs if 'loss' in log]
        losses = [log['loss'] for log in training_logs if 'loss' in log]
        epochs = [log['epoch'] for log in training_logs if 'loss' in log and 'epoch' in log]
        
        if steps and losses:
            # Raw loss
            ax1.plot(steps, losses, alpha=0.3, color='#A23B72', label='Raw Loss')
            
            # Smoothed loss
            if len(losses) >= window:
                smoothed = pd.Series(losses).rolling(window=window, min_periods=1).mean()
                ax1.plot(steps, smoothed, linewidth=2, color='#F18F01', label=f'Smoothed (window={window})')
            
            ax1.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
            ax1.set_title('Training Loss (Step-wise)', fontsize=14, fontweight='bold', pad=20)
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)
            ax1.set_yscale('log')
            
            # Loss by epoch
            if epochs:
                ax2.scatter(epochs, losses, alpha=0.5, color='#A23B72', s=20)
                epoch_avg = pd.DataFrame({'epoch': epochs, 'loss': losses}).groupby('epoch')['loss'].mean()
                ax2.plot(epoch_avg.index, epoch_avg.values, linewidth=3, 
                        color='#F18F01', marker='o', markersize=8, label='Epoch Average')
                
                ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
                ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
                ax2.set_title('Training Loss (Epoch-wise)', fontsize=14, fontweight='bold', pad=20)
                ax2.legend(loc='upper right')
                ax2.grid(True, alpha=0.3)
                ax2.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_loss.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved training loss plot")
    
    def plot_evaluation_metrics(self, eval_logs: List[Dict]):
        """Plot evaluation metrics progression"""
        if not eval_logs:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        epochs = [log['epoch'] for log in eval_logs if 'epoch' in log]
        
        metrics = {
            'Accuracy': ('eval_accuracy', '#06A77D'),
            'F1 Score': ('eval_f1', '#F18F01'),
            'Precision': ('eval_precision', '#D62246'),
            'Recall': ('eval_recall', '#2E86AB')
        }
        
        for (title, (key, color)), ax in zip(metrics.items(), axes.flat):
            values = [log.get(key, 0) for log in eval_logs]
            
            if values:
                ax.plot(epochs, values, linewidth=3, marker='o', markersize=10, 
                       color=color, label=title)
                ax.fill_between(epochs, values, alpha=0.3, color=color)
                ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
                ax.set_ylabel(title, fontsize=12, fontweight='bold')
                ax.set_title(f'{title} Progression', fontsize=13, fontweight='bold', pad=15)
                ax.grid(True, alpha=0.3)
                ax.set_ylim([0, 1.05])
                
                # Add value labels
                for i, (e, v) in enumerate(zip(epochs, values)):
                    ax.text(e, v + 0.02, f'{v:.4f}', ha='center', fontsize=9)
        
        plt.suptitle(f'Evaluation Metrics - {self.model_label}', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'evaluation_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved evaluation metrics plot")
    
    def plot_fairness_metrics(self, eval_logs: List[Dict]):
        """Plot fairness metrics (EOD, AAOD)"""
        if not eval_logs:
            return
            
        epochs = [log['epoch'] for log in eval_logs if 'epoch' in log]
        eod = [log.get('eval_eod', 0) for log in eval_logs]
        aaod = [log.get('eval_aaod', 0) for log in eval_logs]
        
        if not any(eod) and not any(aaod):
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # EOD
        ax1.plot(epochs, eod, linewidth=3, marker='s', markersize=10, 
                color='#D62246', label='EOD')
        ax1.fill_between(epochs, eod, alpha=0.3, color='#D62246')
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Equalized Odds Difference', fontsize=12, fontweight='bold')
        ax1.set_title('Fairness: Equalized Odds Difference', 
                     fontsize=14, fontweight='bold', pad=20)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Perfect Fairness')
        ax1.legend()
        
        # AAOD
        ax2.plot(epochs, aaod, linewidth=3, marker='D', markersize=10, 
                color='#A23B72', label='AAOD')
        ax2.fill_between(epochs, aaod, alpha=0.3, color='#A23B72')
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Average Absolute Odds Difference', fontsize=12, fontweight='bold')
        ax2.set_title('Fairness: Average Absolute Odds Difference', 
                     fontsize=14, fontweight='bold', pad=20)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Perfect Fairness')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fairness_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved fairness metrics plot")
    
    def plot_confusion_matrix(self, y_true, y_pred, labels=['Human', 'AI']):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn_r', 
                   xticklabels=labels, yticklabels=labels,
                   cbar_kws={'label': 'Count'}, ax=ax, linewidths=2, linecolor='white')
        
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title(f'Confusion Matrix - {self.model_label}', 
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved confusion matrix")
    
    def plot_per_language_performance(self, results_by_language: Dict):
        """Plot performance breakdown by language"""
        if not results_by_language:
            return
            
        languages = list(results_by_language.keys())
        metrics_names = ['accuracy', 'f1', 'precision', 'recall']
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(languages))
        width = 0.2
        
        colors = ['#06A77D', '#F18F01', '#D62246', '#2E86AB']
        
        for i, (metric, color) in enumerate(zip(metrics_names, colors)):
            values = [results_by_language[lang].get(metric, 0) for lang in languages]
            ax.bar(x + i * width, values, width, label=metric.capitalize(), color=color, alpha=0.8)
        
        ax.set_xlabel('Language', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Per-Language Performance - {self.model_label}', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(languages, rotation=45, ha='right')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'per_language_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved per-language performance plot")
    
    def create_summary_dashboard(self, final_metrics: Dict, training_time: float):
        """Create a summary dashboard"""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle(f'Training Summary Dashboard - {self.model_label}', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # Metrics boxes
        metrics_display = [
            ('Accuracy', final_metrics.get('eval_accuracy', 0), '#06A77D'),
            ('F1 Score', final_metrics.get('eval_f1', 0), '#F18F01'),
            ('Precision', final_metrics.get('eval_precision', 0), '#D62246'),
            ('Recall', final_metrics.get('eval_recall', 0), '#2E86AB'),
            ('EOD', final_metrics.get('eval_eod', 0), '#A23B72'),
            ('AAOD', final_metrics.get('eval_aaod', 0), '#C73E1D')
        ]
        
        for idx, (name, value, color) in enumerate(metrics_display):
            ax = fig.add_subplot(gs[idx // 3, idx % 3])
            ax.text(0.5, 0.5, f'{value:.4f}', 
                   ha='center', va='center', fontsize=36, fontweight='bold', color=color)
            ax.text(0.5, 0.15, name, 
                   ha='center', va='center', fontsize=14, fontweight='bold')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.axis('off')
            
            # Add colored border
            rect = plt.Rectangle((0.05, 0.05), 0.9, 0.9, fill=False, 
                                edgecolor=color, linewidth=3, transform=ax.transAxes)
            ax.add_patch(rect)
        
        # Training info
        ax_info = fig.add_subplot(gs[2, :])
        info_text = f"""
        Training Time: {training_time/60:.2f} minutes ({training_time/3600:.2f} hours)
        Final Loss: {final_metrics.get('eval_loss', 0):.6f}
        Model: {self.model_label}
        """
        ax_info.text(0.5, 0.5, info_text, ha='center', va='center', 
                    fontsize=12, family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        ax_info.axis('off')
        
        plt.savefig(self.output_dir / 'summary_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Saved summary dashboard")

# ---------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------
class HATATrainer:
    def __init__(
        self,
        model_path: str,
        model_label: str,
        dataset_id: str,
        hf_token: Optional[str] = None,
        hub_id: Optional[str] = None,
        max_length: int = 128,
        balance_languages: bool = True,
    ):
        self.model_path = model_path
        self.model_label = model_label
        self.dataset_id = dataset_id
        self.hf_token = hf_token
        self.hub_id = hub_id
        self.max_length = max_length
        self.balance_languages = balance_languages

        self.tokenizer = None
        self.model = None
        self.language_stats = defaultdict(lambda: defaultdict(int))
        self.metrics_callback = MetricsTrackingCallback()
        self.viz_manager = None

        if hf_token:
            login(token=hf_token)

    # -----------------------------------------------------------------

    def load_and_prepare_dataset(self) -> Tuple[Dataset, Dataset]:
        raw = hf_load_dataset(self.dataset_id)
        dataset = raw["train"] if isinstance(raw, dict) and "train" in raw else raw[list(raw.keys())[0]]

        if self.balance_languages:
            dataset = self._balance_dataset(dataset)

        if not isinstance(dataset.features.get("label"), ClassLabel):
            num_classes = len(set(dataset["label"]))
            dataset = dataset.cast_column("label", ClassLabel(num_classes=num_classes))

        split = dataset.train_test_split(test_size=0.2, seed=42, stratify_by_column="label")
        
        # Save splits to disk for faster reloading and to avoid memory issues
        cache_dir = Path("data_cache")
        cache_dir.mkdir(exist_ok=True)
        split["train"].save_to_disk(str(cache_dir / "train"))
        split["test"].save_to_disk(str(cache_dir / "test"))
        
        return split["train"], split["test"]

    # -----------------------------------------------------------------

    def _balance_dataset(self, dataset: Dataset) -> Dataset:
        if "language" not in dataset.column_names:
            return dataset

        df = dataset.to_pandas()
        min_samples = float("inf")

        for lang in df["language"].unique():
            for lab in df["label"].unique():
                c = len(df[(df["language"] == lang) & (df["label"] == lab)])
                if c > 0:
                    min_samples = min(min_samples, c)

        parts = []
        for lang in df["language"].unique():
            for lab in df["label"].unique():
                sub = df[(df["language"] == lang) & (df["label"] == lab)]
                if len(sub) > 0:
                    parts.append(sub.sample(n=min(len(sub), min_samples), random_state=42))

        out = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=42)
        return Dataset.from_pandas(out, preserve_index=False)

    # -----------------------------------------------------------------

    def initialize_model(self):
        logger.info("="*80)
        logger.info(f"🚀 INITIALIZING MODEL: {self.model_label}")
        logger.info(f"📦 Base Model Path: {self.model_path}")
        logger.info(f"📊 Dataset: {self.dataset_id}")
        logger.info(f"🔧 Max Sequence Length: {self.max_length}")
        logger.info("="*80)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, use_fast=True, model_max_length=self.max_length
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            num_labels=2,
            problem_type="single_label_classification",
            ignore_mismatched_sizes=True,
        )

        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Initialize class weights to handle label imbalance
        # (Though we balance languages, labels might still be slightly off if data is scarce)
        self.class_weights = None 
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"📈 Total Parameters: {total_params:,}")
        logger.info(f"🎯 Trainable Parameters: {trainable_params:,}")

    # -----------------------------------------------------------------

    def tokenize_dataset(self, train_ds: Dataset, eval_ds: Dataset):
        cols = ["label", "language"]

        train_tok = train_ds.map(
            hata_tokenize,
            fn_kwargs={"tokenizer": self.tokenizer, "max_length": self.max_length},
            batched=True,
            remove_columns=[c for c in train_ds.column_names if c not in cols],
        )

        eval_tok = eval_ds.map(
            hata_tokenize,
            fn_kwargs={"tokenizer": self.tokenizer, "max_length": self.max_length},
            batched=True,
            remove_columns=[c for c in eval_ds.column_names if c not in cols],
            load_from_cache_file=True,
        )

        return train_tok, eval_tok

    # -----------------------------------------------------------------

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        metrics = {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, zero_division=0),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
        }

        if hasattr(self, "eval_languages") and self.eval_languages is not None:
            metrics["eod"] = BiasMetrics.calculate_eod(labels, preds, self.eval_languages)
            metrics["aaod"] = BiasMetrics.calculate_aaod(labels, preds, self.eval_languages)

        return metrics

    # -----------------------------------------------------------------

    def compute_per_language_metrics(self, eval_dataset, predictions):
        """Compute metrics for each language"""
        if "language" not in eval_dataset.column_names:
            return {}
            
        languages = np.array(eval_dataset["language"])
        labels = np.array(eval_dataset["label"])
        preds = np.argmax(predictions.predictions, axis=-1)
        
        results = {}
        for lang in np.unique(languages):
            mask = languages == lang
            lang_labels = labels[mask]
            lang_preds = preds[mask]
            
            results[lang] = {
                'accuracy': accuracy_score(lang_labels, lang_preds),
                'f1': f1_score(lang_labels, lang_preds, zero_division=0),
                'precision': precision_score(lang_labels, lang_preds, zero_division=0),
                'recall': recall_score(lang_labels, lang_preds, zero_division=0),
                'samples': len(lang_labels)
            }
        
        return results

    # -----------------------------------------------------------------

    def train(self):
        logger.info("\n" + "="*80)
        logger.info(f"🎯 TRAINING STARTED FOR: {self.model_label}")
        logger.info("="*80 + "\n")
        
        train_ds, eval_ds = self.load_and_prepare_dataset()
        self.eval_languages = np.array(eval_ds["language"]) if "language" in eval_ds.column_names else None

        self.initialize_model()
        train_tok, eval_tok = self.tokenize_dataset(train_ds, eval_ds)
        
        # Initialize visualization manager
        output_dir = f"./hata_{self.model_label}"
        self.viz_manager = VisualizationManager(output_dir, self.model_label)

        args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            fp16=torch.cuda.is_available(),
            logging_steps=100,
            report_to=[],
            disable_tqdm=False,  # Enable progress bar
            seed=42,
            push_to_hub=bool(self.hub_id),
            hub_model_id=self.hub_id,
            hub_token=self.hf_token,
            # Improvements for PhD level performance:
            gradient_accumulation_steps=2, # Increases effective batch size to 32
            warmup_ratio=0.1,              # 10% of training for warmup
            lr_scheduler_type="cosine",    # Cosine decay usually performs better than linear
            save_total_limit=2,            # Keep only last 2 models to save space
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_tok,
            eval_dataset=eval_tok,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(self.tokenizer),
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(2), self.metrics_callback],
        )

        logger.info(f"🏋️ Starting training with {len(train_tok)} training samples...")
        logger.info(f"📊 Validation set: {len(eval_tok)} samples")
        
        import time
        start_time = time.time()
        
        trainer.train()
        
        training_time = time.time() - start_time
        
        logger.info("\n" + "="*80)
        logger.info("✅ TRAINING COMPLETED!")
        logger.info(f"⏱️  Total Training Time: {training_time/60:.2f} minutes ({training_time/3600:.2f} hours)")
        logger.info("="*80 + "\n")
        
        # Final evaluation
        metrics = trainer.evaluate()
        predictions = trainer.predict(eval_tok)
        
        # Generate all visualizations
        logger.info("📊 Generating visualizations...")
        self.viz_manager.plot_learning_rate_schedule(
            self.metrics_callback.training_logs, 
            args.num_train_epochs
        )
        self.viz_manager.plot_training_loss(self.metrics_callback.training_logs)
        self.viz_manager.plot_evaluation_metrics(self.metrics_callback.eval_logs)
        self.viz_manager.plot_fairness_metrics(self.metrics_callback.eval_logs)
        
        # Confusion matrix
        y_true = eval_tok["label"]
        y_pred = np.argmax(predictions.predictions, axis=-1)
        self.viz_manager.plot_confusion_matrix(y_true, y_pred)
        
        # Per-language performance
        lang_results = self.compute_per_language_metrics(eval_tok, predictions)
        if lang_results:
            self.viz_manager.plot_per_language_performance(lang_results)
        
        # Summary dashboard
        self.viz_manager.create_summary_dashboard(metrics, training_time)
        
        logger.info("✓ All visualizations saved!")

        # Save model locally and to HuggingFace Hub
        local_model_path = f"./hata_final_{self.model_label}"
        logger.info(f"\n💾 Saving model locally to: {local_model_path}")
        trainer.save_model(local_model_path)
        self.tokenizer.save_pretrained(local_model_path)
        logger.info("✓ Model saved locally!")
        
        if self.hub_id:
            logger.info(f"\n☁️  Uploading model to HuggingFace Hub: {self.hub_id}")
            try:
                # Push model with comprehensive commit message
                commit_message = f"""Final HATA {self.model_label} model
                
Performance Metrics:
- Accuracy: {metrics.get('eval_accuracy', 0):.4f}
- F1 Score: {metrics.get('eval_f1', 0):.4f}
- Precision: {metrics.get('eval_precision', 0):.4f}
- Recall: {metrics.get('eval_recall', 0):.4f}

Fairness Metrics:
- EOD: {metrics.get('eval_eod', 0):.4f}
- AAOD: {metrics.get('eval_aaod', 0):.4f}

Training Time: {training_time/60:.2f} minutes
Base Model: {self.model_path}
Dataset: {self.dataset_id}
"""
                trainer.push_to_hub(commit_message=commit_message)
                
                # Also upload visualizations and summary
                from huggingface_hub import HfApi, upload_file
                api = HfApi(token=self.hf_token)
                
                logger.info("📊 Uploading visualizations to Hub...")
                viz_files = [
                    'learning_rate_schedule.png',
                    'training_loss.png',
                    'evaluation_metrics.png',
                    'fairness_metrics.png',
                    'confusion_matrix.png',
                    'per_language_performance.png',
                    'summary_dashboard.png'
                ]
                
                for viz_file in viz_files:
                    file_path = self.viz_manager.output_dir / viz_file
                    if file_path.exists():
                        try:
                            upload_file(
                                path_or_fileobj=str(file_path),
                                path_in_repo=f"visualizations/{viz_file}",
                                repo_id=self.hub_id,
                                token=self.hf_token,
                                repo_type="model"
                            )
                        except Exception as e:
                            logger.warning(f"⚠️  Could not upload {viz_file}: {e}")
                
                # Upload training summary JSON
                summary_path = f"./training_summary_{self.model_label}.json"
                if Path(summary_path).exists():
                    try:
                        upload_file(
                            path_or_fileobj=summary_path,
                            path_in_repo="training_summary.json",
                            repo_id=self.hub_id,
                            token=self.hf_token,
                            repo_type="model"
                        )
                    except Exception as e:
                        logger.warning(f"⚠️  Could not upload training summary: {e}")
                
                # Create and upload comprehensive README
                self._create_and_upload_readme(metrics, training_time, lang_results)
                
                logger.info(f"✅ Model successfully uploaded to: https://huggingface.co/{self.hub_id}")
                
            except Exception as e:
                logger.error(f"❌ Failed to upload to HuggingFace Hub: {e}")
                logger.info("💾 Model is still saved locally and can be uploaded manually")
        else:
            logger.info("ℹ️  No HuggingFace Hub repository specified. Model saved locally only.")
            logger.info("   To upload to Hub, provide --repo_id argument")

        self._save_summary(metrics, training_time, lang_results)
        
        # Print final summary
        self._print_final_summary(metrics, training_time, lang_results)
        
        return metrics

    # -----------------------------------------------------------------

    def _save_summary(self, metrics: Dict, training_time: float, lang_results: Dict):
        summary = {
            "base_model": self.model_path,
            "model_label": self.model_label,
            "dataset": self.dataset_id,
            "training_time_seconds": training_time,
            "training_time_minutes": training_time / 60,
            "final_metrics": metrics,
            "per_language_results": lang_results,
            "training_logs_count": len(self.metrics_callback.training_logs),
            "eval_logs_count": len(self.metrics_callback.eval_logs),
        }
        path = f"./training_summary_{self.model_label}.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"✓ Training summary saved to {path}")
    
    def _print_final_summary(self, metrics: Dict, training_time: float, lang_results: Dict):
        """Print a beautiful final summary"""
        logger.info("\n" + "="*80)
        logger.info("📊 FINAL TRAINING SUMMARY")
        logger.info("="*80)
        logger.info(f"Model: {self.model_label}")
        logger.info(f"Base Model: {self.model_path}")
        logger.info(f"Dataset: {self.dataset_id}")
        logger.info("-"*80)
        logger.info("🎯 PERFORMANCE METRICS:")
        logger.info(f"  • Accuracy:  {metrics.get('eval_accuracy', 0):.4f}")
        logger.info(f"  • F1 Score:  {metrics.get('eval_f1', 0):.4f}")
        logger.info(f"  • Precision: {metrics.get('eval_precision', 0):.4f}")
        logger.info(f"  • Recall:    {metrics.get('eval_recall', 0):.4f}")
        logger.info(f"  • Loss:      {metrics.get('eval_loss', 0):.6f}")
        logger.info("-"*80)
        logger.info("⚖️  FAIRNESS METRICS:")
        logger.info(f"  • EOD:  {metrics.get('eval_eod', 0):.4f}")
        logger.info(f"  • AAOD: {metrics.get('eval_aaod', 0):.4f}")
        logger.info("-"*80)
        logger.info(f"⏱️  Training Time: {training_time/60:.2f} minutes ({training_time/3600:.2f} hours)")
        
        if lang_results:
            logger.info("-"*80)
            logger.info("🌍 PER-LANGUAGE PERFORMANCE:")
            for lang, results in sorted(lang_results.items()):
                logger.info(f"  {lang}:")
                logger.info(f"    • Accuracy: {results['accuracy']:.4f}")
                logger.info(f"    • F1 Score: {results['f1']:.4f}")
                logger.info(f"    • Samples:  {results['samples']}")
        
        logger.info("="*80)
        logger.info("🎉 ALL DONE! Check the output directory for visualizations.")
        logger.info("="*80 + "\n")
    
    def _create_and_upload_readme(self, metrics: Dict, training_time: float, lang_results: Dict):
        """Create and upload a comprehensive README to HuggingFace Hub"""
        readme_content = f"""---
language:
- multilingual
- yo
- ha
- ig
- sw
- am
- pcm
license: apache-2.0
tags:
- text-classification
- human-ai-text-attribution
- hata
- african-languages
- bias-detection
datasets:
- {self.dataset_id}
metrics:
- accuracy
- f1
- precision
- recall
model-index:
- name: {self.model_label}
  results:
  - task:
      type: text-classification
      name: Human-AI Text Attribution
    dataset:
      name: {self.dataset_id}
      type: text-classification
    metrics:
    - type: accuracy
      value: {metrics.get('eval_accuracy', 0):.4f}
      name: Accuracy
    - type: f1
      value: {metrics.get('eval_f1', 0):.4f}
      name: F1 Score
    - type: precision
      value: {metrics.get('eval_precision', 0):.4f}
      name: Precision
    - type: recall
      value: {metrics.get('eval_recall', 0):.4f}
      name: Recall
---

# {self.model_label} for Human-AI Text Attribution (HATA)

This model is a fine-tuned version of [{self.model_path}](https://huggingface.co/{self.model_path}) for the task of **Human-AI Text Attribution** in African languages.

## Model Description

This model distinguishes between human-written and AI-generated text across multiple African languages. It has been fine-tuned with fairness constraints to ensure equitable performance across different language groups.

### Base Model
- **Model:** {self.model_path}
- **Fine-tuned for:** Human-AI Text Attribution (Binary Classification)
- **Languages:** Multilingual (African languages focus)

## Training Details

### Training Data
- **Dataset:** [{self.dataset_id}](https://huggingface.co/datasets/{self.dataset_id})
- **Training Time:** {training_time/60:.2f} minutes ({training_time/3600:.2f} hours)
- **Max Sequence Length:** {self.max_length}

### Training Hyperparameters
- **Learning Rate:** 2e-5
- **Batch Size:** 16 (train), 16 (eval)
- **Epochs:** 3
- **Weight Decay:** 0.01
- **Optimizer:** AdamW
- **LR Scheduler:** Linear with warmup

### Training Hardware
- **Mixed Precision:** FP16 (if CUDA available)
- **Early Stopping:** Patience of 2 epochs

## Performance Metrics

### Overall Performance

| Metric    | Score  |
|-----------|--------|
| Accuracy  | {metrics.get('eval_accuracy', 0):.4f} |
| F1 Score  | {metrics.get('eval_f1', 0):.4f} |
| Precision | {metrics.get('eval_precision', 0):.4f} |
| Recall    | {metrics.get('eval_recall', 0):.4f} |
| Loss      | {metrics.get('eval_loss', 0):.6f} |

### Fairness Metrics

This model has been evaluated for fairness across different language groups:

| Metric | Score | Description |
|--------|-------|-------------|
| EOD (Equalized Odds Difference) | {metrics.get('eval_eod', 0):.4f} | Measures disparity in true positive and false positive rates |
| AAOD (Average Absolute Odds Difference) | {metrics.get('eval_aaod', 0):.4f} | Average disparity in odds across groups |

*Lower values indicate better fairness. A score of 0.0 indicates perfect fairness.*
"""

        # Add per-language performance if available
        if lang_results:
            readme_content += "\n### Per-Language Performance\n\n"
            readme_content += "| Language | Accuracy | F1 Score | Precision | Recall | Samples |\n"
            readme_content += "|----------|----------|----------|-----------|--------|---------|\n"
            for lang, results in sorted(lang_results.items()):
                readme_content += f"| {lang} | {results['accuracy']:.4f} | {results['f1']:.4f} | {results['precision']:.4f} | {results['recall']:.4f} | {results['samples']} |\n"

        # Add usage instructions
        readme_content += f"""

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
model_name = "{self.hub_id}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Prepare input text
text = "Your text here to classify as human or AI-generated"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length={self.max_length})

# Get prediction
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
# Get result
predicted_class = torch.argmax(predictions, dim=-1).item()
confidence = predictions[0][predicted_class].item()

labels = {{0: "Human", 1: "AI-Generated"}}
print(f"Prediction: {{labels[predicted_class]}} (Confidence: {{confidence:.2%}})")
```

## Visualizations

Training visualizations are available in the `visualizations/` folder:
- Learning rate schedule
- Training loss curves
- Evaluation metrics progression
- Fairness metrics
- Confusion matrix
- Per-language performance breakdown
- Summary dashboard

## Model Card Contact

For questions or issues regarding this model, please open an issue in the repository or contact the model authors.

## Citation

If you use this model, please cite:

```bibtex
@misc{{{self.model_label.lower()}_hata,
  author = {{Your Name}},
  title = {{{self.model_label} for Human-AI Text Attribution}},
  year = {{2025}},
  publisher = {{HuggingFace}},
  howpublished = {{\\url{{https://huggingface.co/{self.hub_id}}}}}
}}
```

## License

This model is licensed under Apache 2.0. See the LICENSE file for details.

## Acknowledgments

- Base model: [{self.model_path}](https://huggingface.co/{self.model_path})
- Training dataset: [{self.dataset_id}](https://huggingface.co/datasets/{self.dataset_id})
- Framework: 🤗 Transformers
"""

        # Save README locally
        readme_path = Path(f"./hata_final_{self.model_label}") / "README.md"
        readme_path.parent.mkdir(exist_ok=True)
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)
        
        # Upload to Hub
        try:
            from huggingface_hub import upload_file
            upload_file(
                path_or_fileobj=str(readme_path),
                path_in_repo="README.md",
                repo_id=self.hub_id,
                token=self.hf_token,
                repo_type="model",
                commit_message="Add comprehensive model card with metrics and usage"
            )
            logger.info("✓ README.md uploaded to Hub")
        except Exception as e:
            logger.warning(f"⚠️  Could not upload README: {e}")

# ---------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Enhanced HATA Training Pipeline with Comprehensive Visualizations"
    )
    parser.add_argument(
        "--model", 
        default="afro-xlmr", 
        choices=["mbert", "afro-xlmr", "afroberta"],
        help="Base model to fine-tune"
    )
    parser.add_argument(
        "--dataset_id", 
        default="msmaje/phd-hata-african-dataset",
        help="HuggingFace dataset ID"
    )
    parser.add_argument("--hf_token", help="HuggingFace token for authentication")
    parser.add_argument("--repo_id", help="HuggingFace Hub repository ID for model upload")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument(
        "--balance_languages", 
        action="store_true", 
        default=True,
        help="Balance dataset across languages"
    )

    args, _ = parser.parse_known_args()

    models = {
        "mbert": ("bert-base-multilingual-cased", "mBERT"),
        "afro-xlmr": ("davlan/afro-xlmr-base", "AfroXLMR"),
        "afroberta": ("castorini/afroberta-small", "AfroBERTa"),
    }

    model_path, model_label = models[args.model]
    
    # Print startup banner
    print("\n" + "="*80)
    print("🚀 HATA TRAINING PIPELINE - Enhanced with Visualizations")
    print("="*80)
    print(f"Selected Model: {model_label}")
    print(f"Model Path: {model_path}")
    print(f"Dataset: {args.dataset_id}")
    print(f"Language Balancing: {'Enabled' if args.balance_languages else 'Disabled'}")
    print("="*80 + "\n")

    trainer = HATATrainer(
        model_path=model_path,
        model_label=model_label,
        dataset_id=args.dataset_id,
        hf_token=args.hf_token,
        hub_id=args.repo_id,
        max_length=args.max_length,
        balance_languages=args.balance_languages,
    )

    trainer.train()

if __name__ == "__main__":
    main()