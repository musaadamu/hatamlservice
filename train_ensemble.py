import argparse
import logging
import time
import os
from pathlib import Path
from phdmodeltraining import HATATrainer

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Model Ensemble Hub
# ---------------------------------------------------------------------
MODELS = {
    "mbert": ("bert-base-multilingual-cased", "mBERT"),
    "afro-xlmr": ("davlan/afro-xlmr-base", "AfroXLMR"),
    "afroberta": ("castorini/afroberta-small", "AfroBERTa"),
}

def run_ensemble_training(dataset_id, hf_token=None, base_repo_id=None, max_length=128):
    """Run training for all three models sequentially"""
    
    results = {}
    
    print("\n" + "!"*80)
    print("🌍 STARTING HATA ENSEMBLE TRAINING PIPELINE")
    print(f"📡 Dataset: {dataset_id}")
    print(f"🧪 Target Models: {list(MODELS.keys())}")
    print("!"*80 + "\n")

    for model_key, (model_path, model_label) in MODELS.items():
        logger.info(f"\n\n{'#'*60}")
        logger.info(f"🏗️  PHASE: Training {model_label}")
        logger.info(f"{'#'*60}\n")
        
        # Define repo ID for individual model upload if base_repo_id is provided
        # Example: base_repo_id/phd-hata-mbert
        repo_id = f"{base_repo_id}-{model_key}" if base_repo_id else None
        
        try:
            trainer = HATATrainer(
                model_path=model_path,
                model_label=model_label,
                dataset_id=dataset_id,
                hf_token=hf_token,
                hub_id=repo_id,
                max_length=max_length,
                balance_languages=True, # Recommended for African languages
            )
            
            metrics = trainer.train()
            results[model_label] = metrics
            
            logger.info(f"✅ Successfully completed training for {model_label}")
            
        except Exception as e:
            logger.error(f"❌ Error training {model_label}: {e}")
            continue

    # ---------------------------------------------------------------------
    # Final Comparison
    # ---------------------------------------------------------------------
    print("\n\n" + "="*80)
    print("🏆 ENSEMBLE TRAINING RESULTS SUMMARY")
    print("="*80)
    print(f"{'Model':<15} | {'Accuracy':<10} | {'F1 Score':<10} | {'EOD (Fairness)':<15}")
    print("-" * 80)
    
    for label, metrics in results.items():
        acc = metrics.get('eval_accuracy', 0)
        f1 = metrics.get('eval_f1', 0)
        eod = metrics.get('eval_eod', 0)
        print(f"{label:<15} | {acc:<10.4f} | {f1:<10.4f} | {eod:<15.4f}")
    
    print("="*80 + "\n")
    
    # Identify best model
    if results:
        best_model = max(results.items(), key=lambda x: x[1].get('eval_f1', 0))
        print(f"⭐ BEST PERFORMING MODEL: {best_model[0]} with F1: {best_model[1].get('eval_f1', 0):.4f}")
        print("="*80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HATA Ensemble Training Orchestrator")
    parser.add_argument("--dataset_id", default="msmaje/phd-hata-african-dataset", help="HuggingFace dataset ID")
    parser.add_argument("--hf_token", help="HuggingFace token")
    parser.add_argument("--repo_prefix", help="Prefix for your HuggingFace Hub repo IDs")
    parser.add_argument("--max_length", type=int, default=128, help="Max sequence length")
    
    args = parser.parse_known_args()[0]
    
    run_ensemble_training(
        dataset_id=args.dataset_id,
        hf_token=args.hf_token,
        base_repo_id=args.repo_prefix,
        max_length=args.max_length
    )
