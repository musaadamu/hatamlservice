"""
Model Service - AWS Lambda Local Model Inference
Optimized for AWS Lambda deployment with local model loading
Model: msmaje/Quantizedphdhatamodel (INT8 quantized for memory efficiency)
"""

import os
import numpy as np
import gc
from loguru import logger
from config import settings

# Import transformers for local inference on AWS Lambda
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
    logger.info("✅ Transformers and PyTorch available for local inference on AWS Lambda")
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    logger.error(f"❌ CRITICAL: Transformers not available - {e}")
    raise ImportError("PyTorch and Transformers are required for AWS Lambda deployment")

class ModelService:
    """Service for local model inference on AWS Lambda"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.hf_token = settings.HF_TOKEN

        # AWS Lambda uses CPU (no GPU available)
        self.device = "cpu"
        logger.info(f"🖥️ Using device: {self.device}")

        self.load_model()
    
    def load_model(self):
        """Load the quantized model locally on AWS Lambda"""
        try:
            if not TRANSFORMERS_AVAILABLE:
                raise RuntimeError("PyTorch and Transformers are required for AWS Lambda deployment")

            logger.info(f"📦 Loading quantized model from HuggingFace Hub: {settings.MODEL_NAME}")
            logger.info(f"💾 Cache directory: {settings.MODEL_CACHE_DIR}")

            # Create cache directory in /tmp (only writable directory on Lambda)
            os.makedirs(settings.MODEL_CACHE_DIR, exist_ok=True)

            # 1. Load Tokenizer
            logger.info("🔤 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.MODEL_NAME,
                cache_dir=settings.MODEL_CACHE_DIR,
                token=self.hf_token,
                use_fast=True  # Use fast tokenizer for better performance
            )
            logger.info("✅ Tokenizer loaded successfully")

            # 2. Load Quantized Model (INT8 quantized for memory efficiency)
            logger.info("🤖 Loading quantized model (this may take 30-60 seconds on first run)...")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                settings.MODEL_NAME,
                cache_dir=settings.MODEL_CACHE_DIR,
                token=self.hf_token,
                torch_dtype=torch.float32,  # Quantized model uses float32
                low_cpu_mem_usage=True,  # Optimize memory usage for Lambda
                device_map=None  # Explicitly use CPU (no GPU on Lambda)
            )

            # Move model to CPU and set to evaluation mode
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"✅ Quantized model loaded successfully on {self.device}")

            # 3. Clean up memory
            gc.collect()

            # Log memory usage (if available)
            if torch.cuda.is_available():
                logger.info(f"🔋 GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

            logger.info("🎉 AWS Lambda model service initialized successfully!")

        except Exception as e:
            logger.error(f"❌ CRITICAL ERROR initializing model service: {e}")
            raise RuntimeError(f"Failed to load model on AWS Lambda: {e}")
    
    def predict(self, text: str, language: str) -> dict:
        """Make prediction using local model inference on AWS Lambda"""
        return self._predict_local(text, language)

    def _predict_local(self, text: str, language: str) -> dict:
        """Local inference using quantized PyTorch model"""
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                truncation=True,
                max_length=settings.MAX_SEQUENCE_LENGTH,
                padding=True,
                return_tensors="pt"  # PyTorch tensors
            )

            # Inference with no gradient computation (faster)
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            # Convert to numpy and apply softmax
            logits_np = logits.cpu().numpy()[0]
            exp_logits = np.exp(logits_np - np.max(logits_np))
            probabilities = exp_logits / np.sum(exp_logits)

            predicted_class = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_class])

            logger.info(f"Prediction made: {predicted_class} ({confidence:.4f})")

            return {
                "prediction": {
                    "label": predicted_class,
                    "label_text": settings.LABELS[predicted_class],
                    "confidence": confidence,
                    "probabilities": [float(p) for p in probabilities],
                    "human_prob": float(probabilities[0]),
                    "ai_prob": float(probabilities[1])
                },
                "language": language,
                "language_name": settings.LANGUAGE_NAMES.get(language, language),
                "tokens": text.split(),
                "inference_source": "local_quantized"
            }

        except Exception as e:
            logger.error(f"❌ Local prediction error: {e}")
            raise

    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            "model_name": settings.MODEL_NAME,
            "inference_mode": "aws_lambda_local",
            "device": self.device,
            "quantization": "INT8",
            "cache_dir": settings.MODEL_CACHE_DIR
        }

    def batch_predict(self, texts: list, languages: list) -> list:
        """Batch prediction for multiple texts"""
        results = []
        for text, language in zip(texts, languages):
            try:
                result = self.predict(text, language)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch prediction error for text: {e}")
                results.append({"error": str(e)})
        return results

# Create global model service instance
model_service = ModelService()
