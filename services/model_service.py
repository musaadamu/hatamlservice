"""
Model Service - Handles inference via Local Model or HuggingFace API
Optimized for memory efficiency using Quantized Model
Model: msmaje/Quantizedphdhatamodel (INT8 quantized for faster inference)
"""

import os
import httpx
import numpy as np
import gc
from loguru import logger
from config import settings

# Try to import transformers for local inference
TRANSFORMERS_AVAILABLE = False
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
    logger.info("✅ Transformers and PyTorch available for local inference")
except ImportError:
    logger.warning("⚠️ Transformers not available - will use HuggingFace API")

class ModelService:
    """Service for inference supporting both local and API-based models"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.api_endpoint = settings.HF_API_ENDPOINT
        self.hf_token = settings.HF_TOKEN
        self.use_api = settings.USE_HF_INFERENCE_API
        
        # Determine device without importing torch early
        self.device = "cpu" 
        
        self.load_model()
    
    def load_model(self):
        """Initialize the quantized model from HuggingFace"""
        try:
            if self.use_api:
                logger.info(f"🌐 Using HuggingFace Inference API: {self.api_endpoint}")
                logger.info(f"📦 Model: {settings.MODEL_NAME}")
                return

            if not TRANSFORMERS_AVAILABLE:
                logger.warning("⚠️ Transformers not available, falling back to API mode")
                self.use_api = True
                return

            logger.info(f"📦 Loading quantized model from HuggingFace: {settings.MODEL_NAME}")

            # Create cache directory
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

            # 2. Load Quantized Model
            logger.info("🤖 Loading quantized model (this may take a moment)...")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                settings.MODEL_NAME,
                cache_dir=settings.MODEL_CACHE_DIR,
                token=self.hf_token,
                torch_dtype=torch.float32,  # Quantized model uses float32
                low_cpu_mem_usage=True  # Optimize memory usage
            )

            # Set model to evaluation mode
            self.model.eval()
            logger.info("✅ Quantized model loaded successfully")

            # 3. Clean up memory
            gc.collect()
            logger.info("🎉 Model service initialized successfully!")

        except Exception as e:
            logger.error(f"❌ Error initializing model service: {e}")
            logger.info("🔄 Falling back to HuggingFace API mode")
            self.use_api = True
    
    def predict(self, text: str, language: str) -> dict:
        """Make prediction using selected inference method"""
        if self.use_api:
            return self._predict_api(text, language)
        else:
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

    def _predict_api(self, text: str, language: str) -> dict:
        """HuggingFace Inference API implementation"""
        try:
            headers = {"Authorization": f"Bearer {self.hf_token}"} if self.hf_token else {}
            payload = {"inputs": text}

            logger.info(f"🌐 Calling HuggingFace API for prediction...")
            response = httpx.post(
                self.api_endpoint,
                json=payload,
                headers=headers,
                timeout=settings.INFERENCE_API_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()

            # Parse HuggingFace API response
            # Format: [[{"label": "LABEL_0", "score": 0.99}, {"label": "LABEL_1", "score": 0.01}]]
            if isinstance(data, list) and len(data) > 0:
                predictions = data[0] if isinstance(data[0], list) else data

                # Extract probabilities
                label_0_score = next((p['score'] for p in predictions if p['label'] == 'LABEL_0'), 0.5)
                label_1_score = next((p['score'] for p in predictions if p['label'] == 'LABEL_1'), 0.5)

                probabilities = [label_0_score, label_1_score]
                predicted_class = 0 if label_0_score > label_1_score else 1
                confidence = max(probabilities)

                logger.info(f"API Prediction: {predicted_class} ({confidence:.4f})")

                return {
                    "prediction": {
                        "label": predicted_class,
                        "label_text": settings.LABELS[predicted_class],
                        "confidence": confidence,
                        "probabilities": probabilities,
                        "human_prob": probabilities[0],
                        "ai_prob": probabilities[1]
                    },
                    "language": language,
                    "language_name": settings.LANGUAGE_NAMES.get(language, language),
                    "tokens": text.split(),
                    "inference_source": "huggingface_api"
                }
            else:
                raise ValueError(f"Unexpected API response format: {data}")

        except Exception as e:
            logger.error(f"❌ API prediction error: {e}")
            raise

    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            "model_name": settings.MODEL_NAME,
            "inference_mode": "huggingface_api" if self.use_api else "local_quantized",
            "device": self.device
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
