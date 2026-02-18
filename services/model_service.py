"""
Model Service - Handles inference via Local Model or HuggingFace API
Optimized for memory efficiency on CPU
"""

import os
import httpx
import torch
import numpy as np
from loguru import logger
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import settings


class ModelService:
    """Service for inference supporting both local and API-based models"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.api_endpoint = settings.HF_API_ENDPOINT
        self.hf_token = settings.HF_TOKEN
        self.use_api = settings.USE_HF_INFERENCE_API
        
        self.load_model()
    
    def load_model(self):
        """Initialize the model (local or API)"""
        try:
            if self.use_api:
                if not self.hf_token:
                    logger.warning("HF_TOKEN not set. Using HuggingFace API without authentication (Rate limits apply).")
                logger.info(f"Using HuggingFace Inference API: {self.api_endpoint}")
            else:
                logger.info(f"Loading model locally from {settings.MODEL_NAME}...")
                logger.info(f"Target device: {self.device}")
                
                # Create cache directory if it doesn't exist
                os.makedirs(settings.MODEL_CACHE_DIR, exist_ok=True)
                
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    settings.MODEL_NAME,
                    cache_dir=settings.MODEL_CACHE_DIR,
                    token=self.hf_token
                )
                
                # Load model with memory optimizations
                # low_cpu_mem_usage=True and torch_dtype=torch.float32 for CPU
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    settings.MODEL_NAME,
                    cache_dir=settings.MODEL_CACHE_DIR,
                    token=self.hf_token,
                    low_cpu_mem_usage=True
                )
                
                self.model.to(self.device)
                
                # Apply Dynamic Quantization (INT8) to save RAM on CPU
                if settings.USE_DYNAMIC_QUANTIZATION and self.device.type == "cpu":
                    logger.info("⚙️ Applying Dynamic Quantization (INT8) to save RAM...")
                    self.model = torch.quantization.quantize_dynamic(
                        self.model, {torch.nn.Linear}, dtype=torch.qint8
                    )
                    logger.info("✅ Model quantized successfully")
                
                self.model.eval()  # Set to evaluation mode
                
                logger.info("Local model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error initializing model service: {e}")
            if not self.use_api:
                logger.warning("Local load failed. Suggesting switch to API or checking memory.")
            raise
    
    def _prepare_headers(self) -> dict:
        """Prepare headers for API request"""
        headers = {"Content-Type": "application/json"}
        if self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"
        return headers
    
    def predict(self, text: str, language: str) -> dict:
        """Make prediction using selected inference method"""
        if self.use_api:
            return self._predict_api(text, language)
        else:
            return self._predict_local(text, language)

    def _predict_local(self, text: str, language: str) -> dict:
        """Local inference using Transformers/Torch"""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                truncation=True,
                max_length=settings.MAX_SEQUENCE_LENGTH,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                # Apply softmax to get probabilities
                probabilities = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]
                
            # Get the top prediction
            predicted_class = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_class])
            
            # Simple tokenization for response
            tokens = text.split()
            
            result = {
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
                "tokens": tokens,
                "text_length": len(text),
                "token_count": len(tokens),
                "inference_source": "local_inference"
            }
            return result
            
        except Exception as e:
            logger.error(f"Local prediction error: {e}")
            raise

    def _predict_api(self, text: str, language: str) -> dict:
        """Remote inference via HuggingFace API"""
        try:
            payload = {"inputs": text, "parameters": {"truncation": True}}
            headers = self._prepare_headers()
            
            response = httpx.post(
                self.api_endpoint,
                json=payload,
                headers=headers,
                timeout=settings.INFERENCE_API_TIMEOUT
            )
            response.raise_for_status()
            api_response = response.json()
            
            # Handle list responses (common for text classification)
            if isinstance(api_response, list) and len(api_response) > 0:
                predictions = api_response[0]
            else:
                raise ValueError(f"Unexpected API response: {api_response}")
                
            # Sort by score descending
            predictions = sorted(predictions, key=lambda x: x.get('score', 0), reverse=True)
            top_prediction = predictions[0]
            
            # Map labels
            label_text = top_prediction.get('label', 'UNKNOWN')
            confidence = top_prediction.get('score', 0.0)
            predicted_class = 0 if "LABEL_0" in label_text else 1
            
            probabilities = [0.0, 0.0]
            for pred in predictions[:2]:
                label = pred.get('label', '')
                score = pred.get('score', 0.0)
                if "LABEL_0" in label: probabilities[0] = score
                elif "LABEL_1" in label: probabilities[1] = score
            
            return {
                "prediction": {
                    "label": predicted_class,
                    "label_text": settings.LABELS[predicted_class],
                    "confidence": float(confidence),
                    "probabilities": probabilities,
                    "human_prob": float(probabilities[0]),
                    "ai_prob": float(probabilities[1])
                },
                "language": language,
                "language_name": settings.LANGUAGE_NAMES.get(language, language),
                "tokens": text.split(),
                "text_length": len(text),
                "token_count": len(text.split()),
                "inference_source": "huggingface_api"
            }
        except Exception as e:
            logger.error(f"API prediction error: {e}")
            raise

    def batch_predict(self, texts: list, languages: list) -> list:
        """Batch prediction logic"""
        results = []
        for text, language in zip(texts, languages):
            try:
                results.append(self.predict(text, language))
            except Exception as e:
                results.append({"error": str(e), "language": language})
        return results
    
    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            "model_name": settings.MODEL_NAME,
            "inference_mode": "api" if self.use_api else "local",
            "device": str(self.device) if not self.use_api else "n/a",
            "api_endpoint": self.api_endpoint if self.use_api else "n/a",
            "supported_languages": settings.SUPPORTED_LANGUAGES,
            "max_sequence_length": settings.MAX_SEQUENCE_LENGTH,
            "labels": settings.LABELS
        }


# Create global model service instance
model_service = ModelService()

