"""
Model Service — Dual-Mode Inference
Supports both HuggingFace Inference API (remote) and local PyTorch model loading.

Modes:
  - HF Inference API (USE_HF_INFERENCE_API=true):
      Calls api-inference.huggingface.co — ideal for Render free tier
  - Local Model (USE_HF_INFERENCE_API=false):
      Loads the model with PyTorch — ideal for local dev, Lambda, or GPU servers
"""

import os
import numpy as np
import gc
import requests
from loguru import logger
from config import settings


class ModelService:
    """Service for model inference — supports HF API and local modes"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.hf_token = settings.HF_TOKEN
        self.device = "cpu"
        self.inference_mode = "hf_api" if settings.USE_HF_INFERENCE_API else "local"

        logger.info(f"🔧 Inference mode: {self.inference_mode}")

        if settings.USE_HF_INFERENCE_API:
            self._init_hf_api()
        else:
            self._init_local_model()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _init_hf_api(self):
        """Initialize for HuggingFace Inference API mode"""
        logger.info(f"🌐 Using HuggingFace Inference API: {settings.HF_API_ENDPOINT}")
        if not self.hf_token or self.hf_token == "your_hf_token_here":
            logger.warning("⚠️ HF_TOKEN is not set — API calls may fail for private/gated models")
        else:
            logger.info("✅ HF Token configured")

    def _init_local_model(self):
        """Load the model locally with PyTorch"""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
        except ImportError as e:
            logger.error(f"❌ PyTorch/Transformers not installed — {e}")
            raise ImportError(
                "PyTorch and Transformers are required for local inference. "
                "Install with: pip install torch transformers"
            )

        try:
            # Determine model source: local path or HuggingFace Hub
            model_source = settings.LOCAL_MODEL_PATH or settings.MODEL_NAME
            cache_dir = settings.MODEL_CACHE_DIR

            logger.info(f"📦 Loading model from: {model_source}")
            os.makedirs(cache_dir, exist_ok=True)

            # Determine device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"🖥️ Using device: {self.device}")

            # Load tokenizer
            logger.info("🔤 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_source,
                cache_dir=cache_dir,
                token=self.hf_token,
                use_fast=True
            )
            logger.info("✅ Tokenizer loaded")

            # Load model
            logger.info("🤖 Loading model (this may take a moment on first run)...")
            dtype = torch.float16 if (settings.USE_HALF_PRECISION and self.device == "cuda") else torch.float32
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_source,
                cache_dir=cache_dir,
                token=self.hf_token,
                torch_dtype=dtype,
                low_cpu_mem_usage=True
            )

            self.model.to(self.device)
            self.model.eval()
            gc.collect()

            logger.info(f"✅ Model loaded on {self.device}")
            logger.info("🎉 Local model service initialized successfully!")

        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, text: str, language: str) -> dict:
        """Route prediction to the appropriate backend"""
        if settings.USE_HF_INFERENCE_API:
            return self._predict_hf_api(text, language)
        else:
            return self._predict_local(text, language)

    def _predict_hf_api(self, text: str, language: str) -> dict:
        """Predict using HuggingFace Inference API"""
        try:
            headers = {"Content-Type": "application/json"}
            if self.hf_token and self.hf_token != "your_hf_token_here":
                headers["Authorization"] = f"Bearer {self.hf_token}"

            payload = {"inputs": text}
            response = requests.post(
                settings.HF_API_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=settings.INFERENCE_API_TIMEOUT
            )

            if response.status_code == 503:
                # Model is loading — retry after wait
                resp_json = response.json()
                estimated_time = resp_json.get("estimated_time", 30)
                logger.warning(f"⏳ Model loading on HF, estimated wait: {estimated_time}s")
                import time
                time.sleep(min(estimated_time, 60))
                response = requests.post(
                    settings.HF_API_ENDPOINT,
                    headers=headers,
                    json=payload,
                    timeout=settings.INFERENCE_API_TIMEOUT + 60
                )

            response.raise_for_status()
            api_result = response.json()

            # HF API returns [[{"label": "...", "score": ...}, ...]]
            if isinstance(api_result, list) and len(api_result) > 0:
                if isinstance(api_result[0], list):
                    predictions = api_result[0]
                else:
                    predictions = api_result
            else:
                raise ValueError(f"Unexpected API response format: {api_result}")

            # Parse results — map label names to our internal format
            label_scores = {}
            for item in predictions:
                label_scores[item["label"]] = item["score"]

            human_prob = label_scores.get("Human-written", label_scores.get("LABEL_0", 0.5))
            ai_prob = label_scores.get("AI-generated", label_scores.get("LABEL_1", 0.5))

            predicted_class = 1 if ai_prob > human_prob else 0
            confidence = max(human_prob, ai_prob)

            logger.info(f"HF API prediction: {settings.LABELS[predicted_class]} ({confidence:.4f})")

            return {
                "prediction": {
                    "label": predicted_class,
                    "label_text": settings.LABELS[predicted_class],
                    "confidence": confidence,
                    "probabilities": [human_prob, ai_prob],
                    "human_prob": human_prob,
                    "ai_prob": ai_prob
                },
                "language": language,
                "language_name": settings.LANGUAGE_NAMES.get(language, language),
                "tokens": text.split(),
                "inference_source": "huggingface_api"
            }

        except requests.exceptions.Timeout:
            logger.error("❌ HF Inference API timeout")
            raise RuntimeError("HuggingFace Inference API timed out. The model may be loading.")
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ HF Inference API error: {e}")
            raise RuntimeError(f"HuggingFace Inference API error: {e}")

    def _predict_local(self, text: str, language: str) -> dict:
        """Predict using locally loaded PyTorch model"""
        import torch

        try:
            inputs = self.tokenizer(
                text,
                truncation=True,
                max_length=settings.MAX_SEQUENCE_LENGTH,
                padding=True,
                return_tensors="pt"
            )

            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            logits_np = logits.cpu().numpy()[0]
            exp_logits = np.exp(logits_np - np.max(logits_np))
            probabilities = exp_logits / np.sum(exp_logits)

            predicted_class = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_class])

            logger.info(f"Local prediction: {settings.LABELS[predicted_class]} ({confidence:.4f})")

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
                "inference_source": "local_model"
            }

        except Exception as e:
            logger.error(f"❌ Local prediction error: {e}")
            raise

    # ------------------------------------------------------------------
    # Model Info & Batch
    # ------------------------------------------------------------------
    def get_model_info(self) -> dict:
        """Get model information"""
        info = {
            "model_name": settings.MODEL_NAME,
            "inference_mode": self.inference_mode,
            "supported_languages": settings.SUPPORTED_LANGUAGES,
            "labels": settings.LABELS,
        }
        if settings.USE_HF_INFERENCE_API:
            info["api_endpoint"] = settings.HF_API_ENDPOINT
        else:
            info["device"] = self.device
            info["local_model_path"] = settings.LOCAL_MODEL_PATH or "HuggingFace Hub"
        return info

    def batch_predict(self, texts: list, languages: list) -> list:
        """Batch prediction for multiple texts"""
        results = []
        for text, language in zip(texts, languages):
            try:
                result = self.predict(text, language)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch prediction error: {e}")
                results.append({"error": str(e)})
        return results


# Create global model service instance
model_service = ModelService()
