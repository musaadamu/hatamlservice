"""
Explainability Service - LIME-based token attribution
"""

import numpy as np
from lime.lime_text import LimeTextExplainer
from loguru import logger
from config import settings


class ExplainabilityService:
    """Service for generating explanations using LIME"""
    
    def __init__(self, model_service):
        self.model_service = model_service
        self.explainer = LimeTextExplainer(
            class_names=list(settings.LABELS.values()),
            random_state=42
        )
    
    def explain_prediction(self, text: str, language: str, num_features: int = None) -> dict:
        """
        Generate LIME explanation for a prediction
        
        Args:
            text: Input text
            language: Language code
            num_features: Number of top features to return
            
        Returns:
            Dictionary with explanation data
        """
        try:
            if num_features is None:
                num_features = settings.LIME_NUM_FEATURES
            
            # Create prediction function for LIME
            def predict_proba(texts):
                probas = []
                for t in texts:
                    try:
                        result = self.model_service.predict(t, language)
                        probas.append(result['prediction']['probabilities'])
                    except:
                        # Return neutral probabilities on error
                        probas.append([0.5, 0.5])
                return np.array(probas)
            
            # Generate explanation
            explanation = self.explainer.explain_instance(
                text,
                predict_proba,
                num_features=num_features,
                num_samples=settings.LIME_NUM_SAMPLES
            )
            
            # Get feature importance
            feature_weights = explanation.as_list()
            
            # Get tokens and their importance scores
            tokens = text.split()
            token_importances = self._map_features_to_tokens(
                tokens, feature_weights
            )
            
            result = {
                "tokens": tokens,
                "importances": token_importances,
                "top_features": feature_weights[:num_features],
                "method": "lime",
                "num_samples": settings.LIME_NUM_SAMPLES
            }
            
            logger.info(f"Explanation generated for text of length {len(text)}")
            return result
            
        except Exception as e:
            logger.error(f"Explainability error: {e}")
            # Return simple token-based explanation as fallback
            tokens = text.split()
            return {
                "tokens": tokens,
                "importances": [0.5] * len(tokens),
                "top_features": [],
                "method": "fallback",
                "error": str(e)
            }
    
    def _map_features_to_tokens(self, tokens: list, feature_weights: list) -> list:
        """
        Map LIME feature weights to individual tokens
        
        Args:
            tokens: List of tokens
            feature_weights: List of (feature, weight) tuples from LIME
            
        Returns:
            List of importance scores for each token
        """
        # Create a dictionary of feature weights
        weight_dict = {feature: weight for feature, weight in feature_weights}
        
        # Map tokens to weights
        importances = []
        for token in tokens:
            # Find matching feature weight
            weight = 0.0
            for feature, w in feature_weights:
                if token.lower() in feature.lower():
                    weight = abs(w)
                    break
            importances.append(weight)
        
        # Normalize to 0-1 range
        if max(importances) > 0:
            importances = [i / max(importances) for i in importances]
        
        return importances
    
    def get_top_features(self, text: str, language: str, n: int = 10) -> list:
        """
        Get top N most important features
        
        Args:
            text: Input text
            language: Language code
            n: Number of top features
            
        Returns:
            List of (feature, importance) tuples
        """
        explanation = self.explain_prediction(text, language, num_features=n)
        return explanation.get('top_features', [])

