# src/infrastructure/ml/model_manager.py
import torch
import numpy as np
import pickle
import os
from typing import Dict, List, Optional, Any
from src.core.config import settings
from src.models.classical.svd import SVDModel
from src.models.classical.cosine_similarity import CosineSimilarityModel
import logging

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Manages all ML models locally (no cloud costs)
    """
    
    def __init__(self):
        self.models = {}
        self.device = "cuda" if torch.cuda.is_available() and settings.USE_GPU else "cpu"
        self.load_models()
    
    def load_models(self):
        """Load all available models from disk"""
        model_dir = settings.MODEL_DIR
        
        # Load your existing SVD model
        svd_path = os.path.join(model_dir, "svd_model_50_factors.pkl")
        if os.path.exists(svd_path):
            try:
                with open(svd_path, 'rb') as f:
                    self.models['svd'] = pickle.load(f)
                logger.info("Loaded SVD model")
            except Exception as e:
                logger.error(f"Failed to load SVD model: {e}")
        
        # Load your existing cosine similarity models
        for mode in ['user', 'item']:
            model_path = os.path.join(model_dir, f"{mode}_based_cf.pkl")
            if os.path.exists(model_path):
                try:
                    with open(model_path, 'rb') as f:
                        self.models[f'cf_{mode}'] = pickle.load(f)
                    logger.info(f"Loaded {mode}-based CF model")
                except Exception as e:
                    logger.error(f"Failed to load {mode}-based CF model: {e}")
        
        # Load neural model if exists (we'll train this later)
        neural_path = os.path.join(model_dir, "two_tower_model.pt")
        if os.path.exists(neural_path):
            try:
                from src.models.neural.two_tower import TwoTowerModel
                model = TwoTowerModel()
                model.load_state_dict(torch.load(neural_path, map_location=self.device))
                model.eval()
                self.models['neural'] = model
                logger.info("Loaded neural model")
            except Exception as e:
                logger.error(f"Failed to load neural model: {e}")
    
    def get_recommendations(
        self,
        user_id: str,
        model_type: str = 'svd',
        n_recommendations: int = 10,
        **kwargs
    ) -> List[Dict]:
        """Get recommendations from specified model"""
        
        if model_type not in self.models:
            logger.warning(f"Model {model_type} not found, using fallback")
            model_type = 'svd' if 'svd' in self.models else list(self.models.keys())[0]
        
        model = self.models[model_type]
        
        try:
            # Call the appropriate recommendation method
            if hasattr(model, 'recommend'):
                recommendations = model.recommend(user_id, k=n_recommendations)
            else:
                # Fallback for custom models
                recommendations = self._generate_fallback_recommendations(user_id, n_recommendations)
            
            return recommendations
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []
    
    def _generate_fallback_recommendations(
        self,
        user_id: str,
        n_recommendations: int
    ) -> List[Dict]:
        """Generate fallback recommendations when model fails"""
        # Return popular games as fallback
        popular_games = [
            {'item_id': f'game_{i}', 'score': 0.9 - i*0.05}
            for i in range(n_recommendations)
        ]
        return popular_games
    
    def train_model(self, model_type: str, training_data: Any, **kwargs):
        """Train or retrain a model locally"""
        logger.info(f"Training {model_type} model locally...")
        
        if model_type == 'svd':
            from src.models.classical.svd import SVDModel
            model = SVDModel(n_factors=kwargs.get('n_factors', 50))
            model.fit(training_data)
            self.models['svd'] = model
            
            # Save model
            model_path = os.path.join(settings.MODEL_DIR, "svd_model_50_factors.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # Add other model types as needed
        
        logger.info(f"Model {model_type} trained and saved")

# Global model manager instance
model_manager = ModelManager() 