#!/usr/bin/env python
"""
Recommendation API integration with trained Steam recommendation models.
This API provides real-time game recommendations using the trained models.
"""
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import logging
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data models
class GameRecommendation(BaseModel):
    game_id: str
    game_name: str
    genres: List[str]
    tags: List[str]
    price: float
    score: float
    score_breakdown: Optional[Dict[str, float]] = None

class RecommendationRequest(BaseModel):
    user_id: str
    n_recommendations: int = 10
    model_type: str = "hybrid"  # hybrid, popularity, content_based, collaborative

class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[GameRecommendation]
    model_used: str
    total_recommendations: int

class SteamRecommendationAPI:
    """API for Steam game recommendations using trained models."""
    
    def __init__(self, models_dir: str = "data/models", data_dir: str = "data/processed"):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        
        # Load models and data
        self.models = {}
        self.games_df = None
        self.user_item_matrix = None
        
        self._load_models()
        self._load_data()
    
    def _load_models(self) -> None:
        """Load all trained recommendation models."""
        logger.info("Loading trained recommendation models...")
        
        try:
            # Load popularity model
            popularity_file = self.models_dir / "popularity_data.csv"
            if popularity_file.exists():
                self.models['popularity'] = pd.read_csv(popularity_file)
                logger.info("Loaded popularity model")
            
            # Load content-based model
            content_file = self.models_dir / "content_based_similarity.csv"
            if content_file.exists():
                self.models['content_based'] = pd.read_csv(content_file, index_col=0)
                logger.info("Loaded content-based model")
            
            # Load collaborative filtering models
            for method in ['nmf', 'svd']:
                model_file = self.models_dir / f"collaborative_{method}_model.pkl"
                if model_file.exists():
                    with open(model_file, 'rb') as f:
                        self.models[f'collaborative_{method}'] = pickle.load(f)
                    logger.info(f"Loaded collaborative {method.upper()} model")
            
            # Load hybrid model
            hybrid_file = self.models_dir / "hybrid_model.pkl"
            if hybrid_file.exists():
                with open(hybrid_file, 'rb') as f:
                    self.models['hybrid'] = pickle.load(f)
                logger.info("Loaded hybrid model")
            
            # Load model performance
            perf_file = self.models_dir / "model_performance.json"
            if perf_file.exists():
                with open(perf_file, 'r') as f:
                    self.model_performance = json.load(f)
                logger.info("Loaded model performance metrics")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def _load_data(self) -> None:
        """Load processed data for recommendations."""
        logger.info("Loading processed data...")
        
        try:
            # Load games data
            games_file = self.data_dir / "games_processed.csv"
            if games_file.exists():
                self.games_df = pd.read_csv(games_file)
                logger.info(f"Loaded {len(self.games_df)} games")
            
            # Load user-item matrix
            matrix_file = self.data_dir / "user_item_matrix.csv"
            if matrix_file.exists():
                self.user_item_matrix = pd.read_csv(matrix_file, index_col=0)
                logger.info(f"Loaded user-item matrix: {self.user_item_matrix.shape}")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def get_popularity_recommendations(self, n_recommendations: int = 10) -> List[Dict]:
        """Get popularity-based recommendations."""
        if 'popularity' not in self.models:
            raise ValueError("Popularity model not loaded")
        
        # Get top popular games
        top_games = self.models['popularity'].head(n_recommendations)
        
        recommendations = []
        for _, game in top_games.iterrows():
            game_info = self.games_df[self.games_df['item_id'] == game['item_id']].iloc[0]
            
            recommendations.append({
                'game_id': game['item_id'],
                'game_name': game_info['item_name'],
                'genres': eval(game_info['genre']) if isinstance(game_info['genre'], str) else game_info['genre'],
                'tags': eval(game_info['tags']) if isinstance(game_info['tags'], str) else game_info['tags'],
                'price': game_info['price'],
                'score': game['popularity_score'],
                'score_breakdown': {
                    'popularity_score': game['popularity_score'],
                    'mean_rating': game['mean'],
                    'interaction_count': game['count']
                }
            })
        
        return recommendations
    
    def get_content_based_recommendations(self, user_id: str, n_recommendations: int = 10) -> List[Dict]:
        """Get content-based recommendations for a user."""
        if 'content_based' not in self.models or self.user_item_matrix is None:
            raise ValueError("Content-based model or user-item matrix not loaded")
        
        # Get user's liked games
        if user_id not in self.user_item_matrix.index:
            # If user not found, return popular games
            return self.get_popularity_recommendations(n_recommendations)
        
        user_games = self.user_item_matrix.loc[user_id]
        liked_games = user_games[user_games > 0].index.tolist()
        
        if not liked_games:
            return self.get_popularity_recommendations(n_recommendations)
        
        # Calculate average similarity to liked games
        game_scores = []
        similarity_matrix = self.models['content_based']
        
        for _, game in self.games_df.iterrows():
            game_id = game['item_id']
            if game_id in liked_games:
                continue  # Skip already liked games
            
            # Calculate average similarity to liked games
            similarities = []
            for liked_game in liked_games:
                if liked_game in similarity_matrix.index and game_id in similarity_matrix.columns:
                    sim = similarity_matrix.loc[liked_game, game_id]
                    similarities.append(sim)
            
            if similarities:
                avg_similarity = np.mean(similarities)
                game_scores.append({
                    'game_id': game_id,
                    'score': avg_similarity,
                    'game_info': game
                })
        
        # Sort by score and get top recommendations
        game_scores.sort(key=lambda x: x['score'], reverse=True)
        top_games = game_scores[:n_recommendations]
        
        # Format recommendations
        recommendations = []
        for game_data in top_games:
            game = game_data['game_info']
            recommendations.append({
                'game_id': game['item_id'],
                'game_name': game['item_name'],
                'genres': eval(game['genre']) if isinstance(game['genre'], str) else game['genre'],
                'tags': eval(game['tags']) if isinstance(game['tags'], str) else game['tags'],
                'price': game['price'],
                'score': game_data['score'],
                'score_breakdown': {
                    'content_similarity': game_data['score']
                }
            })
        
        return recommendations
    
    def get_collaborative_recommendations(self, user_id: str, method: str = 'nmf', n_recommendations: int = 10) -> List[Dict]:
        """Get collaborative filtering recommendations."""
        model_key = f'collaborative_{method}'
        if model_key not in self.models:
            raise ValueError(f"Collaborative {method.upper()} model not loaded")
        
        model_data = self.models[model_key]
        
        if user_id not in model_data['user_ids']:
            # If user not in training data, return popular games
            return self.get_popularity_recommendations(n_recommendations)
        
        # Get user factors and calculate scores
        user_idx = model_data['user_ids'].index(user_id)
        user_factors = model_data['user_factors'][user_idx]
        item_factors = model_data['item_factors']
        
        # Calculate scores for all items
        scores = np.dot(user_factors, item_factors)
        
        # Get top recommendations
        top_indices = np.argsort(scores)[::-1][:n_recommendations]
        
        recommendations = []
        for idx in top_indices:
            item_id = model_data['item_ids'][idx]
            game_info = self.games_df[self.games_df['item_id'] == item_id].iloc[0]
            
            recommendations.append({
                'game_id': item_id,
                'game_name': game_info['item_name'],
                'genres': eval(game_info['genre']) if isinstance(game_info['genre'], str) else game_info['genre'],
                'tags': eval(game_info['tags']) if isinstance(game_info['tags'], str) else game_info['tags'],
                'price': game_info['price'],
                'score': float(scores[idx]),
                'score_breakdown': {
                    'collaborative_score': float(scores[idx]),
                    'method': method.upper()
                }
            })
        
        return recommendations
    
    def get_hybrid_recommendations(self, user_id: str, n_recommendations: int = 10) -> List[Dict]:
        """Get hybrid recommendations combining multiple approaches."""
        if 'hybrid' not in self.models:
            raise ValueError("Hybrid model not loaded")
        
        # Get recommendations from different models
        try:
            pop_recs = self.get_popularity_recommendations(n_recommendations * 2)
            content_recs = self.get_content_based_recommendations(user_id, n_recommendations * 2)
            cf_recs = self.get_collaborative_recommendations(user_id, 'nmf', n_recommendations * 2)
        except Exception as e:
            logger.warning(f"Error getting individual recommendations: {e}")
            return self.get_popularity_recommendations(n_recommendations)
        
        # Combine and score recommendations
        all_games = {}
        
        # Add popularity recommendations
        for rec in pop_recs:
            game_id = rec['game_id']
            if game_id not in all_games:
                all_games[game_id] = rec.copy()
                all_games[game_id]['hybrid_score'] = rec['score'] * 0.4
            else:
                all_games[game_id]['hybrid_score'] += rec['score'] * 0.4
        
        # Add content-based recommendations
        for rec in content_recs:
            game_id = rec['game_id']
            if game_id not in all_games:
                all_games[game_id] = rec.copy()
                all_games[game_id]['hybrid_score'] = rec['score'] * 0.3
            else:
                all_games[game_id]['hybrid_score'] += rec['score'] * 0.3
        
        # Add collaborative filtering recommendations
        for rec in cf_recs:
            game_id = rec['game_id']
            if game_id not in all_games:
                all_games[game_id] = rec.copy()
                all_games[game_id]['hybrid_score'] = rec['score'] * 0.3
            else:
                all_games[game_id]['hybrid_score'] += rec['score'] * 0.3
        
        # Sort by hybrid score and get top recommendations
        sorted_games = sorted(all_games.values(), key=lambda x: x['hybrid_score'], reverse=True)
        top_recommendations = sorted_games[:n_recommendations]
        
        # Update final scores
        for rec in top_recommendations:
            rec['score'] = rec['hybrid_score']
            rec['score_breakdown']['hybrid_score'] = rec['hybrid_score']
        
        return top_recommendations
    
    def get_recommendations(self, user_id: str, model_type: str = "hybrid", n_recommendations: int = 10) -> List[Dict]:
        """Get recommendations using the specified model type."""
        logger.info(f"Getting {n_recommendations} recommendations for user {user_id} using {model_type} model")
        
        try:
            if model_type == "popularity":
                return self.get_popularity_recommendations(n_recommendations)
            elif model_type == "content_based":
                return self.get_content_based_recommendations(user_id, n_recommendations)
            elif model_type.startswith("collaborative"):
                method = model_type.split("_")[1] if "_" in model_type else "nmf"
                return self.get_collaborative_recommendations(user_id, method, n_recommendations)
            elif model_type == "hybrid":
                return self.get_hybrid_recommendations(user_id, n_recommendations)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            # Fallback to popularity recommendations
            return self.get_popularity_recommendations(n_recommendations)

# Create FastAPI router
router = APIRouter(prefix="/recommendations", tags=["recommendations"])

# Initialize recommendation API
try:
    recommendation_api = SteamRecommendationAPI()
    logger.info("Recommendation API initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize recommendation API: {e}")
    recommendation_api = None

@router.get("/health")
async def health_check():
    """Health check for recommendation service."""
    if recommendation_api is None:
        raise HTTPException(status_code=503, detail="Recommendation service not available")
    
    return {
        "status": "healthy",
        "models_loaded": list(recommendation_api.models.keys()) if recommendation_api else [],
        "service": "steam_recommendations"
    }

@router.post("/generate", response_model=RecommendationResponse)
async def generate_recommendations(request: RecommendationRequest):
    """Generate game recommendations for a user."""
    if recommendation_api is None:
        raise HTTPException(status_code=503, detail="Recommendation service not available")
    
    try:
        recommendations = recommendation_api.get_recommendations(
            user_id=request.user_id,
            model_type=request.model_type,
            n_recommendations=request.n_recommendations
        )
        
        # Convert to response format
        game_recommendations = []
        for rec in recommendations:
            game_rec = GameRecommendation(
                game_id=rec['game_id'],
                game_name=rec['game_name'],
                genres=rec['genres'],
                tags=rec['tags'],
                price=rec['price'],
                score=rec['score'],
                score_breakdown=rec.get('score_breakdown')
            )
            game_recommendations.append(game_rec)
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=game_recommendations,
            model_used=request.model_type,
            total_recommendations=len(game_recommendations)
        )
    
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@router.get("/models")
async def get_available_models():
    """Get information about available recommendation models."""
    if recommendation_api is None:
        raise HTTPException(status_code=503, detail="Recommendation service not available")
    
    models_info = {}
    for model_name in recommendation_api.models.keys():
        if model_name == 'popularity':
            models_info[model_name] = {
                'type': 'popularity_based',
                'description': 'Recommends games based on overall popularity and ratings'
            }
        elif model_name == 'content_based':
            models_info[model_name] = {
                'type': 'content_based',
                'description': 'Recommends games similar to user\'s liked games based on genres and tags'
            }
        elif model_name.startswith('collaborative'):
            method = model_name.split('_')[1].upper()
            models_info[model_name] = {
                'type': 'collaborative_filtering',
                'description': f'Uses {method} matrix factorization to find similar users and games'
            }
        elif model_name == 'hybrid':
            models_info[model_name] = {
                'type': 'hybrid',
                'description': 'Combines multiple approaches for better recommendations'
            }
    
    return {
        "available_models": models_info,
        "total_models": len(models_info)
    }

@router.get("/performance")
async def get_model_performance():
    """Get performance metrics for all models."""
    if recommendation_api is None:
        raise HTTPException(status_code=503, detail="Recommendation service not available")
    
    if hasattr(recommendation_api, 'model_performance'):
        return recommendation_api.model_performance
    else:
        return {"message": "Performance metrics not available"}

@router.get("/popular")
async def get_popular_games(limit: int = 20):
    """Get popular games (popularity-based recommendations)."""
    if recommendation_api is None:
        raise HTTPException(status_code=503, detail="Recommendation service not available")
    
    try:
        recommendations = recommendation_api.get_popularity_recommendations(limit)
        
        # Convert to response format
        game_recommendations = []
        for rec in recommendations:
            game_rec = GameRecommendation(
                game_id=rec['game_id'],
                game_name=rec['game_name'],
                genres=rec['genres'],
                tags=rec['tags'],
                price=rec['price'],
                score=rec['score'],
                score_breakdown=rec.get('score_breakdown')
            )
            game_recommendations.append(game_rec)
        
        return {
            "popular_games": game_recommendations,
            "total_games": len(game_recommendations)
        }
    
    except Exception as e:
        logger.error(f"Error getting popular games: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting popular games: {str(e)}") 