"""Recommendation service for the Steam Game Recommender application."""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple
import logging
from pathlib import Path
import pickle
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from src.domain.entities.recommendation import (
    RecommendationRequest,
    RecommendationResponse,
    RecommendationType,
    UserGameInteraction,
    RecommendationFeedback
)

logger = logging.getLogger(__name__)

class RecommendationService:
    """Service for generating and managing game recommendations."""
    
    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
        self.games_df = None
        self.reviews_df = None
        self.user_item_matrix = None
        self.models = {}
        self.scaler = StandardScaler()
        
        # Load data and models
        self._load_data()
        self._load_or_train_models()
    
    def _load_data(self) -> None:
        """Load processed Steam data."""
        try:
            # Load games data
            games_file = self.data_dir / "games_processed.csv"
            if games_file.exists():
                self.games_df = pd.read_csv(games_file)
                logger.info(f"Loaded {len(self.games_df)} games")
            
            # Load reviews data
            reviews_file = self.data_dir / "reviews_processed.csv"
            if reviews_file.exists():
                self.reviews_df = pd.read_csv(reviews_file)
                logger.info(f"Loaded {len(self.reviews_df)} reviews")
            
            # Load user-item matrix
            matrix_file = self.data_dir / "user_item_matrix.csv"
            if matrix_file.exists():
                self.user_item_matrix = pd.read_csv(matrix_file, index_col=0)
                logger.info(f"Loaded user-item matrix: {self.user_item_matrix.shape}")
                
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            # Create sample data if files don't exist
            self._create_sample_data()
    
    def _create_sample_data(self) -> None:
        """Create sample data for development/testing."""
        logger.info("Creating sample data for development...")
        
        # Create sample games
        np.random.seed(42)
        n_games = 100
        n_users = 50
        
        games_data = []
        for i in range(n_games):
            games_data.append({
                'item_id': f'game_{i}',
                'item_name': f'Sample Game {i}',
                'genre': np.random.choice(['Action', 'RPG', 'Strategy', 'Adventure', 'Sports']),
                'price': np.random.uniform(0, 60),
                'rating': np.random.uniform(2.5, 4.5),
                'playtime_median': np.random.randint(1, 100)
            })
        
        self.games_df = pd.DataFrame(games_data)
        
        # Create sample reviews
        reviews_data = []
        for i in range(n_users):
            n_reviews = np.random.randint(5, 20)
            for j in range(n_reviews):
                game_id = np.random.choice(self.games_df['item_id'])
                reviews_data.append({
                    'user_id': f'user_{i}',
                    'item_id': game_id,
                    'rating': np.random.randint(1, 6),
                    'playtime_forever': np.random.randint(1, 200)
                })
        
        self.reviews_df = pd.DataFrame(reviews_data)
        
        # Create user-item matrix
        self.user_item_matrix = self.reviews_df.pivot_table(
            index='user_id', 
            columns='item_id', 
            values='rating', 
            fill_value=0
        )
        
        logger.info("Sample data created successfully")
    
    def _load_or_train_models(self) -> None:
        """Load pre-trained models or train new ones."""
        try:
            # Try to load pre-trained models
            models_dir = self.data_dir / "models"
            if models_dir.exists():
                self._load_models(models_dir)
            else:
                self._train_models()
        except Exception as e:
            logger.error(f"Error loading/training models: {e}")
            self._train_models()
    
    def _load_models(self, models_dir: Path) -> None:
        """Load pre-trained models from disk."""
        try:
            # Load SVD model
            svd_file = models_dir / "svd_model.pkl"
            if svd_file.exists():
                with open(svd_file, 'rb') as f:
                    self.models['svd'] = pickle.load(f)
                logger.info("Loaded SVD model")
            
            # Load other models as needed
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self._train_models()
    
    def _train_models(self) -> None:
        """Train recommendation models."""
        logger.info("Training recommendation models...")
        
        if self.user_item_matrix is None or self.user_item_matrix.empty:
            logger.error("No user-item matrix available for training")
            return
        
        # Train SVD model (best performer - 26% precision)
        self._train_svd_model()
        
        # Train item-based collaborative filtering
        self._train_item_based_model()
        
        # Build popularity model
        self._build_popularity_model()
        
        logger.info("All models trained successfully")
    
    def _train_svd_model(self) -> None:
        """Train SVD model for matrix factorization."""
        try:
            # Fill NaN values with 0
            matrix_filled = self.user_item_matrix.fillna(0)
            
            # Apply SVD
            n_components = min(50, min(matrix_filled.shape) - 1)
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            
            # Fit and transform
            matrix_transformed = svd.fit_transform(matrix_filled)
            
            # Store model
            self.models['svd'] = {
                'model': svd,
                'matrix_transformed': matrix_transformed,
                'user_item_matrix': matrix_filled,
                'type': 'svd'
            }
            
            logger.info(f"SVD model trained with {n_components} components")
            
        except Exception as e:
            logger.error(f"Error training SVD model: {e}")
    
    def _train_item_based_model(self) -> None:
        """Train item-based collaborative filtering model."""
        try:
            # Calculate item-item similarity matrix
            matrix_filled = self.user_item_matrix.fillna(0)
            item_similarity = cosine_similarity(matrix_filled.T)
            
            # Store model
            self.models['item_based'] = {
                'similarity_matrix': item_similarity,
                'item_ids': matrix_filled.columns.tolist(),
                'user_item_matrix': matrix_filled,
                'type': 'item_based'
            }
            
            logger.info("Item-based collaborative filtering model trained")
            
        except Exception as e:
            logger.error(f"Error training item-based model: {e}")
    
    def _build_popularity_model(self) -> None:
        """Build popularity-based recommendation model."""
        try:
            # Calculate game popularity
            game_stats = self.reviews_df.groupby('item_id').agg({
                'rating': ['mean', 'count'],
                'playtime_forever': 'mean'
            }).reset_index()
            
            game_stats.columns = ['item_id', 'avg_rating', 'review_count', 'avg_playtime']
            
            # Calculate popularity score
            game_stats['popularity_score'] = (
                game_stats['avg_rating'] * 0.6 + 
                (game_stats['review_count'] / game_stats['review_count'].max()) * 0.3 +
                (game_stats['avg_playtime'] / game_stats['avg_playtime'].max()) * 0.1
            )
            
            # Sort by popularity
            game_stats = game_stats.sort_values('popularity_score', ascending=False)
            
            # Store model
            self.models['popularity'] = {
                'popular_games': game_stats,
                'type': 'popularity'
            }
            
            logger.info("Popularity model built successfully")
            
        except Exception as e:
            logger.error(f"Error building popularity model: {e}")
    
    async def generate_recommendations(self, request: RecommendationRequest) -> RecommendationResponse:
        """Generate personalized game recommendations for a user."""
        try:
            user_id = request.user_id
            n_recommendations = request.limit
            recommendation_type = request.recommendation_type
            include_played = request.include_played
            
            logger.info(f"Generating {n_recommendations} recommendations for user {user_id} using {recommendation_type}")
            
            # Map recommendation type to model type
            model_type = self._map_recommendation_type_to_model(recommendation_type)
            
            # Get user's played games
            user_games = self._get_user_games(str(user_id))
            
            # Generate recommendations based on model type
            if model_type == 'svd':
                recommendations = self._get_svd_recommendations(str(user_id), n_recommendations, user_games, include_played)
            elif model_type == 'item_based':
                recommendations = self._get_item_based_recommendations(str(user_id), n_recommendations, user_games, include_played)
            elif model_type == 'popularity':
                recommendations = self._get_popularity_recommendations(n_recommendations, user_games, include_played)
            elif model_type == 'hybrid':
                recommendations = self._get_hybrid_recommendations(str(user_id), n_recommendations, user_games, include_played)
            else:
                # Default to SVD
                recommendations = self._get_svd_recommendations(str(user_id), n_recommendations, user_games, include_played)
            
            # Convert to response format
            game_recommendations = []
            for game_id, score in recommendations:
                game_info = self._get_game_info(game_id)
                if game_info:
                    # Create recommendation score
                    from src.domain.entities.recommendation import RecommendationScore, RecommendationReason
                    
                    rec_score = RecommendationScore(
                        score=min(score, 1.0),  # Ensure score is 0-1
                        confidence=min(score / 5.0, 1.0),  # Normalize confidence
                        reason=RecommendationReason.SIMILAR_GAMES,
                        explanation=f"Recommended based on {model_type} model with score {score:.3f}"
                    )
                    
                    game_recommendations.append({
                        'game_id': int(game_id.split('_')[-1]) if '_' in game_id else 0,
                        'game_title': game_info.get('item_name', 'Unknown Game'),
                        'game_image': None,
                        'score': rec_score,
                        'game_categories': [game_info.get('genre', 'Unknown')],
                        'game_tags': [],
                        'game_price': game_info.get('price', 0.0),
                        'game_rating': game_info.get('rating', 3.0)
                    })
            
            return RecommendationResponse(
                user_id=user_id,
                recommendation_type=recommendation_type,
                recommendations=game_recommendations,
                total_count=len(game_recommendations),
                model_version="1.0.0"
            )
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            raise
    
    def _map_recommendation_type_to_model(self, recommendation_type) -> str:
        """Map recommendation type to internal model type."""
        mapping = {
            'collaborative_filtering': 'svd',
            'content_based': 'item_based',
            'hybrid': 'hybrid',
            'popularity': 'popularity',
            'neural': 'svd',  # Fallback to SVD for neural
            'similarity': 'item_based'  # Fallback to item-based for similarity
        }
        return mapping.get(recommendation_type.value, 'svd')
    
    def _get_user_games(self, user_id: str) -> List[str]:
        """Get games that a user has played/reviewed."""
        if self.reviews_df is None:
            return []
        
        user_reviews = self.reviews_df[self.reviews_df['user_id'] == user_id]
        return user_reviews['item_id'].tolist()
    
    def _get_svd_recommendations(self, user_id: str, n_recommendations: int, 
                                user_games: List[str], include_played: bool) -> List[Tuple[str, float]]:
        """Get SVD-based recommendations."""
        if 'svd' not in self.models:
            return []
        
        try:
            svd_model = self.models['svd']
            user_item_matrix = svd_model['user_item_matrix']
            
            if user_id not in user_item_matrix.index:
                # New user - return popularity recommendations
                return self._get_popularity_recommendations(n_recommendations, [], include_played)
            
            # Get user vector
            user_vector = user_item_matrix.loc[user_id].values.reshape(1, -1)
            
            # Transform user vector
            user_transformed = svd_model['model'].transform(user_vector)
            
            # Reconstruct user vector
            user_reconstructed = svd_model['model'].inverse_transform(user_transformed)
            
            # Get predicted ratings
            predicted_ratings = user_reconstructed[0]
            
            # Create recommendations
            recommendations = []
            for i, game_id in enumerate(user_item_matrix.columns):
                if not include_played and game_id in user_games:
                    continue
                
                score = predicted_ratings[i]
                if score > 0:  # Only positive predictions
                    recommendations.append((game_id, score))
            
            # Sort by score and return top N
            recommendations.sort(key=lambda x: x[1], reverse=True)
            return recommendations[:n_recommendations]
            
        except Exception as e:
            logger.error(f"Error in SVD recommendations: {e}")
            return []
    
    def _get_item_based_recommendations(self, user_id: str, n_recommendations: int,
                                      user_games: List[str], include_played: bool) -> List[Tuple[str, float]]:
        """Get item-based collaborative filtering recommendations."""
        if 'item_based' not in self.models:
            return []
        
        try:
            item_model = self.models['item_based']
            user_item_matrix = item_model['user_item_matrix']
            
            if user_id not in user_item_matrix.index:
                return self._get_popularity_recommendations(n_recommendations, [], include_played)
            
            # Get user's ratings
            user_ratings = user_item_matrix.loc[user_id]
            
            # Calculate weighted average of similar items
            recommendations = []
            for game_id in user_item_matrix.columns:
                if not include_played and game_id in user_games:
                    continue
                
                if user_ratings[game_id] == 0:  # User hasn't rated this game
                    # Find similar games that user has rated
                    game_idx = item_model['item_ids'].index(game_id)
                    similarities = item_model['similarity_matrix'][game_idx]
                    
                    # Calculate predicted rating
                    weighted_sum = 0
                    similarity_sum = 0
                    
                    for rated_game in user_games:
                        if rated_game in item_model['item_ids']:
                            rated_idx = item_model['item_ids'].index(rated_game)
                            similarity = similarities[rated_idx]
                            rating = user_ratings[rated_game]
                            
                            if similarity > 0 and rating > 0:
                                weighted_sum += similarity * rating
                                similarity_sum += similarity
                    
                    if similarity_sum > 0:
                        predicted_rating = weighted_sum / similarity_sum
                        recommendations.append((game_id, predicted_rating))
            
            # Sort by score and return top N
            recommendations.sort(key=lambda x: x[1], reverse=True)
            return recommendations[:n_recommendations]
            
        except Exception as e:
            logger.error(f"Error in item-based recommendations: {e}")
            return []
    
    def _get_popularity_recommendations(self, n_recommendations: int, 
                                      user_games: List[str], include_played: bool) -> List[Tuple[str, float]]:
        """Get popularity-based recommendations."""
        if 'popularity' not in self.models:
            return []
        
        try:
            popular_games = self.models['popularity']['popular_games']
            
            recommendations = []
            for _, game in popular_games.iterrows():
                game_id = game['item_id']
                
                if not include_played and game_id in user_games:
                    continue
                
                recommendations.append((game_id, game['popularity_score']))
                
                if len(recommendations) >= n_recommendations:
                    break
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in popularity recommendations: {e}")
            return []
    
    def _get_hybrid_recommendations(self, user_id: str, n_recommendations: int,
                                  user_games: List[str], include_played: bool) -> List[Tuple[str, float]]:
        """Get hybrid recommendations combining multiple models."""
        try:
            # Get recommendations from different models
            svd_recs = self._get_svd_recommendations(user_id, n_recommendations * 2, user_games, include_played)
            item_recs = self._get_item_based_recommendations(user_id, n_recommendations * 2, user_games, include_played)
            pop_recs = self._get_popularity_recommendations(n_recommendations * 2, user_games, include_played)
            
            # Combine scores with weights
            combined_scores = {}
            
            # SVD: 70% weight (best performer)
            for game_id, score in svd_recs:
                combined_scores[game_id] = combined_scores.get(game_id, 0) + score * 0.7
            
            # Item-based: 20% weight
            for game_id, score in item_recs:
                combined_scores[game_id] = combined_scores.get(game_id, 0) + score * 0.2
            
            # Popularity: 10% weight
            for game_id, score in pop_recs:
                combined_scores[game_id] = combined_scores.get(game_id, 0) + score * 0.1
            
            # Sort by combined score
            recommendations = [(game_id, score) for game_id, score in combined_scores.items()]
            recommendations.sort(key=lambda x: x[1], reverse=True)
            
            return recommendations[:n_recommendations]
            
        except Exception as e:
            logger.error(f"Error in hybrid recommendations: {e}")
            return self._get_popularity_recommendations(n_recommendations, user_games, include_played)
    
    def _get_game_info(self, game_id: str) -> Optional[Dict]:
        """Get game information by ID."""
        if self.games_df is None:
            return None
        
        game_info = self.games_df[self.games_df['item_id'] == game_id]
        if not game_info.empty:
            return game_info.iloc[0].to_dict()
        return None
    
    async def get_user_recommendations(
        self,
        user_id: int,
        recommendation_type: Optional[RecommendationType] = None,
        limit: int = 10
    ) -> List[RecommendationResponse]:
        """Get recommendation history for a user."""
        # Placeholder implementation
        return []
    
    async def record_user_interaction(self, interaction: UserGameInteraction) -> None:
        """Record a user's interaction with a game."""
        # Placeholder implementation
        pass
    
    async def submit_feedback(self, feedback: RecommendationFeedback) -> None:
        """Submit feedback on a recommendation."""
        # Placeholder implementation
        pass
    
    async def get_recommendation_explanation(self, recommendation_id: str) -> dict:
        """Get explanation for why a recommendation was made."""
        # Placeholder implementation
        pass
    
    async def get_performance_metrics(self) -> dict:
        """Get recommendation system performance metrics."""
        try:
            metrics = {
                'total_users': len(self.user_item_matrix.index) if self.user_item_matrix is not None else 0,
                'total_games': len(self.games_df) if self.games_df is not None else 0,
                'total_interactions': len(self.reviews_df) if self.reviews_df is not None else 0,
                'models_available': list(self.models.keys()),
                'data_source': 'Stanford SNAP Steam Dataset (Sample)'
            }
            return metrics
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    async def refresh_user_recommendations(self, user_id: int) -> None:
        """Force refresh of user recommendations."""
        # Placeholder implementation
        pass
    
    async def delete_recommendation(self, recommendation_id: str) -> bool:
        """Delete a specific recommendation."""
        # Placeholder implementation
        pass 
    
    def get_sample_users(self) -> List[Dict]:
        """Get sample users for demo purposes."""
        if self.user_item_matrix is None:
            return []
        
        try:
            # Get users with most interactions
            user_interaction_counts = self.user_item_matrix.astype(bool).sum(axis=1)
            top_users = user_interaction_counts.nlargest(10)
            
            sample_users = []
            for user_id, count in top_users.items():
                # Get user's favorite genre
                user_games = self._get_user_games(user_id)
                if user_games:
                    game_info = self._get_game_info(user_games[0])
                    genre = game_info.get('genre', 'Gamer') if game_info else 'Gamer'
                else:
                    genre = 'Gamer'
                
                sample_users.append({
                    'id': user_id,
                    'reviews': int(count),
                    'description': f"{genre} enthusiast",
                    'genre': genre
                })
            
            return sample_users
            
        except Exception as e:
            logger.error(f"Error getting sample users: {e}")
            return [] 