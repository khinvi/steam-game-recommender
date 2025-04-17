"""
Hybrid recommendation model combining collaborative filtering with content-based features.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional

from src.models.base import BaseRecommender
from src.models.svd import SVDModel
from src.models.cosine_similarity import CosineSimilarityModel

class HybridModel(BaseRecommender):
    """
    Hybrid recommendation model that combines multiple recommendation approaches.
    
    This model can blend recommendations from different models, such as SVD and 
    content-based approaches, to leverage the strengths of each approach.
    """
    
    def __init__(self, models: List[BaseRecommender], weights: Optional[List[float]] = None):
        """
        Initialize the hybrid model.
        
        Args:
            models: List of recommendation models to combine
            weights: List of weights for each model (default: equal weights)
        """
        super().__init__()
        self.models = models
        
        # Set weights (default: equal weights)
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            if len(weights) != len(models):
                raise ValueError("Number of weights must match number of models")
            
            # Normalize weights to sum to 1
            total = sum(weights)
            self.weights = [w / total for w in weights]
        
        self.interaction_matrix = None
        self.user_mapping = None
        self.item_mapping = None
        self.reverse_user_mapping = None
        self.reverse_item_mapping = None
    
    def fit(self, interaction_matrix: pd.DataFrame) -> 'HybridModel':
        """
        Fit all models in the ensemble to the interaction matrix.
        
        Args:
            interaction_matrix: DataFrame with users as index, items as columns,
                               and values representing interactions
        
        Returns:
            The fitted model instance
        """
        self.interaction_matrix = interaction_matrix
        
        # Create mappings between IDs and indices
        self.user_mapping = {user: i for i, user in enumerate(interaction_matrix.index)}
        self.item_mapping = {item: i for i, item in enumerate(interaction_matrix.columns)}
        
        # Create reverse mappings for lookup
        self.reverse_user_mapping = {i: user for user, i in self.user_mapping.items()}
        self.reverse_item_mapping = {i: item for item, i in self.item_mapping.items()}
        
        # Fit each model
        for model in self.models:
            model.fit(interaction_matrix)
        
        self.is_fitted = True
        return self
    
    def recommend(self, user_id: str, k: int = 10, exclude_played: bool = True) -> List[Dict[str, Any]]:
        """
        Generate recommendations for a specific user by combining recommendations from all models.
        
        Args:
            user_id: ID of the user to recommend for
            k: Number of recommendations to generate
            exclude_played: Whether to exclude games the user has already played
            
        Returns:
            List of recommended items with details
        """
        self._check_fitted()
        
        # Check if user exists in the training data
        if user_id not in self.user_mapping:
            raise ValueError(f"User {user_id} not found in the training data.")
        
        # Get recommendations from each model
        all_recommendations = []
        for i, model in enumerate(self.models):
            try:
                model_recs = model.recommend(user_id, k=k*2, exclude_played=exclude_played)
                
                # Add model weight to scores
                for rec in model_recs:
                    rec['original_score'] = rec['score']
                    rec['score'] = rec['score'] * self.weights[i]
                    rec['model'] = model.__class__.__name__
                
                all_recommendations.extend(model_recs)
            except Exception as e:
                print(f"Error getting recommendations from {model.__class__.__name__}: {e}")
        
        # Combine and deduplicate recommendations
        combined_recommendations = {}
        for rec in all_recommendations:
            item_id = rec['item_id']
            
            if item_id in combined_recommendations:
                # If this item was already recommended by another model, add the scores
                combined_recommendations[item_id]['score'] += rec['score']
                combined_recommendations[item_id]['models'].append(rec['model'])
            else:
                # First time seeing this item
                combined_recommendations[item_id] = {
                    'item_id': item_id,
                    'score': rec['score'],
                    'models': [rec['model']]
                }
        
        # Convert to list and sort by score
        recommendations = list(combined_recommendations.values())
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return recommendations[:k]
    
    def evaluate(self, test_data: pd.DataFrame, k: int = 10) -> Dict[str, float]:
        """
        Evaluate the hybrid model on test data.
        
        Args:
            test_data: DataFrame with user-item interactions for testing
            k: Number of recommendations to generate per user
            
        Returns:
            Dictionary with evaluation metrics
        """
        self._check_fitted()
        
        metrics = {
            'precision_at_k': 0.0,
            'hit_rate': 0.0,
            'num_users': 0
        }
        
        # Get unique users in test data
        test_users = test_data['user_id'].unique()
        valid_users = [user for user in test_users if user in self.user_mapping]
        metrics['num_users'] = len(valid_users)
        
        if metrics['num_users'] == 0:
            return metrics
        
        # Calculate metrics for each user
        hit_count = 0
        precision_sum = 0.0
        
        for user_id in valid_users:
            # Get the user's test items
            user_test_items = set(test_data[test_data['user_id'] == user_id]['item_id'])
            
            # Generate recommendations
            try:
                recommendations = self.recommend(user_id, k=k)
                recommended_items = {item['item_id'] for item in recommendations}
                
                # Calculate precision@k
                hits = len(user_test_items.intersection(recommended_items))
                precision = hits / min(k, len(recommended_items)) if recommended_items else 0
                precision_sum += precision
                
                # Track hit rate (at least one hit)
                if hits > 0:
                    hit_count += 1
            except Exception as e:
                print(f"Error generating recommendations for user {user_id}: {e}")
                metrics['num_users'] -= 1
        
        # Calculate final metrics
        if metrics['num_users'] > 0:
            metrics['precision_at_k'] = precision_sum / metrics['num_users']
            metrics['hit_rate'] = hit_count / metrics['num_users']
        
        return metrics
    
    def get_model_weights(self) -> Dict[str, float]:
        """
        Get the weights assigned to each model.
        
        Returns:
            Dictionary mapping model names to their weights
        """
        return {model.__class__.__name__: weight for model, weight in zip(self.models, self.weights)}
    
    def set_model_weights(self, weights: List[float]) -> None:
        """
        Set new weights for the models.
        
        Args:
            weights: List of weights for each model
        """
        if len(weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")
        
        # Normalize weights to sum to 1
        total = sum(weights)
        self.weights = [w / total for w in weights]
    
    def __str__(self) -> str:
        """Return a string representation of the model."""
        models_str = ", ".join([model.__class__.__name__ for model in self.models])
        weights_str = ", ".join([f"{w:.2f}" for w in self.weights])
        return f"HybridModel(models=[{models_str}], weights=[{weights_str}])"

class ContentBasedHybridModel(HybridModel):
    """
    A hybrid model that combines collaborative filtering with content-based features.
    
    This model uses SVD for collaborative filtering and content-based similarity
    for incorporating game metadata.
    """
    
    def __init__(self, n_factors: int = 50, 
                content_weight: float = 0.3,
                metadata_df: Optional[pd.DataFrame] = None,
                genre_col: str = 'genre'):
        """
        Initialize the content-based hybrid model.
        
        Args:
            n_factors: Number of latent factors for SVD model
            content_weight: Weight to assign to content-based recommendations (between 0 and 1)
            metadata_df: DataFrame with game metadata
            genre_col: Name of the column containing genre information
        """
        # Create SVD model for collaborative filtering
        svd_model = SVDModel(n_factors=n_factors)
        
        # Store metadata for content-based filtering
        self.metadata_df = metadata_df
        self.genre_col = genre_col
        self.genre_similarity_matrix = None
        self.item_id_to_idx = {}
        
        # Initialize parent class with SVD model only for now
        # We'll add the content-based model in the fit method once we have the interaction matrix
        super().__init__([svd_model], [1.0])
        
        # Store desired content weight for later
        self.desired_content_weight = content_weight
    
    def fit(self, interaction_matrix: pd.DataFrame) -> 'ContentBasedHybridModel':
        """
        Fit the hybrid model to the interaction matrix and metadata.
        
        Args:
            interaction_matrix: DataFrame with users as index, items as columns,
                               and values representing interactions
        
        Returns:
            The fitted model instance
        """
        # First, fit the SVD model
        super().fit(interaction_matrix)
        
        # Process metadata if provided
        if self.metadata_df is not None and self.genre_col in self.metadata_df.columns:
            # Create genre-based similarity matrix
            self._create_genre_similarity_matrix()
            
            # Create content-based model
            content_model = self._create_content_based_model(interaction_matrix)
            
            # Update models and weights
            self.models = [self.models[0], content_model]
            self.weights = [1.0 - self.desired_content_weight, self.desired_content_weight]
        
        return self
    
    def _create_genre_similarity_matrix(self) -> None:
        """
        Create a genre-based similarity matrix from the metadata.
        """
        # Extract item IDs and genres
        items = self.metadata_df['item_id'].tolist()
        self.item_id_to_idx = {item_id: idx for idx, item_id in enumerate(items)}
        
        # Create genre vectors (one-hot encoding)
        all_genres = set()
        item_genres = {}
        
        for _, row in self.metadata_df.iterrows():
            item_id = row['item_id']
            genre_str = row.get(self.genre_col, '')
            
            if not isinstance(genre_str, str):
                continue
                
            genres = genre_str.split(', ')
            item_genres[item_id] = genres
            all_genres.update(genres)
        
        all_genres = list(all_genres)
        genre_to_idx = {genre: idx for idx, genre in enumerate(all_genres)}
        
        # Create genre vectors
        genre_vectors = np.zeros((len(items), len(all_genres)))
        
        for i, item_id in enumerate(items):
            if item_id in item_genres:
                for genre in item_genres[item_id]:
                    genre_vectors[i, genre_to_idx[genre]] = 1
        
        # Calculate Jaccard similarity between genre vectors
        from sklearn.metrics import pairwise_distances
        self.genre_similarity_matrix = 1 - pairwise_distances(genre_vectors, metric='jaccard')
        
        # Set diagonal to 0 to avoid recommending the same item
        np.fill_diagonal(self.genre_similarity_matrix, 0)
    
    def _create_content_based_model(self, interaction_matrix: pd.DataFrame) -> BaseRecommender:
        """
        Create a content-based model using genre similarity.
        
        Args:
            interaction_matrix: User-item interaction matrix
            
        Returns:
            A model that implements the BaseRecommender interface
        """
        # Create a custom content-based model
        class ContentBasedModel(BaseRecommender):
            def __init__(self, parent):
                super().__init__()
                self.parent = parent
                self.is_fitted = True
            
            def fit(self, _):
                # Nothing to fit for the content-based model
                return self
            
            def recommend(self, user_id, k=10, exclude_played=True):
                # Check if user exists
                if user_id not in self.parent.user_mapping:
                    raise ValueError(f"User {user_id} not found in the training data.")
                
                # Get the user's played games
                user_idx = self.parent.user_mapping[user_id]
                user_interactions = interaction_matrix.iloc[user_idx]
                
                # Get indices of played games with positive interactions
                played_games = []
                for item_id, playtime in user_interactions.items():
                    if playtime > 0 and item_id in self.parent.item_id_to_idx:
                        played_games.append(self.parent.item_id_to_idx[item_id])
                
                if not played_games:
                    # If no played games or none match metadata, return empty list
                    return []
                
                # Calculate similarity scores for all items based on played games
                item_scores = np.zeros(len(self.parent.item_id_to_idx))
                
                for played_idx in played_games:
                    # Add similarity scores from this played game to all other games
                    item_scores += self.parent.genre_similarity_matrix[played_idx]
                
                # Create a mask for played games if needed
                if exclude_played:
                    for played_idx in played_games:
                        item_scores[played_idx] = -1
                
                # Get the top k items
                top_indices = np.argsort(item_scores)[::-1][:k*2]  # Get more than k to account for filtering
                
                # Convert to recommendations
                recommendations = []
                items_list = list(self.parent.item_id_to_idx.keys())
                
                for idx in top_indices:
                    if item_scores[idx] <= 0:
                        continue
                        
                    item_id = items_list[idx]
                    
                    # Only include items that are in the interaction matrix
                    if item_id in interaction_matrix.columns:
                        recommendations.append({
                            'item_id': item_id,
                            'score': float(item_scores[idx])
                        })
                    
                    if len(recommendations) >= k:
                        break
                
                return recommendations
            
            def evaluate(self, test_data, k=10):
                # Reuse parent's evaluation method
                return self.parent.evaluate(test_data, k)
        
        return ContentBasedModel(self)