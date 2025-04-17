"""
Cosine similarity-based recommendation model.
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Any

class CosineSimilarityModel:
    """
    Recommendation model based on cosine similarity.
    
    This model can operate in two modes:
    1. User-based: Recommend games similar to those liked by similar users
    2. Item-based: Recommend games similar to games the user has liked
    """
    
    def __init__(self, mode: str = "item"):
        """
        Initialize the cosine similarity model.
        
        Args:
            mode: Either "user" for user-based collaborative filtering or
                 "item" for item-based collaborative filtering
        """
        self.mode = mode
        self.interaction_matrix = None
        self.similarity_matrix = None
        self.user_mapping = None
        self.item_mapping = None
        self.reverse_user_mapping = None
        self.reverse_item_mapping = None
    
    def fit(self, interaction_matrix: pd.DataFrame):
        """
        Fit the model to an interaction matrix.
        
        Args:
            interaction_matrix: DataFrame with users as index, items as columns,
                               and values representing interactions (e.g., playtime)
        """
        self.interaction_matrix = interaction_matrix
        
        # Create mappings between IDs and indices
        self.user_mapping = {user: i for i, user in enumerate(interaction_matrix.index)}
        self.item_mapping = {item: i for i, item in enumerate(interaction_matrix.columns)}
        
        # Create reverse mappings for lookup
        self.reverse_user_mapping = {i: user for user, i in self.user_mapping.items()}
        self.reverse_item_mapping = {i: item for item, i in self.item_mapping.items()}
        
        # Convert to numpy array for calculations
        matrix = interaction_matrix.values
        
        # Calculate similarity matrix based on the mode
        if self.mode == "user":
            # User-based: Calculate similarity between users
            self.similarity_matrix = cosine_similarity(matrix)
        else:
            # Item-based: Calculate similarity between items
            self.similarity_matrix = cosine_similarity(matrix.T)
        
        return self
    
    def recommend(self, user_id: str, k: int = 10, exclude_played: bool = True) -> List[Dict[str, Any]]:
        """
        Generate recommendations for a specific user.
        
        Args:
            user_id: ID of the user to recommend for
            k: Number of recommendations to generate
            exclude_played: Whether to exclude games the user has already played
            
        Returns:
            List of recommended items with details
        """
        if self.interaction_matrix is None or self.similarity_matrix is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        # Check if user exists in the training data
        if user_id not in self.user_mapping:
            raise ValueError(f"User {user_id} not found in the training data.")
        
        user_idx = self.user_mapping[user_id]
        
        if self.mode == "user":
            # User-based collaborative filtering
            recommendations = self._recommend_user_based(user_idx, k, exclude_played)
        else:
            # Item-based collaborative filtering
            recommendations = self._recommend_item_based(user_idx, k, exclude_played)
        
        return recommendations
    
    def _recommend_user_based(self, user_idx: int, k: int, exclude_played: bool) -> List[Dict[str, Any]]:
        """
        Generate recommendations using user-based collaborative filtering.
        
        Args:
            user_idx: Index of the user in the similarity matrix
            k: Number of recommendations to generate
            exclude_played: Whether to exclude games the user has already played
            
        Returns:
            List of recommended items with details
        """
        # Get the user's row in the interaction matrix
        user_interactions = self.interaction_matrix.iloc[user_idx].values
        
        # Get similarity scores between this user and all other users
        similarities = self.similarity_matrix[user_idx]
        
        # Create a mask for the user's played games
        played_mask = user_interactions > 0 if exclude_played else np.zeros_like(user_interactions, dtype=bool)
        
        # Initialize scores for all items
        item_scores = np.zeros(self.interaction_matrix.shape[1])
        
        # Calculate weighted sum of other users' interactions
        for other_user_idx in range(self.interaction_matrix.shape[0]):
            if other_user_idx != user_idx:  # Skip the user themselves
                sim_score = similarities[other_user_idx]
                other_interactions = self.interaction_matrix.iloc[other_user_idx].values
                
                # Add the weighted interactions to the scores
                item_scores += sim_score * other_interactions
        
        # Mask out already played games if required
        if exclude_played:
            item_scores[played_mask] = -1
        
        # Get the indices of the top-k scoring items
        top_items_idx = np.argsort(item_scores)[::-1][:k]
        
        # Convert to actual item IDs and gather scores
        recommendations = []
        for idx in top_items_idx:
            item_id = self.reverse_item_mapping[idx]
            score = item_scores[idx]
            
            # Skip items with negative scores (already played)
            if score < 0:
                continue
                
            recommendations.append({
                'item_id': item_id,
                'score': float(score)
            })
        
        return recommendations
    
    def _recommend_item_based(self, user_idx: int, k: int, exclude_played: bool) -> List[Dict[str, Any]]:
        """
        Generate recommendations using item-based collaborative filtering.
        
        Args:
            user_idx: Index of the user in the interaction matrix
            k: Number of recommendations to generate
            exclude_played: Whether to exclude games the user has already played
            
        Returns:
            List of recommended items with details
        """
        # Get the user's row in the interaction matrix
        user_interactions = self.interaction_matrix.iloc[user_idx].values
        
        # Create a mask for the user's played games
        played_mask = user_interactions > 0
        played_indices = np.where(played_mask)[0]
        
        # Initialize scores for all items
        item_scores = np.zeros(self.interaction_matrix.shape[1])
        
        # For each game the user has played
        for played_idx in played_indices:
            # Get similarity scores between this game and all other games
            similarities = self.similarity_matrix[played_idx]
            
            # Add the weighted similarities to the scores
            # Weighted by the user's interaction with the played game
            item_scores += similarities * user_interactions[played_idx]
        
        # Mask out already played games if required
        if exclude_played:
            item_scores[played_mask] = -1
        
        # Get the indices of the top-k scoring items
        top_items_idx = np.argsort(item_scores)[::-1][:k]
        
        # Convert to actual item IDs and gather scores
        recommendations = []
        for idx in top_items_idx:
            item_id = self.reverse_item_mapping[idx]
            score = item_scores[idx]
            
            # Skip items with negative scores (already played)
            if score < 0:
                continue
                
            recommendations.append({
                'item_id': item_id,
                'score': float(score)
            })
        
        return recommendations[:k]  # Ensure we return at most k items
    
    def evaluate(self, test_data: pd.DataFrame, k: int = 10) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_data: DataFrame with user-item interactions for testing
            k: Number of recommendations to generate per user
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.interaction_matrix is None or self.similarity_matrix is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
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