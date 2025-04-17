"""
SVD-based recommendation model.
"""
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from typing import List, Dict, Tuple, Any, Optional

class SVDModel:
    """
    Recommendation model based on Singular Value Decomposition (SVD).
    
    This model uses matrix factorization to decompose the user-item interaction
    matrix into latent factors, which are then used to predict user preferences
    for items they haven't interacted with yet.
    """
    
    def __init__(self, n_factors: int = 50, regularization: float = 0.1, random_state: int = 42):
        """
        Initialize the SVD model.
        
        Args:
            n_factors: Number of latent factors to use
            regularization: Regularization parameter for the model
            random_state: Random seed for reproducibility
        """
        self.n_factors = n_factors
        self.regularization = regularization
        self.random_state = random_state
        
        self.interaction_matrix = None
        self.user_mapping = None
        self.item_mapping = None
        self.reverse_user_mapping = None
        self.reverse_item_mapping = None
        
        self.user_factors = None
        self.sigma = None
        self.item_factors = None
        
        self.user_biases = None
        self.item_biases = None
        self.global_mean = None
    
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
        
        # Calculate global mean
        self.global_mean = np.mean(matrix[matrix > 0])
        
        # Calculate user and item biases
        self._calculate_biases(matrix)
        
        # Center the matrix (subtract biases)
        centered_matrix = self._center_matrix(matrix)
        
        # Perform SVD
        u, sigma, vt = svds(centered_matrix, k=min(self.n_factors, min(matrix.shape) - 1))
        
        # Sort the factors (SVD returns them in ascending order)
        sorted_indices = np.argsort(-sigma)
        sigma = sigma[sorted_indices]
        u = u[:, sorted_indices]
        vt = vt[sorted_indices, :]
        
        # Store the factorization results
        self.user_factors = u
        self.sigma = sigma
        self.item_factors = vt.T
        
        return self
    
    def _calculate_biases(self, matrix: np.ndarray):
        """
        Calculate user and item biases.
        
        Args:
            matrix: User-item interaction matrix
        """
        # Calculate user biases
        user_biases = np.zeros(matrix.shape[0])
        for i in range(matrix.shape[0]):
            user_interactions = matrix[i, matrix[i, :] > 0]
            if len(user_interactions) > 0:
                user_biases[i] = np.mean(user_interactions) - self.global_mean
        
        # Calculate item biases
        item_biases = np.zeros(matrix.shape[1])
        for j in range(matrix.shape[1]):
            item_interactions = matrix[matrix[:, j] > 0, j]
            if len(item_interactions) > 0:
                item_biases[j] = np.mean(item_interactions) - self.global_mean
        
        self.user_biases = user_biases
        self.item_biases = item_biases
    
    def _center_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """
        Center the matrix by subtracting biases.
        
        Args:
            matrix: Original interaction matrix
            
        Returns:
            Centered matrix
        """
        centered = matrix.copy()
        
        # Only modify non-zero entries
        mask = centered > 0
        
        for i in range(centered.shape[0]):
            for j in range(centered.shape[1]):
                if mask[i, j]:
                    centered[i, j] = centered[i, j] - self.global_mean - self.user_biases[i] - self.item_biases[j]
        
        return centered
    
    def predict(self, user_idx: int, item_idx: int) -> float:
        """
        Predict the rating/preference of a user for an item.
        
        Args:
            user_idx: Index of the user
            item_idx: Index of the item
            
        Returns:
            Predicted rating/preference
        """
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        # Predict using the factorized matrices
        pred = self.global_mean
        pred += self.user_biases[user_idx] if user_idx < len(self.user_biases) else 0
        pred += self.item_biases[item_idx] if item_idx < len(self.item_biases) else 0
        
        # Add the factorization term
        pred += np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        
        return max(0, pred)  # Ensure non-negative prediction
    
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
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        # Check if user exists in the training data
        if user_id not in self.user_mapping:
            raise ValueError(f"User {user_id} not found in the training data.")
        
        user_idx = self.user_mapping[user_id]
        
        # Get the user's row in the interaction matrix
        user_interactions = self.interaction_matrix.iloc[user_idx].values
        
        # Create a mask for the user's played games
        played_mask = user_interactions > 0 if exclude_played else np.zeros_like(user_interactions, dtype=bool)
        
        # Calculate predicted ratings for all items
        item_scores = np.zeros(self.interaction_matrix.shape[1])
        
        for item_idx in range(self.interaction_matrix.shape[1]):
            item_scores[item_idx] = self.predict(user_idx, item_idx)
        
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
        if self.user_factors is None or self.item_factors is None:
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