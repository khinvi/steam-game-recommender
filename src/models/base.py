"""
Base class for recommendation models.
"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict, Any, Optional

class BaseRecommender(ABC):
    """
    Abstract base class for recommendation models.
    
    All recommendation models should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self):
        """Initialize the recommendation model."""
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, interaction_matrix: pd.DataFrame) -> 'BaseRecommender':
        """
        Fit the model to an interaction matrix.
        
        Args:
            interaction_matrix: DataFrame with users as index, items as columns,
                               and values representing interactions (e.g., playtime)
        
        Returns:
            The fitted model instance
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def evaluate(self, test_data: pd.DataFrame, k: int = 10) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_data: DataFrame with user-item interactions for testing
            k: Number of recommendations to generate per user
            
        Returns:
            Dictionary with evaluation metrics
        """
        pass
    
    def _check_fitted(self):
        """
        Check if the model has been fitted.
        
        Raises:
            ValueError: If the model has not been fitted
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
    
    def __str__(self) -> str:
        """Return a string representation of the model."""
        return f"{self.__class__.__name__}()"
    
    def __repr__(self) -> str:
        """Return a string representation of the model."""
        return self.__str__()