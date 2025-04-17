"""
Unit tests for recommendation models.
"""
import os
import sys
import unittest
import pandas as pd
import numpy as np
import tempfile

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.cosine_similarity import CosineSimilarityModel
from src.models.svd import SVDModel
from src.models.hybrid import HybridModel, ContentBasedHybridModel


class TestModels(unittest.TestCase):
    """Test cases for recommendation models."""

    def setUp(self):
        """Set up test data."""
        # Create a small interaction matrix for testing
        data = {
            'user_id': ['U1', 'U1', 'U1', 'U2', 'U2', 'U3', 'U3', 'U3', 'U4', 'U4'],
            'item_id': ['I1', 'I2', 'I3', 'I1', 'I2', 'I2', 'I3', 'I4', 'I1', 'I4'],
            'playtime_forever': [10, 20, 30, 15, 25, 40, 35, 20, 5, 10]
        }
        self.interactions_df = pd.DataFrame(data)
        
        # Create interaction matrix
        self.interaction_matrix = pd.pivot_table(
            self.interactions_df,
            values='playtime_forever',
            index='user_id',
            columns='item_id',
            fill_value=0
        )
        
        # Create train and test data
        self.train_df = self.interactions_df.sample(frac=0.8, random_state=42)
        self.test_df = self.interactions_df.drop(self.train_df.index)
        
    def test_cosine_similarity_user_based(self):
        """Test user-based cosine similarity model."""
        # Create and fit the model
        model = CosineSimilarityModel(mode="user")
        model.fit(self.interaction_matrix)
        
        # Test recommendation
        user_id = self.interaction_matrix.index[0]
        recommendations = model.recommend(user_id, k=2)
        
        # Check recommendation format
        self.assertIsInstance(recommendations, list)
        self.assertTrue(len(recommendations) <= 2)
        
        if recommendations:
            self.assertIn('item_id', recommendations[0])
            self.assertIn('score', recommendations[0])
            
        # Test evaluation
        metrics = model.evaluate(self.test_df, k=2)
        self.assertIn('precision_at_k', metrics)
        self.assertIn('hit_rate', metrics)
        
    def test_cosine_similarity_item_based(self):
        """Test item-based cosine similarity model."""
        # Create and fit the model
        model = CosineSimilarityModel(mode="item")
        model.fit(self.interaction_matrix)
        
        # Test recommendation
        user_id = self.interaction_matrix.index[0]
        recommendations = model.recommend(user_id, k=2)
        
        # Check recommendation format
        self.assertIsInstance(recommendations, list)
        self.assertTrue(len(recommendations) <= 2)
        
        if recommendations:
            self.assertIn('item_id', recommendations[0])
            self.assertIn('score', recommendations[0])
            
        # Test evaluation
        metrics = model.evaluate(self.test_df, k=2)
        self.assertIn('precision_at_k', metrics)
        self.assertIn('hit_rate', metrics)
    
    def test_svd_model(self):
        """Test SVD model."""
        # Create and fit the model
        model = SVDModel(n_factors=2)  # Small number of factors for testing
        model.fit(self.interaction_matrix)
        
        # Test recommendation
        user_id = self.interaction_matrix.index[0]
        recommendations = model.recommend(user_id, k=2)
        
        # Check recommendation format
        self.assertIsInstance(recommendations, list)
        self.assertTrue(len(recommendations) <= 2)
        
        if recommendations:
            self.assertIn('item_id', recommendations[0])
            self.assertIn('score', recommendations[0])
            
        # Test evaluation
        metrics = model.evaluate(self.test_df, k=2)
        self.assertIn('precision_at_k', metrics)
        self.assertIn('hit_rate', metrics)
    
    def test_hybrid_model(self):
        """Test hybrid model."""
        # Create component models
        user_cf = CosineSimilarityModel(mode="user")
        item_cf = CosineSimilarityModel(mode="item")
        
        # Create and fit the hybrid model
        model = HybridModel([user_cf, item_cf], weights=[0.7, 0.3])
        model.fit(self.interaction_matrix)
        
        # Test recommendation
        user_id = self.interaction_matrix.index[0]
        recommendations = model.recommend(user_id, k=2)
        
        # Check recommendation format
        self.assertIsInstance(recommendations, list)
        self.assertTrue(len(recommendations) <= 2)
        
        if recommendations:
            self.assertIn('item_id', recommendations[0])
            self.assertIn('score', recommendations[0])
            
        # Test evaluation
        metrics = model.evaluate(self.test_df, k=2)
        self.assertIn('precision_at_k', metrics)
        self.assertIn('hit_rate', metrics)
    
    def test_content_based_hybrid_model(self):
        """Test content-based hybrid model."""
        # Create metadata
        metadata = {
            'item_id': ['I1', 'I2', 'I3', 'I4'],
            'genre': ['Action, Adventure', 'RPG, Adventure', 'Action, RPG', 'Simulation']
        }
        metadata_df = pd.DataFrame(metadata)
        
        # Create and fit the model
        model = ContentBasedHybridModel(n_factors=2, content_weight=0.3, metadata_df=metadata_df)
        model.fit(self.interaction_matrix)
        
        # Test recommendation
        user_id = self.interaction_matrix.index[0]
        recommendations = model.recommend(user_id, k=2)
        
        # Check recommendation format
        self.assertIsInstance(recommendations, list)
        self.assertTrue(len(recommendations) <= 2)
        
        if recommendations:
            self.assertIn('item_id', recommendations[0])
            self.assertIn('score', recommendations[0])
            
        # Test evaluation
        metrics = model.evaluate(self.test_df, k=2)
        self.assertIn('precision_at_k', metrics)
        self.assertIn('hit_rate', metrics)
    
    def test_model_saving_loading(self):
        """Test saving and loading models."""
        # Create a temporary directory for saving models
        with tempfile.TemporaryDirectory() as temp_dir:
            # Train and save an SVD model
            model = SVDModel(n_factors=2)
            model.fit(self.interaction_matrix)
            
            # Save the model
            import pickle
            model_path = os.path.join(temp_dir, "svd_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Load the model
            with open(model_path, 'rb') as f:
                loaded_model = pickle.load(f)
            
            # Check that the loaded model works
            user_id = self.interaction_matrix.index[0]
            recommendations = loaded_model.recommend(user_id, k=2)
            
            # Verify recommendations
            self.assertIsInstance(recommendations, list)


if __name__ == '__main__':
    unittest.main()