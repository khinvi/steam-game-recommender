"""
Unit tests for data processing functions.
"""
import os
import sys
import unittest
import pandas as pd
import numpy as np
import tempfile
import json
import gzip

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import load_json, convert_to_dataframes, create_user_game_matrix
from src.data.preprocessor import (
    extract_interactions,
    filter_games,
    handle_missing_values,
    normalize_playtime,
    train_test_split_interactions,
    create_interaction_matrix
)


class TestDataProcessing(unittest.TestCase):
    """Test cases for data processing functions."""

    def setUp(self):
        """Set up test data."""
        # Create sample data
        self.sample_reviews = [
            {
                'user_id': 'U1',
                'item_id': 'I1',
                'playtime_forever': 100,
                'playtime_2weeks': 20
            },
            {
                'user_id': 'U1',
                'item_id': 'I2',
                'playtime_forever': 200,
                'playtime_2weeks': 30
            },
            {
                'user_id': 'U2',
                'item_id': 'I1',
                'playtime_forever': 150,
                'playtime_2weeks': None
            },
            {
                'user_id': 'U2',
                'item_id': 'I3',
                'playtime_forever': 300,
                'playtime_2weeks': 50
            },
            {
                'user_id': 'U3',
                'item_id': 'I2',
                'playtime_forever': 250,
                'playtime_2weeks': 40
            }
        ]
        
        self.sample_metadata = [
            {
                'item_id': 'I1',
                'item_name': 'Game 1',
                'genre': 'Action, Adventure'
            },
            {
                'item_id': 'I2',
                'item_name': 'Game 2',
                'genre': 'RPG, Strategy'
            },
            {
                'item_id': 'I3',
                'item_name': 'Game 3',
                'genre': 'Simulation, Sport'
            }
        ]
        
        # Convert to DataFrames
        self.reviews_df = pd.DataFrame(self.sample_reviews)
        self.metadata_df = pd.DataFrame(self.sample_metadata)
        
    def test_extract_interactions(self):
        """Test extracting interactions from reviews data."""
        interactions = extract_interactions(self.reviews_df)
        
        # Check that the required columns are present
        self.assertIn('user_id', interactions.columns)
        self.assertIn('item_id', interactions.columns)
        self.assertIn('playtime_forever', interactions.columns)
        
        # Check that all user-item pairs are preserved
        self.assertEqual(len(interactions), len(self.sample_reviews))
        
    def test_filter_games(self):
        """Test filtering games and users."""
        # Create a DataFrame with some test games
        df = pd.DataFrame({
            'user_id': ['U1', 'U1', 'U1', 'U2', 'U2', 'U3', 'U4', 'U4', 'U4', 'U5'],
            'item_id': ['I1', 'I2', 'I3', 'I1', 'I2', 'I1', 'I4', 'I5', 'I6', 'I7'],
            'item_name': ['Game 1', 'Game 2', 'Game 3', 'Game 1', 'Game 2', 
                          'Game 1', 'Game Test', 'Game 5', 'Game 6', 'Game 7']
        })
        
        # Original counts
        orig_users = df['user_id'].nunique()
        orig_games = df['item_id'].nunique()
        
        # Filter games
        filtered = filter_games(df)
        
        # Check that test games are removed (if item_name is in the DataFrame)
        if 'item_name' in filtered.columns:
            test_games = filtered[filtered['item_name'].str.lower().str.contains('test', na=False)]
            self.assertEqual(len(test_games), 0)
        
        # Check that the number of users and games has decreased or remained the same
        self.assertTrue(filtered['user_id'].nunique() <= orig_users)
        self.assertTrue(filtered['item_id'].nunique() <= orig_games)
        
    def test_handle_missing_values(self):
        """Test handling missing values in interactions data."""
        # Create DataFrame with missing values
        df = pd.DataFrame({
            'user_id': ['U1', 'U2', 'U3'],
            'item_id': ['I1', 'I2', 'I3'],
            'playtime_forever': [100, None, 200],
            'playtime_2weeks': [10, 20, None]
        })
        
        # Handle missing values
        filled_df = handle_missing_values(df)
        
        # Check that there are no more missing values
        self.assertEqual(filled_df['playtime_forever'].isna().sum(), 0)
        self.assertEqual(filled_df['playtime_2weeks'].isna().sum(), 0)
        
        # Check that missing values are filled with 0
        self.assertEqual(filled_df.loc[1, 'playtime_forever'], 0)
        self.assertEqual(filled_df.loc[2, 'playtime_2weeks'], 0)
        
    def test_normalize_playtime(self):
        """Test normalizing playtime values."""
        # Create DataFrame with playtime values
        df = pd.DataFrame({
            'user_id': ['U1', 'U1', 'U2', 'U2'],
            'item_id': ['I1', 'I2', 'I1', 'I3'],
            'playtime_forever': [100, 200, 1000, 2000]
        })
        
        # Normalize playtime
        normalized_df = normalize_playtime(df)
        
        # Check that normalized playtime column is added
        self.assertIn('playtime_normalized', normalized_df.columns)
        
        # Check that normalized values are between 0 and 1
        self.assertTrue((normalized_df['playtime_normalized'] >= 0).all())
        self.assertTrue((normalized_df['playtime_normalized'] <=.1).all())
        
        # Check that each user's values are normalized separately
        user1_max = normalized_df[normalized_df['user_id'] == 'U1']['playtime_normalized'].max()
        user2_max = normalized_df[normalized_df['user_id'] == 'U2']['playtime_normalized'].max()
        
        self.assertAlmostEqual(user1_max, 1.0, places=6)
        self.assertAlmostEqual(user2_max, 1.0, places=6)
        
    def test_train_test_split_interactions(self):
        """Test splitting interactions into training and testing sets."""
        # Create a DataFrame with several interactions per user
        df = pd.DataFrame({
            'user_id': ['U1', 'U1', 'U1', 'U2', 'U2', 'U2', 'U3', 'U3', 'U3'],
            'item_id': ['I1', 'I2', 'I3', 'I1', 'I2', 'I3', 'I1', 'I2', 'I3'],
            'playtime_forever': [100, 200, 300, 400, 500, 600, 700, 800, 900]
        })
        
        # Split into train and test
        train_df, test_df = train_test_split_interactions(df, test_size=0.2)
        
        # Check that the total number of rows is preserved
        self.assertEqual(len(train_df) + len(test_df), len(df))
        
        # Check that the test size is approximately correct
        self.assertAlmostEqual(len(test_df) / len(df), 0.2, delta=0.1)
        
        # Check that all users appear in the training set
        self.assertEqual(train_df['user_id'].nunique(), df['user_id'].nunique())
        
    def test_create_interaction_matrix(self):
        """Test creating an interaction matrix from a DataFrame."""
        # Create a DataFrame
        df = pd.DataFrame({
            'user_id': ['U1', 'U1', 'U2', 'U2', 'U3'],
            'item_id': ['I1', 'I2', 'I1', 'I3', 'I2'],
            'playtime_forever': [100, 200, 300, 400, 500]
        })
        
        # Create interaction matrix
        matrix = create_interaction_matrix(df, value_col='playtime_forever')
        
        # Check matrix dimensions
        self.assertEqual(matrix.shape, (3, 3))  # 3 users, 3 items
        
        # Check that values are correctly placed
        self.assertEqual(matrix.loc['U1', 'I1'], 100)
        self.assertEqual(matrix.loc['U1', 'I2'], 200)
        self.assertEqual(matrix.loc['U2', 'I1'], 300)
        self.assertEqual(matrix.loc['U2', 'I3'], 400)
        self.assertEqual(matrix.loc['U3', 'I2'], 500)
        
        # Check that missing interactions are filled with 0
        self.assertEqual(matrix.loc['U1', 'I3'], 0)
        self.assertEqual(matrix.loc['U3', 'I1'], 0)
        self.assertEqual(matrix.loc['U3', 'I3'], 0)


class TestDataLoader(unittest.TestCase):
    """Test cases for data loading functions."""
    
    def setUp(self):
        """Set up test data files."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create sample data
        self.sample_reviews = [
            {
                'user_id': 'U1',
                'item_id': 'I1',
                'playtime_forever': 100
            },
            {
                'user_id': 'U2',
                'item_id': 'I2',
                'playtime_forever': 200
            }
        ]
        
        # Write sample data to gzipped JSON file
        reviews_path = os.path.join(self.temp_dir.name, 'reviews.json.gz')
        with gzip.open(reviews_path, 'wt') as f:
            for review in self.sample_reviews:
                f.write(json.dumps(review) + '\n')
    
    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()
    
    def test_load_json(self):
        """Test loading JSON file."""
        # Create a temporary JSON file
        json_path = os.path.join(self.temp_dir.name, 'test.json')
        with open(json_path, 'w') as f:
            for item in self.sample_reviews:
                f.write(json.dumps(item) + '\n')
        
        # Load the JSON file
        data = load_json(json_path)
        
        # Check that the data is loaded correctly
        self.assertEqual(len(data), len(self.sample_reviews))
        self.assertEqual(data[0]['user_id'], 'U1')
        self.assertEqual(data[1]['user_id'], 'U2')
    
    def test_convert_to_dataframes(self):
        """Test converting JSON data to DataFrames."""
        # Create sample data dictionary
        data = {
            'reviews': self.sample_reviews,
            'metadata': [
                {
                    'item_id': 'I1',
                    'item_name': 'Game 1'
                },
                {
                    'item_id': 'I2',
                    'item_name': 'Game 2'
                }
            ]
        }
        
        # Convert to DataFrames
        dfs = convert_to_dataframes(data)
        
        # Check that the DataFrames are created correctly
        self.assertIn('reviews', dfs)
        self.assertIn('metadata', dfs)
        self.assertEqual(len(dfs['reviews']), len(data['reviews']))
        self.assertEqual(len(dfs['metadata']), len(data['metadata']))
    
    def test_create_user_game_matrix(self):
        """Test creating a user-game matrix."""
        # Create a DataFrame
        df = pd.DataFrame({
            'user_id': ['U1', 'U1', 'U2', 'U2'],
            'item_id': ['I1', 'I2', 'I1', 'I3'],
            'playtime_forever': [100, 200, 300, 400]
        })
        
        # Create user-game matrix
        matrix, user_mapping, game_mapping = create_user_game_matrix(df)
        
        # Check matrix dimensions
        self.assertEqual(matrix.shape, (2, 3))  # 2 users, 3 items
        
        # Check mappings
        self.assertEqual(len(user_mapping), 2)
        self.assertEqual(len(game_mapping), 3)
        self.assertIn('U1', user_mapping)
        self.assertIn('I1', game_mapping)


if __name__ == '__main__':
    unittest.main()