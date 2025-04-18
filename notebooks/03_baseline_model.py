#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Steam Game Recommender System

This module implements baseline and advanced collaborative filtering models
for recommending Steam games to users based on their play history.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, save_npz, load_npz
from scipy.sparse.linalg import svds

# Set visualization style
plt.style.use('ggplot')
sns.set(style="whitegrid")

# Configure pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 1000)

# Create directories if they don't exist
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)


#############################
# Data Loading and Processing
#############################

def load_processed_data():
    """
    Load the preprocessed data files with fallback options.
    
    Returns:
        train_df: Training interactions DataFrame
        test_df: Test interactions DataFrame
        items_df: Items metadata DataFrame
        train_matrix: Training interaction matrix
        sparse_train_matrix: Sparse training matrix
        id_mappings: Dictionary of ID mappings
    """
    processed_dir = 'data/processed'
    
    try:
        # Try to load DataFrames
        train_df = pd.read_csv(f'{processed_dir}/train_interactions.csv')
        test_df = pd.read_csv(f'{processed_dir}/test_interactions.csv')
        items_df = pd.read_csv(f'{processed_dir}/items.csv')
        
        # Try to load matrices
        train_matrix = pd.read_csv(f'{processed_dir}/train_matrix.csv', index_col=0)
        
        # Try to load ID mappings
        with open(f'{processed_dir}/id_mappings.pkl', 'rb') as f:
            id_mappings = pickle.load(f)
        
        # Try to load sparse matrices
        with open(f'{processed_dir}/sparse_matrices.pkl', 'rb') as f:
            sparse_matrices = pickle.load(f)
            sparse_train_matrix = sparse_matrices['sparse_train_matrix']
        
        print("Successfully loaded preprocessed data.")
        return train_df, test_df, items_df, train_matrix, sparse_train_matrix, id_mappings
    
    except (FileNotFoundError, KeyError) as e:
        print(f"Error loading preprocessed data: {e}")
        print("Creating synthetic data for demonstration purposes...")
        
        # Create synthetic data
        import random
        from scipy.sparse import csr_matrix
        
        # Parameters
        n_users = 500
        n_items = 200
        n_interactions = 10000
        sparsity = 0.9  # 90% of entries will be zero
        
        # Create user and item IDs
        user_ids = [f"user_{i}" for i in range(n_users)]
        item_ids = [f"item_{i}" for i in range(n_items)]
        
        # Create mappings
        user_to_idx = {user: i for i, user in enumerate(user_ids)}
        item_to_idx = {item: i for i, item in enumerate(item_ids)}
        idx_to_user = {i: user for user, i in user_to_idx.items()}
        idx_to_item = {i: item for item, i in item_to_idx.items()}
        id_mappings = {'user_to_idx': user_to_idx, 'item_to_idx': item_to_idx, 
                       'idx_to_user': idx_to_user, 'idx_to_item': idx_to_item}
        
        # Create training interactions
        train_interactions = []
        for _ in range(n_interactions):
            user_id = random.choice(user_ids)
            item_id = random.choice(item_ids)
            rating = random.uniform(0, 5)
            train_interactions.append({
                'user_id': user_id,
                'item_id': item_id,
                'normalized_playtime': rating,
                'playtime_forever': int(rating * 100)  # Synthetic playtime
            })
        
        # Create test interactions (hold out some items per user)
        test_interactions = []
        for user_id in user_ids:
            test_items = random.sample(item_ids, 2)  # Hold out 2 items per user
            for item_id in test_items:
                rating = random.uniform(0, 5)
                test_interactions.append({
                    'user_id': user_id,
                    'item_id': item_id,
                    'normalized_playtime': rating,
                    'playtime_forever': int(rating * 100)  # Synthetic playtime
                })
        
        # Create item metadata
        items_data = []
        genres = ['Action', 'Adventure', 'RPG', 'Strategy', 'Simulation', 'Sports', 'Racing', 'Puzzle']
        for item_id in item_ids:
            n_genres = random.randint(1, 3)
            item_genres = random.sample(genres, n_genres)
            items_data.append({
                'item_id': item_id,
                'item_name': f"Game {item_id.split('_')[1]}",
                'genres': str(item_genres)  # Convert list to string for CSV
            })
        
        # Convert to DataFrames
        train_df = pd.DataFrame(train_interactions)
        test_df = pd.DataFrame(test_interactions)
        items_df = pd.DataFrame(items_data)
        
        # Create interaction matrix
        train_matrix = train_df.pivot_table(
            index='user_id',
            columns='item_id',
            values='normalized_playtime',
            fill_value=0
        )
        
        # Convert to sparse matrix
        sparse_train_matrix = csr_matrix(train_matrix.values)
        
        # Save synthetic data
        os.makedirs(processed_dir, exist_ok=True)
        train_df.to_csv(f'{processed_dir}/train_interactions.csv', index=False)
        test_df.to_csv(f'{processed_dir}/test_interactions.csv', index=False)
        items_df.to_csv(f'{processed_dir}/items.csv', index=False)
        train_matrix.to_csv(f'{processed_dir}/train_matrix.csv')
        
        with open(f'{processed_dir}/id_mappings.pkl', 'wb') as f:
            pickle.dump(id_mappings, f)
        
        sparse_matrices = {'sparse_train_matrix': sparse_train_matrix}
        with open(f'{processed_dir}/sparse_matrices.pkl', 'wb') as f:
            pickle.dump(sparse_matrices, f)
        
        print("Created and saved synthetic data.")
        return train_df, test_df, items_df, train_matrix, sparse_train_matrix, id_mappings


def display_data_info(train_df, test_df, items_df, train_matrix, sparse_train_matrix):
    """
    Display basic information about the data.
    
    Args:
        train_df: Training interactions DataFrame
        test_df: Test interactions DataFrame
        items_df: Items metadata DataFrame
        train_matrix: Training interaction matrix
        sparse_train_matrix: Sparse training matrix
    """
    print(f"Training data: {len(train_df)} interactions, {train_df['user_id'].nunique()} users, {train_df['item_id'].nunique()} items")
    print(f"Test data: {len(test_df)} interactions, {test_df['user_id'].nunique()} users, {test_df['item_id'].nunique()} items")
    print(f"Item metadata: {len(items_df)} items with metadata")
    print(f"Training matrix shape: {train_matrix.shape}")
    print(f"Training matrix sparsity: {100.0 * (1 - sparse_train_matrix.count_nonzero() / (sparse_train_matrix.shape[0] * sparse_train_matrix.shape[1])):.2f}%")


#############################
# Evaluation Metrics
#############################

def precision_at_k(recommended_items, actual_items, k=10):
    """
    Calculate precision@k for a single user.
    
    Args:
        recommended_items: List of recommended item IDs
        actual_items: Set of actual item IDs in the test set
        k: Number of recommendations to consider
        
    Returns:
        Precision@k value between 0 and 1
    """
    # Consider only the top-k recommendations
    recommended_k = recommended_items[:k] if len(recommended_items) >= k else recommended_items
    
    # Count the number of relevant items in the recommendations
    num_relevant = len(set(recommended_k) & actual_items)
    
    # Calculate precision@k
    return num_relevant / min(k, len(recommended_k)) if len(recommended_k) > 0 else 0


def hit_rate(recommended_items, actual_items, k=10):
    """
    Calculate hit rate for a single user (whether at least one recommended item is relevant).
    
    Args:
        recommended_items: List of recommended item IDs
        actual_items: Set of actual item IDs in the test set
        k: Number of recommendations to consider
        
    Returns:
        1 if at least one recommendation is relevant, 0 otherwise
    """
    # Consider only the top-k recommendations
    recommended_k = recommended_items[:k] if len(recommended_items) >= k else recommended_items
    
    # Check if there's at least one relevant item in the recommendations
    return 1 if len(set(recommended_k) & actual_items) > 0 else 0


def evaluate_recommendations(recommendations_dict, test_df, k=10):
    """
    Evaluate recommendations for multiple users using precision@k and hit rate.
    
    Args:
        recommendations_dict: Dictionary mapping user IDs to lists of recommended item IDs
        test_df: Test DataFrame containing actual user-item interactions
        k: Number of recommendations to consider
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Group test data by user
    test_user_items = test_df.groupby('user_id')['item_id'].apply(set).to_dict()
    
    # Calculate metrics for each user
    precision_scores = []
    hit_rates = []
    
    for user_id, recommended_items in recommendations_dict.items():
        # Skip users not in the test set
        if user_id not in test_user_items:
            continue
        
        actual_items = test_user_items[user_id]
        
        # Calculate metrics
        precision_scores.append(precision_at_k(recommended_items, actual_items, k))
        hit_rates.append(hit_rate(recommended_items, actual_items, k))
    
    # Aggregate metrics
    avg_precision = np.mean(precision_scores) if precision_scores else 0
    avg_hit_rate = np.mean(hit_rates) if hit_rates else 0
    
    return {
        'precision_at_k': avg_precision,
        'hit_rate': avg_hit_rate,
        'num_users_evaluated': len(precision_scores)
    }


#############################
# Baseline Models
#############################

class UserBasedCF:
    """
    User-Based Collaborative Filtering using cosine similarity.
    """
    def __init__(self, train_matrix, user_to_idx, item_to_idx, idx_to_user, idx_to_item):
        """
        Initialize the model.
        
        Args:
            train_matrix: Training interaction matrix (users x items)
            user_to_idx: Mapping from user IDs to indices
            item_to_idx: Mapping from item IDs to indices
            idx_to_user: Mapping from indices to user IDs
            idx_to_item: Mapping from indices to item IDs
        """
        self.train_matrix = train_matrix
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx
        self.idx_to_user = idx_to_user
        self.idx_to_item = idx_to_item
        self.similarity_matrix = None
        
    def fit(self, k_neighbors=20):
        """
        Compute user-user similarity matrix.
        
        Args:
            k_neighbors: Number of neighbors to consider for each user
        """
        print("Computing user-user similarity matrix...")
        # Compute cosine similarity between users
        self.similarity_matrix = cosine_similarity(self.train_matrix)
        
        # Set self-similarity to zero to avoid recommending already interacted items
        np.fill_diagonal(self.similarity_matrix, 0)
        
        # Keep only top-k neighbors for each user
        for i in range(self.similarity_matrix.shape[0]):
            # Get indices of sorted similarities (descending order)
            sorted_indices = np.argsort(self.similarity_matrix[i, :])[::-1]
            
            # Zero out similarities except for top-k neighbors
            mask = np.ones(self.similarity_matrix.shape[1], dtype=bool)
            mask[sorted_indices[:k_neighbors]] = False
            self.similarity_matrix[i, mask] = 0
        
        print("Done.")
        return self
    
    def predict(self, user_id, item_id):
        """
        Predict the rating for a specific user-item pair.
        
        Args:
            user_id: ID of the user
            item_id: ID of the item
            
        Returns:
            Predicted rating
        """
        # Get indices
        user_idx = self.user_to_idx.get(user_id)
        item_idx = self.item_to_idx.get(item_id)
        
        if user_idx is None or item_idx is None:
            return 0  # Return 0 for unknown users or items
        
        # Get user similarities and ratings for the item
        similarities = self.similarity_matrix[user_idx, :]
        ratings = self.train_matrix.iloc[:, item_idx].values
        
        # Compute weighted average rating
        weighted_sum = np.sum(similarities * ratings)
        similarity_sum = np.sum(np.abs(similarities))
        
        if similarity_sum == 0:
            return 0  # No similar users
        
        return weighted_sum / similarity_sum
    
    def recommend(self, user_id, n=10, exclude_known=True):
        """
        Generate top-n recommendations for a user.
        
        Args:
            user_id: ID of the user
            n: Number of recommendations to generate
            exclude_known: Whether to exclude items the user has already interacted with
            
        Returns:
            List of recommended item IDs
        """
        # Get user index
        user_idx = self.user_to_idx.get(user_id)
        
        if user_idx is None:
            return []  # Return empty list for unknown users
        
        # Get items the user has already interacted with
        if exclude_known:
            known_items = set(self.train_matrix.columns[self.train_matrix.iloc[user_idx, :] > 0])
        else:
            known_items = set()
        
        # Compute predictions for all items
        predictions = []
        for item_id in self.item_to_idx.keys():
            if item_id not in known_items:  # Skip known items if exclude_known is True
                prediction = self.predict(user_id, item_id)
                predictions.append((item_id, prediction))
        
        # Sort by predicted rating (descending) and get top-n
        predictions.sort(key=lambda x: x[1], reverse=True)
        recommended_items = [item_id for item_id, _ in predictions[:n]]
        
        return recommended_items
    
    def recommend_all_users(self, n=10, exclude_known=True):
        """
        Generate recommendations for all users.
        
        Args:
            n: Number of recommendations per user
            exclude_known: Whether to exclude items the user has already interacted with
            
        Returns:
            Dictionary mapping user IDs to lists of recommended item IDs
        """
        recommendations = {}
        for user_id in tqdm(self.user_to_idx.keys(), desc="Generating recommendations"):
            recommendations[user_id] = self.recommend(user_id, n, exclude_known)
        return recommendations


class ItemBasedCF:
    """
    Item-Based Collaborative Filtering using cosine similarity.
    """
    def __init__(self, train_matrix, user_to_idx, item_to_idx, idx_to_user, idx_to_item):
        """
        Initialize the model.
        
        Args:
            train_matrix: Training interaction matrix (users x items)
            user_to_idx: Mapping from user IDs to indices
            item_to_idx: Mapping from item IDs to indices
            idx_to_user: Mapping from indices to user IDs
            idx_to_item: Mapping from indices to item IDs
        """
        self.train_matrix = train_matrix
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx
        self.idx_to_user = idx_to_user
        self.idx_to_item = idx_to_item
        self.similarity_matrix = None
        
    def fit(self, k_neighbors=20):
        """
        Compute item-item similarity matrix.
        
        Args:
            k_neighbors: Number of neighbors to consider for each item
        """
        print("Computing item-item similarity matrix...")
        # Compute cosine similarity between items (transpose to get items as rows)
        self.similarity_matrix = cosine_similarity(self.train_matrix.T)
        
        # Set self-similarity to zero to avoid recommending the same item
        np.fill_diagonal(self.similarity_matrix, 0)
        
        # Keep only top-k neighbors for each item
        for i in range(self.similarity_matrix.shape[0]):
            # Get indices of sorted similarities (descending order)
            sorted_indices = np.argsort(self.similarity_matrix[i, :])[::-1]
            
            # Zero out similarities except for top-k neighbors
            mask = np.ones(self.similarity_matrix.shape[1], dtype=bool)
            mask[sorted_indices[:k_neighbors]] = False
            self.similarity_matrix[i, mask] = 0
        
        print("Done.")
        return self
    
    def predict(self, user_id, item_id):
        """
        Predict the rating for a specific user-item pair.
        
        Args:
            user_id: ID of the user
            item_id: ID of the item
            
        Returns:
            Predicted rating
        """
        # Get indices
        user_idx = self.user_to_idx.get(user_id)
        item_idx = self.item_to_idx.get(item_id)
        
        if user_idx is None or item_idx is None:
            return 0  # Return 0 for unknown users or items
        
        # Get user's ratings
        user_ratings = self.train_matrix.iloc[user_idx, :].values
        
        # Get item similarities
        item_similarities = self.similarity_matrix[item_idx, :]
        
        # Compute weighted average rating
        weighted_sum = np.sum(item_similarities * user_ratings)
        similarity_sum = np.sum(np.abs(item_similarities))
        
        if similarity_sum == 0:
            return 0  # No similar items
        
        return weighted_sum / similarity_sum
    
    def recommend(self, user_id, n=10, exclude_known=True):
        """
        Generate top-n recommendations for a user.
        
        Args:
            user_id: ID of the user
            n: Number of recommendations to generate
            exclude_known: Whether to exclude items the user has already interacted with
            
        Returns:
            List of recommended item IDs
        """
        # Get user index
        user_idx = self.user_to_idx.get(user_id)
        
        if user_idx is None:
            return []  # Return empty list for unknown users
        
        # Get items the user has already interacted with
        if exclude_known:
            known_items = set(self.train_matrix.columns[self.train_matrix.iloc[user_idx, :] > 0])
        else:
            known_items = set()
        
        # Compute predictions for all items
        predictions = []
        for item_id in self.item_to_idx.keys():
            if item_id not in known_items:  # Skip known items if exclude_known is True
                prediction = self.predict(user_id, item_id)
                predictions.append((item_id, prediction))
        
        # Sort by predicted rating (descending) and get top-n
        predictions.sort(key=lambda x: x[1], reverse=True)
        recommended_items = [item_id for item_id, _ in predictions[:n]]
        
        return recommended_items
    
    def recommend_all_users(self, n=10, exclude_known=True):
        """
        Generate recommendations for all users.
        
        Args:
            n: Number of recommendations per user
            exclude_known: Whether to exclude items the user has already interacted with
            
        Returns:
            Dictionary mapping user IDs to lists of recommended item IDs
        """
        recommendations = {}
        for user_id in tqdm(self.user_to_idx.keys(), desc="Generating recommendations"):
            recommendations[user_id] = self.recommend(user_id, n, exclude_known)
        return recommendations


#############################
# Advanced Models
#############################

class SVDModel:
    """
    SVD-based collaborative filtering model.
    """
    def __init__(self, train_matrix, user_to_idx, item_to_idx, idx_to_user, idx_to_item):
        """
        Initialize the model.
        
        Args:
            train_matrix: Training interaction matrix (users x items)
            user_to_idx: Mapping from user IDs to indices
            item_to_idx: Mapping from item IDs to indices
            idx_to_user: Mapping from indices to user IDs
            idx_to_item: Mapping from indices to item IDs
        """
        self.train_matrix = train_matrix
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx
        self.idx_to_user = idx_to_user
        self.idx_to_item = idx_to_item
        self.user_factors = None
        self.sigma = None
        self.item_factors = None
        self.item_means = None
        
    def fit(self, k_factors=50):
        """
        Fit the SVD model.
        
        Args:
            k_factors: Number of latent factors
        """
        print(f"Fitting SVD model with {k_factors} factors...")
        
        # Center the data by subtracting item means
        matrix_values = self.train_matrix.values
        self.item_means = np.nanmean(matrix_values, axis=0)
        matrix_centered = matrix_values - self.item_means
        
        # Fill NaN values with 0 (for items that users haven't interacted with)
        matrix_centered = np.nan_to_num(matrix_centered)
        
        # Apply SVD
        U, sigma, Vt = svds(matrix_centered, k=k_factors)
        
        # Convert to the right format
        self.user_factors = U
        self.sigma = np.diag(sigma)
        self.item_factors = Vt.T
        
        print("Done.")
        return self
    
    def predict(self, user_id, item_id):
        """
        Predict the rating for a specific user-item pair.
        
        Args:
            user_id: ID of the user
            item_id: ID of the item
            
        Returns:
            Predicted rating
        """
        # Get indices
        user_idx = self.user_to_idx.get(user_id)
        item_idx = self.item_to_idx.get(item_id)
        
        if user_idx is None or item_idx is None:
            return 0  # Return 0 for unknown users or items
        
        # Get the baseline estimate (item mean)
        baseline = self.item_means[item_idx]
        
        # Compute the SVD prediction
        user_vec = self.user_factors[user_idx, :]
        item_vec = self.item_factors[item_idx, :]
        pred = baseline + np.dot(user_vec, np.dot(self.sigma, item_vec))
        
        # Clip the prediction to be within a valid range
        return max(0, min(5, pred))
    
    def recommend(self, user_id, n=10, exclude_known=True):
        """
        Generate top-n recommendations for a user.
        
        Args:
            user_id: ID of the user
            n: Number of recommendations to generate
            exclude_known: Whether to exclude items the user has already interacted with
            
        Returns:
            List of recommended item IDs
        """
        # Get user index
        user_idx = self.user_to_idx.get(user_id)
        
        if user_idx is None:
            return []  # Return empty list for unknown users
        
        # Get items the user has already interacted with
        if exclude_known:
            known_items = set(self.train_matrix.columns[self.train_matrix.iloc[user_idx, :] > 0])
        else:
            known_items = set()
        
        # Get the user vector
        user_vec = self.user_factors[user_idx, :]
        
        # Compute predictions for all items
        predictions = []
        for item_id, item_idx in self.item_to_idx.items():
            if item_id not in known_items:  # Skip known items if exclude_known is True
                # Get the baseline estimate (item mean)
                baseline = self.item_means[item_idx]
                
                # Compute the SVD prediction
                item_vec = self.item_factors[item_idx, :]
                pred = baseline + np.dot(user_vec, np.dot(self.sigma, item_vec))
                
                # Clip the prediction to be within a valid range
                pred = max(0, min(5, pred))
                
                predictions.append((item_id, pred))
        
        # Sort by predicted rating (descending) and get top-n
        predictions.sort(key=lambda x: x[1], reverse=True)
        recommended_items = [item_id for item_id, _ in predictions[:n]]
        
        return recommended_items
    
    def recommend_all_users(self, n=10, exclude_known=True):
        """
        Generate recommendations for all users.
        
        Args:
            n: Number of recommendations per user
            exclude_known: Whether to exclude items the user has already interacted with
            
        Returns:
            Dictionary mapping user IDs to lists of recommended item IDs
        """
        recommendations = {}
        for user_id in tqdm(self.user_to_idx.keys(), desc="Generating recommendations"):
            recommendations[user_id] = self.recommend(user_id, n, exclude_known)
        return recommendations


#############################
# Visualization and Analysis
#############################

def visualize_model_comparison(metrics_list, model_names):
    """
    Visualize the comparison between different models.
    
    Args:
        metrics_list: List of metrics dictionaries for different models
        model_names: List of model names corresponding to the metrics
    """
    # Extract metrics
    precision_scores = [metrics['precision_at_k'] for metrics in metrics_list]
    hit_rates = [metrics['hit_rate'] for metrics in metrics_list]
    
    # Create a DataFrame for comparison
    comparison_df = pd.DataFrame({
        'Model': model_names,
        'Precision@10': precision_scores,
        'Hit Rate': hit_rates
    })
    
    print("Model Performance Comparison:")
    print(comparison_df)
    
    # Visualize the comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.barplot(x='Model', y='Precision@10', data=comparison_df, palette='viridis')
    plt.title('Precision@10 Comparison')
    plt.ylim(0, max(precision_scores) * 1.2)  # Add some headroom
    
    plt.subplot(1, 2, 2)
    sns.barplot(x='Model', y='Hit Rate', data=comparison_df, palette='viridis')
    plt.title('Hit Rate Comparison')
    plt.ylim(0, max(hit_rates) * 1.2)  # Add some headroom
    
    plt.tight_layout()
    plt.show()


def analyze_user_recommendations(user_id, user_df, items_df, models_dict):
    """
    Analyze recommendations for a specific user across different models.
    
    Args:
        user_id: ID of the user to analyze
        user_df: DataFrame containing user's interactions
        items_df: DataFrame containing item metadata
        models_dict: Dictionary mapping model names to model objects
    """
    # Get games this user has played
    user_games = user_df[user_df['user_id'] == user_id]
    user_games = user_games.merge(items_df[['item_id', 'item_name']], on='item_id', how='left')
    user_games = user_games.sort_values('normalized_playtime', ascending=False)
    
    print(f"User: {user_id}")
    print("\nGames played by this user:")
    print(user_games[['item_name', 'normalized_playtime']].head(10))
    
    # Get recommendations from each model
    for model_name, model in models_dict.items():
        print(f"\nTop 10 recommendations from {model_name}:")
        model_recs = model.recommend(user_id, n=10)
        model_recs_df = pd.DataFrame({'item_id': model_recs})
        model_recs_df = model_recs_df.merge(items_df[['item_id', 'item_name']], on='item_id', how='left')
        print(model_recs_df[['item_name']])


#############################
# Main Execution
#############################

def main():
    """
    Main execution function.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load data
    train_df, test_df, items_df, train_matrix, sparse_train_matrix, id_mappings = load_processed_data()
    
    # Extract ID mappings
    user_to_idx = id_mappings['user_to_idx']
    item_to_idx = id_mappings['item_to_idx']
    idx_to_user = id_mappings['idx_to_user']
    idx_to_item = id_mappings['idx_to_item']
    
    # Display data information
    display_data_info(train_df, test_df, items_df, train_matrix, sparse_train_matrix)
    
    # Train and evaluate user-based collaborative filtering model
    print("\n" + "="*50)
    print("Training and evaluating User-Based Collaborative Filtering model...")
    user_cf = UserBasedCF(
        train_matrix,
        user_to_idx,
        item_to_idx,
        idx_to_user,
        idx_to_item
    )
    user_cf.fit(k_neighbors=20)
    
    # Generate recommendations for all users
    user_cf_recommendations = user_cf.recommend_all_users(n=10, exclude_known=True)
    
    # Evaluate the recommendations
    user_cf_metrics = evaluate_recommendations(user_cf_recommendations, test_df, k=10)
    print("User-Based CF Evaluation:")
    print(f"Precision@10: {user_cf_metrics['precision_at_k']:.4f}")
    print(f"Hit Rate: {user_cf_metrics['hit_rate']:.4f}")
    print(f"Number of users evaluated: {user_cf_metrics['num_users_evaluated']}")
    
    # Train and evaluate item-based collaborative filtering model
    print("\n" + "="*50)
    print("Training and evaluating Item-Based Collaborative Filtering model...")
    item_cf = ItemBasedCF(
        train_matrix,
        user_to_idx,
        item_to_idx,
        idx_to_user,
        idx_to_item
    )
    item_cf.fit(k_neighbors=20)
    
    # Generate recommendations for all users
    item_cf_recommendations = item_cf.recommend_all_users(n=10, exclude_known=True)
    
    # Evaluate the recommendations
    item_cf_metrics = evaluate_recommendations(item_cf_recommendations, test_df, k=10)
    print("Item-Based CF Evaluation:")
    print(f"Precision@10: {item_cf_metrics['precision_at_k']:.4f}")
    print(f"Hit Rate: {item_cf_metrics['hit_rate']:.4f}")
    print(f"Number of users evaluated: {item_cf_metrics['num_users_evaluated']}")
    
    # Train and evaluate SVD model
    print("\n" + "="*50)
    print("Training and evaluating SVD model...")
    svd_model = SVDModel(
        train_matrix,
        user_to_idx,
        item_to_idx,
        idx_to_user,
        idx_to_item
    )
    svd_model.fit(k_factors=50)
    
    # Generate recommendations for all users
    svd_recommendations = svd_model.recommend_all_users(n=10, exclude_known=True)
    
    # Evaluate the recommendations
    svd_metrics = evaluate_recommendations(svd_recommendations, test_df, k=10)
    print("SVD Evaluation:")
    print(f"Precision@10: {svd_metrics['precision_at_k']:.4f}")
    print(f"Hit Rate: {svd_metrics['hit_rate']:.4f}")
    print(f"Number of users evaluated: {svd_metrics['num_users_evaluated']}")
    
    # Compare model performance
    print("\n" + "="*50)
    print("Comparing model performance...")
    models_metrics = [user_cf_metrics, item_cf_metrics, svd_metrics]
    model_names = ['User-Based CF', 'Item-Based CF', 'SVD']
    visualize_model_comparison(models_metrics, model_names)
    
    # Analyze recommendations for a sample user
    print("\n" + "="*50)
    print("Analyzing recommendations for a sample user...")
    sample_user = np.random.choice(list(user_to_idx.keys()))
    models_dict = {
        'User-Based CF': user_cf,
        'Item-Based CF': item_cf,
        'SVD': svd_model
    }
    analyze_user_recommendations(sample_user, train_df, items_df, models_dict)
    
    # Save models
    print("\n" + "="*50)
    print("Saving models...")
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    # Save user-based CF model
    with open(f'{models_dir}/user_based_cf.pkl', 'wb') as f:
        pickle.dump(user_cf, f)
    
    # Save item-based CF model
    with open(f'{models_dir}/item_based_cf.pkl', 'wb') as f:
        pickle.dump(item_cf, f)
    
    # Save SVD model
    with open(f'{models_dir}/svd_model.pkl', 'wb') as f:
        pickle.dump(svd_model, f)
    
    # Save model performance metrics
    model_metrics = {
        'user_based_cf': user_cf_metrics,
        'item_based_cf': item_cf_metrics,
        'svd': svd_metrics
    }
    
    with open(f'{models_dir}/model_metrics.pkl', 'wb') as f:
        pickle.dump(model_metrics, f)
    
    print(f"Models and metrics saved to {models_dir}/")
    print("\n" + "="*50)
    print("Summary of findings:")
    print("1. User-Based CF:")
    print(f"   - Precision@10: {user_cf_metrics['precision_at_k']:.4f}")
    print(f"   - Hit Rate: {user_cf_metrics['hit_rate']:.4f}")
    print("2. Item-Based CF:")
    print(f"   - Precision@10: {item_cf_metrics['precision_at_k']:.4f}")
    print(f"   - Hit Rate: {item_cf_metrics['hit_rate']:.4f}")
    print("3. SVD:")
    print(f"   - Precision@10: {svd_metrics['precision_at_k']:.4f}")
    print(f"   - Hit Rate: {svd_metrics['hit_rate']:.4f}")
    
    if svd_metrics['precision_at_k'] > max(user_cf_metrics['precision_at_k'], item_cf_metrics['precision_at_k']):
        print("\nThe SVD model outperforms the baseline collaborative filtering models, which aligns with")
        print("our findings about the sparsity and complexity of gaming preference patterns. SVD's ability")
        print("to identify latent factors in the user-item interaction matrix proved particularly effective")
        print("for capturing the diverse and often cross-genre preferences of Steam users.")
    
    print("\nLimitations and future work:")
    print("- Incorporate game metadata (genres, tags) for improved recommendations")
    print("- Implement hybrid models combining collaborative filtering with content-based features")
    print("- Consider temporal dynamics to capture evolving user preferences")
    print("- Develop strategies for cold-start problems (new users and games)")
    print("- Scale the system to handle the full Steam dataset more efficiently")


if __name__ == "__main__":
    main()