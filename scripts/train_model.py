#!/usr/bin/env python
"""
Script to train recommendation models.
"""
import os
import sys
import argparse
import pickle
import pandas as pd

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import load_steam_data, convert_to_dataframes
from src.data.preprocessor import (
    preprocess_data, 
    train_test_split_interactions, 
    create_interaction_matrix
)
from src.models.cosine_similarity import CosineSimilarityModel
from src.models.svd import SVDModel
from src.evaluation.metrics import evaluate_model

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train recommendation models for Steam games.')
    parser.add_argument('--model', type=str, default='svd', choices=['user_cf', 'item_cf', 'svd'],
                        help='Model to train: user_cf, item_cf, or svd')
    parser.add_argument('--n_factors', type=int, default=50,
                        help='Number of latent factors for SVD model')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Fraction of data to use for testing')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='Number of interactions to sample (for quicker testing)')
    parser.add_argument('--output_dir', type=str, default='../models',
                        help='Directory to save the trained model')
    
    return parser.parse_args()

def load_data(sample_size=None):
    """Load and preprocess the data."""
    print("Loading data...")
    
    # Check if processed data exists
    processed_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                     "data", "processed")
    train_path = os.path.join(processed_data_dir, "train_interactions.csv")
    test_path = os.path.join(processed_data_dir, "test_interactions.csv")
    matrix_path = os.path.join(processed_data_dir, "interaction_matrix.csv")
    
    if os.path.exists(train_path) and os.path.exists(test_path) and os.path.exists(matrix_path):
        print("Loading preprocessed data...")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        interaction_matrix = pd.read_csv(matrix_path, index_col=0)
        
        # Apply sampling if requested
        if sample_size is not None:
            print(f"Sampling {sample_size} interactions...")
            if len(train_df) > sample_size:
                train_df = train_df.sample(sample_size, random_state=42)
            if len(test_df) > sample_size // 4:
                test_df = test_df.sample(sample_size // 4, random_state=42)
            
            # Recreate the interaction matrix
            interaction_matrix = create_interaction_matrix(
                train_df,
                user_col='user_id',
                item_col='item_id',
                value_col='playtime_normalized' if 'playtime_normalized' in train_df.columns else None
            )
        
        return train_df, test_df, interaction_matrix
    else:
        print("Preprocessed data not found. Loading and preprocessing raw data...")
        
        # Load raw data
        raw_data = load_steam_data()
        dfs = convert_to_dataframes(raw_data)
        
        if 'reviews' not in dfs:
            raise ValueError("Reviews data not found. Please run download_data.py first.")
        
        # Preprocess data
        reviews_df = dfs['reviews']
        
        # Apply sampling if requested
        if sample_size is not None and len(reviews_df) > sample_size:
            print(f"Sampling {sample_size} interactions...")
            reviews_df = reviews_df.sample(sample_size, random_state=42)
        
        processed = preprocess_data({'reviews': reviews_df})
        
        # Split into train and test
        train_df, test_df = train_test_split_interactions(
            processed['interactions'], 
            test_size=args.test_size
        )
        
        # Create interaction matrix
        interaction_matrix = create_interaction_matrix(
            train_df,
            user_col='user_id',
            item_col='item_id',
            value_col='playtime_normalized' if 'playtime_normalized' in train_df.columns else None
        )
        
        return train_df, test_df, interaction_matrix

def train_and_evaluate(model_name, interaction_matrix, train_df, test_df, n_factors=50):
    """Train and evaluate a model."""
    print(f"Training {model_name} model...")
    
    if model_name == 'user_cf':
        model = CosineSimilarityModel(mode='user')
    elif model_name == 'item_cf':
        model = CosineSimilarityModel(mode='item')
    elif model_name == 'svd':
        model = SVDModel(n_factors=n_factors)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Train the model
    model.fit(interaction_matrix)
    
    # Evaluate the model
    print("Evaluating model...")
    metrics = evaluate_model(model, test_df)
    print(f"Evaluation metrics: {metrics}")
    
    return model, metrics

def save_model(model, model_name, output_dir, n_factors=None):
    """Save the trained model."""
    os.makedirs(output_dir, exist_ok=True)
    
    if model_name == 'svd' and n_factors is not None:
        filepath = os.path.join(output_dir, f"{model_name}_{n_factors}_factors.pkl")
    else:
        filepath = os.path.join(output_dir, f"{model_name}.pkl")
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {filepath}")

def main(args):
    """Main function to train and evaluate models."""
    # Load data
    train_df, test_df, interaction_matrix = load_data(args.sample_size)
    
    # Train and evaluate model
    model, metrics = train_and_evaluate(
        args.model, 
        interaction_matrix, 
        train_df, 
        test_df, 
        args.n_factors
    )
    
    # Save model
    save_model(model, args.model, args.output_dir, args.n_factors)
    
    print("\nTraining completed successfully!")
    print(f"Model: {args.model}")
    print(f"Precision@10: {metrics['precision_at_k']:.4f}")
    print(f"Hit Rate: {metrics['hit_rate']:.4f}")

if __name__ == "__main__":
    args = parse_args()
    main(args)