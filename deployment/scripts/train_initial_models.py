# deployment/scripts/train_initial_models.py
"""
Train initial models for the recommendation system
"""
import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.classical.svd import SVDModel
from src.models.classical.cosine_similarity import CosineSimilarityModel

def train_models():
    """Train all initial models"""
    print("Training initial models...")
    
    # Load interaction matrix
    if os.path.exists('data/processed/interaction_matrix.csv'):
        interaction_matrix = pd.read_csv('data/processed/interaction_matrix.csv', index_col=0)
    else:
        print("No interaction matrix found. Creating sample data...")
        from download_sample_data import create_sample_data
        interaction_matrix = create_sample_data()
    
    print(f"Loaded interaction matrix: {interaction_matrix.shape}")
    
    # Create models directory
    os.makedirs('data/models', exist_ok=True)
    
    # Train SVD model
    print("Training SVD model...")
    svd_model = SVDModel(n_factors=50)
    svd_model.fit(interaction_matrix)
    
    # Save SVD model
    with open('data/models/svd_model_50_factors.pkl', 'wb') as f:
        pickle.dump(svd_model, f)
    print("✓ SVD model saved")
    
    # Train user-based collaborative filtering
    print("Training user-based CF model...")
    user_cf_model = CosineSimilarityModel(mode='user')
    user_cf_model.fit(interaction_matrix)
    
    # Save user-based CF model
    with open('data/models/user_based_cf.pkl', 'wb') as f:
        pickle.dump(user_cf_model, f)
    print("✓ User-based CF model saved")
    
    # Train item-based collaborative filtering
    print("Training item-based CF model...")
    item_cf_model = CosineSimilarityModel(mode='item')
    item_cf_model.fit(interaction_matrix)
    
    # Save item-based CF model
    with open('data/models/item_based_cf.pkl', 'wb') as f:
        pickle.dump(item_cf_model, f)
    print("✓ Item-based CF model saved")
    
    # Create sample embeddings for vector store
    print("Creating game embeddings...")
    
    # Use SVD components as embeddings
    item_embeddings = svd_model.item_factors if hasattr(svd_model, 'item_factors') else None
    
    if item_embeddings is not None:
        # Save embeddings
        np.save('data/models/game_embeddings.npy', item_embeddings)
        print("✓ Game embeddings saved")
    
    print("\n✅ All models trained successfully!")
    print("\nModels saved in data/models/:")
    print("  - svd_model_50_factors.pkl")
    print("  - user_based_cf.pkl")
    print("  - item_based_cf.pkl")
    print("  - game_embeddings.npy")

if __name__ == "__main__":
    train_models() 