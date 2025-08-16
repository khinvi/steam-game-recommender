# deployment/scripts/download_sample_data.py
"""
Download and prepare sample data for local development
"""
import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import random

def create_sample_data():
    """Create sample data for testing"""
    print("Creating sample data...")
    
    # Create sample users
    n_users = 1000
    users = []
    for i in range(n_users):
        users.append({
            'user_id': f'user_{i}',
            'username': f'player_{i}',
            'created_at': datetime.now() - timedelta(days=random.randint(1, 365))
        })
    
    # Create sample games
    n_games = 500
    genres = ['Action', 'Adventure', 'RPG', 'Strategy', 'Simulation', 'Sports', 'Racing', 'Puzzle']
    games = []
    for i in range(n_games):
        game_genres = random.sample(genres, k=random.randint(1, 3))
        games.append({
            'game_id': f'game_{i}',
            'name': f'Game Title {i}',
            'genres': game_genres,
            'price': round(random.uniform(0, 59.99), 2),
            'release_date': datetime.now() - timedelta(days=random.randint(1, 1000))
        })
    
    # Create sample interactions
    interactions = []
    for user in users[:200]:  # Use subset for interactions
        n_interactions = random.randint(5, 50)
        user_games = random.sample(games, k=min(n_interactions, len(games)))
        
        for game in user_games:
            playtime = random.randint(0, 1000)
            interactions.append({
                'user_id': user['user_id'],
                'game_id': game['game_id'],
                'playtime_forever': playtime,
                'playtime_2weeks': min(playtime, random.randint(0, 100)),
                'timestamp': datetime.now() - timedelta(days=random.randint(1, 30))
            })
    
    # Save to CSV
    os.makedirs('data/processed', exist_ok=True)
    
    pd.DataFrame(users).to_csv('data/processed/users.csv', index=False)
    pd.DataFrame(games).to_csv('data/processed/games.csv', index=False)
    pd.DataFrame(interactions).to_csv('data/processed/interactions.csv', index=False)
    
    # Create interaction matrix
    interaction_df = pd.DataFrame(interactions)
    interaction_matrix = interaction_df.pivot_table(
        index='user_id',
        columns='game_id',
        values='playtime_forever',
        fill_value=0
    )
    interaction_matrix.to_csv('data/processed/interaction_matrix.csv')
    
    print(f"Created sample data:")
    print(f"  - {len(users)} users")
    print(f"  - {len(games)} games")
    print(f"  - {len(interactions)} interactions")
    print(f"  - Interaction matrix shape: {interaction_matrix.shape}")
    
    return interaction_matrix

if __name__ == "__main__":
    create_sample_data() 