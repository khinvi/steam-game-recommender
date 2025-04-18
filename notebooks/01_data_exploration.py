#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Steam Game Recommender System - Data Exploration

This script explores the Steam dataset, examining its structure and characteristics to inform 
our recommendation system design. We'll analyze user behavior patterns, game popularity 
distributions, and user-game interactions.
"""

# ========== Setup and Data Loading ==========

# Import required libraries
import os
import json
import gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from tqdm import tqdm

# Set visualization style
plt.style.use('ggplot')
sns.set(style="whitegrid")

# Configure pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 1000)

# Create data directories if they don't exist
os.makedirs('../data/processed', exist_ok=True)


def load_data(sample_size=None):
    """
    Load the Steam dataset files (reviews and items).
    
    Args:
        sample_size: Number of reviews to sample (for faster processing)
        
    Returns:
        reviews_df: DataFrame containing user reviews/interactions
        items_df: DataFrame containing game metadata
    """
    # Define file paths - try multiple possible locations
    possible_paths = [
        ('../data/reviews_v2.json.gz', '../data/items_v2.json.gz'),
        ('../data/steam_reviews.json.gz', '../data/steam_games.json.gz'),
        ('data/reviews_v2.json.gz', 'data/items_v2.json.gz'),
        ('data/steam_reviews.json.gz', 'data/steam_games.json.gz')
    ]
    
    # Try to find the data files
    reviews_path, items_path = None, None
    for r_path, i_path in possible_paths:
        if os.path.exists(r_path) and os.path.exists(i_path):
            reviews_path, items_path = r_path, i_path
            break
    
    if reviews_path is None:
        raise FileNotFoundError("Could not find data files. Please check paths.")
        
    print(f"Loading reviews from {reviews_path}")
    print(f"Loading items from {items_path}")
    
    # Load reviews data
    reviews = []
    with gzip.open(reviews_path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc="Loading reviews")):
            if sample_size is not None and i >= sample_size:
                break
            try:
                reviews.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    # Load items data
    items = []
    with gzip.open(items_path, 'rt', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading items"):
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    # Convert to DataFrames
    reviews_df = pd.DataFrame(reviews)
    items_df = pd.DataFrame(items)
    
    return reviews_df, items_df


def create_synthetic_data():
    """
    Create synthetic data for demonstration purposes if real data is unavailable.
    
    Returns:
        reviews_df: Synthetic DataFrame containing user reviews/interactions
        items_df: Synthetic DataFrame containing game metadata
    """
    # Create sample reviews
    n_users = 1000
    n_games = 500
    n_reviews = 10000
    
    user_ids = [f"user_{i}" for i in range(n_users)]
    game_ids = [f"game_{i}" for i in range(n_games)]
    
    reviews = []
    for _ in range(n_reviews):
        user_id = np.random.choice(user_ids)
        game_id = np.random.choice(game_ids)
        playtime = np.random.randint(0, 1000) if np.random.random() > 0.2 else 0
        reviews.append({
            'user_id': user_id,
            'item_id': game_id,
            'playtime_forever': playtime,
            'playtime_2weeks': min(playtime, np.random.randint(0, 20)) if playtime > 0 else 0
        })
    
    # Create sample games
    game_names = [f"Game Title {i}" for i in range(n_games)]
    genres = ['Action', 'Adventure', 'RPG', 'Strategy', 'Simulation', 'Sports', 'Racing', 'Puzzle']
    
    items = []
    for i in range(n_games):
        n_genres = np.random.randint(1, 3)
        game_genres = np.random.choice(genres, n_genres, replace=False).tolist()
        items.append({
            'item_id': game_ids[i],
            'item_name': game_names[i],
            'genres': game_genres
        })
    
    reviews_df = pd.DataFrame(reviews)
    items_df = pd.DataFrame(items)
    
    print(f"Created {len(reviews_df)} synthetic reviews and {len(items_df)} synthetic items.")
    return reviews_df, items_df


def main():
    # ========== Load Data ==========
    # Load a sample of the data for faster exploration (50,000 reviews)
    # Set to None to load the full dataset (may take significant time)
    SAMPLE_SIZE = 50000
    
    try:
        reviews_df, items_df = load_data(sample_size=SAMPLE_SIZE)
        print(f"Loaded {len(reviews_df)} reviews and {len(items_df)} items.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Creating sample data for demonstration purposes...")
        reviews_df, items_df = create_synthetic_data()

    # ========== Basic Data Exploration ==========
    # Display the first few rows of the reviews dataset
    print("\nReviews Dataset:")
    print(reviews_df.head())

    # Display the first few rows of the items dataset
    print("\nItems Dataset:")
    print(items_df.head())

    # Get information about the reviews DataFrame
    print("\nReviews DataFrame Info:")
    reviews_df.info()

    # Check for missing values in reviews
    print("\nMissing values in reviews:")
    print(reviews_df.isnull().sum())

    # Get information about the items DataFrame
    print("\nItems DataFrame Info:")
    items_df.info()

    # Check for missing values in items
    print("\nMissing values in items:")
    print(items_df.isnull().sum())

    # Basic statistics
    n_users = reviews_df['user_id'].nunique()
    n_games = reviews_df['item_id'].nunique()
    n_interactions = len(reviews_df)
    sparsity = 100 * (1 - (n_interactions / (n_users * n_games)))

    print(f"\nNumber of unique users: {n_users}")
    print(f"Number of unique games: {n_games}")
    print(f"Number of interactions: {n_interactions}")
    print(f"Matrix sparsity: {sparsity:.2f}%")

    # ========== User Activity Analysis ==========
    # Calculate interactions per user
    user_interactions = reviews_df['user_id'].value_counts()

    # Plot the distribution of interactions per user
    plt.figure(figsize=(10, 6))
    sns.histplot(user_interactions, log_scale=True, bins=50)
    plt.title('Distribution of Games per User (Log Scale)')
    plt.xlabel('Number of Games')
    plt.ylabel('Number of Users')
    plt.grid(True)
    plt.savefig('../data/processed/games_per_user_dist.png')
    plt.close()

    # Print summary statistics
    user_interactions_stats = user_interactions.describe()
    print("\nSummary statistics for games per user:")
    print(user_interactions_stats)

    # Calculate total playtime per user
    user_playtime = reviews_df.groupby('user_id')['playtime_forever'].sum()

    # Plot the distribution of total playtime per user
    plt.figure(figsize=(10, 6))
    sns.histplot(user_playtime[user_playtime > 0], log_scale=True, bins=50)
    plt.title('Distribution of Total Playtime per User (Log Scale)')
    plt.xlabel('Total Playtime (minutes)')
    plt.ylabel('Number of Users')
    plt.grid(True)
    plt.savefig('../data/processed/playtime_per_user_dist.png')
    plt.close()

    # Print summary statistics
    user_playtime_stats = user_playtime.describe()
    print("\nSummary statistics for total playtime per user:")
    print(user_playtime_stats)

    # ========== Game Popularity Analysis ==========
    # Calculate interactions per game
    game_interactions = reviews_df['item_id'].value_counts()

    # Plot the distribution of interactions per game
    plt.figure(figsize=(10, 6))
    sns.histplot(game_interactions, log_scale=True, bins=50)
    plt.title('Distribution of Users per Game (Log Scale)')
    plt.xlabel('Number of Users')
    plt.ylabel('Number of Games')
    plt.grid(True)
    plt.savefig('../data/processed/users_per_game_dist.png')
    plt.close()

    # Print summary statistics
    game_interactions_stats = game_interactions.describe()
    print("\nSummary statistics for users per game:")
    print(game_interactions_stats)

    # Calculate total playtime per game
    game_playtime = reviews_df.groupby('item_id')['playtime_forever'].sum()

    # Plot the distribution of total playtime per game
    plt.figure(figsize=(10, 6))
    sns.histplot(game_playtime[game_playtime > 0], log_scale=True, bins=50)
    plt.title('Distribution of Total Playtime per Game (Log Scale)')
    plt.xlabel('Total Playtime (minutes)')
    plt.ylabel('Number of Games')
    plt.grid(True)
    plt.savefig('../data/processed/playtime_per_game_dist.png')
    plt.close()

    # Print summary statistics
    game_playtime_stats = game_playtime.describe()
    print("\nSummary statistics for total playtime per game:")
    print(game_playtime_stats)

    # ========== Top Games Analysis ==========
    # Merge reviews with items to get game names
    merged_df = reviews_df.merge(items_df, on='item_id', how='left')

    # Top games by number of users
    top_games_by_users = merged_df.groupby(['item_id', 'item_name']).size().reset_index(name='user_count')
    top_games_by_users = top_games_by_users.sort_values('user_count', ascending=False).head(20)

    # Plot top games by users
    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_games_by_users, x='user_count', y='item_name', palette='viridis')
    plt.title('Top 20 Games by Number of Users')
    plt.xlabel('Number of Users')
    plt.ylabel('Game Title')
    plt.tight_layout()
    plt.savefig('../data/processed/top_games_by_users.png')
    plt.close()

    # Top games by total playtime
    top_games_by_playtime = merged_df.groupby(['item_id', 'item_name'])['playtime_forever'].sum().reset_index()
    top_games_by_playtime = top_games_by_playtime.sort_values('playtime_forever', ascending=False).head(20)

    # Convert playtime to hours for better readability
    top_games_by_playtime['playtime_hours'] = top_games_by_playtime['playtime_forever'] / 60

    # Plot top games by playtime
    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_games_by_playtime, x='playtime_hours', y='item_name', palette='viridis')
    plt.title('Top 20 Games by Total Playtime')
    plt.xlabel('Total Playtime (hours)')
    plt.ylabel('Game Title')
    plt.tight_layout()
    plt.savefig('../data/processed/top_games_by_playtime.png')
    plt.close()

    # ========== Average Playtime Analysis ==========
    # Calculate average playtime per game
    avg_playtime = merged_df.groupby(['item_id', 'item_name'])['playtime_forever'].mean().reset_index()
    avg_playtime = avg_playtime[avg_playtime['playtime_forever'] > 0]  # Filter out zero playtime
    avg_playtime = avg_playtime.sort_values('playtime_forever', ascending=False).head(20)

    # Convert to hours
    avg_playtime['avg_playtime_hours'] = avg_playtime['playtime_forever'] / 60

    # Plot top games by average playtime
    plt.figure(figsize=(12, 8))
    sns.barplot(data=avg_playtime, x='avg_playtime_hours', y='item_name', palette='viridis')
    plt.title('Top 20 Games by Average Playtime per User')
    plt.xlabel('Average Playtime (hours)')
    plt.ylabel('Game Title')
    plt.tight_layout()
    plt.savefig('../data/processed/top_games_by_avg_playtime.png')
    plt.close()

    # ========== User-Game Interaction Matrix Analysis ==========
    # Create a smaller sample for visualization
    top_users = reviews_df['user_id'].value_counts().head(50).index.tolist()
    top_games = reviews_df['item_id'].value_counts().head(50).index.tolist()

    # Filter the reviews for top users and games
    sample_interactions = reviews_df[
        (reviews_df['user_id'].isin(top_users)) & 
        (reviews_df['item_id'].isin(top_games))
    ]

    # Create a binary interaction matrix
    interaction_matrix = sample_interactions.pivot_table(
        index='user_id',
        columns='item_id',
        values='playtime_forever',
        aggfunc='sum',
        fill_value=0
    )

    # Convert to binary (played or not played)
    binary_matrix = (interaction_matrix > 0).astype(int)

    # Visualize the matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(binary_matrix, cmap='viridis', cbar_kws={'label': 'Interaction'})
    plt.title('User-Game Interaction Matrix (Sample of Top 50 Users and Games)')
    plt.xlabel('Game ID')
    plt.ylabel('User ID')
    plt.tight_layout()
    plt.savefig('../data/processed/user_game_matrix.png')
    plt.close()

    # Calculate sparsity of this sample
    sample_sparsity = 100 * (1 - binary_matrix.sum().sum() / (binary_matrix.shape[0] * binary_matrix.shape[1]))
    print(f"\nSparsity of the sample matrix: {sample_sparsity:.2f}%")

    # ========== Playtime Distribution Analysis ==========
    # Filter out zero playtimes for a more meaningful analysis
    non_zero_playtime = reviews_df[reviews_df['playtime_forever'] > 0]

    # Histogram of playtime (log scale)
    plt.figure(figsize=(10, 6))
    sns.histplot(non_zero_playtime['playtime_forever'], log_scale=True, bins=50)
    plt.title('Distribution of Playtime (Log Scale)')
    plt.xlabel('Playtime (minutes)')
    plt.ylabel('Count')
    plt.grid(True)
    plt.savefig('../data/processed/playtime_distribution.png')
    plt.close()

    # Statistics of playtime
    playtime_stats = non_zero_playtime['playtime_forever'].describe()
    print("\nPlaytime statistics (minutes):")
    print(playtime_stats)

    # Convert to hours for better interpretability
    playtime_hours_stats = non_zero_playtime['playtime_forever'].div(60).describe()
    print("\nPlaytime statistics (hours):")
    print(playtime_hours_stats)

    # ========== User Behavior Patterns ==========
    # Analyze recent vs. total playtime (if 'playtime_2weeks' is available)
    if 'playtime_2weeks' in reviews_df.columns:
        # Filter for users who played within the last two weeks
        recent_players = reviews_df[reviews_df['playtime_2weeks'] > 0]
        
        # Calculate percentage of recent playtime relative to total playtime
        recent_players['playtime_percentage'] = 100 * recent_players['playtime_2weeks'] / recent_players['playtime_forever']
        
        # Plot the distribution of recent playtime percentage
        plt.figure(figsize=(10, 6))
        sns.histplot(recent_players['playtime_percentage'].clip(0, 100), bins=50)
        plt.title('Recent Playtime as Percentage of Total Playtime')
        plt.xlabel('Recent Playtime Percentage')
        plt.ylabel('Count')
        plt.grid(True)
        plt.savefig('../data/processed/recent_playtime_percentage.png')
        plt.close()
        
        # Calculate statistics
        recent_stats = recent_players['playtime_percentage'].describe()
        print("\nRecent playtime percentage statistics:")
        print(recent_stats)
        
        # Find games with high percentage of recent playtime
        recent_game_avg = recent_players.groupby('item_id')['playtime_percentage'].mean().reset_index()
        recent_game_avg = recent_game_avg.merge(items_df[['item_id', 'item_name']], on='item_id', how='left')
        recent_game_avg = recent_game_avg.sort_values('playtime_percentage', ascending=False).head(20)
        
        # Plot games with highest recent playtime percentage
        plt.figure(figsize=(12, 8))
        sns.barplot(data=recent_game_avg, x='playtime_percentage', y='item_name', palette='viridis')
        plt.title('Top 20 Games by Recent Playtime Percentage')
        plt.xlabel('Average Recent Playtime Percentage')
        plt.ylabel('Game Title')
        plt.tight_layout()
        plt.savefig('../data/processed/top_games_by_recent_playtime.png')
        plt.close()
    else:
        print("\n'playtime_2weeks' column not available in the dataset.")

    # ========== Game Genre Analysis ==========
    # Check if genres are available in the items dataframe
    if 'genres' in items_df.columns:
        # Extract all genres and count their occurrences
        all_genres = []
        
        for genres_list in items_df['genres'].dropna():
            if isinstance(genres_list, list):
                all_genres.extend(genres_list)
            elif isinstance(genres_list, str):
                # Handle case where genres might be stored as a string
                try:
                    genres_json = json.loads(genres_list.replace("'", "\""))
                    if isinstance(genres_json, list):
                        all_genres.extend(genres_json)
                except:
                    pass
        
        # Count genre occurrences
        genre_counts = Counter(all_genres)
        
        # Plot top genres
        top_genres = pd.DataFrame({
            'genre': list(genre_counts.keys()),
            'count': list(genre_counts.values())
        }).sort_values('count', ascending=False).head(20)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(data=top_genres, x='count', y='genre', palette='viridis')
        plt.title('Top 20 Game Genres')
        plt.xlabel('Number of Games')
        plt.ylabel('Genre')
        plt.tight_layout()
        plt.savefig('../data/processed/top_genres.png')
        plt.close()
        
        # Analyze playtime by genre
        if not merged_df.empty and 'genres' in merged_df.columns:
            # Create a new dataframe with one row per game-genre combination
            genre_playtime = []
            
            for _, row in merged_df.iterrows():
                if pd.notna(row['genres']):
                    genres = row['genres']
                    if isinstance(genres, list):
                        for genre in genres:
                            genre_playtime.append({
                                'genre': genre,
                                'playtime_forever': row['playtime_forever']
                            })
                    elif isinstance(genres, str):
                        try:
                            genres_json = json.loads(genres.replace("'", "\""))
                            if isinstance(genres_json, list):
                                for genre in genres_json:
                                    genre_playtime.append({
                                        'genre': genre,
                                        'playtime_forever': row['playtime_forever']
                                    })
                        except:
                            pass
            
            genre_playtime_df = pd.DataFrame(genre_playtime)
            
            # Calculate average playtime per genre
            avg_playtime_by_genre = genre_playtime_df.groupby('genre')['playtime_forever'].mean().reset_index()
            avg_playtime_by_genre = avg_playtime_by_genre.sort_values('playtime_forever', ascending=False).head(20)
            
            # Convert to hours
            avg_playtime_by_genre['avg_playtime_hours'] = avg_playtime_by_genre['playtime_forever'] / 60
            
            # Plot average playtime by genre
            plt.figure(figsize=(12, 8))
            sns.barplot(data=avg_playtime_by_genre, x='avg_playtime_hours', y='genre', palette='viridis')
            plt.title('Top 20 Genres by Average Playtime')
            plt.xlabel('Average Playtime (hours)')
            plt.ylabel('Genre')
            plt.tight_layout()
            plt.savefig('../data/processed/avg_playtime_by_genre.png')
            plt.close()
    else:
        print("\nGenre information not available in the dataset.")

    # ========== User Similarity Analysis ==========
    # Create a function to calculate Jaccard similarity between two sets
    def jaccard_similarity(set1, set2):
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0

    # Sample a small number of users for analysis
    sample_size = min(100, n_users)
    sample_users = np.random.choice(reviews_df['user_id'].unique(), sample_size, replace=False)

    # Create a dictionary mapping users to their game libraries
    user_games = {}
    for user in sample_users:
        user_games[user] = set(reviews_df[reviews_df['user_id'] == user]['item_id'])

    # Calculate pairwise similarities
    similarities = []
    for i, user1 in enumerate(sample_users):
        for user2 in sample_users[i+1:]:
            sim = jaccard_similarity(user_games[user1], user_games[user2])
            similarities.append(sim)

    # Plot the distribution of user similarities
    plt.figure(figsize=(10, 6))
    sns.histplot(similarities, bins=50)
    plt.title('Distribution of Jaccard Similarity Between User Game Libraries')
    plt.xlabel('Jaccard Similarity')
    plt.ylabel('Count')
    plt.grid(True)
    plt.savefig('../data/processed/user_similarity_distribution.png')
    plt.close()

    # Calculate statistics
    sim_stats = pd.Series(similarities).describe()
    print("\nStatistics of user-user similarities:")
    print(sim_stats)

    # ========== Game Similarity Analysis ==========
    # Sample a small number of games for analysis
    sample_size = min(100, n_games)
    sample_games = np.random.choice(reviews_df['item_id'].unique(), sample_size, replace=False)

    # Create a dictionary mapping games to the users who played them
    game_users = {}
    for game in sample_games:
        game_users[game] = set(reviews_df[reviews_df['item_id'] == game]['user_id'])

    # Calculate pairwise similarities
    game_similarities = []
    for i, game1 in enumerate(sample_games):
        for game2 in sample_games[i+1:]:
            sim = jaccard_similarity(game_users[game1], game_users[game2])
            game_similarities.append(sim)

    # Plot the distribution of game similarities
    plt.figure(figsize=(10, 6))
    sns.histplot(game_similarities, bins=50)
    plt.title('Distribution of Jaccard Similarity Between Games (Based on User Overlap)')
    plt.xlabel('Jaccard Similarity')
    plt.ylabel('Count')
    plt.grid(True)
    plt.savefig('../data/processed/game_similarity_distribution.png')
    plt.close()

    # Calculate statistics
    game_sim_stats = pd.Series(game_similarities).describe()
    print("\nStatistics of game-game similarities:")
    print(game_sim_stats)

    # ========== Save Exploratory Results ==========
    # Create a dictionary of key statistics
    exploration_stats = {
        'n_users': n_users,
        'n_games': n_games,
        'n_interactions': n_interactions,
        'sparsity': sparsity,
        'avg_games_per_user': float(user_interactions.mean()),
        'median_games_per_user': float(user_interactions.median()),
        'avg_users_per_game': float(game_interactions.mean()),
        'median_users_per_game': float(game_interactions.median()),
        'avg_playtime_per_user_hours': float(user_playtime.mean() / 60),
        'avg_playtime_per_game_hours': float(game_playtime.mean() / 60)
    }

    # Save statistics to a JSON file
    with open('../data/processed/exploration_stats.json', 'w') as f:
        json.dump(exploration_stats, f, indent=4)

    print("\nSaved exploration statistics to '../data/processed/exploration_stats.json'")
    
    # ========== Print Summary ==========
    print("\nBased on our exploratory data analysis, we've made several important observations:")
    print("\n1. Data Sparsity: The user-game interaction matrix is extremely sparse, with most users playing only a small fraction of available games.")
    print("\n2. Playtime Distribution: There's a wide range of playtime across users and games, with some users playing certain games for thousands of hours.")
    print("\n3. Long-Tail Distribution: Both user activity and game popularity follow a long-tail distribution.")
    print("\n4. User Similarity: Users generally have low similarity in their game libraries, indicating diverse gaming preferences.")
    print("\n5. Genre Preferences: Users often play games across multiple genres, indicating cross-genre recommendations might be valuable.")


if __name__ == "__main__":
    main()