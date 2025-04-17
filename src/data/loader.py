"""
Data loading utilities for the Steam Game Recommender System.
"""
import os
import json
import gzip
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

def load_steam_data(data_dir: str = "data") -> Dict[str, Any]:
    """
    Load Steam dataset files.
    
    Args:
        data_dir: Directory containing the data files
        
    Returns:
        Dictionary containing the loaded data
    """
    data_paths = {
        "reviews": os.path.join(data_dir, "reviews_v2.json.gz"),
        "metadata": os.path.join(data_dir, "items_v2.json.gz"),
        "bundles": os.path.join(data_dir, "bundles.json"),
    }
    
    data = {}
    
    # Check if files exist
    for name, path in data_paths.items():
        if not os.path.exists(path):
            print(f"Warning: {path} not found. Download the data files first.")
            return {}
    
    # Load reviews
    print("Loading reviews data...")
    data["reviews"] = load_json_gz(data_paths["reviews"])
    
    # Load item metadata
    print("Loading item metadata...")
    data["metadata"] = load_json_gz(data_paths["metadata"])
    
    # Load bundles
    print("Loading bundles data...")
    data["bundles"] = load_json(data_paths["bundles"])
    
    return data

def load_json_gz(filepath: str) -> List[Dict[str, Any]]:
    """
    Load a gzipped JSON file line by line.
    
    Args:
        filepath: Path to the .json.gz file
        
    Returns:
        List of dictionaries, one for each JSON line
    """
    data = []
    with gzip.open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_json(filepath: str) -> List[Dict[str, Any]]:
    """
    Load a JSON file.
    
    Args:
        filepath: Path to the .json file
        
    Returns:
        List of dictionaries, one for each JSON line
    """
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def convert_to_dataframes(data: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Convert the loaded JSON data to pandas DataFrames.
    
    Args:
        data: Dictionary containing the loaded data
        
    Returns:
        Dictionary containing pandas DataFrames
    """
    dfs = {}
    
    # Convert reviews to DataFrame
    if "reviews" in data and data["reviews"]:
        print("Converting reviews to DataFrame...")
        dfs["reviews"] = pd.DataFrame(data["reviews"])
    
    # Convert metadata to DataFrame
    if "metadata" in data and data["metadata"]:
        print("Converting metadata to DataFrame...")
        dfs["metadata"] = pd.DataFrame(data["metadata"])
    
    # Process bundles
    if "bundles" in data and data["bundles"]:
        print("Converting bundles to DataFrame...")
        # Flatten bundle data
        bundle_rows = []
        for bundle in data["bundles"]:
            for item in bundle.get("items", []):
                bundle_row = {
                    "bundle_id": bundle.get("bundle_id"),
                    "bundle_name": bundle.get("bundle_name"),
                    "bundle_price": bundle.get("bundle_price"),
                    "bundle_final_price": bundle.get("bundle_final_price"),
                    "bundle_discount": bundle.get("bundle_discount"),
                    "item_id": item.get("item_id"),
                    "item_name": item.get("item_name"),
                    "item_url": item.get("item_url"),
                    "discounted_price": item.get("discounted_price"),
                    "genre": item.get("genre")
                }
                bundle_rows.append(bundle_row)
        dfs["bundles"] = pd.DataFrame(bundle_rows)
    
    return dfs

def create_user_game_matrix(review_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int], Dict[str, int]]:
    """
    Create a user-game interaction matrix from the review data.
    
    Args:
        review_df: DataFrame containing review data
        
    Returns:
        Tuple containing:
        - The user-game interaction matrix (DataFrame)
        - Dictionary mapping user IDs to matrix indices
        - Dictionary mapping game IDs to matrix indices
    """
    # Extract relevant columns
    if "user_id" not in review_df.columns or "item_id" not in review_df.columns:
        # Try to extract from nested structure if necessary
        if "reviews" in review_df.columns:
            # Handle case where reviews are nested
            pass
    
    # Keep only necessary columns
    interaction_df = review_df[['user_id', 'item_id']]
    if 'playtime_forever' in review_df.columns:
        interaction_df['playtime_forever'] = review_df['playtime_forever']
    else:
        interaction_df['playtime_forever'] = 1  # Binary interaction as fallback
    
    # Create dictionaries to map IDs to indices
    unique_users = interaction_df['user_id'].unique()
    unique_games = interaction_df['item_id'].unique()
    
    user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
    game_to_idx = {game: idx for idx, game in enumerate(unique_games)}
    
    # Create the interaction matrix
    matrix = pd.pivot_table(
        interaction_df,
        values='playtime_forever',
        index='user_id',
        columns='item_id',
        fill_value=0
    )
    
    return matrix, user_to_idx, game_to_idx

def get_sample_data(df: pd.DataFrame, sample_size: int = 10000) -> pd.DataFrame:
    """
    Get a sample of the data for testing or quick exploration.
    
    Args:
        df: DataFrame to sample from
        sample_size: Number of rows to sample
        
    Returns:
        Sampled DataFrame
    """
    if len(df) > sample_size:
        return df.sample(sample_size, random_state=42)
    return df