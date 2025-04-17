"""
Data preprocessing utilities for the Steam Game Recommender System.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from sklearn.model_selection import train_test_split

def preprocess_data(data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Preprocess the loaded data for model training.
    
    Args:
        data: Dictionary containing raw DataFrames
        
    Returns:
        Dictionary containing preprocessed data
    """
    processed = {}
    
    # Process reviews data
    if "reviews" in data:
        reviews_df = data["reviews"]
        
        # Extract user-item interactions
        interactions = extract_interactions(reviews_df)
        
        # Filter out test versions or non-standard games
        interactions = filter_games(interactions)
        
        # Handle missing values
        interactions = handle_missing_values(interactions)
        
        # Normalize playtime
        interactions = normalize_playtime(interactions)
        
        processed["interactions"] = interactions
    
    # Process metadata if available
    if "metadata" in data:
        metadata_df = data["metadata"]
        processed["metadata"] = process_metadata(metadata_df)
    
    return processed

def extract_interactions(reviews_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract user-item interactions from the reviews data.
    
    Args:
        reviews_df: DataFrame containing review data
        
    Returns:
        DataFrame with user-item interactions
    """
    # Check if the dataframe has the expected columns
    required_cols = ["user_id", "item_id"]
    optional_cols = ["playtime_forever", "playtime_2weeks", "recommended"]
    
    # Ensure required columns exist
    for col in required_cols:
        if col not in reviews_df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Select relevant columns
    cols_to_keep = required_cols + [col for col in optional_cols if col in reviews_df.columns]
    interactions = reviews_df[cols_to_keep].copy()
    
    # Convert item_id and user_id to string if they aren't already
    interactions["item_id"] = interactions["item_id"].astype(str)
    interactions["user_id"] = interactions["user_id"].astype(str)
    
    # If playtime columns don't exist, create a binary interaction column
    if "playtime_forever" not in interactions.columns:
        interactions["playtime_forever"] = 1
    
    return interactions

def filter_games(interactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out test versions or non-standard games.
    
    Args:
        interactions_df: DataFrame with user-item interactions
        
    Returns:
        Filtered DataFrame
    """
    # Filter out games with "test" in their name (if item_name column exists)
    if "item_name" in interactions_df.columns:
        interactions_df = interactions_df[~interactions_df["item_name"].str.lower().str.contains("test", na=False)]
    
    # Filter out users with very few interactions (e.g., less than 5)
    user_counts = interactions_df["user_id"].value_counts()
    active_users = user_counts[user_counts >= 5].index
    interactions_df = interactions_df[interactions_df["user_id"].isin(active_users)]
    
    # Filter out games with very few interactions (e.g., less than 10)
    game_counts = interactions_df["item_id"].value_counts()
    popular_games = game_counts[game_counts >= 10].index
    interactions_df = interactions_df[interactions_df["item_id"].isin(popular_games)]
    
    return interactions_df

def handle_missing_values(interactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the interactions data.
    
    Args:
        interactions_df: DataFrame with user-item interactions
        
    Returns:
        DataFrame with handled missing values
    """
    # Fill missing playtime values with 0
    if "playtime_forever" in interactions_df.columns:
        interactions_df["playtime_forever"] = interactions_df["playtime_forever"].fillna(0)
    
    if "playtime_2weeks" in interactions_df.columns:
        interactions_df["playtime_2weeks"] = interactions_df["playtime_2weeks"].fillna(0)
    
    # Fill missing recommended values with False
    if "recommended" in interactions_df.columns:
        interactions_df["recommended"] = interactions_df["recommended"].fillna(False)
    
    return interactions_df

def normalize_playtime(interactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize playtime values to reduce the impact of outliers.
    
    Args:
        interactions_df: DataFrame with user-item interactions
        
    Returns:
        DataFrame with normalized playtime
    """
    if "playtime_forever" in interactions_df.columns:
        # Log transformation for playtime (adding 1 to avoid log(0))
        interactions_df["playtime_log"] = np.log1p(interactions_df["playtime_forever"])
        
        # Min-max scaling per user
        def min_max_scale_by_user(group):
            min_val = group["playtime_log"].min()
            max_val = group["playtime_log"].max()
            
            if max_val > min_val:
                group["playtime_normalized"] = (group["playtime_log"] - min_val) / (max_val - min_val)
            else:
                group["playtime_normalized"] = 1.0  # All games have same playtime
                
            return group
        
        interactions_df = interactions_df.groupby("user_id").apply(min_max_scale_by_user)
        
        # Reset index if groupby created a multi-index
        if isinstance(interactions_df.index, pd.MultiIndex):
            interactions_df = interactions_df.reset_index(drop=True)
    
    return interactions_df

def process_metadata(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process game metadata.
    
    Args:
        metadata_df: DataFrame containing game metadata
        
    Returns:
        Processed metadata DataFrame
    """
    # Copy to avoid modifying the original
    metadata = metadata_df.copy()
    
    # Convert item_id to string if it isn't already
    metadata["item_id"] = metadata["item_id"].astype(str)
    
    # Extract genre information if available
    if "genre" in metadata.columns:
        # Create one-hot encoded genre columns
        genres = metadata["genre"].str.split(", ", expand=False)
        genre_dummies = pd.get_dummies(genres.explode()).groupby(level=0).sum()
        
        # Join the genre dummies with the metadata
        metadata = metadata.join(genre_dummies)
    
    return metadata

def train_test_split_interactions(interactions_df: pd.DataFrame, 
                                  test_size: float = 0.2, 
                                  random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the interactions data into training and testing sets.
    
    Args:
        interactions_df: DataFrame with user-item interactions
        test_size: Fraction of interactions to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, test_df)
    """
    # Strategy: For each user, split their interactions into train and test
    train_interactions = []
    test_interactions = []
    
    for user_id, user_data in interactions_df.groupby("user_id"):
        # Only split if the user has multiple interactions
        if len(user_data) > 1:
            user_train, user_test = train_test_split(
                user_data, 
                test_size=test_size, 
                random_state=random_state
            )
            train_interactions.append(user_train)
            test_interactions.append(user_test)
        else:
            # If only one interaction, add to training
            train_interactions.append(user_data)
    
    train_df = pd.concat(train_interactions, ignore_index=True)
    
    # Only create test_df if there are test interactions
    if test_interactions:
        test_df = pd.concat(test_interactions, ignore_index=True)
    else:
        test_df = pd.DataFrame(columns=interactions_df.columns)
    
    return train_df, test_df

def create_interaction_matrix(interactions_df: pd.DataFrame, 
                              user_col: str = "user_id", 
                              item_col: str = "item_id", 
                              value_col: str = "playtime_normalized") -> pd.DataFrame:
    """
    Create an interaction matrix from a DataFrame of interactions.
    
    Args:
        interactions_df: DataFrame with user-item interactions
        user_col: Name of the user ID column
        item_col: Name of the item ID column
        value_col: Name of the column to use as values in the matrix
        
    Returns:
        User-item interaction matrix as DataFrame
    """
    # If value_col doesn't exist, fall back to a binary indicator
    if value_col not in interactions_df.columns:
        interactions_df["interaction"] = 1
        value_col = "interaction"
    
    # Create the pivot table
    interaction_matrix = interactions_df.pivot_table(
        index=user_col,
        columns=item_col,
        values=value_col,
        fill_value=0
    )
    
    return interaction_matrix