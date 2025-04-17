"""
Visualization utilities for the Steam Game Recommender System.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional

def plot_interaction_matrix(interaction_matrix: pd.DataFrame, 
                           n_users: int = 20, 
                           n_items: int = 20, 
                           title: str = "User-Game Interaction Matrix",
                           cmap: str = "Blues"):
    """
    Visualize a portion of the user-item interaction matrix.
    
    Args:
        interaction_matrix: DataFrame with users as index, items as columns
        n_users: Number of users to show
        n_items: Number of items to show
        title: Plot title
        cmap: Colormap to use
    
    Returns:
        matplotlib figure
    """
    # Take a subset of the matrix
    subset = interaction_matrix.iloc[:n_users, :n_items]
    
    # Create figure
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(subset, cmap=cmap, cbar=True)
    plt.title(title)
    plt.xlabel("Games")
    plt.ylabel("Users")
    
    return plt.gcf()

def plot_distribution(data: pd.Series, 
                     title: str, 
                     xlabel: str, 
                     ylabel: str = "Frequency",
                     bins: int = 30,
                     log_scale: bool = False):
    """
    Plot the distribution of a series of values.
    
    Args:
        data: Series of values to plot
        title: Plot title
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        bins: Number of bins for histogram
        log_scale: Whether to use log scale for x-axis
    
    Returns:
        matplotlib figure
    """
    plt.figure(figsize=(10, 6))
    ax = data.hist(bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    if log_scale:
        plt.xscale('log')
    
    return plt.gcf()

def plot_precision_recall_curve(precisions: List[float], 
                              recalls: List[float], 
                              thresholds: List[float],
                              title: str = "Precision-Recall Curve"):
    """
    Plot a precision-recall curve.
    
    Args:
        precisions: List of precision values
        recalls: List of recall values
        thresholds: List of threshold values
        title: Plot title
    
    Returns:
        matplotlib figure
    """
    plt.figure(figsize=(10, 6))
    plt.plot(recalls, precisions, marker='o')
    plt.title(title)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    
    # Add threshold annotations for some points
    step = max(1, len(thresholds) // 10)
    for i in range(0, len(thresholds), step):
        plt.annotate(f"{thresholds[i]:.2f}", 
                     (recalls[i], precisions[i]),
                     textcoords="offset points",
                     xytext=(0,10),
                     ha='center')
    
    return plt.gcf()

def plot_model_comparison(results_df: pd.DataFrame, 
                        metric_col: str = 'precision_at_k',
                        title: Optional[str] = None,
                        figsize: tuple = (10, 6)):
    """
    Plot a comparison of models based on a metric.
    
    Args:
        results_df: DataFrame with model comparison results
        metric_col: Column name of the metric to plot
        title: Plot title (if None, will be auto-generated)
        figsize: Figure size
    
    Returns:
        matplotlib figure
    """
    if 'model' not in results_df.columns or metric_col not in results_df.columns:
        raise ValueError(f"results_df must contain 'model' and '{metric_col}' columns")
    
    plt.figure(figsize=figsize)
    ax = sns.barplot(x='model', y=metric_col, data=results_df)
    
    # Add value labels on top of bars
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}", 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='bottom', 
                   xytext=(0, 5), 
                   textcoords='offset points')
    
    if title is None:
        title = f"Comparison of Models by {metric_col.replace('_', ' ').title()}"
    
    plt.title(title)
    plt.xlabel('Model')
    plt.ylabel(metric_col.replace('_', ' ').title())
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return plt.gcf()

def plot_latent_factors(item_factors: np.ndarray, n_factors: int = 3):
    """
    Visualize the distribution and relationships of latent factors.
    
    Args:
        item_factors: Matrix of item factors from SVD
        n_factors: Number of factors to visualize
    
    Returns:
        List of matplotlib figures
    """
    figures = []
    
    if item_factors is None or item_factors.shape[1] < n_factors:
        print("Not enough latent factors to visualize")
        return figures
    
    # Distribution of each factor
    plt.figure(figsize=(18, 6))
    for i in range(n_factors):
        plt.subplot(1, n_factors, i+1)
        plt.hist(item_factors[:, i], bins=30)
        plt.title(f'Distribution of Latent Factor {i+1}')
    
    figures.append(plt.gcf())
    
    # Pairwise relationships between factors
    if n_factors >= 2:
        for i in range(n_factors):
            for j in range(i+1, n_factors):
                plt.figure(figsize=(10, 8))
                plt.scatter(item_factors[:, i], item_factors[:, j], alpha=0.5)
                plt.title(f'Item Factors: Factor {i+1} vs Factor {j+1}')
                plt.xlabel(f'Factor {i+1}')
                plt.ylabel(f'Factor {j+1}')
                plt.grid(True)
                figures.append(plt.gcf())
    
    return figures

def plot_recommendations_for_user(user_id: str, 
                                 recommendations: List[Dict[str, Any]], 
                                 game_metadata: Optional[pd.DataFrame] = None):
    """
    Visualize recommendations for a user, optionally with game metadata.
    
    Args:
        user_id: ID of the user
        recommendations: List of recommendation dictionaries with 'item_id' and 'score' keys
        game_metadata: DataFrame with game metadata (optional)
    
    Returns:
        matplotlib figure
    """
    plt.figure(figsize=(12, 6))
    
    # Extract item IDs and scores
    item_ids = [rec['item_id'] for rec in recommendations]
    scores = [rec['score'] for rec in recommendations]
    
    # Get game names if metadata is provided
    if game_metadata is not None and 'item_name' in game_metadata.columns:
        game_metadata = game_metadata.set_index('item_id')
        labels = [game_metadata.loc[item_id, 'item_name'] if item_id in game_metadata.index 
                 else item_id for item_id in item_ids]
    else:
        labels = item_ids
    
    # Truncate long game names
    labels = [label[:30] + '...' if isinstance(label, str) and len(label) > 30 else label 
             for label in labels]
    
    # Create horizontal bar chart
    y_pos = np.arange(len(labels))
    plt.barh(y_pos, scores, align='center')
    plt.yticks(y_pos, labels)
    plt.xlabel('Recommendation Score')
    plt.title(f'Top Recommendations for User {user_id}')
    plt.tight_layout()
    
    return plt.gcf()

def plot_playtime_distribution(interactions_df: pd.DataFrame, 
                              log_scale: bool = True,
                              bins: int = 50):
    """
    Plot the distribution of playtime across all interactions.
    
    Args:
        interactions_df: DataFrame with user-game interactions
        log_scale: Whether to use log scale for x-axis
        bins: Number of bins for histogram
    
    Returns:
        matplotlib figure
    """
    if 'playtime_forever' not in interactions_df.columns:
        raise ValueError("interactions_df must contain 'playtime_forever' column")
    
    plt.figure(figsize=(10, 6))
    plt.hist(interactions_df['playtime_forever'], bins=bins)
    plt.title('Distribution of Playtime')
    plt.xlabel('Playtime (minutes)')
    plt.ylabel('Frequency')
    
    if log_scale:
        plt.xscale('log')
    
    return plt.gcf()

def plot_games_per_user_distribution(interactions_df: pd.DataFrame, 
                                    bins: int = 30):
    """
    Plot the distribution of number of games per user.
    
    Args:
        interactions_df: DataFrame with user-game interactions
        bins: Number of bins for histogram
    
    Returns:
        matplotlib figure
    """
    games_per_user = interactions_df.groupby('user_id')['item_id'].count()
    
    plt.figure(figsize=(10, 6))
    plt.hist(games_per_user, bins=bins)
    plt.title('Distribution of Number of Games per User')
    plt.xlabel('Number of Games')
    plt.ylabel('Number of Users')
    
    return plt.gcf()

def plot_users_per_game_distribution(interactions_df: pd.DataFrame, 
                                   bins: int = 30):
    """
    Plot the distribution of number of users per game.
    
    Args:
        interactions_df: DataFrame with user-game interactions
        bins: Number of bins for histogram
    
    Returns:
        matplotlib figure
    """
    users_per_game = interactions_df.groupby('item_id')['user_id'].count()
    
    plt.figure(figsize=(10, 6))
    plt.hist(users_per_game, bins=bins)
    plt.title('Distribution of Number of Users per Game')
    plt.xlabel('Number of Users')
    plt.ylabel('Number of Games')
    
    return plt.gcf()

def plot_genre_distribution(metadata_df: pd.DataFrame, 
                          top_n: int = 15,
                          genre_col: str = 'genre'):
    """
    Plot the distribution of game genres.
    
    Args:
        metadata_df: DataFrame with game metadata
        top_n: Number of top genres to show
        genre_col: Name of the genre column
    
    Returns:
        matplotlib figure
    """
    if genre_col not in metadata_df.columns:
        raise ValueError(f"metadata_df must contain '{genre_col}' column")
    
    # Count genres
    genre_counts = {}
    
    for genre_list in metadata_df[genre_col].dropna():
        if isinstance(genre_list, str):
            genres = genre_list.split(', ')
            for genre in genres:
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
    
    # Convert to DataFrame for plotting
    genre_df = pd.DataFrame({
        'genre': list(genre_counts.keys()),
        'count': list(genre_counts.values())
    })
    genre_df = genre_df.sort_values('count', ascending=False).head(top_n)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='count', y='genre', data=genre_df)
    plt.title('Most Common Game Genres')
    plt.xlabel('Number of Games')
    plt.ylabel('Genre')
    
    return plt.gcf()