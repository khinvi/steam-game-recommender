"""
Evaluation metrics for recommendation systems.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Union, Callable

def precision_at_k(recommendations: List[Dict[str, Any]], 
                   ground_truth: List[str], 
                   k: int = 10) -> float:
    """
    Calculate precision@k for a single user.
    
    Args:
        recommendations: List of recommended items with 'item_id' field
        ground_truth: List of items that the user has actually interacted with
        k: The number of recommendations to consider
        
    Returns:
        Precision@k value
    """
    # Ensure we only consider the top k recommendations
    recommendations = recommendations[:k]
    
    # Extract the item IDs from the recommendations
    recommended_items = [rec['item_id'] for rec in recommendations]
    
    # Convert to sets for intersection
    recommended_set = set(recommended_items)
    ground_truth_set = set(ground_truth)
    
    # Calculate precision@k
    hits = len(recommended_set.intersection(ground_truth_set))
    precision = hits / min(k, len(recommended_items)) if recommended_items else 0
    
    return precision

def hit_rate(recommendations: List[Dict[str, Any]], 
             ground_truth: List[str]) -> float:
    """
    Calculate hit rate for a single user (1 if at least one recommended item is relevant, 0 otherwise).
    
    Args:
        recommendations: List of recommended items with 'item_id' field
        ground_truth: List of items that the user has actually interacted with
        
    Returns:
        Hit rate value (0 or 1)
    """
    # Extract the item IDs from the recommendations
    recommended_items = [rec['item_id'] for rec in recommendations]
    
    # Convert to sets for intersection
    recommended_set = set(recommended_items)
    ground_truth_set = set(ground_truth)
    
    # Calculate hit rate
    hits = len(recommended_set.intersection(ground_truth_set))
    hit_rate_value = 1 if hits > 0 else 0
    
    return hit_rate_value

def evaluate_model(model: Any, 
                   test_data: pd.DataFrame, 
                   k: int = 10, 
                   user_col: str = 'user_id', 
                   item_col: str = 'item_id') -> Dict[str, float]:
    """
    Evaluate a recommendation model on test data.
    
    Args:
        model: A recommendation model with a recommend() method
        test_data: DataFrame with user-item interactions for testing
        k: Number of recommendations to generate per user
        user_col: Name of the user ID column in test_data
        item_col: Name of the item ID column in test_data
        
    Returns:
        Dictionary with evaluation metrics
    """
    metrics = {
        'precision_at_k': 0.0,
        'hit_rate': 0.0,
        'num_users': 0
    }
    
    # Get unique users in test data
    test_users = test_data[user_col].unique()
    
    # Track metrics for each user
    hit_count = 0
    precision_sum = 0.0
    valid_user_count = 0
    
    for user_id in test_users:
        try:
            # Get the user's test items
            user_test_items = test_data[test_data[user_col] == user_id][item_col].tolist()
            
            # Generate recommendations
            recommendations = model.recommend(user_id, k=k)
            
            # Calculate precision@k
            user_precision = precision_at_k(recommendations, user_test_items, k)
            precision_sum += user_precision
            
            # Calculate hit rate
            user_hit_rate = hit_rate(recommendations, user_test_items)
            hit_count += user_hit_rate
            
            valid_user_count += 1
        except Exception as e:
            print(f"Error evaluating user {user_id}: {e}")
    
    # Calculate average metrics
    if valid_user_count > 0:
        metrics['precision_at_k'] = precision_sum / valid_user_count
        metrics['hit_rate'] = hit_count / valid_user_count
        metrics['num_users'] = valid_user_count
    
    return metrics

def evaluate_multiple_models(models: Dict[str, Any], 
                            test_data: pd.DataFrame, 
                            k: int = 10) -> pd.DataFrame:
    """
    Evaluate multiple recommendation models on the same test data.
    
    Args:
        models: Dictionary mapping model names to model objects
        test_data: DataFrame with user-item interactions for testing
        k: Number of recommendations to generate per user
        
    Returns:
        DataFrame with evaluation metrics for each model
    """
    results = []
    
    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        metrics = evaluate_model(model, test_data, k)
        
        # Add model name to metrics
        metrics['model'] = model_name
        results.append(metrics)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Reorder columns to put model name first
    cols = ['model'] + [col for col in results_df.columns if col != 'model']
    results_df = results_df[cols]
    
    return results_df

def plot_evaluation_results(results_df: pd.DataFrame, metric: str = 'precision_at_k'):
    """
    Plot evaluation results for multiple models.
    
    Args:
        results_df: DataFrame with evaluation results
        metric: Metric to plot
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(results_df['model'], results_df[metric])
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom')
    
    plt.xlabel('Model')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f'Comparison of Models by {metric.replace("_", " ").title()}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return plt