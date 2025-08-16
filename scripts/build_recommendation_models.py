#!/usr/bin/env python
"""
Build and train recommendation models using processed Steam data.
This script implements several recommendation algorithms and evaluates their performance.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SteamRecommendationEngine:
    """Build and train recommendation models for Steam games."""
    
    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
        self.games_df = None
        self.reviews_df = None
        self.user_item_matrix = None
        self.train_df = None
        self.test_df = None
        
        # Models
        self.models = {}
        self.model_performance = {}
        
    def load_processed_data(self) -> None:
        """Load the processed Steam data."""
        logger.info("Loading processed Steam data...")
        
        # Load games data
        games_file = self.data_dir / "games_processed.csv"
        self.games_df = pd.read_csv(games_file)
        logger.info(f"Loaded {len(self.games_df)} processed games")
        
        # Load reviews data
        reviews_file = self.data_dir / "reviews_processed.csv"
        self.reviews_df = pd.read_csv(reviews_file)
        logger.info(f"Loaded {len(self.reviews_df)} processed reviews")
        
        # Load user-item matrix
        matrix_file = self.data_dir / "user_item_matrix.csv"
        self.user_item_matrix = pd.read_csv(matrix_file, index_col=0)
        logger.info(f"Loaded user-item matrix: {self.user_item_matrix.shape}")
        
        # Load training and test data
        train_file = self.data_dir / "train_data.csv"
        test_file = self.data_dir / "test_data.csv"
        self.train_df = pd.read_csv(train_file)
        self.test_df = pd.read_csv(test_file)
        logger.info(f"Loaded training data: {len(self.train_df)} samples")
        logger.info(f"Loaded test data: {len(self.test_df)} samples")
    
    def build_popularity_based_model(self) -> Dict:
        """Build a popularity-based recommendation model."""
        logger.info("Building popularity-based recommendation model...")
        
        # Calculate game popularity based on interaction strength
        game_popularity = self.reviews_df.groupby('item_id')['interaction_strength'].agg([
            'mean', 'count', 'sum'
        ]).reset_index()
        
        # Sort by popularity score (combination of mean rating and number of interactions)
        game_popularity['popularity_score'] = (
            game_popularity['mean'] * 0.7 + 
            game_popularity['count'] / game_popularity['count'].max() * 0.3
        )
        
        game_popularity = game_popularity.sort_values('popularity_score', ascending=False)
        
        # Store model
        self.models['popularity'] = {
            'type': 'popularity_based',
            'data': game_popularity,
            'top_games': game_popularity.head(20)['item_id'].tolist()
        }
        
        logger.info(f"Built popularity model with top {len(self.models['popularity']['top_games'])} games")
        return self.models['popularity']
    
    def build_content_based_model(self) -> Dict:
        """Build a content-based recommendation model using game features."""
        logger.info("Building content-based recommendation model...")
        
        # Get feature columns (genres and tags)
        feature_cols = [col for col in self.games_df.columns if col.startswith(('genre_', 'tag_'))]
        
        if not feature_cols:
            logger.warning("No feature columns found for content-based model")
            return {}
        
        # Create feature matrix
        feature_matrix = self.games_df[feature_cols].fillna(0)
        
        # Calculate cosine similarity between games
        game_similarity = cosine_similarity(feature_matrix)
        game_similarity_df = pd.DataFrame(
            game_similarity,
            index=self.games_df['item_id'],
            columns=self.games_df['item_id']
        )
        
        # Store model
        self.models['content_based'] = {
            'type': 'content_based',
            'similarity_matrix': game_similarity_df,
            'feature_matrix': feature_matrix,
            'feature_columns': feature_cols
        }
        
        logger.info(f"Built content-based model with {len(feature_cols)} features")
        return self.models['content_based']
    
    def build_collaborative_filtering_model(self, method: str = 'nmf') -> Dict:
        """Build collaborative filtering model using matrix factorization."""
        logger.info(f"Building collaborative filtering model using {method.upper()}...")
        
        if self.user_item_matrix is None:
            logger.error("User-item matrix not loaded")
            return {}
        
        # Remove users/items with too few interactions
        min_interactions = 3
        user_counts = (self.user_item_matrix > 0).sum(axis=1)
        item_counts = (self.user_item_matrix > 0).sum(axis=0)
        
        valid_users = user_counts[user_counts >= min_interactions].index
        valid_items = item_counts[item_counts >= min_interactions].index
        
        filtered_matrix = self.user_item_matrix.loc[valid_users, valid_items]
        
        if method.lower() == 'nmf':
            # Non-negative Matrix Factorization
            model = NMF(n_components=50, random_state=42, max_iter=200)
            user_factors = model.fit_transform(filtered_matrix)
            item_factors = model.components_
        elif method.lower() == 'svd':
            # Singular Value Decomposition
            model = TruncatedSVD(n_components=50, random_state=42)
            user_factors = model.fit_transform(filtered_matrix)
            item_factors = model.components_
        else:
            logger.error(f"Unknown method: {method}")
            return {}
        
        # Store model
        self.models[f'collaborative_{method}'] = {
            'type': f'collaborative_filtering_{method}',
            'model': model,
            'user_factors': user_factors,
            'item_factors': item_factors,
            'user_ids': valid_users.tolist(),
            'item_ids': valid_items.tolist(),
            'original_matrix': filtered_matrix
        }
        
        logger.info(f"Built collaborative filtering model with {method.upper()}")
        return self.models[f'collaborative_{method}']
    
    def build_hybrid_model(self) -> Dict:
        """Build a hybrid recommendation model combining multiple approaches."""
        logger.info("Building hybrid recommendation model...")
        
        # Ensure we have the base models
        if 'popularity' not in self.models:
            self.build_popularity_based_model()
        
        if 'content_based' not in self.models:
            self.build_content_based_model()
        
        if 'collaborative_nmf' not in self.models:
            self.build_collaborative_filtering_model('nmf')
        
        # Create hybrid scoring function
        def hybrid_score(user_id: str, item_id: str, alpha: float = 0.4, beta: float = 0.3, gamma: float = 0.3):
            """Calculate hybrid recommendation score."""
            scores = {}
            
            # Popularity score
            if 'popularity' in self.models:
                pop_data = self.models['popularity']['data']
                pop_score = pop_data[pop_data['item_id'] == item_id]['popularity_score'].iloc[0] if len(pop_data[pop_data['item_id'] == item_id]) > 0 else 0
                scores['popularity'] = pop_score
            
            # Content-based score (average similarity to user's liked games)
            if 'content_based' in self.models and user_id in self.user_item_matrix.index:
                user_games = self.user_item_matrix.loc[user_id]
                liked_games = user_games[user_games > 0].index.tolist()
                
                if liked_games:
                    similarities = []
                    for liked_game in liked_games:
                        if liked_game in self.models['content_based']['similarity_matrix'].index and item_id in self.models['content_based']['similarity_matrix'].columns:
                            sim = self.models['content_based']['similarity_matrix'].loc[liked_game, item_id]
                            similarities.append(sim)
                    
                    content_score = np.mean(similarities) if similarities else 0
                    scores['content'] = content_score
            
            # Collaborative filtering score
            if 'collaborative_nmf' in self.models and user_id in self.models['collaborative_nmf']['user_ids']:
                user_idx = self.models['collaborative_nmf']['user_ids'].index(user_id)
                item_idx = self.models['collaborative_nmf']['item_ids'].index(item_id) if item_id in self.models['collaborative_nmf']['item_ids'] else -1
                
                if item_idx >= 0:
                    cf_score = np.dot(
                        self.models['collaborative_nmf']['user_factors'][user_idx],
                        self.models['collaborative_nmf']['item_factors'][:, item_idx]
                    )
                    scores['collaborative'] = cf_score
            
            # Calculate weighted hybrid score
            hybrid_score = (
                scores.get('popularity', 0) * alpha +
                scores.get('content', 0) * beta +
                scores.get('collaborative', 0) * gamma
            )
            
            return hybrid_score, scores
        
        # Store hybrid model
        self.models['hybrid'] = {
            'type': 'hybrid',
            'scoring_function': hybrid_score,
            'weights': {'alpha': 0.4, 'beta': 0.3, 'gamma': 0.3}
        }
        
        logger.info("Built hybrid recommendation model")
        return self.models['hybrid']
    
    def evaluate_models(self) -> Dict:
        """Evaluate all built models on test data."""
        logger.info("Evaluating recommendation models...")
        
        if self.test_df is None:
            logger.error("Test data not loaded")
            return {}
        
        results = {}
        
        for model_name, model_data in self.models.items():
            logger.info(f"Evaluating {model_name} model...")
            
            if model_name == 'popularity':
                # Evaluate popularity model
                predictions = []
                actuals = []
                
                for _, test_row in self.test_df.iterrows():
                    user_id = test_row['user_id']
                    item_id = test_row['item_id']
                    actual = test_row['interaction_strength']
                    
                    # Get popularity score for this item
                    pop_data = model_data['data']
                    pred = pop_data[pop_data['item_id'] == item_id]['popularity_score'].iloc[0] if len(pop_data[pop_data['item_id'] == item_id]) > 0 else 0
                    
                    predictions.append(pred)
                    actuals.append(actual)
                
                # Calculate metrics
                mse = mean_squared_error(actuals, predictions)
                mae = mean_absolute_error(actuals, predictions)
                
                results[model_name] = {
                    'mse': mse,
                    'mae': mae,
                    'rmse': np.sqrt(mse)
                }
            
            elif model_name.startswith('collaborative'):
                # Evaluate collaborative filtering model
                predictions = []
                actuals = []
                
                for _, test_row in self.test_df.iterrows():
                    user_id = test_row['user_id']
                    item_id = test_row['item_id']
                    actual = test_row['interaction_strength']
                    
                    # Get prediction from collaborative model
                    if user_id in model_data['user_ids'] and item_id in model_data['item_ids']:
                        user_idx = model_data['user_ids'].index(user_id)
                        item_idx = model_data['item_ids'].index(item_id)
                        
                        pred = np.dot(
                            model_data['user_factors'][user_idx],
                            model_data['item_factors'][:, item_idx]
                        )
                    else:
                        pred = 0
                    
                    predictions.append(pred)
                    actuals.append(actual)
                
                # Calculate metrics
                mse = mean_squared_error(actuals, predictions)
                mae = mean_absolute_error(actuals, predictions)
                
                results[model_name] = {
                    'mse': mse,
                    'mae': mae,
                    'rmse': np.sqrt(mse)
                }
        
        self.model_performance = results
        logger.info("Model evaluation complete")
        return results
    
    def generate_recommendations(self, user_id: str, n_recommendations: int = 10) -> List[Dict]:
        """Generate recommendations for a specific user."""
        logger.info(f"Generating {n_recommendations} recommendations for user {user_id}")
        
        if 'hybrid' not in self.models:
            self.build_hybrid_model()
        
        hybrid_model = self.models['hybrid']
        scoring_function = hybrid_model['scoring_function']
        
        # Get all games
        all_games = self.games_df['item_id'].tolist()
        
        # Calculate scores for all games
        game_scores = []
        for game_id in all_games:
            score, breakdown = scoring_function(user_id, game_id)
            game_scores.append({
                'game_id': game_id,
                'score': score,
                'breakdown': breakdown
            })
        
        # Sort by score and get top recommendations
        game_scores.sort(key=lambda x: x['score'], reverse=True)
        top_recommendations = game_scores[:n_recommendations]
        
        # Add game details
        recommendations = []
        for rec in top_recommendations:
            game_info = self.games_df[self.games_df['item_id'] == rec['game_id']].iloc[0]
            recommendations.append({
                'game_id': rec['game_id'],
                'game_name': game_info['item_name'],
                'genres': game_info['genre'],
                'tags': game_info['tags'],
                'price': game_info['price'],
                'score': rec['score'],
                'score_breakdown': rec['breakdown']
            })
        
        logger.info(f"Generated {len(recommendations)} recommendations")
        return recommendations
    
    def save_models(self, output_dir: str = "data/models") -> None:
        """Save trained models to disk."""
        logger.info("Saving trained models...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save each model
        for model_name, model_data in self.models.items():
            model_file = output_path / f"{model_name}_model.pkl"
            
            # Handle different model types
            if model_name == 'popularity':
                # Save popularity data
                model_data['data'].to_csv(output_path / f"{model_name}_data.csv", index=False)
            
            elif model_name.startswith('collaborative'):
                # Save collaborative filtering model
                with open(model_file, 'wb') as f:
                    pickle.dump({
                        'model': model_data['model'],
                        'user_factors': model_data['user_factors'],
                        'item_factors': model_data['item_factors'],
                        'user_ids': model_data['user_ids'],
                        'item_ids': model_data['item_ids']
                    }, f)
            
            elif model_name == 'content_based':
                # Save similarity matrix
                model_data['similarity_matrix'].to_csv(output_path / f"{model_name}_similarity.csv")
            
            elif model_name == 'hybrid':
                # Save hybrid model configuration
                with open(model_file, 'wb') as f:
                    pickle.dump({
                        'weights': model_data['weights'],
                        'model_name': 'hybrid'
                    }, f)
        
        # Save model performance
        if self.model_performance:
            with open(output_path / "model_performance.json", 'w') as f:
                json.dump(self.model_performance, f, indent=2)
        
        logger.info(f"Saved models to {output_path}")
    
    def generate_model_summary(self) -> Dict:
        """Generate a summary of all built models."""
        summary = {
            "models_built": list(self.models.keys()),
            "model_count": len(self.models),
            "performance_metrics": self.model_performance,
            "data_summary": {
                "games": len(self.games_df) if self.games_df is not None else 0,
                "users": len(self.reviews_df['user_id'].unique()) if self.reviews_df is not None else 0,
                "interactions": len(self.reviews_df) if self.reviews_df is not None else 0
            }
        }
        
        return summary

def main():
    """Main function to build and train recommendation models."""
    print("🎮 Building Steam Recommendation Models...")
    print("=" * 50)
    
    # Initialize recommendation engine
    engine = SteamRecommendationEngine()
    
    try:
        # Load processed data
        engine.load_processed_data()
        
        # Build models
        print("\n🔨 Building Recommendation Models...")
        
        # 1. Popularity-based model
        engine.build_popularity_based_model()
        
        # 2. Content-based model
        engine.build_content_based_model()
        
        # 3. Collaborative filtering models
        engine.build_collaborative_filtering_model('nmf')
        engine.build_collaborative_filtering_model('svd')
        
        # 4. Hybrid model
        engine.build_hybrid_model()
        
        # Evaluate models
        print("\n📊 Evaluating Models...")
        performance = engine.evaluate_models()
        
        # Generate sample recommendations
        print("\n🎯 Generating Sample Recommendations...")
        sample_user = engine.reviews_df['user_id'].iloc[0]
        recommendations = engine.generate_recommendations(sample_user, n_recommendations=5)
        
        # Save models
        print("\n💾 Saving Models...")
        engine.save_models()
        
        # Generate summary
        summary = engine.generate_model_summary()
        
        print("\n" + "=" * 50)
        print("🎉 Recommendation Models Built Successfully!")
        print("=" * 50)
        
        print(f"\n📊 Models Built:")
        for model_name in summary["models_built"]:
            print(f"- {model_name}")
        
        print(f"\n📈 Performance Metrics:")
        for model_name, metrics in summary["performance_metrics"].items():
            print(f"- {model_name}: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")
        
        print(f"\n🎯 Sample Recommendations for User {sample_user}:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['game_name']} (Score: {rec['score']:.4f})")
            print(f"   Genres: {', '.join(rec['genres'])}")
            print(f"   Price: ${rec['price']:.2f}")
        
        print(f"\n📁 Models Saved:")
        print(f"- data/models/ (all model files)")
        print(f"- data/models/model_performance.json")
        
        print(f"\n🚀 Next Steps:")
        print("1. Integrate models with your API")
        print("2. Deploy models for real-time recommendations")
        print("3. Monitor model performance")
        print("4. Retrain models with new data")
        
    except Exception as e:
        print(f"❌ Error building models: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 