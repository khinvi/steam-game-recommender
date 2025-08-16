#!/usr/bin/env python
"""
Comprehensive Steam data processing pipeline.
This script loads, preprocesses, and prepares the Steam dataset for recommendation models.
"""
import json
import gzip
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional
import logging

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.config import settings
from src.infrastructure.database.connection import SessionLocal, engine, Base
from src.infrastructure.database.models import User, Game, Recommendation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SteamDataProcessor:
    """Process Steam dataset and prepare it for recommendation models."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.games_df = None
        self.reviews_df = None
        self.bundles_df = None
        
    def load_data(self) -> None:
        """Load all Steam dataset files."""
        logger.info("Loading Steam dataset...")
        
        # Load games data
        games_file = self.data_dir / "items_v2.json.gz"
        games_data = []
        with gzip.open(games_file, 'rt', encoding='utf-8') as f:
            for line in f:
                games_data.append(json.loads(line.strip()))
        
        self.games_df = pd.DataFrame(games_data)
        logger.info(f"Loaded {len(self.games_df)} games")
        
        # Load reviews data
        reviews_file = self.data_dir / "reviews_v2.json.gz"
        reviews_data = []
        with gzip.open(reviews_file, 'rt', encoding='utf-8') as f:
            for line in f:
                reviews_data.append(json.loads(line.strip()))
        
        self.reviews_df = pd.DataFrame(reviews_data)
        logger.info(f"Loaded {len(self.reviews_df)} reviews")
        
        # Load bundles data
        bundles_file = self.data_dir / "bundles.json"
        with open(bundles_file, 'r', encoding='utf-8') as f:
            bundles_data = json.load(f)
        
        self.bundles_df = pd.DataFrame(bundles_data)
        logger.info(f"Loaded {len(self.bundles_df)} bundles")
        
    def preprocess_games(self) -> pd.DataFrame:
        """Preprocess games data."""
        logger.info("Preprocessing games data...")
        
        if self.games_df is None:
            raise ValueError("Games data not loaded. Call load_data() first.")
        
        # Clean and normalize games data
        games_clean = self.games_df.copy()
        
        # Convert genres and tags to proper lists if they're strings
        if 'genre' in games_clean.columns:
            games_clean['genre'] = games_clean['genre'].apply(
                lambda x: x if isinstance(x, list) else [x] if pd.notna(x) else []
            )
        
        if 'tags' in games_clean.columns:
            games_clean['tags'] = games_clean['tags'].apply(
                lambda x: x if isinstance(x, list) else [x] if pd.notna(x) else []
            )
        
        # Fill missing values
        games_clean['price'] = games_clean['price'].fillna(0.0)
        games_clean['rating'] = games_clean['rating'].fillna(3.0)
        games_clean['playtime_median'] = games_clean['playtime_median'].fillna(0)
        games_clean['playtime_mean'] = games_clean['playtime_mean'].fillna(0)
        
        # Create feature vectors for genres and tags
        all_genres = set()
        all_tags = set()
        
        for genres in games_clean['genre']:
            all_genres.update(genres)
        
        for tags in games_clean['tags']:
            all_tags.update(tags)
        
        # Create one-hot encoded features
        for genre in all_genres:
            games_clean[f'genre_{genre.lower().replace(" ", "_")}'] = games_clean['genre'].apply(
                lambda x: 1 if genre in x else 0
            )
        
        for tag in all_tags:
            games_clean[f'tag_{tag.lower().replace(" ", "_")}'] = games_clean['tags'].apply(
                lambda x: 1 if tag in x else 0
            )
        
        logger.info(f"Preprocessed games data: {len(games_clean)} games, {len(all_genres)} genres, {len(all_tags)} tags")
        return games_clean
    
    def preprocess_reviews(self) -> pd.DataFrame:
        """Preprocess reviews data."""
        logger.info("Preprocessing reviews data...")
        
        if self.reviews_df is None:
            raise ValueError("Reviews data not loaded. Call load_data() first.")
        
        # Clean and normalize reviews data
        reviews_clean = self.reviews_df.copy()
        
        # Fill missing values
        reviews_clean['playtime_forever'] = reviews_clean['playtime_forever'].fillna(0)
        reviews_clean['playtime_2weeks'] = reviews_clean['playtime_2weeks'].fillna(0)
        reviews_clean['rating'] = reviews_clean['rating'].fillna(3)
        
        # Create normalized playtime features
        reviews_clean['playtime_normalized'] = reviews_clean['playtime_forever'] / reviews_clean['playtime_forever'].max()
        
        # Create interaction strength (combination of rating and playtime)
        reviews_clean['interaction_strength'] = (
            reviews_clean['rating'] * 0.4 + 
            reviews_clean['playtime_normalized'] * 0.6
        )
        
        logger.info(f"Preprocessed reviews data: {len(reviews_clean)} reviews")
        return reviews_clean
    
    def create_user_item_matrix(self, reviews_df: pd.DataFrame) -> pd.DataFrame:
        """Create user-item interaction matrix."""
        logger.info("Creating user-item interaction matrix...")
        
        # Create pivot table
        user_item_matrix = reviews_df.pivot_table(
            index='user_id',
            columns='item_id',
            values='interaction_strength',
            fill_value=0
        )
        
        logger.info(f"Created user-item matrix: {user_item_matrix.shape[0]} users x {user_item_matrix.shape[1]} items")
        return user_item_matrix
    
    def create_training_data(self, reviews_df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into training and testing sets."""
        logger.info("Creating training and testing datasets...")
        
        # Sort by timestamp (if available) or random
        reviews_sorted = reviews_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Split into train and test
        split_idx = int(len(reviews_sorted) * (1 - test_size))
        train_df = reviews_sorted.iloc[:split_idx]
        test_df = reviews_sorted.iloc[split_idx:]
        
        logger.info(f"Training set: {len(train_df)} samples, Test set: {len(test_df)} samples")
        return train_df, test_df
    
    def save_processed_data(self, output_dir: str = "data/processed") -> None:
        """Save processed data to files."""
        logger.info("Saving processed data...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save preprocessed data
        if self.games_df is not None:
            games_clean = self.preprocess_games()
            games_clean.to_csv(output_path / "games_processed.csv", index=False)
            logger.info(f"Saved processed games to {output_path / 'games_processed.csv'}")
        
        if self.reviews_df is not None:
            reviews_clean = self.preprocess_reviews()
            reviews_clean.to_csv(output_path / "reviews_processed.csv", index=False)
            logger.info(f"Saved processed reviews to {output_path / 'reviews_processed.csv'}")
            
            # Create and save user-item matrix
            user_item_matrix = self.create_user_item_matrix(reviews_clean)
            user_item_matrix.to_csv(output_path / "user_item_matrix.csv")
            logger.info(f"Saved user-item matrix to {output_path / 'user_item_matrix.csv'}")
            
            # Create and save training/test split
            train_df, test_df = self.create_training_data(reviews_clean)
            train_df.to_csv(output_path / "train_data.csv", index=False)
            test_df.to_csv(output_path / "test_data.csv", index=False)
            logger.info(f"Saved training data to {output_path / 'train_data.csv'}")
            logger.info(f"Saved test data to {output_path / 'test_data.csv'}")
    
    def load_to_database(self) -> None:
        """Load processed data into the database."""
        logger.info("Loading data into database...")
        
        # Create database tables
        Base.metadata.create_all(bind=engine)
        
        # Create database session
        db = SessionLocal()
        
        try:
            # Load games
            if self.games_df is not None:
                games_clean = self.preprocess_games()
                
                for _, game_row in games_clean.iterrows():
                    # Check if game already exists
                    existing_game = db.query(Game).filter(Game.steam_id == game_row['item_id']).first()
                    
                    if not existing_game:
                        game = Game(
                            steam_id=game_row['item_id'],
                            name=game_row['item_name'],
                            genres=game_row['genre'],
                            tags=game_row['tags'],
                            price=game_row['price'],
                            game_data={
                                'release_year': game_row.get('release_year', 2020),
                                'developer': game_row.get('developer', 'Unknown'),
                                'publisher': game_row.get('publisher', 'Unknown'),
                                'rating': game_row['rating'],
                                'playtime_median': game_row['playtime_median'],
                                'playtime_mean': game_row['playtime_mean']
                            }
                        )
                        db.add(game)
                
                db.commit()
                logger.info(f"Loaded {len(games_clean)} games to database")
            
            # Load users (from reviews)
            if self.reviews_df is not None:
                unique_users = self.reviews_df['user_id'].unique()
                
                for user_id in unique_users:
                    # Check if user already exists
                    existing_user = db.query(User).filter(User.steam_id == user_id).first()
                    
                    if not existing_user:
                        user = User(
                            username=f"user_{user_id}",
                            email=f"{user_id}@steam.com",
                            hashed_password="placeholder_hash",
                            steam_id=user_id,
                            preferences={}
                        )
                        db.add(user)
                
                db.commit()
                logger.info(f"Loaded {len(unique_users)} users to database")
            
            # Load recommendations (from reviews)
            if self.reviews_df is not None:
                reviews_clean = self.preprocess_reviews()
                
                # Get user and game mappings
                user_map = {user.steam_id: user.id for user in db.query(User).all()}
                game_map = {game.steam_id: game.id for game in db.query(Game).all()}
                
                for _, review_row in reviews_clean.iterrows():
                    user_id = user_map.get(review_row['user_id'])
                    game_id = game_map.get(review_row['item_id'])
                    
                    if user_id and game_id:
                        # Check if recommendation already exists
                        existing_rec = db.query(Recommendation).filter(
                            Recommendation.user_id == user_id,
                            Recommendation.game_id == game_id
                        ).first()
                        
                        if not existing_rec:
                            recommendation = Recommendation(
                                user_id=user_id,
                                game_id=game_id,
                                score=review_row['interaction_strength'],
                                type='user_review'
                            )
                            db.add(recommendation)
                
                db.commit()
                logger.info(f"Loaded recommendations to database")
                
        except Exception as e:
            db.rollback()
            logger.error(f"Error loading data to database: {e}")
            raise
        finally:
            db.close()
    
    def generate_data_summary(self) -> Dict:
        """Generate summary statistics of the processed data."""
        logger.info("Generating data summary...")
        
        summary = {
            "dataset_info": {
                "name": "Processed Steam Dataset",
                "description": "Preprocessed Steam data ready for recommendation models",
                "processed_at": "2025-08-16"
            },
            "data_files": {},
            "statistics": {}
        }
        
        if self.games_df is not None:
            games_clean = self.preprocess_games()
            summary["data_files"]["games"] = {
                "file": "games_processed.csv",
                "records": len(games_clean),
                "features": list(games_clean.columns)
            }
            summary["statistics"]["total_games"] = len(games_clean)
            summary["statistics"]["avg_price"] = games_clean['price'].mean()
            summary["statistics"]["avg_rating"] = games_clean['rating'].mean()
        
        if self.reviews_df is not None:
            reviews_clean = self.preprocess_reviews()
            summary["data_files"]["reviews"] = {
                "file": "reviews_processed.csv",
                "records": len(reviews_clean),
                "features": list(reviews_clean.columns)
            }
            summary["statistics"]["total_reviews"] = len(reviews_clean)
            summary["statistics"]["total_users"] = reviews_clean['user_id'].nunique()
            summary["statistics"]["avg_rating"] = reviews_clean['rating'].mean()
            summary["statistics"]["avg_playtime"] = reviews_clean['playtime_forever'].mean()
        
        # Save summary
        output_path = Path("data/processed")
        output_path.mkdir(parents=True, exist_ok=True)
        
        with open(output_path / "processed_data_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved data summary to {output_path / 'processed_data_summary.json'}")
        return summary

def main():
    """Main function to process Steam dataset."""
    print("🎮 Processing Steam Dataset...")
    print("=" * 50)
    
    # Initialize processor
    processor = SteamDataProcessor()
    
    try:
        # Load data
        processor.load_data()
        
        # Save processed data
        processor.save_processed_data()
        
        # Load to database
        processor.load_to_database()
        
        # Generate summary
        summary = processor.generate_data_summary()
        
        print("\n" + "=" * 50)
        print("🎉 Steam Dataset Processing Complete!")
        print("=" * 50)
        
        print(f"\n📊 Processing Summary:")
        print(f"- Games processed: {summary['statistics'].get('total_games', 0):,}")
        print(f"- Reviews processed: {summary['statistics'].get('total_reviews', 0):,}")
        print(f"- Users processed: {summary['statistics'].get('total_users', 0):,}")
        
        print(f"\n📁 Files Created:")
        print(f"- data/processed/games_processed.csv")
        print(f"- data/processed/reviews_processed.csv")
        print(f"- data/processed/user_item_matrix.csv")
        print(f"- data/processed/train_data.csv")
        print(f"- data/processed/test_data.csv")
        print(f"- data/processed/processed_data_summary.json")
        
        print(f"\n🗄️ Database Updated:")
        print(f"- Games loaded to database")
        print(f"- Users loaded to database")
        print(f"- Recommendations loaded to database")
        
        print(f"\n🚀 Next Steps:")
        print("1. Train recommendation models using the processed data")
        print("2. Evaluate model performance on test data")
        print("3. Deploy models to your API")
        print("4. Generate real-time recommendations")
        
    except Exception as e:
        print(f"❌ Error processing data: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 