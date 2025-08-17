"""Game service for the Steam Game Recommender application."""

from typing import List, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from src.domain.entities.game import Game

logger = logging.getLogger(__name__)

class GameService:
    def __init__(self):
        self.games_data = None
        self._load_games_data()
    
    def _load_games_data(self):
        """Load games data from processed files or create sample data"""
        try:
            # Try to load processed data
            data_dir = Path("data/processed")
            games_file = data_dir / "games.csv"
            
            if games_file.exists():
                self.games_data = pd.read_csv(games_file)
                logger.info(f"Loaded {len(self.games_data)} games from processed data")
            else:
                # Create sample games data
                self._create_sample_games()
                logger.info("Created sample games data")
                
        except Exception as e:
            logger.error(f"Error loading games data: {e}")
            self._create_sample_games()
    
    def _create_sample_games(self):
        """Create sample games data for the dashboard"""
        np.random.seed(42)
        
        # Sample game titles
        game_titles = [
            "The Witcher 3: Wild Hunt", "Red Dead Redemption 2", "Cyberpunk 2077",
            "Elden Ring", "God of War", "Horizon Zero Dawn", "Death Stranding",
            "Control", "Disco Elysium", "Hades", "Stardew Valley", "Undertale",
            "Portal 2", "Half-Life 2", "BioShock", "Mass Effect 2", "Dragon Age: Origins",
            "Divinity: Original Sin 2", "Pillars of Eternity", "Baldur's Gate 3",
            "XCOM 2", "Civilization VI", "Total War: Warhammer", "Crusader Kings III",
            "RimWorld", "Factorio", "Cities: Skylines", "Planet Coaster", "Two Point Hospital",
            "FIFA 23", "NBA 2K23", "Madden NFL 23", "Rocket League", "Fall Guys",
            "Among Us", "Phasmophobia", "Dead by Daylight", "Resident Evil Village",
            "Resident Evil 4 Remake", "Silent Hill 2", "Outlast", "Amnesia: The Dark Descent",
            "Layers of Fear", "Little Nightmares", "Inside", "Limbo", "Gris", "Journey",
            "Abzû", "Flower", "The Last Guardian", "Shadow of the Colossus", "Ico"
        ]
        
        # Sample categories
        categories = [
            "Action", "Adventure", "RPG", "Strategy", "Simulation", "Sports", "Racing",
            "Fighting", "Shooter", "Stealth", "Survival", "Horror", "Puzzle", "Platformer",
            "Visual Novel", "Point & Click", "Roguelike", "Metroidvania", "Open World",
            "Linear", "Multiplayer", "Single Player", "Co-op", "Competitive", "Casual"
        ]
        
        # Create sample data
        n_games = 1000
        self.games_data = pd.DataFrame({
            'game_id': range(n_games),
            'game_title': np.random.choice(game_titles, n_games, replace=True),
            'game_categories': [np.random.choice(categories, np.random.randint(1, 4), replace=False).tolist() for _ in range(n_games)],
            'game_price': np.random.uniform(9.99, 69.99, n_games).round(2),
            'game_rating': np.random.uniform(2.0, 5.0, n_games).round(1),
            'game_tags': [[] for _ in range(n_games)],
            'game_image': [None] * n_games
        })
        
        # Ensure some games have realistic combinations
        for i in range(0, n_games, 10):
            self.games_data.iloc[i, self.games_data.columns.get_loc('game_categories')] = ['RPG', 'Action']
            self.games_data.iloc[i, self.games_data.columns.get_loc('game_price')] = np.random.uniform(39.99, 59.99)
            self.games_data.iloc[i, self.games_data.columns.get_loc('game_rating')] = np.random.uniform(4.0, 5.0)
    
    async def get_games(
        self,
        limit: int = 100,
        offset: int = 0,
        category: Optional[str] = None,
        min_rating: Optional[float] = None,
        max_price: Optional[float] = None
    ) -> List[Game]:
        """Get games with optional filtering"""
        try:
            data = self.games_data.copy()
            
            # Apply filters
            if category:
                data = data[data['game_categories'].apply(lambda x: category in x)]
            
            if min_rating is not None:
                data = data[data['game_rating'] >= min_rating]
            
            if max_price is not None:
                data = data[data['game_price'] <= max_price]
            
            # Apply pagination
            data = data.iloc[offset:offset + limit]
            
            # Convert to Game entities
            games = []
            for _, row in data.iterrows():
                game = Game(
                    game_id=row['game_id'],
                    game_title=row['game_title'],
                    game_categories=row['game_categories'],
                    game_price=row['game_price'],
                    game_rating=row['game_rating'],
                    game_tags=row['game_tags'],
                    game_image=row['game_image']
                )
                games.append(game)
            
            return games
            
        except Exception as e:
            logger.error(f"Error getting games: {e}")
            return []
    
    async def get_game_by_id(self, game_id: int) -> Optional[Game]:
        """Get a specific game by ID"""
        try:
            game_data = self.games_data[self.games_data['game_id'] == game_id]
            if game_data.empty:
                return None
            
            row = game_data.iloc[0]
            return Game(
                game_id=row['game_id'],
                game_title=row['game_title'],
                game_categories=row['game_categories'],
                game_price=row['game_price'],
                game_rating=row['game_rating'],
                game_tags=row['game_tags'],
                game_image=row['game_image']
            )
            
        except Exception as e:
            logger.error(f"Error getting game by ID: {e}")
            return None
    
    async def get_game_categories(self) -> List[str]:
        """Get list of available game categories"""
        try:
            all_categories = []
            for categories in self.games_data['game_categories']:
                all_categories.extend(categories)
            return sorted(list(set(all_categories)))
        except Exception as e:
            logger.error(f"Error getting categories: {e}")
            return []
    
    async def search_games(self, query: str, limit: int = 20) -> List[Game]:
        """Search games by title"""
        try:
            query_lower = query.lower()
            matching_games = self.games_data[
                self.games_data['game_title'].str.lower().str.contains(query_lower)
            ].head(limit)
            
            games = []
            for _, row in matching_games.iterrows():
                game = Game(
                    game_id=row['game_id'],
                    game_title=row['game_title'],
                    game_categories=row['game_categories'],
                    game_price=row['game_price'],
                    game_rating=row['game_rating'],
                    game_tags=row['game_tags'],
                    game_image=row['game_image']
                )
                games.append(game)
            
            return games
            
        except Exception as e:
            logger.error(f"Error searching games: {e}")
            return [] 