#!/usr/bin/env python
"""
Script to create a realistic sample Steam dataset for development and testing.
This creates a dataset similar to the original Steam data structure.
"""
import json
import gzip
import random
import os
from pathlib import Path

# Data directory
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Sample data generation
def create_sample_steam_data():
    """Create realistic sample Steam data."""
    
    # Sample game genres and tags
    GENRES = [
        "Action", "Adventure", "RPG", "Strategy", "Simulation", 
        "Sports", "Racing", "Puzzle", "Indie", "Casual",
        "Shooter", "Fighting", "Platformer", "Horror", "Visual Novel"
    ]
    
    TAGS = [
        "Multiplayer", "Single-player", "Co-op", "Competitive", "Story Rich",
        "Atmospheric", "Great Soundtrack", "2D", "3D", "Pixel Graphics",
        "Open World", "Linear", "Sandbox", "Crafting", "Building",
        "Exploration", "Combat", "Stealth", "Roguelike", "Roguelite"
    ]
    
    # Sample games data
    games_data = []
    for i in range(1000):  # 1000 sample games
        game = {
            "item_id": f"I{i:06d}",
            "item_name": f"Sample Game {i+1}",
            "genre": random.sample(GENRES, random.randint(1, 3)),
            "tags": random.sample(TAGS, random.randint(2, 5)),
            "price": round(random.uniform(0, 59.99), 2),
            "release_year": random.randint(2010, 2024),
            "developer": f"Developer {random.randint(1, 50)}",
            "publisher": f"Publisher {random.randint(1, 20)}",
            "rating": round(random.uniform(1.0, 5.0), 1),
            "playtime_median": random.randint(10, 500),
            "playtime_mean": random.randint(15, 600)
        }
        games_data.append(game)
    
    # Sample user-game interactions (reviews/playtime)
    reviews_data = []
    for i in range(5000):  # 5000 sample interactions
        review = {
            "user_id": f"U{random.randint(1, 500):06d}",
            "item_id": f"I{random.randint(1, 1000):06d}",
            "playtime_forever": random.randint(0, 1000),
            "playtime_2weeks": random.randint(0, 100),
            "rating": random.randint(1, 5),
            "review_text": f"Sample review {i+1}",
            "helpful_votes": random.randint(0, 50),
            "funny_votes": random.randint(0, 20)
        }
        reviews_data.append(review)
    
    # Sample bundles data
    bundles_data = []
    for i in range(50):  # 50 sample bundles
        bundle = {
            "bundle_id": f"B{i:06d}",
            "bundle_name": f"Sample Bundle {i+1}",
            "bundle_price": round(random.uniform(20, 200), 2),
            "bundle_final_price": round(random.uniform(10, 150), 2),
            "bundle_discount": f"{random.randint(10, 70)}%",
            "items": random.sample(games_data, random.randint(2, 5))
        }
        bundles_data.append(bundle)
    
    return games_data, reviews_data, bundles_data

def save_data_files(games_data, reviews_data, bundles_data):
    """Save the sample data to files."""
    
    # Save games data (compressed)
    games_file = DATA_DIR / "items_v2.json.gz"
    with gzip.open(games_file, 'wt', encoding='utf-8') as f:
        for game in games_data:
            f.write(json.dumps(game) + '\n')
    print(f"✅ Saved {len(games_data)} games to {games_file}")
    
    # Save reviews data (compressed)
    reviews_file = DATA_DIR / "reviews_v2.json.gz"
    with gzip.open(reviews_file, 'wt', encoding='utf-8') as f:
        for review in reviews_data:
            f.write(json.dumps(review) + '\n')
    print(f"✅ Saved {len(reviews_data)} reviews to {reviews_file}")
    
    # Save bundles data (uncompressed JSON)
    bundles_file = DATA_DIR / "bundles.json"
    with open(bundles_file, 'w', encoding='utf-8') as f:
        json.dump(bundles_data, f, indent=2)
    print(f"✅ Saved {len(bundles_data)} bundles to {bundles_file}")

def create_data_summary():
    """Create a summary of the generated data."""
    summary = {
        "dataset_info": {
            "name": "Sample Steam Dataset",
            "description": "Realistic sample data for Steam game recommender development",
            "generated_at": "2025-08-16"
        },
        "data_files": {
            "games": {
                "file": "items_v2.json.gz",
                "records": 1000,
                "fields": ["item_id", "item_name", "genre", "tags", "price", "release_year", "developer", "publisher", "rating", "playtime_median", "playtime_mean"]
            },
            "reviews": {
                "file": "reviews_v2.json.gz", 
                "records": 5000,
                "fields": ["user_id", "item_id", "playtime_forever", "playtime_2weeks", "rating", "review_text", "helpful_votes", "funny_votes"]
            },
            "bundles": {
                "file": "bundles.json",
                "records": 50,
                "fields": ["bundle_id", "bundle_name", "bundle_price", "bundle_final_price", "bundle_discount", "items"]
            }
        },
        "statistics": {
            "total_games": 1000,
            "total_users": 500,
            "total_reviews": 5000,
            "total_bundles": 50,
            "avg_rating": 3.0,
            "avg_playtime": 250
        }
    }
    
    summary_file = DATA_DIR / "dataset_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Created dataset summary: {summary_file}")
    
    return summary

def main():
    """Main function to create sample Steam dataset."""
    print("🎮 Creating Sample Steam Dataset...")
    print("=" * 50)
    
    # Generate sample data
    games_data, reviews_data, bundles_data = create_sample_steam_data()
    
    # Save data files
    save_data_files(games_data, reviews_data, bundles_data)
    
    # Create summary
    summary = create_data_summary()
    
    print("\n" + "=" * 50)
    print("🎉 Sample Steam Dataset Created Successfully!")
    print("=" * 50)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"- Games: {summary['statistics']['total_games']:,}")
    print(f"- Users: {summary['statistics']['total_users']:,}")
    print(f"- Reviews: {summary['statistics']['total_reviews']:,}")
    print(f"- Bundles: {summary['statistics']['total_bundles']:,}")
    
    print(f"\n📁 Files Created:")
    print(f"- {DATA_DIR / 'items_v2.json.gz'}")
    print(f"- {DATA_DIR / 'reviews_v2.json.gz'}")
    print(f"- {DATA_DIR / 'bundles.json'}")
    print(f"- {DATA_DIR / 'dataset_summary.json'}")
    
    print(f"\n🚀 Next Steps:")
    print("1. Explore the data structure")
    print("2. Preprocess and clean the data")
    print("3. Create training datasets")
    print("4. Build recommendation models")
    print("5. Integrate with your database system")

if __name__ == "__main__":
    main() 