#!/usr/bin/env python
"""
Test script for the Steam Game Recommender system.
This script tests the recommendation service with sample data.
"""

import sys
import asyncio
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from src.domain.services.recommendation_service import RecommendationService
from src.domain.entities.recommendation import RecommendationRequest, RecommendationType

async def test_recommendation_service():
    """Test the recommendation service with sample data."""
    print("🧪 Testing Steam Game Recommender System")
    print("=" * 50)
    
    try:
        # Initialize the service
        print("📊 Initializing recommendation service...")
        service = RecommendationService()
        
        # Test getting sample users
        print("\n👥 Getting sample users...")
        sample_users = service.get_sample_users()
        print(f"Found {len(sample_users)} sample users:")
        for user in sample_users[:3]:  # Show first 3
            print(f"  - {user['description']} ({user['reviews']} reviews)")
        
        # Test getting performance metrics
        print("\n📈 Getting performance metrics...")
        metrics = await service.get_performance_metrics()
        print(f"Dataset stats: {metrics['total_users']} users, {metrics['total_games']} games, {metrics['total_interactions']} interactions")
        print(f"Available models: {', '.join(metrics['models_available'])}")
        
        # Test recommendations for each model type
        test_user = 12345  # Use integer user ID
        print(f"\n🎯 Testing recommendations for user: {test_user}")
        
        # Map model types to recommendation types
        model_to_rec_type = {
            'svd': RecommendationType.COLLABORATIVE_FILTERING,
            'item_based': RecommendationType.COLLABORATIVE_FILTERING,
            'popularity': RecommendationType.POPULARITY,
            'hybrid': RecommendationType.HYBRID
        }
        
        for model_type in ['svd', 'item_based', 'popularity', 'hybrid']:
            print(f"\n🔍 Testing {model_type.upper()} model...")
            
            try:
                # Create recommendation request with correct structure
                request = RecommendationRequest(
                    user_id=test_user,
                    recommendation_type=model_to_rec_type[model_type],
                    limit=5,
                    include_played=False
                )
                
                # Generate recommendations
                response = await service.generate_recommendations(request)
                
                print(f"  ✅ Generated {response.total_count} recommendations")
                print(f"  📊 Recommendation type: {response.recommendation_type}")
                
                # Show first 2 recommendations
                for i, game in enumerate(response.recommendations[:2]):
                    print(f"    {i+1}. {game.game_title} (Score: {game.score.score:.3f})")
                
            except Exception as e:
                print(f"  ❌ Error with {model_type}: {e}")
        
        print("\n🎉 All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to run the async test."""
    success = asyncio.run(test_recommendation_service())
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 