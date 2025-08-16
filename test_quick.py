#!/usr/bin/env python3
"""
Quick test script for the Steam Game Recommender system.
Run this to test basic functionality without setting up pytest.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        from src.core.config import get_settings, settings
        print("✅ Config imported successfully")
        
        from src.infrastructure.database.connection import get_db, Base, engine
        print("✅ Database connection imported successfully")
        
        from src.infrastructure.database.models import User, Game, Recommendation
        print("✅ Database models imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_config():
    """Test configuration loading."""
    print("\n🔧 Testing configuration...")
    
    try:
        from src.core.config import get_settings, settings
        
        # Test settings
        assert hasattr(settings, 'APP_NAME')
        assert hasattr(settings, 'DATABASE_URL')
        assert hasattr(settings, 'DEBUG')
        
        print(f"✅ App name: {settings.APP_NAME}")
        print(f"✅ Database URL: {settings.DATABASE_URL}")
        print(f"✅ Debug mode: {settings.DEBUG}")
        
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_database_creation():
    """Test database creation and table creation."""
    print("\n🗄️ Testing database creation...")
    
    try:
        from src.infrastructure.database.connection import engine, Base
        from src.infrastructure.database.models import User, Game, Recommendation
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created successfully")
        
        # Test that we can connect
        from sqlalchemy import text
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).scalar()
            assert result == 1
            print("✅ Database connection successful")
        
        return True
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_model_creation():
    """Test creating and manipulating models."""
    print("\n🎮 Testing model creation...")
    
    try:
        from src.infrastructure.database.connection import SessionLocal
        from src.infrastructure.database.models import User, Game, Recommendation
        
        # Create a session
        db = SessionLocal()
        
        # Create a test user
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password="test_hash",
            steam_id="12345"
        )
        
        # Create a test game
        game = Game(
            steam_id="67890",
            name="Test Game",
            genres=["Action", "Adventure"],
            tags=["fun", "exciting"],
            price=29.99
        )
        
        # Add to database
        db.add(user)
        db.add(game)
        db.commit()
        
        print("✅ User and game created successfully")
        
        # Test relationships
        user.games.append(game)
        db.commit()
        
        print(f"✅ User has {len(user.games)} games")
        print(f"✅ Game has {len(game.users)} users")
        
        # Clean up
        db.delete(user)
        db.delete(game)
        db.commit()
        db.close()
        
        return True
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting Steam Game Recommender tests...\n")
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Database Creation", test_database_creation),
        ("Model Creation", test_model_creation),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your system is working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 