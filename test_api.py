#!/usr/bin/env python
"""
Test script to verify the Steam Game Recommender API works with processed data.
"""

import requests
import time
import subprocess
import sys
from pathlib import Path

def test_api_health():
    """Test if the API is responding."""
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("✅ API is responding")
            return True
        else:
            print(f"❌ API returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API - server may not be running")
        return False

def test_recommendations_endpoint():
    """Test the recommendations endpoint."""
    try:
        # Test getting popular games
        response = requests.get("http://localhost:8000/recommendations/popular")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Popular recommendations endpoint working - returned {len(data)} games")
            return True
        else:
            print(f"❌ Recommendations endpoint returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to recommendations endpoint")
        return False

def test_games_endpoint():
    """Test the games endpoint."""
    try:
        response = requests.get("http://localhost:8000/games/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Games endpoint working - returned {len(data)} games")
            return True
        else:
            print(f"❌ Games endpoint returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to games endpoint")
        return False

def main():
    """Main test function."""
    print("🧪 Testing Steam Game Recommender API...")
    print("=" * 50)
    
    # Check if processed data exists
    processed_dir = Path("data/processed")
    if not processed_dir.exists():
        print("❌ Processed data directory not found")
        print("Please run: python scripts/process_steam_data.py")
        return False
    
    # Check if models exist
    models_dir = Path("data/models")
    if not models_dir.exists():
        print("❌ Models directory not found")
        print("Please run: python scripts/build_recommendation_models.py")
        return False
    
    print("✅ Data and models found")
    
    # Try to start the API server
    print("\n🚀 Starting API server...")
    try:
        # Start server in background
        server_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "src.main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Wait for server to start
        print("⏳ Waiting for server to start...")
        time.sleep(5)
        
        # Test API endpoints
        print("\n🔍 Testing API endpoints...")
        
        if test_api_health():
            if test_games_endpoint():
                if test_recommendations_endpoint():
                    print("\n🎉 All tests passed! API is working correctly.")
                    print("\n📱 You can now:")
                    print("- Access the API at: http://localhost:8000")
                    print("- View API docs at: http://localhost:8000/docs")
                    print("- Start the frontend with: cd frontend && npm start")
                    
                    # Keep server running for user to test
                    print("\n🔄 API server is running. Press Ctrl+C to stop...")
                    try:
                        server_process.wait()
                    except KeyboardInterrupt:
                        print("\n🛑 Stopping server...")
                        server_process.terminate()
                        server_process.wait()
                        print("✅ Server stopped")
                    
                    return True
                else:
                    print("❌ Recommendations endpoint test failed")
            else:
                print("❌ Games endpoint test failed")
        else:
            print("❌ API health check failed")
        
        # Clean up
        server_process.terminate()
        server_process.wait()
        return False
        
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 