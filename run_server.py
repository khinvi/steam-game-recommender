#!/usr/bin/env python3
"""
Simple script to run the Steam Game Recommender server.
Run this from the project root directory.
"""

import sys
import os
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Now we can import from src
from core.config import settings
from infrastructure.database.connection import engine, Base
from infrastructure.database.models import User, Game, Recommendation

print("🚀 Starting Steam Game Recommender Server...")
print(f"📁 Project root: {Path(__file__).parent}")
print(f"📁 Source path: {src_path}")
print(f"🔧 Config loaded: {settings.APP_NAME}")
print(f"🗄️ Database URL: {settings.DATABASE_URL}")

# Create database tables
print("🗄️ Creating database tables...")
Base.metadata.create_all(bind=engine)
print("✅ Database tables created successfully")

# Test database connection
print("🔌 Testing database connection...")
with engine.connect() as conn:
    from sqlalchemy import text
    result = conn.execute(text("SELECT 1")).scalar()
    print(f"✅ Database connection successful: {result}")

print("\n🎮 Your Steam Game Recommender is ready!")
print("📊 Database is working and tables are created")
print("\n🔗 Next steps:")
print("1. Run: python src/simple_server.py (from src directory)")
print("2. Or run: python -m uvicorn src.simple_server:app --host 0.0.0.0 --port 8000")
print("3. Visit: http://localhost:8000/docs for API documentation")
print("4. Visit: http://localhost:8000/health for health check") 