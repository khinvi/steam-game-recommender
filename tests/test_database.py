"""Tests for database connection and models."""

import pytest
import tempfile
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.infrastructure.database.connection import get_db, Base
from src.infrastructure.database.models import User, Game, Recommendation, user_games


class TestDatabaseConnection:
    """Test database connection and session management."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        # Create temporary file
        fd, path = tempfile.mkstemp()
        os.close(fd)
        
        # Create engine for temp database
        engine = create_engine(f"sqlite:///{path}")
        Base.metadata.create_all(bind=engine)
        
        yield path
        
        # Cleanup
        os.unlink(path)
    
    def test_database_creation(self, temp_db):
        """Test that database can be created."""
        engine = create_engine(f"sqlite:///{temp_db}")
        assert engine is not None
        
        # Test that tables can be created
        Base.metadata.create_all(bind=engine)
        
        # Check that tables exist
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        assert "users" in tables
        assert "games" in tables
        assert "recommendations" in tables
        assert "user_games" in tables
    
    def test_session_context_manager(self, temp_db):
        """Test the get_db context manager."""
        engine = create_engine(f"sqlite:///{temp_db}")
        Base.metadata.create_all(bind=engine)
        
        # Mock the SessionLocal to use our temp database
        from src.infrastructure.database.connection import SessionLocal
        original_session = SessionLocal
        
        try:
            SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
            
            # Test context manager
            with get_db() as db:
                assert db is not None
                # Test that we can query
                from sqlalchemy import text
                result = db.execute(text("SELECT 1")).scalar()
                assert result == 1
                
        finally:
            # Restore original SessionLocal
            SessionLocal = original_session


class TestDatabaseModels:
    """Test database models and relationships."""
    
    @pytest.fixture
    def db_session(self):
        """Create a database session for testing."""
        # Create in-memory database
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        
        Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        session = Session()
        
        yield session
        
        session.close()
    
    def test_user_creation(self, db_session):
        """Test creating a user."""
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password="hashed_password_123",
            steam_id="12345"
        )
        
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
        
        assert user.id is not None
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.steam_id == "12345"
        assert user.created_at is not None
    
    def test_game_creation(self, db_session):
        """Test creating a game."""
        game = Game(
            steam_id="123456",
            name="Test Game",
            genres=["Action", "Adventure"],
            tags=["fun", "exciting"],
            price=29.99
        )
        
        db_session.add(game)
        db_session.commit()
        db_session.refresh(game)
        
        assert game.id is not None
        assert game.steam_id == "123456"
        assert game.name == "Test Game"
        assert game.genres == ["Action", "Adventure"]
        assert game.tags == ["fun", "exciting"]
        assert game.price == 29.99
    
    def test_user_game_relationship(self, db_session):
        """Test the many-to-many relationship between users and games."""
        # Create user and game
        user = User(username="testuser", email="test@example.com", hashed_password="hash")
        game = Game(steam_id="123", name="Test Game")
        
        db_session.add_all([user, game])
        db_session.commit()
        
        # Add game to user
        user.games.append(game)
        db_session.commit()
        
        # Test relationship
        assert len(user.games) == 1
        assert user.games[0].name == "Test Game"
        assert len(game.users) == 1
        assert game.users[0].username == "testuser"
    
    def test_recommendation_creation(self, db_session):
        """Test creating a recommendation."""
        # Create user and game first
        user = User(username="testuser", email="test@example.com", hashed_password="hash")
        game = Game(steam_id="123", name="Test Game")
        
        db_session.add_all([user, game])
        db_session.commit()
        
        # Create recommendation
        recommendation = Recommendation(
            user_id=user.id,
            game_id=game.id,
            score=0.85,
            type="personalized"
        )
        
        db_session.add(recommendation)
        db_session.commit()
        db_session.refresh(recommendation)
        
        assert recommendation.id is not None
        assert recommendation.user_id == user.id
        assert recommendation.game_id == game.id
        assert recommendation.score == 0.85
        assert recommendation.type == "personalized"
        assert recommendation.created_at is not None


class TestDatabaseConstraints:
    """Test database constraints and validation."""
    
    @pytest.fixture
    def db_session(self):
        """Create a database session for testing."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        
        Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        session = Session()
        
        yield session
        
        session.close()
    
    def test_unique_username_constraint(self, db_session):
        """Test that usernames must be unique."""
        user1 = User(username="testuser", email="test1@example.com", hashed_password="hash")
        user2 = User(username="testuser", email="test2@example.com", hashed_password="hash")
        
        db_session.add(user1)
        db_session.commit()
        
        # Adding user with same username should fail
        db_session.add(user2)
        with pytest.raises(Exception):  # SQLAlchemy will raise an integrity error
            db_session.commit()
    
    def test_unique_email_constraint(self, db_session):
        """Test that emails must be unique."""
        user1 = User(username="user1", email="test@example.com", hashed_password="hash")
        user2 = User(username="user2", email="test@example.com", hashed_password="hash")
        
        db_session.add(user1)
        db_session.commit()
        
        # Adding user with same email should fail
        db_session.add(user2)
        with pytest.raises(Exception):  # SQLAlchemy will raise an integrity error
            db_session.commit() 