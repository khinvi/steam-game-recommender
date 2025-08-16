from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey, Table
from sqlalchemy.orm import relationship
from datetime import datetime
from .connection import Base

# Association tables
user_games = Table(
    'user_games',
    Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id')),
    Column('game_id', Integer, ForeignKey('games.id')),
    Column('playtime', Integer, default=0),
    Column('rating', Float, nullable=True)
)

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    steam_id = Column(String, nullable=True, unique=True)
    preferences = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    
    games = relationship("Game", secondary=user_games, back_populates="users")
    recommendations = relationship("Recommendation", back_populates="user")

class Game(Base):
    __tablename__ = "games"
    
    id = Column(Integer, primary_key=True, index=True)
    steam_id = Column(String, unique=True, index=True)
    name = Column(String)
    genres = Column(JSON, default=[])
    tags = Column(JSON, default=[])
    price = Column(Float, default=0.0)
    game_data = Column(JSON, default={})  # Changed from 'metadata' to 'game_data'
    
    users = relationship("User", secondary=user_games, back_populates="games")

class Recommendation(Base):
    __tablename__ = "recommendations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    game_id = Column(Integer, ForeignKey('games.id'))
    score = Column(Float)
    type = Column(String)  # personalized, similar, trending
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="recommendations") 