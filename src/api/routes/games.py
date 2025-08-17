# src/api/routes/games.py - BASIC VERSION
from fastapi import APIRouter, HTTPException
from typing import List, Optional
from src.domain.entities.game import Game
from src.domain.services.game_service import GameService

router = APIRouter()
game_service = GameService()

@router.get("/", response_model=List[Game])
async def get_games(
    limit: Optional[int] = 100,
    offset: Optional[int] = 0,
    category: Optional[str] = None,
    min_rating: Optional[float] = None,
    max_price: Optional[float] = None
):
    """Get a list of games with optional filtering"""
    try:
        games = await game_service.get_games(
            limit=limit,
            offset=offset,
            category=category,
            min_rating=min_rating,
            max_price=max_price
        )
        return games
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching games: {str(e)}")

@router.get("/{game_id}", response_model=Game)
async def get_game(game_id: int):
    """Get a specific game by ID"""
    try:
        game = await game_service.get_game_by_id(game_id)
        if not game:
            raise HTTPException(status_code=404, detail="Game not found")
        return game
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching game: {str(e)}")

@router.get("/categories/list")
async def get_game_categories():
    """Get list of available game categories"""
    try:
        categories = await game_service.get_game_categories()
        return {"categories": categories}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching categories: {str(e)}")

@router.get("/search/{query}")
async def search_games(query: str, limit: Optional[int] = 20):
    """Search games by title or description"""
    try:
        games = await game_service.search_games(query, limit=limit)
        return {"results": games, "query": query, "count": len(games)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching games: {str(e)}") 