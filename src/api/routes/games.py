# src/api/routes/games.py - BASIC VERSION
from fastapi import APIRouter, HTTPException

router = APIRouter()

@router.get("/{game_id}")
async def get_game(game_id: str):
    # Mock data for now
    return {"game_id": game_id, "name": f"Game {game_id}"}

@router.get("/search")
async def search_games(q: str):
    # Mock search
    return {"results": []} 