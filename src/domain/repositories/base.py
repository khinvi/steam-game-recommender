"""Base repository interface for the Steam Game Recommender application."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List, Optional, Any, Dict
from pydantic import BaseModel

# Generic type for entities
T = TypeVar('T', bound=BaseModel)


class BaseRepository(ABC, Generic[T]):
    """Base repository interface for CRUD operations."""
    
    @abstractmethod
    async def create(self, entity: T) -> T:
        """Create a new entity."""
        pass
    
    @abstractmethod
    async def get_by_id(self, entity_id: Any) -> Optional[T]:
        """Get an entity by ID."""
        pass
    
    @abstractmethod
    async def get_all(self, skip: int = 0, limit: int = 100) -> List[T]:
        """Get all entities with pagination."""
        pass
    
    @abstractmethod
    async def update(self, entity_id: Any, entity: T) -> Optional[T]:
        """Update an entity."""
        pass
    
    @abstractmethod
    async def delete(self, entity_id: Any) -> bool:
        """Delete an entity."""
        pass
    
    @abstractmethod
    async def count(self) -> int:
        """Get the total count of entities."""
        pass
    
    @abstractmethod
    async def exists(self, entity_id: Any) -> bool:
        """Check if an entity exists."""
        pass 