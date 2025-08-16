"""Redis cache implementation for the Steam Game Recommender application."""

import json
import pickle
from typing import Optional, Any
from datetime import timedelta
import redis
from cachetools import TTLCache
from src.core.config import settings
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    """
    Hybrid cache manager - uses Redis if available, falls back to in-memory
    """
    
    def __init__(self):
        self.redis_client = None
        self.memory_cache = TTLCache(maxsize=1000, ttl=300)  # 5 min TTL
        
        if settings.USE_REDIS and settings.redis_url:
            try:
                self.redis_client = redis.from_url(settings.redis_url)
                self.redis_client.ping()
                logger.info("Connected to Redis cache")
            except Exception as e:
                logger.warning(f"Redis connection failed, using in-memory cache: {e}")
                self.redis_client = None
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if self.redis_client:
            try:
                value = self.redis_client.get(key)
                if value:
                    return pickle.loads(value)
            except Exception as e:
                logger.error(f"Redis get error: {e}")
        
        # Fallback to memory cache
        return self.memory_cache.get(key)
    
    async def set(self, key: str, value: Any, expire: int = 300):
        """Set value in cache with expiration"""
        if self.redis_client:
            try:
                self.redis_client.setex(
                    key,
                    expire,
                    pickle.dumps(value)
                )
                return
            except Exception as e:
                logger.error(f"Redis set error: {e}")
        
        # Fallback to memory cache
        self.memory_cache[key] = value
    
    async def delete(self, key: str):
        """Delete key from cache"""
        if self.redis_client:
            try:
                self.redis_client.delete(key)
            except Exception as e:
                logger.error(f"Redis delete error: {e}")
        
        if key in self.memory_cache:
            del self.memory_cache[key]
    
    async def clear(self):
        """Clear all cache"""
        if self.redis_client:
            try:
                self.redis_client.flushdb()
            except Exception as e:
                logger.error(f"Redis flush error: {e}")
        
        self.memory_cache.clear()

# Global cache instance
cache = CacheManager() 