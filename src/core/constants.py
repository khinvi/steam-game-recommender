"""Application constants and configuration values."""

# API Constants
API_V1_STR = "/api/v1"
PROJECT_NAME = "Steam Game Recommender"
VERSION = "0.1.0"
DESCRIPTION = "A comprehensive Steam game recommendation system with ML models and web API"

# Security Constants
SECURITY_BCRYPT_ROUNDS = 12
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Steam API Constants
STEAM_API_BASE_URL = "https://store.steampowered.com/api"
STEAM_STORE_BASE_URL = "https://store.steampowered.com"
STEAM_COMMUNITY_BASE_URL = "https://steamcommunity.com"

# Game Categories
GAME_CATEGORIES = [
    "Action", "Adventure", "RPG", "Strategy", "Simulation", "Sports",
    "Racing", "Puzzle", "Indie", "Casual", "Arcade", "Platformer",
    "Shooter", "Fighting", "Stealth", "Survival", "Horror", "Visual Novel"
]

# Game Tags
POPULAR_GAME_TAGS = [
    "Multiplayer", "Single-player", "Co-op", "Competitive", "Story Rich",
    "Open World", "Sandbox", "Roguelike", "Roguelite", "Metroidvania",
    "RPG", "Strategy", "Simulation", "Sports", "Racing", "Puzzle",
    "Indie", "Casual", "Arcade", "Platformer", "Shooter", "Fighting"
]

# ML Model Constants
DEFAULT_EMBEDDING_DIMENSION = 128
DEFAULT_LATENT_FACTORS = 50
DEFAULT_TOP_K_RECOMMENDATIONS = 10
MODEL_CACHE_TTL = 3600  # 1 hour in seconds

# Cache Constants
CACHE_TTL_DEFAULT = 300  # 5 minutes
CACHE_TTL_USER_PROFILE = 1800  # 30 minutes
CACHE_TTL_GAME_DATA = 3600  # 1 hour
CACHE_TTL_RECOMMENDATIONS = 1800  # 30 minutes

# Rate Limiting Constants
RATE_LIMIT_PER_MINUTE = 60
RATE_LIMIT_PER_HOUR = 1000
RATE_LIMIT_PER_DAY = 10000

# Database Constants
MAX_CONNECTION_POOL_SIZE = 20
CONNECTION_POOL_TIMEOUT = 30
CONNECTION_POOL_RECYCLE = 3600

# File Paths
DEFAULT_DATA_DIR = "./data"
DEFAULT_MODELS_DIR = "./data/models"
DEFAULT_CACHE_DIR = "./data/cache"
DEFAULT_VECTOR_STORE_DIR = "./data/vector_store"

# HTTP Status Messages
HTTP_200_MESSAGE = "Success"
HTTP_201_MESSAGE = "Created"
HTTP_400_MESSAGE = "Bad Request"
HTTP_401_MESSAGE = "Unauthorized"
HTTP_403_MESSAGE = "Forbidden"
HTTP_404_MESSAGE = "Not Found"
HTTP_422_MESSAGE = "Validation Error"
HTTP_429_MESSAGE = "Too Many Requests"
HTTP_500_MESSAGE = "Internal Server Error"

# Error Messages
ERROR_INVALID_CREDENTIALS = "Invalid credentials"
ERROR_INSUFFICIENT_PERMISSIONS = "Insufficient permissions"
ERROR_RESOURCE_NOT_FOUND = "Resource not found"
ERROR_VALIDATION_FAILED = "Validation failed"
ERROR_RATE_LIMIT_EXCEEDED = "Rate limit exceeded"
ERROR_INTERNAL_SERVER = "Internal server error"
ERROR_EXTERNAL_API_FAILED = "External API call failed"

# Success Messages
SUCCESS_USER_CREATED = "User created successfully"
SUCCESS_USER_UPDATED = "User updated successfully"
SUCCESS_USER_DELETED = "User deleted successfully"
SUCCESS_RECOMMENDATIONS_GENERATED = "Recommendations generated successfully"
SUCCESS_GAME_ADDED = "Game added successfully"
SUCCESS_GAME_UPDATED = "Game updated successfully" 