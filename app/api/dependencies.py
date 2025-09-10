"""API dependencies for dependency injection."""

from functools import lru_cache

from app.core.config import rag_config, settings


@lru_cache()
def get_settings():
    """Get application settings."""
    return settings


@lru_cache()
def get_rag_config():
    """Get RAG configuration."""
    return rag_config
