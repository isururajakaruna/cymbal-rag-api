"""API dependencies for dependency injection."""

from functools import lru_cache

from app.core.config import rag_config, settings
from app.services.rag_service import RAGService


@lru_cache()
def get_rag_service() -> RAGService:
    """Get RAG service instance."""
    return RAGService()


@lru_cache()
def get_settings():
    """Get application settings."""
    return settings


@lru_cache()
def get_rag_config():
    """Get RAG configuration."""
    return rag_config
