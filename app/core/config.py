"""Configuration management for the RAG API."""

import json
from pathlib import Path
from typing import Any, Dict, List

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Google Cloud Configuration
    google_cloud_project_id: str = Field(..., env="GOOGLE_CLOUD_PROJECT_ID")
    google_application_credentials: str = Field(
        ..., env="GOOGLE_APPLICATION_CREDENTIALS"
    )
    google_cloud_region: str = Field(default="us-central1", env="GOOGLE_CLOUD_REGION")

    # Document AI Configuration
    document_ai_processor_id: str = Field(..., env="DOCUMENT_AI_PROCESSOR_ID")
    document_ai_location: str = Field(default="us", env="DOCUMENT_AI_LOCATION")

    # Vertex AI Configuration
    vertex_ai_location: str = Field(default="us-central1", env="VERTEX_AI_LOCATION")
    vertex_ai_model_name: str = Field(
        default="gemini-2.5-flash", env="VERTEX_AI_MODEL_NAME"
    )
    vertex_ai_embedding_model_name: str = Field(
        default="gemini-embedding-001", env="VERTEX_AI_EMBEDDING_MODEL_NAME"
    )

    # Vector Search Configuration
    vector_search_index_id: str = Field(..., env="VECTOR_SEARCH_INDEX_ID")
    vector_search_index_endpoint_id: str = Field(
        ..., env="VECTOR_SEARCH_INDEX_ENDPOINT_ID"
    )
    vector_search_deployed_index_id: str = Field(..., env="VECTOR_SEARCH_DEPLOYED_INDEX_ID")

    # Storage Configuration
    storage_bucket_name: str = Field(..., env="STORAGE_BUCKET_NAME")

    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_workers: int = Field(default=1, env="API_WORKERS")

    class Config:
        env_file = ".env"
        case_sensitive = False


class RAGConfig:
    """RAG-specific configuration loaded from config.json."""

    def __init__(self, config_path: str = "config/config.json"):
        self.config_path = Path(config_path)
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")

    @property
    def rag(self) -> Dict[str, Any]:
        """RAG configuration parameters."""
        return self._config.get("rag", {})

    @property
    def document_processing(self) -> Dict[str, Any]:
        """Document processing configuration."""
        return self._config.get("document_processing", {})

    @property
    def vector_search(self) -> Dict[str, Any]:
        """Vector search configuration."""
        return self._config.get("vector_search", {})

    @property
    def api(self) -> Dict[str, Any]:
        """API configuration."""
        return self._config.get("api", {})

    @property
    def chunk_size(self) -> int:
        """Text chunk size for processing."""
        return self.rag.get("chunk_size", 1000)

    @property
    def chunk_overlap(self) -> int:
        """Overlap between chunks."""
        return self.rag.get("chunk_overlap", 200)

    @property
    def max_chunks_per_document(self) -> int:
        """Maximum chunks per document."""
        return self.rag.get("max_chunks_per_document", 50)

    @property
    def similarity_threshold(self) -> float:
        """Similarity threshold for search results."""
        return self.rag.get("similarity_threshold", 0.7)

    @property
    def max_results(self) -> int:
        """Maximum number of search results."""
        return self.rag.get("max_results", 10)

    @property
    def supported_formats(self) -> List[str]:
        """Supported file formats."""
        return self.document_processing.get("supported_formats", ["pdf", "txt"])

    @property
    def max_file_size_mb(self) -> int:
        """Maximum file size in MB."""
        return self.document_processing.get("max_file_size_mb", 10)

    @property
    def ocr_enabled(self) -> bool:
        """Whether OCR is enabled."""
        return self.document_processing.get("ocr_enabled", True)

    @property
    def table_extraction_enabled(self) -> bool:
        """Whether table extraction is enabled."""
        return self.document_processing.get("table_extraction_enabled", True)


# Global configuration instances
settings = Settings()
rag_config = RAGConfig()
