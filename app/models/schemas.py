"""Pydantic models for the RAG API."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class FileUploadResponse(BaseModel):
    """Response model for file upload."""

    file_id: str
    filename: str
    file_size: int
    content_type: str
    upload_timestamp: datetime
    status: str = "uploaded"


class FileInfo(BaseModel):
    """File information model."""

    file_id: str
    filename: str
    file_size: int
    content_type: str
    upload_timestamp: datetime
    last_modified: datetime
    status: str
    chunks_count: Optional[int] = None


class FileListResponse(BaseModel):
    """Response model for file listing."""

    files: List[FileInfo]
    total_count: int
    page: int
    page_size: int


class SearchRequest(BaseModel):
    """Request model for RAG search."""

    query: str = Field(..., min_length=1, max_length=1000)
    max_results: Optional[int] = Field(default=10, ge=1, le=50)
    similarity_threshold: Optional[float] = Field(default=0.7, ge=0.0, le=1.0)
    file_ids: Optional[List[str]] = Field(default=None, max_items=10)


class SearchResult(BaseModel):
    """Individual search result model."""

    content: str
    file_id: str
    filename: str
    chunk_index: int
    similarity_score: float
    metadata: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    """Response model for RAG search."""

    query: str
    results: List[SearchResult]
    total_results: int
    processing_time_ms: float


class ChunkInfo(BaseModel):
    """Chunk information model."""

    chunk_id: str
    content: str
    chunk_index: int
    metadata: Dict[str, Any]


class DocumentProcessingStatus(BaseModel):
    """Document processing status model."""

    file_id: str
    status: str  # processing, completed, failed
    progress_percentage: float
    chunks_processed: int
    total_chunks: int
    error_message: Optional[str] = None
    processing_start_time: datetime
    processing_end_time: Optional[datetime] = None


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class HealthCheckResponse(BaseModel):
    """Health check response model."""

    status: str
    timestamp: datetime
    version: str
    services: Dict[str, str]  # service_name -> status
