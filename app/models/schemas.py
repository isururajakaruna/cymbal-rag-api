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
    tags: Optional[List[str]] = Field(default=[], description="Tags associated with the file")


class FileListResponse(BaseModel):
    """Response model for file listing."""

    files: List[FileInfo]
    total_count: int
    page: int
    page_size: int


class SearchRequest(BaseModel):
    """Request model for RAG search."""

    query: str = Field(..., min_length=1, max_length=1000)
    ktop: Optional[int] = Field(default=None, ge=1, le=50, description="Number of top results to retrieve")
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Similarity threshold")
    file_ids: Optional[List[str]] = Field(default=None, max_items=10, description="Optional list of file IDs to search within")
    tags: Optional[List[str]] = Field(default=None, max_items=10, description="Optional list of tags to filter by")


class SearchResult(BaseModel):
    """Individual search result model."""

    content: str
    file_id: str
    filename: str
    chunk_index: int
    distance: float  # Distance score (lower = more similar)
    metadata: Optional[Dict[str, Any]] = None


class RAGSearchFileInfo(BaseModel):
    """File information for RAG search results."""
    name: str
    path: str
    file_type: str
    last_updated: datetime
    size: int
    tags: Optional[List[str]] = Field(default=[], description="Tags associated with the file")
    matched_chunks: List[SearchResult] = Field(default_factory=list)


class RAGSearchResponse(BaseModel):
    """Enhanced response model for RAG search with file list and RAG response."""

    success: bool
    query: str
    files: List[RAGSearchFileInfo]
    total_files: int
    total_chunks: int
    rag_response: str
    processing_time_ms: float
    search_parameters: Dict[str, Any]


class SearchResponse(BaseModel):
    """Legacy response model for RAG search."""

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


class ContentAnalysis(BaseModel):
    """Content analysis result from Gemini."""

    content_quality: Dict[str, Any]
    faq_structure: Dict[str, Any]


class FileValidationResponse(BaseModel):
    """Response model for file validation."""

    success: bool
    validation_id: Optional[str] = None
    filename: Optional[str] = None
    content_type: Optional[str] = None
    file_size: Optional[int] = None
    temp_path: Optional[str] = None
    file_exists: Optional[bool] = None
    content_analysis: Optional[ContentAnalysis] = None
    message: Optional[str] = None
    error: Optional[str] = None
    supported_extensions: Optional[List[str]] = None
    provided_type: Optional[str] = None
    provided_extension: Optional[str] = None
    quality_score: Optional[int] = None
    reasoning: Optional[str] = None
    suggestion: Optional[str] = None


class FileValidationRequest(BaseModel):
    """Request model for file validation."""

    replace_existing: bool = False
