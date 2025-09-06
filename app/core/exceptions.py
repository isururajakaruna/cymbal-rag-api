"""Custom exceptions for the RAG API."""

from typing import Optional


class RAGAPIException(Exception):
    """Base exception for RAG API."""

    def __init__(self, message: str, error_code: Optional[str] = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class DocumentProcessingError(RAGAPIException):
    """Exception raised during document processing."""

    pass


class VectorSearchError(RAGAPIException):
    """Exception raised during vector search operations."""

    pass


class StorageError(RAGAPIException):
    """Exception raised during storage operations."""

    pass


class ValidationError(RAGAPIException):
    """Exception raised during input validation."""

    pass


class ConfigurationError(RAGAPIException):
    """Exception raised due to configuration issues."""

    pass


class FileNotFoundError(RAGAPIException):
    """Exception raised when a file is not found."""

    pass


class UnsupportedFileFormatError(RAGAPIException):
    """Exception raised for unsupported file formats."""

    pass


class FileSizeExceededError(RAGAPIException):
    """Exception raised when file size exceeds limits."""

    pass
