"""Tests for RAG search API endpoints."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.mark.asyncio
async def test_rag_search_documents_post_success(client, sample_search_results):
    """Test successful RAG document search via POST."""
    with patch("app.api.v1.search.RAGSearchService") as mock_rag_service_class:
        mock_service = AsyncMock()
        mock_response = {
            "success": True,
            "query": "test query",
            "files": [],
            "total_files": 0,
            "total_chunks": len(sample_search_results),
            "rag_response": "Generated response",
            "processing_time_ms": 100.0,
            "search_parameters": {}
        }
        mock_service.search_documents.return_value = mock_response
        mock_rag_service_class.return_value = mock_service

        search_data = {
            "query": "test query",
            "ktop": 10,
            "threshold": 0.7,
        }
        response = client.post("/api/v1/search/rag", json=search_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["query"] == "test query"
        assert data["total_chunks"] == len(sample_search_results)


@pytest.mark.asyncio
async def test_rag_search_documents_get_success(client, sample_search_results):
    """Test successful RAG document search via GET."""
    with patch("app.api.v1.search.RAGSearchService") as mock_rag_service_class:
        mock_service = AsyncMock()
        mock_response = {
            "success": True,
            "query": "test query",
            "files": [],
            "total_files": 0,
            "total_chunks": len(sample_search_results),
            "rag_response": "Generated response",
            "processing_time_ms": 100.0,
            "search_parameters": {}
        }
        mock_service.search_documents.return_value = mock_response
        mock_rag_service_class.return_value = mock_service

        response = client.get("/api/v1/search/rag?query=test%20query&ktop=10&threshold=0.7")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["query"] == "test query"


@pytest.mark.asyncio
async def test_rag_search_documents_with_file_ids(client, sample_search_results):
    """Test RAG document search with file IDs filter."""
    with patch("app.api.v1.search.RAGSearchService") as mock_rag_service_class:
        mock_service = AsyncMock()
        mock_response = {
            "success": True,
            "query": "test query",
            "files": [],
            "total_files": 0,
            "total_chunks": len(sample_search_results),
            "rag_response": "Generated response",
            "processing_time_ms": 100.0,
            "search_parameters": {"file_ids": ["file1.pdf", "file2.pdf"]}
        }
        mock_service.search_documents.return_value = mock_response
        mock_rag_service_class.return_value = mock_service

        search_data = {
            "query": "test query",
            "ktop": 10,
            "threshold": 0.7,
            "file_ids": ["file1.pdf", "file2.pdf"]
        }
        response = client.post("/api/v1/search/rag", json=search_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["query"] == "test query"


@pytest.mark.asyncio
async def test_rag_search_documents_with_tags(client, sample_search_results):
    """Test RAG document search with tags filter."""
    with patch("app.api.v1.search.RAGSearchService") as mock_rag_service_class:
        mock_service = AsyncMock()
        mock_response = {
            "success": True,
            "query": "test query",
            "files": [],
            "total_files": 0,
            "total_chunks": len(sample_search_results),
            "rag_response": "Generated response",
            "processing_time_ms": 100.0,
            "search_parameters": {"tags": ["product", "catalog"]}
        }
        mock_service.search_documents.return_value = mock_response
        mock_rag_service_class.return_value = mock_service

        search_data = {
            "query": "test query",
            "ktop": 10,
            "threshold": 0.7,
            "tags": ["product", "catalog"]
        }
        response = client.post("/api/v1/search/rag", json=search_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["query"] == "test query"


@pytest.mark.asyncio
async def test_rag_search_documents_service_error(client):
    """Test RAG document search with service error."""
    with patch("app.api.v1.search.RAGSearchService") as mock_rag_service_class:
        mock_service = AsyncMock()
        mock_service.search_documents.side_effect = Exception("Service error")
        mock_rag_service_class.return_value = mock_service

        search_data = {
            "query": "test query",
            "ktop": 10,
            "threshold": 0.7,
        }
        response = client.post("/api/v1/search/rag", json=search_data)

        assert response.status_code == 500
        data = response.json()
        assert "Internal server error" in data["detail"]


def test_search_health_check(client):
    """Test search health check endpoint."""
    response = client.get("/api/v1/search/health")
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "search"