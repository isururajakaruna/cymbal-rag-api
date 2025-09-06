"""Tests for search API endpoints."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.mark.asyncio
async def test_search_documents_post_success(client, sample_search_results):
    """Test successful document search via POST."""
    with patch("app.api.v1.search.get_rag_service") as mock_get_service:
        mock_service = AsyncMock()
        mock_response = {
            "query": "test query",
            "results": sample_search_results,
            "total_results": len(sample_search_results),
            "processing_time_ms": 100.0,
        }
        mock_service.search_documents.return_value = mock_response
        mock_get_service.return_value = mock_service

        search_data = {
            "query": "test query",
            "max_results": 10,
            "similarity_threshold": 0.7,
        }
        response = client.post("/api/v1/search/", json=search_data)

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "test query"
        assert len(data["results"]) == len(sample_search_results)


@pytest.mark.asyncio
async def test_search_documents_get_success(client, sample_search_results):
    """Test successful document search via GET."""
    with patch("app.api.v1.search.get_rag_service") as mock_get_service:
        mock_service = AsyncMock()
        mock_response = {
            "query": "test query",
            "results": sample_search_results,
            "total_results": len(sample_search_results),
            "processing_time_ms": 100.0,
        }
        mock_service.search_documents.return_value = mock_response
        mock_get_service.return_value = mock_service

        response = client.get("/api/v1/search/?query=test%20query&max_results=10")

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "test query"


@pytest.mark.asyncio
async def test_search_documents_with_file_ids(client, sample_search_results):
    """Test document search with file IDs filter."""
    with patch("app.api.v1.search.get_rag_service") as mock_get_service:
        mock_service = AsyncMock()
        mock_response = {
            "query": "test query",
            "results": sample_search_results,
            "total_results": len(sample_search_results),
            "processing_time_ms": 100.0,
        }
        mock_service.search_documents.return_value = mock_response
        mock_get_service.return_value = mock_service

        response = client.get("/api/v1/search/?query=test%20query&file_ids=file1,file2")

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "test query"


@pytest.mark.asyncio
async def test_search_documents_invalid_query(client):
    """Test search with invalid query parameters."""
    # Empty query
    response = client.get("/api/v1/search/?query=")
    assert response.status_code == 422

    # Query too long
    long_query = "a" * 1001
    response = client.get(f"/api/v1/search/?query={long_query}")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_search_documents_invalid_max_results(client):
    """Test search with invalid max_results parameter."""
    # Max results too high
    response = client.get("/api/v1/search/?query=test&max_results=100")
    assert response.status_code == 422

    # Max results too low
    response = client.get("/api/v1/search/?query=test&max_results=0")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_search_documents_invalid_similarity_threshold(client):
    """Test search with invalid similarity_threshold parameter."""
    # Threshold too high
    response = client.get("/api/v1/search/?query=test&similarity_threshold=1.5")
    assert response.status_code == 422

    # Threshold too low
    response = client.get("/api/v1/search/?query=test&similarity_threshold=-0.1")
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_search_health_check(client):
    """Test search health check endpoint."""
    response = client.get("/api/v1/search/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "search"


@pytest.mark.asyncio
async def test_search_documents_service_error(client):
    """Test search when service raises an error."""
    with patch("app.api.v1.search.get_rag_service") as mock_get_service:
        mock_service = AsyncMock()
        mock_service.search_documents.side_effect = Exception("Service error")
        mock_get_service.return_value = mock_service

        search_data = {
            "query": "test query",
            "max_results": 10,
            "similarity_threshold": 0.7,
        }
        response = client.post("/api/v1/search/", json=search_data)

        assert response.status_code == 500
        assert "Service error" in response.json()["detail"]


@pytest.mark.asyncio
async def test_search_documents_post_missing_query(client):
    """Test POST search with missing query."""
    search_data = {"max_results": 10, "similarity_threshold": 0.7}
    response = client.post("/api/v1/search/", json=search_data)

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_search_documents_post_invalid_data(client):
    """Test POST search with invalid data."""
    search_data = {
        "query": "test query",
        "max_results": "invalid",
        "similarity_threshold": 0.7,
    }
    response = client.post("/api/v1/search/", json=search_data)

    assert response.status_code == 422
