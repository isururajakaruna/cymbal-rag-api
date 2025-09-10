"""Tests for file management API endpoints."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.mark.asyncio
async def test_list_files_success(client):
    """Test successful file listing."""
    with patch("app.api.v1.files.StorageService") as mock_storage_class:
        mock_storage = AsyncMock()
        mock_files = [
            {
                "name": "test1.pdf",
                "path": "uploads/test1.pdf",
                "file_type": "application/pdf (.pdf)",
                "last_updated": "2023-01-01T00:00:00Z",
                "size": 1000,
                "tags": ["test", "document"]
            },
            {
                "name": "test2.txt",
                "path": "uploads/test2.txt",
                "file_type": "text/plain (.txt)",
                "last_updated": "2023-01-02T00:00:00Z",
                "size": 500,
                "tags": []
            }
        ]
        mock_storage.list_files.return_value = mock_files
        mock_storage_class.return_value = mock_storage

        response = client.get("/api/v1/files/list")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["files"]) == 2
        assert data["files"][0]["name"] == "test1.pdf"


@pytest.mark.asyncio
async def test_list_files_with_pagination(client):
    """Test file listing with pagination."""
    with patch("app.api.v1.files.StorageService") as mock_storage_class:
        mock_storage = AsyncMock()
        mock_files = [
            {
                "name": f"test{i}.pdf",
                "path": f"uploads/test{i}.pdf",
                "file_type": "application/pdf (.pdf)",
                "last_updated": "2023-01-01T00:00:00Z",
                "size": 1000,
                "tags": []
            }
            for i in range(5)
        ]
        mock_storage.list_files.return_value = mock_files
        mock_storage_class.return_value = mock_storage

        response = client.get("/api/v1/files/list?limit=3&offset=1")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["limit"] == 3
        assert data["offset"] == 1


@pytest.mark.asyncio
async def test_list_files_with_search(client):
    """Test file listing with search query."""
    with patch("app.api.v1.files.StorageService") as mock_storage_class:
        mock_storage = AsyncMock()
        mock_files = [
            {
                "name": "test_document.pdf",
                "path": "uploads/test_document.pdf",
                "file_type": "application/pdf (.pdf)",
                "last_updated": "2023-01-01T00:00:00Z",
                "size": 1000,
                "tags": []
            }
        ]
        mock_storage.list_files.return_value = mock_files
        mock_storage_class.return_value = mock_storage

        response = client.get("/api/v1/files/list?search=document")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["search_query"] == "document"


@pytest.mark.asyncio
async def test_list_files_with_tags(client):
    """Test file listing with tag filtering."""
    with patch("app.api.v1.files.StorageService") as mock_storage_class:
        mock_storage = AsyncMock()
        mock_files = [
            {
                "name": "test.pdf",
                "path": "uploads/test.pdf",
                "file_type": "application/pdf (.pdf)",
                "last_updated": "2023-01-01T00:00:00Z",
                "size": 1000,
                "tags": ["product", "catalog"]
            }
        ]
        mock_storage.list_files.return_value = mock_files
        mock_storage_class.return_value = mock_storage

        response = client.get("/api/v1/files/list?tags=product,catalog")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["tags_filter"] == ["product", "catalog"]


@pytest.mark.asyncio
async def test_view_file_success(client):
    """Test successful file download."""
    with patch("app.api.v1.files.StorageService") as mock_storage_class:
        mock_storage = AsyncMock()
        mock_content = b"Test file content"
        mock_storage.download_file.return_value = mock_content
        mock_storage_class.return_value = mock_storage

        response = client.get("/api/v1/files/view?filename=test.pdf")

        assert response.status_code == 200
        assert response.content == mock_content
        assert response.headers["content-type"] == "application/pdf"


@pytest.mark.asyncio
async def test_view_file_not_found(client):
    """Test file download when file not found."""
    with patch("app.api.v1.files.StorageService") as mock_storage_class:
        mock_storage = AsyncMock()
        mock_storage.download_file.side_effect = FileNotFoundError("File not found")
        mock_storage_class.return_value = mock_storage

        response = client.get("/api/v1/files/view?filename=nonexistent.pdf")

        assert response.status_code == 404
        data = response.json()
        assert data["success"] is False
        assert "not found" in data["error"].lower()


@pytest.mark.asyncio
async def test_get_embedding_stats_success(client):
    """Test successful embedding stats retrieval."""
    with patch("app.api.v1.files.StorageService") as mock_storage_class, \
         patch("app.api.v1.files.VectorSearchService") as mock_vector_class:
        
        mock_storage = AsyncMock()
        mock_storage.get_file_metadata.return_value = {
            "name": "test.pdf",
            "path": "uploads/test.pdf",
            "file_type": "application/pdf (.pdf)",
            "last_updated": "2023-01-01T00:00:00Z",
            "size": 1000
        }
        mock_storage_class.return_value = mock_storage
        
        mock_vector = AsyncMock()
        mock_vector.get_embeddings_by_metadata.return_value = {
            "total_embeddings": 5,
            "datapoint_ids": ["test_0", "test_1", "test_2", "test_3", "test_4"],
            "has_embeddings": True
        }
        mock_vector_class.return_value = mock_vector

        response = client.get("/api/v1/files/embedding-stats?filename=test.pdf")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["embedding_stats"]["total_embeddings"] == 5
        assert data["embedding_stats"]["has_embeddings"] is True


@pytest.mark.asyncio
async def test_get_embedding_stats_file_not_found(client):
    """Test embedding stats when file not found."""
    with patch("app.api.v1.files.StorageService") as mock_storage_class:
        mock_storage = AsyncMock()
        mock_storage.get_file_metadata.side_effect = FileNotFoundError("File not found")
        mock_storage_class.return_value = mock_storage

        response = client.get("/api/v1/files/embedding-stats?filename=nonexistent.pdf")

        assert response.status_code == 404
        data = response.json()
        assert data["success"] is False
        assert "not found" in data["error"].lower()