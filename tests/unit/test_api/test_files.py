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
async def test_upload_file_success(client, sample_file_content):
    """Test successful file upload."""
    with patch("app.api.v1.files.get_rag_service") as mock_get_service:
        mock_service = AsyncMock()
        mock_service.process_and_store_document.return_value = "test-file-id"
        mock_get_service.return_value = mock_service

        files = {"file": ("test.txt", sample_file_content, "text/plain")}
        response = client.post("/api/v1/files/upload", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["file_id"] == "test-file-id"
        assert data["filename"] == "test.txt"
        assert data["status"] == "uploaded"


@pytest.mark.asyncio
async def test_upload_file_unsupported_format(client, sample_file_content):
    """Test file upload with unsupported format."""
    files = {"file": ("test.xyz", sample_file_content, "application/xyz")}
    response = client.post("/api/v1/files/upload", files=files)

    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


@pytest.mark.asyncio
async def test_upload_file_missing_filename(client, sample_file_content):
    """Test file upload without filename."""
    files = {"file": (None, sample_file_content, "text/plain")}
    response = client.post("/api/v1/files/upload", files=files)

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_update_file_success(client, sample_file_content):
    """Test successful file update."""
    with patch("app.api.v1.files.get_rag_service") as mock_get_service:
        mock_service = AsyncMock()
        mock_service.update_document.return_value = True
        mock_get_service.return_value = mock_service

        files = {"file": ("updated.txt", sample_file_content, "text/plain")}
        response = client.put("/api/v1/files/test-file-id", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["file_id"] == "test-file-id"
        assert data["filename"] == "updated.txt"
        assert data["status"] == "updated"


@pytest.mark.asyncio
async def test_update_file_not_found(client, sample_file_content):
    """Test file update when file not found."""
    with patch("app.api.v1.files.get_rag_service") as mock_get_service:
        mock_service = AsyncMock()
        mock_service.update_document.return_value = False
        mock_get_service.return_value = mock_service

        files = {"file": ("updated.txt", sample_file_content, "text/plain")}
        response = client.put("/api/v1/files/nonexistent-file-id", files=files)

        assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_file_success(client):
    """Test successful file deletion."""
    with patch("app.api.v1.files.get_rag_service") as mock_get_service:
        mock_service = AsyncMock()
        mock_service.delete_document.return_value = True
        mock_get_service.return_value = mock_service

        response = client.delete("/api/v1/files/test-file-id")

        assert response.status_code == 200
        data = response.json()
        assert "deleted successfully" in data["message"]


@pytest.mark.asyncio
async def test_delete_file_not_found(client):
    """Test file deletion when file not found."""
    with patch("app.api.v1.files.get_rag_service") as mock_get_service:
        mock_service = AsyncMock()
        mock_service.delete_document.return_value = False
        mock_get_service.return_value = mock_service

        response = client.delete("/api/v1/files/nonexistent-file-id")

        assert response.status_code == 404


@pytest.mark.asyncio
async def test_list_files_success(client):
    """Test successful file listing."""
    with patch("app.api.v1.files.get_rag_service") as mock_get_service:
        mock_service = AsyncMock()
        mock_files = [
            {
                "file_id": "file1",
                "filename": "test1.txt",
                "size": 100,
                "content_type": "text/plain",
                "created": "2023-01-01T00:00:00Z",
                "updated": "2023-01-01T00:00:00Z",
            }
        ]
        mock_service.list_documents.return_value = mock_files
        mock_get_service.return_value = mock_service

        response = client.get("/api/v1/files/")

        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 1
        assert len(data["files"]) == 1
        assert data["files"][0]["file_id"] == "file1"


@pytest.mark.asyncio
async def test_list_files_pagination(client):
    """Test file listing with pagination."""
    with patch("app.api.v1.files.get_rag_service") as mock_get_service:
        mock_service = AsyncMock()
        mock_files = [
            {
                "file_id": f"file{i}",
                "filename": f"test{i}.txt",
                "size": 100,
                "content_type": "text/plain",
                "created": "2023-01-01T00:00:00Z",
                "updated": "2023-01-01T00:00:00Z",
            }
            for i in range(5)
        ]
        mock_service.list_documents.return_value = mock_files
        mock_get_service.return_value = mock_service

        response = client.get("/api/v1/files/?page=1&page_size=2")

        assert response.status_code == 200
        data = response.json()
        assert data["page"] == 1
        assert data["page_size"] == 2
        assert len(data["files"]) == 2


@pytest.mark.asyncio
async def test_get_file_info_not_implemented(client):
    """Test get file info endpoint (not implemented)."""
    response = client.get("/api/v1/files/test-file-id")

    assert response.status_code == 501


@pytest.mark.asyncio
async def test_get_processing_status(client):
    """Test get processing status endpoint."""
    response = client.get("/api/v1/files/test-file-id/status")

    assert response.status_code == 200
    data = response.json()
    assert data["file_id"] == "test-file-id"
    assert data["status"] == "completed"
