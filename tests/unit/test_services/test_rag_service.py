"""Tests for RAG service."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.core.exceptions import RAGAPIException
from app.models.schemas import ChunkInfo, SearchRequest
from app.services.rag_service import RAGService


@pytest.fixture
def rag_service():
    """Create RAG service instance for testing."""
    with patch("app.services.rag_service.DocumentProcessor"), patch(
        "app.services.rag_service.VectorStore"
    ), patch("app.services.rag_service.StorageService"), patch(
        "app.services.rag_service.TextGenerationModel"
    ):
        return RAGService()


@pytest.mark.asyncio
async def test_process_and_store_document_success(rag_service, sample_file_content):
    """Test successful document processing and storage."""
    # Mock the service methods
    rag_service.storage_service.upload_file = AsyncMock(return_value="test-file-id")
    rag_service.document_processor.process_document = AsyncMock(return_value=[])
    rag_service.vector_store.add_chunks = AsyncMock(return_value=True)

    result = await rag_service.process_and_store_document(
        file_content=sample_file_content, filename="test.txt", content_type="text/plain"
    )

    assert result == "test-file-id"
    rag_service.storage_service.upload_file.assert_called_once()
    rag_service.document_processor.process_document.assert_called_once()
    rag_service.vector_store.add_chunks.assert_called_once()


@pytest.mark.asyncio
async def test_process_and_store_document_failure(rag_service, sample_file_content):
    """Test document processing failure with cleanup."""
    # Mock storage upload to succeed but processing to fail
    rag_service.storage_service.upload_file = AsyncMock(return_value="test-file-id")
    rag_service.document_processor.process_document = AsyncMock(
        side_effect=Exception("Processing failed")
    )
    rag_service.storage_service.delete_file = AsyncMock(return_value=True)

    with pytest.raises(RAGAPIException):
        await rag_service.process_and_store_document(
            file_content=sample_file_content,
            filename="test.txt",
            content_type="text/plain",
        )

    # Verify cleanup was called
    rag_service.storage_service.delete_file.assert_called_once_with("test-file-id")


@pytest.mark.asyncio
async def test_search_documents_success(
    rag_service, sample_search_request, sample_search_results
):
    """Test successful document search."""
    # Mock the search methods
    rag_service.vector_store.search_similar = AsyncMock(
        return_value=sample_search_results
    )
    rag_service.generation_model.predict = Mock(
        return_value=Mock(text="Generated response")
    )

    result = await rag_service.search_documents(sample_search_request)

    assert result.query == sample_search_request.query
    assert len(result.results) == len(sample_search_results)
    rag_service.vector_store.search_similar.assert_called_once()


@pytest.mark.asyncio
async def test_search_documents_no_results(rag_service, sample_search_request):
    """Test document search with no results."""
    # Mock empty search results
    rag_service.vector_store.search_similar = AsyncMock(return_value=[])

    result = await rag_service.search_documents(sample_search_request)

    assert result.query == sample_search_request.query
    assert len(result.results) == 0
    assert (
        "No relevant documents found" in result.results[0].content
        if result.results
        else True
    )


@pytest.mark.asyncio
async def test_update_document_success(rag_service, sample_file_content):
    """Test successful document update."""
    # Mock the update methods
    rag_service.vector_store.delete_file_chunks = AsyncMock(return_value=True)
    rag_service.document_processor.process_document = AsyncMock(return_value=[])
    rag_service.storage_service.upload_file = AsyncMock(return_value="test-file-id")
    rag_service.vector_store.add_chunks = AsyncMock(return_value=True)

    result = await rag_service.update_document(
        file_id="test-file-id",
        file_content=sample_file_content,
        filename="updated.txt",
        content_type="text/plain",
    )

    assert result is True
    rag_service.vector_store.delete_file_chunks.assert_called_once()
    rag_service.document_processor.process_document.assert_called_once()
    rag_service.storage_service.upload_file.assert_called_once()
    rag_service.vector_store.add_chunks.assert_called_once()


@pytest.mark.asyncio
async def test_delete_document_success(rag_service):
    """Test successful document deletion."""
    # Mock the delete methods
    rag_service.vector_store.delete_file_chunks = AsyncMock(return_value=True)
    rag_service.storage_service.delete_file = AsyncMock(return_value=True)

    result = await rag_service.delete_document("test-file-id")

    assert result is True
    rag_service.vector_store.delete_file_chunks.assert_called_once_with("test-file-id")
    rag_service.storage_service.delete_file.assert_called_once_with("test-file-id")


@pytest.mark.asyncio
async def test_list_documents_success(rag_service):
    """Test successful document listing."""
    # Mock the list method
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
    rag_service.storage_service.list_files = AsyncMock(return_value=mock_files)

    result = await rag_service.list_documents()

    assert len(result) == 1
    assert result[0]["file_id"] == "file1"
    rag_service.storage_service.list_files.assert_called_once()


@pytest.mark.asyncio
async def test_build_context(rag_service, sample_search_results):
    """Test context building from search results."""
    context = rag_service._build_context(sample_search_results)

    assert "Document 1" in context
    assert "test.pdf" in context
    assert "relevant document chunk" in context


@pytest.mark.asyncio
async def test_generate_response_success(rag_service):
    """Test successful response generation."""
    # Mock the generation model
    mock_response = Mock()
    mock_response.text = "Generated answer"
    rag_service.generation_model.predict = Mock(return_value=mock_response)

    result = await rag_service._generate_response("test query", "test context")

    assert result == "Generated answer"
    rag_service.generation_model.predict.assert_called_once()


@pytest.mark.asyncio
async def test_generate_response_failure(rag_service):
    """Test response generation failure."""
    # Mock the generation model to raise an exception
    rag_service.generation_model.predict = Mock(
        side_effect=Exception("Generation failed")
    )

    with pytest.raises(RAGAPIException):
        await rag_service._generate_response("test query", "test context")
