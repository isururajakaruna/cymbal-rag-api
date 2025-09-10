"""Pytest configuration and fixtures."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from app.core.config import rag_config, settings
from app.main import app
from app.services.rag_search_service import RAGSearchService


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def mock_rag_search_service():
    """Create a mock RAG search service."""
    service = Mock(spec=RAGSearchService)
    service.search_documents = AsyncMock()
    return service


@pytest.fixture
def sample_file_content():
    """Sample file content for testing."""
    return b"This is a sample document content for testing purposes."


@pytest.fixture
def sample_pdf_content():
    """Sample PDF content for testing."""
    # This would be actual PDF bytes in a real test
    return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"


@pytest.fixture
def sample_image_content():
    """Sample image content for testing."""
    # This would be actual image bytes in a real test
    return b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xdb\x00\x00\x00\x00IEND\xaeB`\x82"


@pytest.fixture
def sample_search_request():
    """Sample search request for testing."""
    from app.models.schemas import SearchRequest

    return SearchRequest(query="test query", max_results=10, similarity_threshold=0.7)


@pytest.fixture
def sample_chunks():
    """Sample chunks for testing."""
    from app.models.schemas import ChunkInfo

    return [
        ChunkInfo(
            chunk_id="chunk_1",
            content="This is the first chunk of text.",
            chunk_index=0,
            metadata={"type": "text", "page": 1},
        ),
        ChunkInfo(
            chunk_id="chunk_2",
            content="This is the second chunk of text.",
            chunk_index=1,
            metadata={"type": "text", "page": 1},
        ),
    ]


@pytest.fixture
def sample_search_results():
    """Sample search results for testing."""
    from app.models.schemas import SearchResult

    return [
        SearchResult(
            content="This is a relevant document chunk.",
            file_id="file_1",
            filename="test.pdf",
            chunk_index=0,
            similarity_score=0.95,
            metadata={"type": "text"},
        )
    ]


@pytest.fixture
def mock_document_processor():
    """Create a mock document processor."""
    processor = Mock()
    processor.process_document = AsyncMock(return_value=[])
    return processor


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    store = Mock()
    store.add_chunks = AsyncMock(return_value=True)
    store.search_similar = AsyncMock(return_value=[])
    store.delete_file_chunks = AsyncMock(return_value=True)
    return store


@pytest.fixture
def mock_storage_service():
    """Create a mock storage service."""
    storage = Mock()
    storage.upload_file = AsyncMock(return_value="test-file-id")
    storage.download_file = AsyncMock(return_value=b"test content")
    storage.delete_file = AsyncMock(return_value=True)
    storage.list_files = AsyncMock(return_value=[])
    storage.get_file_metadata = AsyncMock(return_value={})
    return storage
