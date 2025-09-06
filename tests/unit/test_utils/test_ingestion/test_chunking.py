"""Tests for chunking utilities."""

import pytest

from app.models.schemas import ChunkInfo
from app.utils.ingestion.chunking import ChunkingStrategy


@pytest.fixture
def chunking_strategy():
    """Create chunking strategy instance for testing."""
    return ChunkingStrategy()


@pytest.mark.asyncio
async def test_chunk_text_basic(chunking_strategy):
    """Test basic text chunking."""
    text = "This is a test document. " * 100  # Create a long text
    chunks = await chunking_strategy.chunk_text(text)

    assert len(chunks) > 0
    assert all(isinstance(chunk, ChunkInfo) for chunk in chunks)
    assert all(chunk.content.strip() for chunk in chunks)


@pytest.mark.asyncio
async def test_chunk_text_empty(chunking_strategy):
    """Test chunking empty text."""
    chunks = await chunking_strategy.chunk_text("")
    assert len(chunks) == 0


@pytest.mark.asyncio
async def test_chunk_text_short(chunking_strategy):
    """Test chunking short text."""
    text = "Short text"
    chunks = await chunking_strategy.chunk_text(text)

    assert len(chunks) == 1
    assert chunks[0].content == text
    assert chunks[0].chunk_index == 0


@pytest.mark.asyncio
async def test_chunk_table_basic(chunking_strategy):
    """Test basic table chunking."""
    table_text = "Header1 | Header2 | Header3\nValue1 | Value2 | Value3\nValue4 | Value5 | Value6"
    chunks = await chunking_strategy.chunk_table(table_text)

    assert len(chunks) > 0
    assert all(chunk.metadata["type"] == "table_chunk" for chunk in chunks)


@pytest.mark.asyncio
async def test_chunk_table_empty(chunking_strategy):
    """Test chunking empty table."""
    chunks = await chunking_strategy.chunk_table("")
    assert len(chunks) == 0


@pytest.mark.asyncio
async def test_chunk_paragraphs_basic(chunking_strategy):
    """Test paragraph-based chunking."""
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = await chunking_strategy.chunk_paragraphs(text)

    assert len(chunks) > 0
    assert all(chunk.metadata["type"] == "paragraph_chunk" for chunk in chunks)


@pytest.mark.asyncio
async def test_optimize_chunks_under_limit(chunking_strategy, sample_chunks):
    """Test chunk optimization when under limit."""
    optimized = await chunking_strategy.optimize_chunks(sample_chunks, 10)
    assert len(optimized) == len(sample_chunks)


@pytest.mark.asyncio
async def test_optimize_chunks_over_limit(chunking_strategy):
    """Test chunk optimization when over limit."""
    # Create many chunks
    chunks = [
        ChunkInfo(
            chunk_id=f"chunk_{i}",
            content=f"Content {i}",
            chunk_index=i,
            metadata={"type": "text"},
        )
        for i in range(20)
    ]

    optimized = await chunking_strategy.optimize_chunks(chunks, 5)
    assert len(optimized) <= 5


@pytest.mark.asyncio
async def test_get_chunking_strategy(chunking_strategy):
    """Test chunking strategy selection."""
    assert chunking_strategy.get_chunking_strategy("application/pdf") == "mixed"
    assert chunking_strategy.get_chunking_strategy("text/plain") == "paragraphs"
    assert chunking_strategy.get_chunking_strategy("image/png") == "single"
    assert chunking_strategy.get_chunking_strategy("unknown") == "text"


@pytest.mark.asyncio
async def test_chunk_text_with_metadata(chunking_strategy):
    """Test chunking with metadata."""
    text = "Test content"
    metadata = {"source": "test", "page": 1}
    chunks = await chunking_strategy.chunk_text(text, metadata)

    assert len(chunks) == 1
    assert chunks[0].metadata["source"] == "test"
    assert chunks[0].metadata["page"] == 1


@pytest.mark.asyncio
async def test_chunk_table_with_metadata(chunking_strategy):
    """Test table chunking with metadata."""
    table_text = "Header1 | Header2\nValue1 | Value2"
    metadata = {"source": "table", "page": 1}
    chunks = await chunking_strategy.chunk_table(table_text, metadata)

    assert len(chunks) > 0
    assert all(chunk.metadata["source"] == "table" for chunk in chunks)
    assert all(chunk.metadata["page"] == 1 for chunk in chunks)
