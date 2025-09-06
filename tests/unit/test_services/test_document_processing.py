"""Tests for document processing with different file types."""

import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.core.exceptions import (DocumentProcessingError,
                                 UnsupportedFileFormatError)
from app.services.gemini_document_processor import GeminiDocumentProcessor


class MockGeminiDocumentProcessor(GeminiDocumentProcessor):
    """Mock version that doesn't require authentication."""

    def __init__(self):
        # Initialize without calling parent __init__ to avoid authentication
        from app.utils.ingestion.chunking import ChunkingStrategy
        from app.utils.ingestion.text_cleaner import TextCleaner

        self.chunking_strategy = ChunkingStrategy()
        self.text_cleaner = TextCleaner()

    async def _analyze_with_gemini(self, content: str, prompt: str) -> str:
        """Mock Gemini analysis that returns sample content."""
        if "table" in prompt.lower():
            return """
            **Table Analysis:**
            - Product Name: Apple, Price: $1.50, Quantity: 10, Total: $15.00
            - Product Name: Banana, Price: $0.75, Quantity: 20, Total: $15.00
            - Product Name: Orange, Price: $2.00, Quantity: 5, Total: $10.00
            - Product Name: Grape, Price: $3.00, Quantity: 8, Total: $24.00
            - **Total Sales: $64.00**
            """
        elif "diagram" in prompt.lower() or "flowchart" in prompt.lower():
            return """
            **Flowchart Analysis:**
            This is a RAG system flowchart showing:
            1. Document Ingestion → Text Chunking
            2. Text Chunking → Vector Embedding  
            3. Vector Embedding → Vector Storage
            4. Query Processing → Document Retrieval
            5. Document Retrieval → Response Generation
            """
        elif "business card" in prompt.lower():
            return """
            **Business Card Analysis:**
            Company: Tech Solutions Inc.
            Name: John Smith
            Title: Senior AI Engineer
            Email: john.smith@techsolutions.com
            Phone: (555) 123-4567
            Address: 123 Tech Street, Silicon Valley, CA 94000
            """
        else:
            return """
            **Document Analysis:**
            This document contains technical information about RAG systems.
            Key topics include:
            - Retrieval-Augmented Generation concepts
            - Implementation strategies
            - Benefits and use cases
            - Future trends in AI
            """

    async def _process_pdf_with_gemini(self, file_content: bytes, filename: str):
        """Mock PDF processing."""
        from pdf2image import convert_from_bytes

        from app.models.schemas import ChunkInfo

        try:
            # Convert PDF to images
            images = convert_from_bytes(file_content)
            chunks = []

            for i, image in enumerate(images):
                # Mock analysis for each page
                prompt = "Analyze this document page for tables, text, and diagrams."
                analysis = await self._analyze_with_gemini("", prompt)

                chunk = ChunkInfo(
                    chunk_id=f"{filename}_page_{i+1}",
                    content=analysis,
                    chunk_index=i,
                    metadata={
                        "type": "pdf_page",
                        "processor": "gemini_mock",
                        "filename": filename,
                        "page_number": i + 1,
                        "total_pages": len(images),
                    },
                )
                chunks.append(chunk)

            return chunks
        except Exception as e:
            raise DocumentProcessingError(f"Failed to process PDF {filename}: {str(e)}")

    async def _process_image_with_gemini(self, file_content: bytes, filename: str):
        """Mock image processing."""
        from app.models.schemas import ChunkInfo

        try:
            # Mock analysis for image
            prompt = "Analyze this image for text, tables, or diagrams."
            analysis = await self._analyze_with_gemini("", prompt)

            chunk = ChunkInfo(
                chunk_id=f"{filename}_image",
                content=analysis,
                chunk_index=0,
                metadata={
                    "type": "image",
                    "processor": "gemini_mock",
                    "filename": filename,
                },
            )

            return [chunk]
        except Exception as e:
            raise DocumentProcessingError(
                f"Failed to process image {filename}: {str(e)}"
            )


@pytest.fixture
def mock_processor():
    """Create mock document processor for testing."""
    return MockGeminiDocumentProcessor()


@pytest.fixture
def test_data_dir():
    """Get the test data directory."""
    return Path(__file__).parent.parent.parent.parent / "test_data"


@pytest.fixture
def verify_test_data_files(test_data_dir):
    """Verify that test_data directory exists and contains expected files."""
    assert test_data_dir.exists(), f"test_data directory not found: {test_data_dir}"

    # Check for expected subdirectories
    expected_dirs = ["tables", "images", "excel", "text"]
    for subdir in expected_dirs:
        subdir_path = test_data_dir / subdir
        if not subdir_path.exists():
            print(f"Warning: {subdir} directory not found in test_data")

    return test_data_dir


@pytest.mark.asyncio
async def test_process_pdf_document(mock_processor, test_data_dir):
    """Test PDF document processing with Gemini multimodal analysis.

    This test verifies that PDF files are correctly:
    - Converted from PDF to images using pdf2image
    - Processed with mock Gemini analysis for each page
    - Chunked with proper metadata including page numbers
    - Handled with appropriate error handling
    """
    pdf_file = test_data_dir / "tables" / "simple_table.pdf"

    if not pdf_file.exists():
        pytest.skip(f"Test PDF file not found: {pdf_file}")

    with open(pdf_file, "rb") as f:
        file_content = f.read()

    chunks = await mock_processor.process_document(
        file_content, "simple_table.pdf", "application/pdf"
    )

    assert len(chunks) > 0
    assert all(chunk.chunk_id for chunk in chunks)
    assert all(chunk.content for chunk in chunks)
    assert all(chunk.metadata["type"] == "pdf_page" for chunk in chunks)
    assert all(chunk.metadata["processor"] == "gemini_mock" for chunk in chunks)


@pytest.mark.asyncio
async def test_process_image_document(mock_processor, test_data_dir):
    """Test image document processing with Gemini vision capabilities.

    This test verifies that image files are correctly:
    - Processed with mock Gemini vision analysis
    - Analyzed for text, tables, and diagram content
    - Chunked as single chunks with image metadata
    - Handled with appropriate content type detection
    """
    image_file = test_data_dir / "images" / "image_with_text.png"

    if not image_file.exists():
        pytest.skip(f"Test image file not found: {image_file}")

    with open(image_file, "rb") as f:
        file_content = f.read()

    chunks = await mock_processor.process_document(
        file_content, "image_with_text.png", "image/png"
    )

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "image_with_text.png_image"
    assert chunks[0].content
    assert chunks[0].metadata["type"] == "image"
    assert chunks[0].metadata["processor"] == "gemini_mock"


@pytest.mark.asyncio
async def test_process_excel_document(mock_processor, test_data_dir):
    """Test Excel document processing with pandas data extraction.

    This test verifies that Excel files are correctly:
    - Read using pandas with proper sheet detection
    - Processed to extract structured data from multiple sheets
    - Converted to descriptive text format for chunking
    - Chunked with sheet-specific metadata and statistics
    """
    excel_file = test_data_dir / "excel" / "sample_data.xlsx"

    if not excel_file.exists():
        pytest.skip(f"Test Excel file not found: {excel_file}")

    with open(excel_file, "rb") as f:
        file_content = f.read()

    chunks = await mock_processor.process_document(
        file_content,
        "sample_data.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    assert len(chunks) > 0
    assert all(chunk.chunk_id for chunk in chunks)
    assert all(chunk.content for chunk in chunks)
    assert all(chunk.metadata["type"] == "excel_sheet" for chunk in chunks)
    assert all(chunk.metadata["processor"] == "pandas" for chunk in chunks)


@pytest.mark.asyncio
async def test_process_text_document(mock_processor, test_data_dir):
    """Test text document processing with standard text chunking.

    This test verifies that text documents are correctly:
    - Processed as PDF files (since our test data uses PDF format)
    - Converted to images and analyzed with Gemini
    - Chunked with appropriate text processing metadata
    - Handled with proper content type detection
    """
    text_file = test_data_dir / "text" / "sample_text.pdf"

    if not text_file.exists():
        pytest.skip(f"Test text file not found: {text_file}")

    with open(text_file, "rb") as f:
        file_content = f.read()

    chunks = await mock_processor.process_document(
        file_content, "sample_text.pdf", "application/pdf"
    )

    assert len(chunks) > 0
    assert all(chunk.chunk_id for chunk in chunks)
    assert all(chunk.content for chunk in chunks)
    assert all(chunk.metadata["type"] == "pdf_page" for chunk in chunks)


@pytest.mark.asyncio
async def test_unsupported_file_format(mock_processor):
    """Test error handling for unsupported file formats.

    This test verifies that the system correctly:
    - Detects unsupported file formats and content types
    - Raises appropriate DocumentProcessingError exceptions
    - Provides meaningful error messages for debugging
    - Handles edge cases gracefully without crashing
    """
    with pytest.raises(DocumentProcessingError):  # The exception gets wrapped
        await mock_processor.process_document(
            b"some content", "test.xyz", "application/unknown"
        )


@pytest.mark.asyncio
async def test_chunking_strategy_selection(mock_processor, test_data_dir):
    """Test intelligent chunking strategy selection based on file types.

    This test verifies that the system correctly:
    - Selects PDF processing strategy for PDF files (page-by-page analysis)
    - Selects image processing strategy for image files (single chunk analysis)
    - Uses appropriate processing methods for each content type
    - Maintains consistent metadata structure across different strategies
    """
    # Test PDF chunking strategy with real PDF from test_data
    pdf_file = test_data_dir / "tables" / "simple_table.pdf"

    if not pdf_file.exists():
        pytest.skip(f"Test PDF file not found: {pdf_file}")

    with open(pdf_file, "rb") as f:
        pdf_content = f.read()

    pdf_chunks = await mock_processor._process_pdf_with_gemini(
        pdf_content, "simple_table.pdf"
    )
    assert len(pdf_chunks) > 0
    assert all(chunk.metadata["type"] == "pdf_page" for chunk in pdf_chunks)

    # Test image chunking strategy with real image from test_data
    image_file = test_data_dir / "images" / "image_with_text.png"

    if not image_file.exists():
        pytest.skip(f"Test image file not found: {image_file}")

    with open(image_file, "rb") as f:
        image_content = f.read()

    image_chunks = await mock_processor._process_image_with_gemini(
        image_content, "image_with_text.png"
    )
    assert len(image_chunks) == 1
    assert image_chunks[0].metadata["type"] == "image"


@pytest.mark.asyncio
async def test_chunk_metadata_structure(mock_processor, test_data_dir):
    """Test comprehensive chunk metadata structure and validation.

    This test verifies that all chunks have proper metadata including:
    - Required fields: type, processor, filename
    - PDF-specific metadata: page_number, total_pages
    - Consistent metadata structure across different file types
    - Proper data types and values for all metadata fields
    """
    pdf_file = test_data_dir / "tables" / "simple_table.pdf"

    if not pdf_file.exists():
        pytest.skip(f"Test PDF file not found: {pdf_file}")

    with open(pdf_file, "rb") as f:
        file_content = f.read()

    chunks = await mock_processor.process_document(
        file_content, "simple_table.pdf", "application/pdf"
    )

    for chunk in chunks:
        # Check required fields
        assert "type" in chunk.metadata
        assert "processor" in chunk.metadata
        assert "filename" in chunk.metadata

        # Check PDF-specific metadata
        if chunk.metadata["type"] == "pdf_page":
            assert "page_number" in chunk.metadata
            assert "total_pages" in chunk.metadata


@pytest.mark.asyncio
async def test_chunk_content_quality(mock_processor, test_data_dir):
    """Test chunk content quality and meaningful data extraction.

    This test verifies that chunk content is:
    - Non-empty and properly formatted
    - Contains structured information relevant to the source
    - Includes appropriate keywords for the content type
    - Maintains data integrity through the processing pipeline
    """
    excel_file = test_data_dir / "excel" / "sample_data.xlsx"

    if not excel_file.exists():
        pytest.skip(f"Test Excel file not found: {excel_file}")

    with open(excel_file, "rb") as f:
        file_content = f.read()

    chunks = await mock_processor.process_document(
        file_content,
        "sample_data.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    for chunk in chunks:
        # Content should not be empty
        assert chunk.content.strip()

        # Content should contain structured information
        assert any(
            keyword in chunk.content.lower()
            for keyword in ["sheet", "column", "row", "data", "dimensions"]
        )

        # Metadata should contain sheet information
        if chunk.metadata["type"] == "excel_sheet":
            assert "sheet_name" in chunk.metadata
            assert "rows" in chunk.metadata
            assert "columns" in chunk.metadata


@pytest.mark.asyncio
async def test_multiple_file_types_processing(mock_processor, test_data_dir):
    """Test end-to-end processing of multiple file types in sequence.

    This comprehensive test verifies that the system can:
    - Process different file types (PDF, image, Excel) in sequence
    - Maintain consistent chunk structure across file types
    - Handle mixed content types without conflicts
    - Provide appropriate metadata for each processed file type
    - Scale processing capabilities for diverse document collections
    """
    test_files = [
        ("tables/simple_table.pdf", "application/pdf"),
        ("images/image_with_text.png", "image/png"),
        (
            "excel/sample_data.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ),
    ]

    results = {}

    for file_path, content_type in test_files:
        full_path = test_data_dir / file_path

        if not full_path.exists():
            print(f"Skipping {file_path} - file not found in test_data")
            continue

        with open(full_path, "rb") as f:
            file_content = f.read()

        filename = full_path.name
        chunks = await mock_processor.process_document(
            file_content, filename, content_type
        )

        results[filename] = {
            "chunks_count": len(chunks),
            "content_types": [chunk.metadata["type"] for chunk in chunks],
            "processors": [chunk.metadata["processor"] for chunk in chunks],
        }

    # Verify we processed at least one file from test_data
    assert len(results) > 0, "No files from test_data folder were processed"

    # Verify each file type was processed correctly
    for filename, result in results.items():
        assert result["chunks_count"] > 0
        assert all(
            ct in ["pdf_page", "image", "excel_sheet"] for ct in result["content_types"]
        )
        assert all(p in ["gemini_mock", "pandas"] for p in result["processors"])
