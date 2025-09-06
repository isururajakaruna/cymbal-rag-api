"""Document processing service using Google Document AI."""

import io
from typing import Any, Dict, List, Optional

import PyPDF2
from google.cloud import documentai, storage
from PIL import Image

from app.core.config import settings
from app.core.exceptions import (DocumentProcessingError,
                                 UnsupportedFileFormatError)
from app.models.schemas import ChunkInfo


class DocumentProcessor:
    """Service for processing documents using Google Document AI."""

    def __init__(self):
        self.client = documentai.DocumentProcessorServiceClient()
        self.storage_client = storage.Client()
        self.processor_name = f"projects/{settings.google_cloud_project_id}/locations/{settings.document_ai_location}/processors/{settings.document_ai_processor_id}"

    async def process_document(
        self, file_content: bytes, filename: str, content_type: str
    ) -> List[ChunkInfo]:
        """
        Process a document and extract text chunks.

        Args:
            file_content: Raw file content
            filename: Name of the file
            content_type: MIME type of the file

        Returns:
            List of ChunkInfo objects
        """
        try:
            # Determine processing method based on file type
            if content_type == "application/pdf":
                return await self._process_pdf(file_content, filename)
            elif content_type.startswith("image/"):
                return await self._process_image(file_content, filename)
            elif content_type == "text/plain":
                return await self._process_text(file_content, filename)
            else:
                raise UnsupportedFileFormatError(
                    f"Unsupported file format: {content_type}"
                )

        except Exception as e:
            raise DocumentProcessingError(
                f"Failed to process document {filename}: {str(e)}"
            )

    async def _process_pdf(self, file_content: bytes, filename: str) -> List[ChunkInfo]:
        """Process PDF documents using Document AI."""
        try:
            # Create document request
            raw_document = documentai.RawDocument(
                content=file_content, mime_type="application/pdf"
            )

            request = documentai.ProcessRequest(
                name=self.processor_name, raw_document=raw_document
            )

            # Process the document
            result = self.client.process_document(request=request)
            document = result.document

            # Extract text and structure
            chunks = []
            page_num = 0

            for page in document.pages:
                page_num += 1

                # Extract paragraphs
                for paragraph in page.paragraphs:
                    text = self._get_text_from_layout(paragraph.layout, document.text)
                    if text.strip():
                        chunks.append(
                            ChunkInfo(
                                chunk_id=f"{filename}_page{page_num}_para{len(chunks)}",
                                content=text.strip(),
                                chunk_index=len(chunks),
                                metadata={
                                    "page_number": page_num,
                                    "type": "paragraph",
                                    "confidence": paragraph.layout.confidence,
                                },
                            )
                        )

                # Extract tables
                for table in page.tables:
                    table_text = self._extract_table_text(table, document.text)
                    if table_text.strip():
                        chunks.append(
                            ChunkInfo(
                                chunk_id=f"{filename}_page{page_num}_table{len([c for c in chunks if c.metadata.get('type') == 'table'])}",
                                content=table_text.strip(),
                                chunk_index=len(chunks),
                                metadata={
                                    "page_number": page_num,
                                    "type": "table",
                                    "confidence": table.layout.confidence
                                    if hasattr(table, "layout")
                                    else 0.0,
                                },
                            )
                        )

            return chunks

        except Exception as e:
            raise DocumentProcessingError(f"Failed to process PDF {filename}: {str(e)}")

    async def _process_image(
        self, file_content: bytes, filename: str
    ) -> List[ChunkInfo]:
        """Process image documents using Document AI OCR."""
        try:
            # Create document request
            raw_document = documentai.RawDocument(
                content=file_content,
                mime_type="image/png",  # Document AI will auto-detect
            )

            request = documentai.ProcessRequest(
                name=self.processor_name, raw_document=raw_document
            )

            # Process the document
            result = self.client.process_document(request=request)
            document = result.document

            # Extract text from OCR
            chunks = []
            if document.text.strip():
                chunks.append(
                    ChunkInfo(
                        chunk_id=f"{filename}_ocr_text",
                        content=document.text.strip(),
                        chunk_index=0,
                        metadata={
                            "type": "ocr_text",
                            "confidence": document.pages[0].layout.confidence
                            if document.pages
                            else 0.0,
                        },
                    )
                )

            return chunks

        except Exception as e:
            raise DocumentProcessingError(
                f"Failed to process image {filename}: {str(e)}"
            )

    async def _process_text(
        self, file_content: bytes, filename: str
    ) -> List[ChunkInfo]:
        """Process plain text documents."""
        try:
            text = file_content.decode("utf-8")

            # Simple chunking for text files
            chunks = []
            lines = text.split("\n")
            current_chunk = []
            chunk_size = 0
            max_chunk_size = 1000  # From config

            for line in lines:
                if chunk_size + len(line) > max_chunk_size and current_chunk:
                    chunks.append(
                        ChunkInfo(
                            chunk_id=f"{filename}_chunk{len(chunks)}",
                            content="\n".join(current_chunk),
                            chunk_index=len(chunks),
                            metadata={"type": "text_chunk"},
                        )
                    )
                    current_chunk = [line]
                    chunk_size = len(line)
                else:
                    current_chunk.append(line)
                    chunk_size += len(line)

            # Add remaining content
            if current_chunk:
                chunks.append(
                    ChunkInfo(
                        chunk_id=f"{filename}_chunk{len(chunks)}",
                        content="\n".join(current_chunk),
                        chunk_index=len(chunks),
                        metadata={"type": "text_chunk"},
                    )
                )

            return chunks

        except Exception as e:
            raise DocumentProcessingError(
                f"Failed to process text file {filename}: {str(e)}"
            )

    def _get_text_from_layout(
        self, layout: documentai.Document.Page.Layout, text: str
    ) -> str:
        """Extract text from a layout element."""
        if not layout.text_anchor:
            return ""

        start_index = layout.text_anchor.text_segments[0].start_index
        end_index = layout.text_anchor.text_segments[0].end_index

        return text[start_index:end_index]

    def _extract_table_text(
        self, table: documentai.Document.Page.Table, text: str
    ) -> str:
        """Extract text from a table in a structured format."""
        table_text = []

        for row in table.body_rows:
            row_text = []
            for cell in row.cells:
                cell_text = self._get_text_from_layout(cell.layout, text)
                row_text.append(cell_text.strip())
            table_text.append(" | ".join(row_text))

        return "\n".join(table_text)
