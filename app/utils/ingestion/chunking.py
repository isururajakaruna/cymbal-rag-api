"""Text chunking strategies for different document types."""

import re
from typing import Any, Dict, List

from app.core.config import rag_config
from app.models.schemas import ChunkInfo


class ChunkingStrategy:
    """Different chunking strategies for various content types."""

    def __init__(self):
        self.chunk_size = rag_config.chunk_size
        self.chunk_overlap = rag_config.chunk_overlap

    async def chunk_text(
        self, text: str, metadata: Dict[str, Any] = None
    ) -> List[ChunkInfo]:
        """
        Basic text chunking with fixed size and overlap.

        Args:
            text: Text to chunk
            metadata: Additional metadata for chunks

        Returns:
            List of ChunkInfo objects
        """
        if not text.strip():
            return []

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            # Try to break at sentence boundary
            if end < len(text):
                sentence_end = text.rfind(".", start, end)
                if sentence_end > start + self.chunk_size // 2:
                    end = sentence_end + 1

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    ChunkInfo(
                        chunk_id=f"chunk_{chunk_index}",
                        content=chunk_text,
                        chunk_index=chunk_index,
                        metadata={
                            "type": "text_chunk",
                            "start_pos": start,
                            "end_pos": end,
                            **(metadata or {}),
                        },
                    )
                )
                chunk_index += 1

            # Move start position with overlap
            start = max(start + self.chunk_size - self.chunk_overlap, end)

        return chunks

    async def chunk_table(
        self, table_text: str, metadata: Dict[str, Any] = None
    ) -> List[ChunkInfo]:
        """
        Chunk table content preserving table structure.

        Args:
            table_text: Table text to chunk
            metadata: Additional metadata for chunks

        Returns:
            List of ChunkInfo objects
        """
        if not table_text.strip():
            return []

        lines = table_text.split("\n")
        chunks = []
        chunk_index = 0
        current_chunk = []
        current_size = 0

        for line in lines:
            line_size = len(line)

            # If adding this line would exceed chunk size, finalize current chunk
            if current_size + line_size > self.chunk_size and current_chunk:
                chunks.append(
                    ChunkInfo(
                        chunk_id=f"table_chunk_{chunk_index}",
                        content="\n".join(current_chunk),
                        chunk_index=chunk_index,
                        metadata={
                            "type": "table_chunk",
                            "row_count": len(current_chunk),
                            **(metadata or {}),
                        },
                    )
                )
                chunk_index += 1
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size

        # Add remaining content
        if current_chunk:
            chunks.append(
                ChunkInfo(
                    chunk_id=f"table_chunk_{chunk_index}",
                    content="\n".join(current_chunk),
                    chunk_index=chunk_index,
                    metadata={
                        "type": "table_chunk",
                        "row_count": len(current_chunk),
                        **(metadata or {}),
                    },
                )
            )

        return chunks

    async def chunk_paragraphs(
        self, text: str, metadata: Dict[str, Any] = None
    ) -> List[ChunkInfo]:
        """
        Chunk text by paragraphs, respecting paragraph boundaries.

        Args:
            text: Text to chunk
            metadata: Additional metadata for chunks

        Returns:
            List of ChunkInfo objects
        """
        if not text.strip():
            return []

        # Split by double newlines (paragraphs)
        paragraphs = re.split(r"\n\s*\n", text)
        chunks = []
        chunk_index = 0
        current_chunk = []
        current_size = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            paragraph_size = len(paragraph)

            # If adding this paragraph would exceed chunk size, finalize current chunk
            if current_size + paragraph_size > self.chunk_size and current_chunk:
                chunks.append(
                    ChunkInfo(
                        chunk_id=f"para_chunk_{chunk_index}",
                        content="\n\n".join(current_chunk),
                        chunk_index=chunk_index,
                        metadata={
                            "type": "paragraph_chunk",
                            "paragraph_count": len(current_chunk),
                            **(metadata or {}),
                        },
                    )
                )
                chunk_index += 1
                current_chunk = [paragraph]
                current_size = paragraph_size
            else:
                current_chunk.append(paragraph)
                current_size += paragraph_size

        # Add remaining content
        if current_chunk:
            chunks.append(
                ChunkInfo(
                    chunk_id=f"para_chunk_{chunk_index}",
                    content="\n\n".join(current_chunk),
                    chunk_index=chunk_index,
                    metadata={
                        "type": "paragraph_chunk",
                        "paragraph_count": len(current_chunk),
                        **(metadata or {}),
                    },
                )
            )

        return chunks

    async def optimize_chunks(
        self, chunks: List[ChunkInfo], max_chunks: int
    ) -> List[ChunkInfo]:
        """
        Optimize chunks to fit within maximum chunk limit.

        Args:
            chunks: List of chunks to optimize
            max_chunks: Maximum number of chunks allowed

        Returns:
            Optimized list of chunks
        """
        if len(chunks) <= max_chunks:
            return chunks

        # Strategy: Merge smaller chunks together
        optimized_chunks = []
        current_chunk = None
        current_size = 0
        chunk_index = 0

        for chunk in chunks:
            chunk_size = len(chunk.content)

            # If we can merge with current chunk and stay under limit
            if (
                current_chunk
                and current_size + chunk_size <= self.chunk_size
                and len(optimized_chunks) < max_chunks - 1
            ):
                # Merge chunks
                current_chunk.content += "\n\n" + chunk.content
                current_size += chunk_size
                current_chunk.metadata["merged_chunks"] = (
                    current_chunk.metadata.get("merged_chunks", 1) + 1
                )
            else:
                # Finalize current chunk
                if current_chunk:
                    optimized_chunks.append(current_chunk)

                # Start new chunk
                current_chunk = ChunkInfo(
                    chunk_id=f"optimized_chunk_{chunk_index}",
                    content=chunk.content,
                    chunk_index=chunk_index,
                    metadata=chunk.metadata.copy(),
                )
                current_size = chunk_size
                chunk_index += 1

        # Add final chunk
        if current_chunk:
            optimized_chunks.append(current_chunk)

        return optimized_chunks[:max_chunks]

    def get_chunking_strategy(self, content_type: str) -> str:
        """
        Get the appropriate chunking strategy for a content type.

        Args:
            content_type: MIME type of the content

        Returns:
            Strategy name
        """
        if content_type == "application/pdf":
            return "mixed"  # Will be determined by content analysis
        elif content_type.startswith("text/"):
            return "paragraphs"
        elif content_type.startswith("image/"):
            return "single"  # OCR text is usually short
        else:
            return "text"
