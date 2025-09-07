"""
Chunking service for the Cymbal RAG API.

This module provides functionality to chunk documents into smaller pieces
for better embedding and retrieval performance.
"""

from typing import List, Dict, Any
from app.core.config import rag_config
from app.models.schemas import ChunkInfo
import re

class ChunkingService:
    """Service for chunking documents into smaller pieces."""
    
    def __init__(self):
        """Initialize the chunking service."""
        self.chunk_size = rag_config.chunk_size
        self.chunk_overlap = rag_config.chunk_overlap
    
    def chunk_document(
        self, 
        content: str, 
        metadata: Dict[str, Any], 
        chunk_size: int = None, 
        chunk_overlap: int = None
    ) -> List[ChunkInfo]:
        """
        Chunk a document into smaller pieces.
        
        Args:
            content: The document content to chunk
            metadata: Metadata associated with the document
            chunk_size: Size of each chunk (overrides config if provided)
            chunk_overlap: Overlap between chunks (overrides config if provided)
            
        Returns:
            List of ChunkInfo objects representing the chunks
        """
        if chunk_size is None:
            chunk_size = self.chunk_size
        if chunk_overlap is None:
            chunk_overlap = self.chunk_overlap
        
        # Clean and prepare content
        content = self._clean_content(content)
        
        # Split content into chunks
        chunks = self._split_content(content, chunk_size, chunk_overlap)
        
        # Create ChunkInfo objects
        chunk_infos = []
        for i, chunk_content in enumerate(chunks):
            chunk_info = ChunkInfo(
                chunk_id=f"{metadata.get('document_id', 'unknown')}_{i}",
                chunk_index=i,
                content=chunk_content,
                metadata={
                    **metadata,
                    "chunk_size": len(chunk_content),
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            )
            chunk_infos.append(chunk_info)
        
        return chunk_infos
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize content before chunking."""
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove special characters that might interfere with chunking
        content = content.strip()
        
        return content
    
    def _split_content(self, content: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """
        Split content into chunks with overlap.
        
        Args:
            content: Content to split
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks
            
        Returns:
            List of chunk strings
        """
        if len(content) <= chunk_size:
            return [content]
        
        chunks = []
        start = 0
        
        while start < len(content):
            # Calculate end position
            end = start + chunk_size
            
            # If this is not the last chunk, try to break at a sentence boundary
            if end < len(content):
                # Look for sentence endings within the last 100 characters
                search_start = max(start, end - 100)
                sentence_end = self._find_sentence_boundary(content, search_start, end)
                
                if sentence_end > start:
                    end = sentence_end
            
            # Extract chunk
            chunk = content[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - chunk_overlap
            if start >= len(content):
                break
        
        return chunks
    
    def _find_sentence_boundary(self, content: str, start: int, end: int) -> int:
        """
        Find a good sentence boundary within the given range.
        
        Args:
            content: Content to search in
            start: Start position
            end: End position
            
        Returns:
            Position of sentence boundary, or end if none found
        """
        # Look for sentence endings
        sentence_endings = ['.', '!', '?', '\n\n']
        
        for i in range(end - 1, start, -1):
            if content[i] in sentence_endings:
                # Make sure it's followed by whitespace or end of string
                if i + 1 >= len(content) or content[i + 1].isspace():
                    return i + 1
        
        return end
    
    def chunk_by_paragraphs(self, content: str, metadata: Dict[str, Any]) -> List[ChunkInfo]:
        """
        Chunk content by paragraphs.
        
        Args:
            content: Content to chunk
            metadata: Metadata associated with the document
            
        Returns:
            List of ChunkInfo objects
        """
        # Split by double newlines (paragraph breaks)
        paragraphs = content.split('\n\n')
        
        chunks = []
        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if paragraph:
                chunk_info = ChunkInfo(
                    chunk_id=f"{metadata.get('document_id', 'unknown')}_para_{i}",
                    chunk_index=i,
                    content=paragraph,
                    metadata={
                        **metadata,
                        "chunk_type": "paragraph",
                        "chunk_size": len(paragraph),
                        "chunk_index": i,
                        "total_chunks": len(paragraphs)
                    }
                )
                chunks.append(chunk_info)
        
        return chunks
    
    def chunk_by_sentences(self, content: str, metadata: Dict[str, Any], max_sentences: int = 3) -> List[ChunkInfo]:
        """
        Chunk content by sentences, grouping multiple sentences together.
        
        Args:
            content: Content to chunk
            metadata: Metadata associated with the document
            max_sentences: Maximum number of sentences per chunk
            
        Returns:
            List of ChunkInfo objects
        """
        # Split by sentence endings
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        for i in range(0, len(sentences), max_sentences):
            chunk_sentences = sentences[i:i + max_sentences]
            chunk_content = '. '.join(chunk_sentences)
            
            if chunk_content:
                chunk_info = ChunkInfo(
                    chunk_id=f"{metadata.get('document_id', 'unknown')}_sent_{i}",
                    chunk_index=i // max_sentences,
                    content=chunk_content,
                    metadata={
                        **metadata,
                        "chunk_type": "sentences",
                        "chunk_size": len(chunk_content),
                        "chunk_index": i // max_sentences,
                        "total_chunks": (len(sentences) + max_sentences - 1) // max_sentences,
                        "sentence_count": len(chunk_sentences)
                    }
                )
                chunks.append(chunk_info)
        
        return chunks
