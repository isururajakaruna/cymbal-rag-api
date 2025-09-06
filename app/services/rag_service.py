"""RAG service orchestrating document processing, vector search, and generation."""

from typing import Any, Dict, List, Optional

from vertexai.preview.language_models import TextGenerationModel

from app.core.config import rag_config, settings
from app.core.exceptions import RAGAPIException
from app.models.schemas import (ChunkInfo, SearchRequest, SearchResponse,
                                SearchResult)
from app.services.gemini_document_processor import GeminiDocumentProcessor
from app.services.storage_service import StorageService
from app.services.vector_store import VectorStore
from app.utils.ingestion.chunking import ChunkingStrategy


class RAGService:
    """Main RAG service orchestrating all components."""

    def __init__(self):
        self.document_processor = GeminiDocumentProcessor()
        self.vector_store = VectorStore()
        self.storage_service = StorageService()
        self.generation_model = TextGenerationModel.from_pretrained(
            settings.vertex_ai_model_name
        )
        self.chunking_strategy = ChunkingStrategy()

    async def process_and_store_document(
        self, file_content: bytes, filename: str, content_type: str
    ) -> str:
        """
        Process a document and store it in the vector database.

        Args:
            file_content: Raw file content
            filename: Original filename
            content_type: MIME type of the file

        Returns:
            File ID of the processed document
        """
        try:
            # Upload file to storage
            file_id = await self.storage_service.upload_file(
                file_content=file_content, filename=filename, content_type=content_type
            )

            # Process document to extract chunks using Gemini multimodal
            chunks = await self.document_processor.process_document(
                file_content=file_content, filename=filename, content_type=content_type
            )

            # Apply chunking strategy if needed
            if len(chunks) > rag_config.max_chunks_per_document:
                chunks = await self.chunking_strategy.optimize_chunks(
                    chunks, rag_config.max_chunks_per_document
                )

            # Store chunks in vector database
            await self.vector_store.add_chunks(chunks, file_id)

            return file_id

        except Exception as e:
            # Clean up uploaded file if processing failed
            try:
                await self.storage_service.delete_file(file_id)
            except:
                pass
            raise RAGAPIException(f"Failed to process document {filename}: {str(e)}")

    async def search_documents(self, search_request: SearchRequest) -> SearchResponse:
        """
        Search documents using RAG.

        Args:
            search_request: Search parameters

        Returns:
            SearchResponse with results
        """
        try:
            # Perform vector search
            search_results = await self.vector_store.search_similar(
                query=search_request.query,
                max_results=search_request.max_results or rag_config.max_results,
                similarity_threshold=search_request.similarity_threshold
                or rag_config.similarity_threshold,
                file_ids=search_request.file_ids,
            )

            # Generate response using the language model
            if search_results:
                context = self._build_context(search_results)
                generated_response = await self._generate_response(
                    query=search_request.query, context=context
                )
            else:
                generated_response = "No relevant documents found for your query."

            return SearchResponse(
                query=search_request.query,
                results=search_results,
                total_results=len(search_results),
                processing_time_ms=0.0,  # Would be calculated in real implementation
            )

        except Exception as e:
            raise RAGAPIException(f"Failed to search documents: {str(e)}")

    async def update_document(
        self, file_id: str, file_content: bytes, filename: str, content_type: str
    ) -> bool:
        """
        Update an existing document.

        Args:
            file_id: ID of the file to update
            file_content: New file content
            filename: New filename
            content_type: New content type

        Returns:
            True if successful
        """
        try:
            # Delete old chunks from vector store
            await self.vector_store.delete_file_chunks(file_id)

            # Process new document using Gemini multimodal
            chunks = await self.document_processor.process_document(
                file_content=file_content, filename=filename, content_type=content_type
            )

            # Apply chunking strategy if needed
            if len(chunks) > rag_config.max_chunks_per_document:
                chunks = await self.chunking_strategy.optimize_chunks(
                    chunks, rag_config.max_chunks_per_document
                )

            # Update storage
            await self.storage_service.upload_file(
                file_content=file_content,
                filename=filename,
                content_type=content_type,
                metadata={"file_id": file_id},
            )

            # Add new chunks to vector store
            await self.vector_store.add_chunks(chunks, file_id)

            return True

        except Exception as e:
            raise RAGAPIException(f"Failed to update document {file_id}: {str(e)}")

    async def delete_document(self, file_id: str) -> bool:
        """
        Delete a document and all its chunks.

        Args:
            file_id: ID of the file to delete

        Returns:
            True if successful
        """
        try:
            # Delete from vector store
            await self.vector_store.delete_file_chunks(file_id)

            # Delete from storage
            await self.storage_service.delete_file(file_id)

            return True

        except Exception as e:
            raise RAGAPIException(f"Failed to delete document {file_id}: {str(e)}")

    async def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all documents.

        Returns:
            List of document metadata
        """
        try:
            return await self.storage_service.list_files()
        except Exception as e:
            raise RAGAPIException(f"Failed to list documents: {str(e)}")

    def _build_context(self, search_results: List[SearchResult]) -> str:
        """Build context string from search results."""
        context_parts = []
        for i, result in enumerate(search_results, 1):
            context_parts.append(
                f"Document {i} (from {result.filename}):\n{result.content}\n"
            )
        return "\n".join(context_parts)

    async def _generate_response(self, query: str, context: str) -> str:
        """Generate response using the language model."""
        try:
            prompt = f"""
            Based on the following context, please answer the question.
            
            Context:
            {context}
            
            Question: {query}
            
            Answer:
            """

            response = self.generation_model.predict(
                prompt=prompt, max_output_tokens=1024, temperature=0.1
            )

            return response.text

        except Exception as e:
            raise RAGAPIException(f"Failed to generate response: {str(e)}")
