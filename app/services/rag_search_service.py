"""Enhanced RAG search service with real vector search integration and reranking."""

import os
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
import mimetypes

from google.cloud import storage
from google.cloud import aiplatform
from google.cloud.aiplatform import MatchingEngineIndex, MatchingEngineIndexEndpoint
from google.cloud.aiplatform_v1.types import IndexDatapoint
try:
    from google.cloud import discoveryengine_v1 as discoveryengine
    DISCOVERY_ENGINE_AVAILABLE = True
except ImportError:
    discoveryengine = None
    DISCOVERY_ENGINE_AVAILABLE = False
import vertexai
from vertexai.generative_models import GenerativeModel
import google.generativeai as genai

from app.core.config import settings
from app.core.exceptions import RAGAPIException
from app.models.schemas import (
    SearchRequest, 
    SearchResult, 
    RAGSearchFileInfo, 
    RAGSearchResponse
)


class RAGSearchService:
    """Enhanced RAG search service with real vector search, reranking, and Gemini integration."""
    
    def __init__(self):
        """Initialize the RAG search service."""
        self.project_id = settings.google_cloud_project_id
        self.location = settings.google_cloud_region
        self.index_id = settings.vector_search_index_id
        self.endpoint_id = settings.vector_search_index_endpoint_id
        
        # Initialize AI Platform
        aiplatform.init(project=self.project_id, location=self.location)
        
        # Get index and endpoint resource names
        self.index_name = f"projects/{self.project_id}/locations/{self.location}/indexes/{self.index_id}"
        self.endpoint_name = f"projects/{self.project_id}/locations/{self.location}/indexEndpoints/{self.endpoint_id}"
        
        # The endpoint_id is actually the deployed_index_id for search
        self.deployed_index_id = self.endpoint_id
        
        # Initialize Gemini model
        vertexai.init(project=self.project_id, location=self.location)
        self.gemini_model = GenerativeModel(settings.vertex_ai_model_name)
        
        # Initialize Discovery Engine client for reranking (if available)
        if DISCOVERY_ENGINE_AVAILABLE and discoveryengine:
            self.discovery_client = discoveryengine.RankServiceClient()
        else:
            self.discovery_client = None
        
        print(f"RAGSearchService initialized for project {self.project_id}")
        print(f"Index: {self.index_name}")
        print(f"Endpoint: {self.endpoint_name}")
    
    async def search_documents(self, search_request: SearchRequest) -> RAGSearchResponse:
        """
        Search documents using RAG with vector search, reranking, and Gemini response generation.
        
        Args:
            search_request: Search parameters including query, ktop, threshold
            
        Returns:
            RAGSearchResponse with file list, matched chunks, and RAG response
        """
        start_time = time.time()
        
        try:
            # Get query embedding
            query_embedding = await self._get_query_embedding(search_request.query)
            
            # Perform vector search (no threshold initially to get more candidates)
            ktop = search_request.ktop if search_request.ktop is not None else 10
            threshold = search_request.threshold if search_request.threshold is not None else 0.5
            
            # Get more results than needed for better reranking
            vector_search_limit = max(ktop * 3, 50)  # Get 3x more results for reranking
            
            search_results = await self._perform_vector_search(
                query_embedding=query_embedding,
                ktop=vector_search_limit,  # Get more results initially
                threshold=0.0,  # No threshold for initial search
                file_ids=search_request.file_ids,
                tags=search_request.tags
            )
            
            print(f"Vector search returned {len(search_results)} candidates for reranking")
            
            # Apply reranking using Google's semantic reranker
            reranked_results = await self._rerank_results(
                query=search_request.query,
                search_results=search_results
            )
            
            print(f"Reranked to {len(reranked_results)} results")
            
            # Apply threshold after reranking
            filtered_results = [
                result for result in reranked_results 
                if result.distance >= threshold  # Higher distance = better similarity
            ]
            
            print(f"After threshold {threshold}: {len(filtered_results)} results")
            
            # Limit to requested ktop
            final_results = filtered_results[:ktop]
            
            # Group results by file
            files_dict = await self._group_results_by_file(final_results)
            
            # Get file metadata from GCS
            files_with_metadata = await self._enrich_with_file_metadata(files_dict)
            
            # Generate RAG response using Gemini
            rag_response = await self._generate_rag_response(
                query=search_request.query,
                search_results=final_results
            )
            
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            return RAGSearchResponse(
                success=True,
                query=search_request.query,
                files=files_with_metadata,
                total_files=len(files_with_metadata),
                total_chunks=len(final_results),
                rag_response=rag_response,
                processing_time_ms=processing_time,
                search_parameters={
                    "ktop": ktop,
                    "threshold": threshold,
                    "file_ids": search_request.file_ids,
                    "tags": search_request.tags
                }
            )
            
        except Exception as e:
            raise RAGAPIException(f"Error performing RAG search: {str(e)}")
    
    async def _rerank_results(
        self, 
        query: str, 
        search_results: List[SearchResult]
    ) -> List[SearchResult]:
        """Rerank search results using Google's semantic reranker."""
        try:
            if not search_results:
                return []
            
            # Check if Discovery Engine is available
            if not DISCOVERY_ENGINE_AVAILABLE or not discoveryengine or not self.discovery_client:
                print("Discovery Engine not available, skipping reranking")
                # Return original results sorted by distance (descending)
                search_results.sort(key=lambda x: x.distance, reverse=True)
                return search_results
            
            # Prepare ranking records for the reranker
            ranking_records = []
            for i, result in enumerate(search_results):
                # Extract title from metadata if available
                title = result.metadata.get("title", "") if result.metadata else ""
                
                ranking_record = discoveryengine.RankingRecord(
                    id=str(i),  # Use index as ID
                    title=title,
                    content=result.content
                )
                ranking_records.append(ranking_record)
            
            # Get ranking config path
            ranking_config = self.discovery_client.ranking_config_path(
                project=self.project_id,
                location="global",
                ranking_config="default_ranking_config"
            )
            
            # Create rerank request
            request = discoveryengine.RankRequest(
                ranking_config=ranking_config,
                model="semantic-ranker-default@latest",
                top_n=len(search_results),  # Rerank all results
                query=query,
                records=ranking_records
            )
            
            # Perform reranking
            print(f"Reranking {len(search_results)} results with Discovery Engine")
            response = self.discovery_client.rank(request=request)
            
            # Map reranked results back to SearchResult objects
            reranked_results = []
            for ranked_record in response.records:
                # Get original result by ID (index)
                original_index = int(ranked_record.id)
                if original_index < len(search_results):
                    original_result = search_results[original_index]
                    # Update distance with rerank score (higher is better)
                    original_result.distance = ranked_record.score
                    reranked_results.append(original_result)
            
            # Sort by rerank score (descending - higher is better)
            reranked_results.sort(key=lambda x: x.distance, reverse=True)
            
            print(f"Reranking completed, returning {len(reranked_results)} results")
            return reranked_results
            
        except Exception as e:
            print(f"Reranking failed, falling back to original order: {e}")
            # If reranking fails, return original results sorted by distance
            search_results.sort(key=lambda x: x.distance, reverse=True)
            return search_results
    
    async def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for the search query using Gemini embedding model."""
        try:
            # Use Google Generative AI with QUESTION_ANSWERING task type
            result = genai.embed_content(
                model=settings.vertex_ai_embedding_model_name,
                content=query,
                task_type="QUESTION_ANSWERING"
            )
            query_embedding = result['embedding']
            
            return query_embedding
        except Exception as e:
            raise RAGAPIException(f"Error generating query embedding: {str(e)}")
    
    async def _perform_vector_search(
        self, 
        query_embedding: List[float], 
        ktop: int, 
        threshold: float,
        file_ids: Optional[List[str]] = None,
        tags: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """Perform vector search using the existing VectorSearchService."""
        try:
            from app.utils.vector_search import VectorSearchService
            
            # Use the existing vector search service
            vector_service = VectorSearchService()
            
            # Perform search using the existing service
            results = await vector_service.search_similar(
                query_embedding=query_embedding,
                index_id=self.index_id,
                endpoint_id=settings.vector_search_deployed_index_id,
                top_k=ktop,
                filter_expression=None,
                tags=tags
            )
            
            # Process results (no threshold filtering here)
            print(f"Vector search returned {len(results)} results")
            search_results = []
            for i, result in enumerate(results):
                distance_score = result["score"]  # score is now distance
                print(f"Result {i}: distance={distance_score:.3f}")
                
                metadata = result.get("metadata", {})
                
                # Apply file filter if specified
                if file_ids and metadata.get("filename") not in file_ids:
                    continue
                
                search_result = SearchResult(
                    content=metadata.get("content", ""),
                    file_id=metadata.get("filename", ""),
                    filename=metadata.get("filename", ""),
                    chunk_index=int(metadata.get("chunk_index", 0)),
                    distance=distance_score,
                    metadata=metadata
                )
                search_results.append(search_result)
            
            print(f"Processed {len(search_results)} results from vector search")
            return search_results
            
        except Exception as e:
            print(f"Vector search error details: {str(e)}")
            raise RAGAPIException(f"Error performing vector search: {str(e)}")
    
    async def _group_results_by_file(self, search_results: List[SearchResult]) -> Dict[str, List[SearchResult]]:
        """Group search results by filename."""
        files_dict = {}
        for result in search_results:
            filename = result.filename
            if not filename:
                continue
            if filename not in files_dict:
                files_dict[filename] = []
            files_dict[filename].append(result)
        return files_dict
    
    async def _enrich_with_file_metadata(self, files_dict: Dict[str, List[SearchResult]]) -> List[RAGSearchFileInfo]:
        """Enrich file information with metadata from Google Cloud Storage."""
        try:
            storage_client = storage.Client(project=self.project_id)
            bucket = storage_client.bucket(settings.storage_bucket_name)
            
            files_with_metadata = []
            
            for filename, chunks in files_dict.items():
                # Get file metadata from GCS
                clean_filename = filename.replace(" ", "_").replace(":", "-").replace("/", "-")
                file_path = f"uploads/{clean_filename}"
                blob = bucket.blob(file_path)
                
                if blob.exists():
                    blob.reload()
                    content_type = blob.content_type or "application/octet-stream"
                    file_extension = os.path.splitext(filename)[1].lower()
                    
                    if file_extension:
                        file_type = f"{content_type} ({file_extension})"
                    else:
                        file_type = content_type
                    
                    # Extract tags from blob metadata
                    file_tags = []
                    if blob.metadata and "tags" in blob.metadata:
                        tags_str = blob.metadata["tags"]
                        if tags_str:
                            file_tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()]
                    
                    # Extract title from chunk metadata
                    document_title = None
                    if chunks and chunks[0].metadata and "title" in chunks[0].metadata:
                        document_title = chunks[0].metadata["title"]
                    
                    file_info = RAGSearchFileInfo(
                        name=filename,
                        path=file_path,
                        file_type=file_type,
                        last_updated=blob.updated,
                        size=blob.size,
                        tags=file_tags,
                        title=document_title,
                        matched_chunks=chunks
                    )
                    files_with_metadata.append(file_info)
            
            return files_with_metadata
            
        except Exception as e:
            raise RAGAPIException(f"Error enriching file metadata: {str(e)}")
    
    async def _generate_rag_response(self, query: str, search_results: List[SearchResult]) -> str:
        """Generate RAG response using Gemini model."""
        try:
            if not search_results:
                return "No relevant documents found for your query. Please try a different search term or check if the knowledge base contains relevant information."
            
            # Build context from search results
            context_parts = []
            for i, result in enumerate(search_results, 1):
                context_parts.append(
                    f"Source {i} (from {result.filename}, chunk {result.chunk_index}):\n{result.content}\n"
                )
            context = "\n".join(context_parts)
            
            # Create prompt for Gemini
            prompt = f"""Based on the following context from our knowledge base, please provide a clear and concise answer to the user's question.

Context:
{context}

User Question: {query}

Please provide a direct answer based on the context above. If the context doesn't contain enough information to answer the question, simply state that the information is not available in the knowledge base. Keep your response focused and avoid mentioning specific chunks or sources."""

            # Generate response using Gemini
            response = self.gemini_model.generate_content(prompt)
            
            return response.text if response.text else "I apologize, but I couldn't generate a response at this time. Please try again."
            
        except Exception as e:
            return f"Error generating response: {str(e)}. Please try again or contact support."
