"""RAG search endpoints."""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.dependencies import get_rag_service
from app.core.exceptions import RAGAPIException
from app.models.schemas import ErrorResponse, SearchRequest, SearchResponse
from app.services.rag_service import RAGService

router = APIRouter()


@router.post("/", response_model=SearchResponse)
async def search_documents(
    search_request: SearchRequest, rag_service: RAGService = Depends(get_rag_service)
):
    """
    Search documents using RAG (Retrieval-Augmented Generation).

    - **query**: Search query text
    - **max_results**: Maximum number of results to return (1-50)
    - **similarity_threshold**: Minimum similarity score (0.0-1.0)
    - **file_ids**: Optional list of file IDs to search within
    """
    try:
        response = await rag_service.search_documents(search_request)
        return response

    except RAGAPIException as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=SearchResponse)
async def search_documents_get(
    query: str = Query(..., min_length=1, max_length=1000, description="Search query"),
    max_results: Optional[int] = Query(
        10, ge=1, le=50, description="Maximum number of results"
    ),
    similarity_threshold: Optional[float] = Query(
        0.7, ge=0.0, le=1.0, description="Similarity threshold"
    ),
    file_ids: Optional[str] = Query(
        None, description="Comma-separated list of file IDs to search within"
    ),
    rag_service: RAGService = Depends(get_rag_service),
):
    """
    Search documents using RAG (GET version for simple queries).

    - **query**: Search query text
    - **max_results**: Maximum number of results to return (1-50)
    - **similarity_threshold**: Minimum similarity score (0.0-1.0)
    - **file_ids**: Comma-separated list of file IDs to search within
    """
    try:
        # Parse file_ids if provided
        file_ids_list = None
        if file_ids:
            file_ids_list = [fid.strip() for fid in file_ids.split(",") if fid.strip()]

        # Create search request
        search_request = SearchRequest(
            query=query,
            max_results=max_results,
            similarity_threshold=similarity_threshold,
            file_ids=file_ids_list,
        )

        response = await rag_service.search_documents(search_request)
        return response

    except RAGAPIException as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def search_health_check():
    """
    Health check for search service.
    """
    return {
        "status": "healthy",
        "service": "search",
        "message": "Search service is operational",
    }
