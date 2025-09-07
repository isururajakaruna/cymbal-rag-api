"""RAG search endpoints."""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.dependencies import get_rag_service
from app.core.exceptions import RAGAPIException
from app.models.schemas import ErrorResponse, SearchRequest, SearchResponse, RAGSearchResponse
from app.services.rag_service import RAGService
from app.services.rag_search_service import RAGSearchService

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


@router.post("/rag", response_model=RAGSearchResponse)
async def rag_search_documents(
    search_request: SearchRequest
):
    """
    Enhanced RAG search with vector search and Gemini response generation.
    
    - **query**: Search query text
    - **ktop**: Number of top results to retrieve (default: 10)
    - **threshold**: Similarity threshold (default: 0.7)
    - **file_ids**: Optional list of file IDs to search within
    """
    try:
        rag_search_service = RAGSearchService()
        response = await rag_search_service.search_documents(search_request)
        return response
        
    except RAGAPIException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/rag", response_model=RAGSearchResponse)
async def rag_search_documents_get(
    query: str = Query(..., min_length=1, max_length=1000, description="Search query"),
    ktop: Optional[int] = Query(
        10, ge=1, le=50, description="Number of top results to retrieve"
    ),
    threshold: Optional[float] = Query(
        0.7, ge=0.0, le=1.0, description="Similarity threshold"
    ),
    file_ids: Optional[str] = Query(
        None, description="Comma-separated list of file IDs to search within"
    ),
):
    """
    Enhanced RAG search with vector search and Gemini response generation (GET version).
    
    - **query**: Search query text
    - **ktop**: Number of top results to retrieve (default: 10)
    - **threshold**: Similarity threshold (default: 0.7)
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
            ktop=ktop,
            threshold=threshold,
            file_ids=file_ids_list,
        )

        rag_search_service = RAGSearchService()
        response = await rag_search_service.search_documents(search_request)
        return response
        
    except RAGAPIException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


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
