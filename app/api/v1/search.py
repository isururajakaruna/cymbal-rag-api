"""RAG search endpoints."""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from app.core.exceptions import RAGAPIException
from app.models.schemas import SearchRequest, RAGSearchResponse
from app.services.rag_search_service import RAGSearchService

router = APIRouter()


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
