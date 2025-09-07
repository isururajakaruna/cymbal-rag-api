"""Main FastAPI application."""

from contextlib import asynccontextmanager
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1 import files, search, validate, upload
from app.core.config import rag_config, settings
from app.core.exceptions import RAGAPIException
from app.models.schemas import ErrorResponse, HealthCheckResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    print("Starting Cymbal RAG API...")
    print(f"Configuration loaded: {rag_config.api}")

    yield

    # Shutdown
    print("Shutting down Cymbal RAG API...")


# Create FastAPI application
app = FastAPI(
    title=rag_config.api.get("title", "Cymbal RAG API"),
    description=rag_config.api.get(
        "description", "RAG API using Google Vertex AI services"
    ),
    version=rag_config.api.get("version", "1.0.0"),
    docs_url=rag_config.api.get("docs_url", "/docs"),
    redoc_url=rag_config.api.get("redoc_url", "/redoc"),
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(files.router, prefix="/api/v1/files", tags=["files"])
app.include_router(search.router, prefix="/api/v1/search", tags=["search"])
app.include_router(validate.router, prefix="/api/v1/file", tags=["file-validation"])
app.include_router(upload.router, prefix="/api/v1", tags=["upload"])


@app.get("/", response_model=HealthCheckResponse)
async def root():
    """Root endpoint with API information."""
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version=rag_config.api.get("version", "1.0.0"),
        services={
            "api": "healthy",
            "document_processor": "healthy",
            "vector_store": "healthy",
            "storage": "healthy",
        },
    )


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint.
    """
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version=rag_config.api.get("version", "1.0.0"),
        services={
            "api": "healthy",
            "document_processor": "healthy",
            "vector_store": "healthy",
            "storage": "healthy",
        },
    )


@app.exception_handler(RAGAPIException)
async def rag_api_exception_handler(request, exc: RAGAPIException):
    """Handle RAG API exceptions."""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(error=exc.message, error_code=exc.error_code).dict(),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error=exc.detail, error_code=str(exc.status_code)).dict(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle general exceptions."""
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error", error_code="INTERNAL_ERROR"
        ).dict(),
    )


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=True,
    )
