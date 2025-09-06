"""File management endpoints."""

from datetime import datetime
from typing import List, Optional

from fastapi import (APIRouter, Depends, File, Form, HTTPException, Query,
                     UploadFile)
from fastapi.responses import JSONResponse

from app.api.dependencies import get_rag_service
from app.core.config import rag_config
from app.core.exceptions import (FileSizeExceededError, RAGAPIException,
                                 UnsupportedFileFormatError, ValidationError)
from app.models.schemas import (DocumentProcessingStatus, ErrorResponse,
                                FileInfo, FileListResponse, FileUploadResponse)
from app.services.rag_service import RAGService

router = APIRouter()


@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...), rag_service: RAGService = Depends(get_rag_service)
):
    """
    Upload a new file for processing and indexing.

    - **file**: File to upload (PDF, TXT, DOCX, PNG, JPG, JPEG)
    - **description**: Upload a document to be processed and added to the vector database
    """
    try:
        # Validate file
        await _validate_upload_file(file)

        # Read file content
        file_content = await file.read()

        # Process and store document
        file_id = await rag_service.process_and_store_document(
            file_content=file_content,
            filename=file.filename,
            content_type=file.content_type,
        )

        return FileUploadResponse(
            file_id=file_id,
            filename=file.filename,
            file_size=len(file_content),
            content_type=file.content_type,
            upload_timestamp=datetime.utcnow(),
            status="uploaded",
        )

    except UnsupportedFileFormatError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileSizeExceededError as e:
        raise HTTPException(status_code=413, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RAGAPIException as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{file_id}", response_model=FileUploadResponse)
async def update_file(
    file_id: str,
    file: UploadFile = File(...),
    rag_service: RAGService = Depends(get_rag_service),
):
    """
    Update an existing file.

    - **file_id**: ID of the file to update
    - **file**: New file content
    """
    try:
        # Validate file
        await _validate_upload_file(file)

        # Read file content
        file_content = await file.read()

        # Update document
        success = await rag_service.update_document(
            file_id=file_id,
            file_content=file_content,
            filename=file.filename,
            content_type=file.content_type,
        )

        if not success:
            raise HTTPException(status_code=404, detail="File not found")

        return FileUploadResponse(
            file_id=file_id,
            filename=file.filename,
            file_size=len(file_content),
            content_type=file.content_type,
            upload_timestamp=datetime.utcnow(),
            status="updated",
        )

    except UnsupportedFileFormatError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileSizeExceededError as e:
        raise HTTPException(status_code=413, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RAGAPIException as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{file_id}")
async def delete_file(file_id: str, rag_service: RAGService = Depends(get_rag_service)):
    """
    Delete a file and all its chunks.

    - **file_id**: ID of the file to delete
    """
    try:
        success = await rag_service.delete_document(file_id)

        if not success:
            raise HTTPException(status_code=404, detail="File not found")

        return {"message": f"File {file_id} deleted successfully"}

    except RAGAPIException as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=FileListResponse)
async def list_files(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Number of files per page"),
    rag_service: RAGService = Depends(get_rag_service),
):
    """
    List all uploaded files with pagination.

    - **page**: Page number (starting from 1)
    - **page_size**: Number of files per page (1-100)
    """
    try:
        files_data = await rag_service.list_documents()

        # Apply pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_files = files_data[start_idx:end_idx]

        # Convert to FileInfo objects
        files = []
        for file_data in paginated_files:
            files.append(
                FileInfo(
                    file_id=file_data["file_id"],
                    filename=file_data["filename"],
                    file_size=file_data["size"],
                    content_type=file_data["content_type"],
                    upload_timestamp=file_data["created"],
                    last_modified=file_data["updated"],
                    status="processed",
                )
            )

        return FileListResponse(
            files=files, total_count=len(files_data), page=page, page_size=page_size
        )

    except RAGAPIException as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{file_id}", response_model=FileInfo)
async def get_file_info(
    file_id: str, rag_service: RAGService = Depends(get_rag_service)
):
    """
    Get information about a specific file.

    - **file_id**: ID of the file
    """
    try:
        # This would typically involve getting file metadata from storage
        # For now, we'll return a placeholder
        raise HTTPException(status_code=501, detail="Not implemented yet")

    except RAGAPIException as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{file_id}/status", response_model=DocumentProcessingStatus)
async def get_processing_status(
    file_id: str, rag_service: RAGService = Depends(get_rag_service)
):
    """
    Get the processing status of a file.

    - **file_id**: ID of the file
    """
    try:
        # This would typically involve checking processing status
        # For now, we'll return a placeholder
        return DocumentProcessingStatus(
            file_id=file_id,
            status="completed",
            progress_percentage=100.0,
            chunks_processed=0,
            total_chunks=0,
            processing_start_time=datetime.utcnow(),
            processing_end_time=datetime.utcnow(),
        )

    except RAGAPIException as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _validate_upload_file(file: UploadFile) -> None:
    """Validate uploaded file."""
    # Check file size
    max_size = rag_config.max_file_size_mb * 1024 * 1024  # Convert to bytes
    if file.size and file.size > max_size:
        raise FileSizeExceededError(
            f"File size exceeds maximum allowed size of {rag_config.max_file_size_mb}MB"
        )

    # Check file type
    if file.content_type not in rag_config.supported_formats:
        raise UnsupportedFileFormatError(f"Unsupported file type: {file.content_type}")

    # Check filename
    if not file.filename:
        raise ValidationError("Filename is required")
