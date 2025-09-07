"""File management API endpoints."""

import os
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from google.cloud import storage
from datetime import datetime
import mimetypes

from app.core.config import settings
from app.core.exceptions import RAGAPIException

router = APIRouter()


class FileInfo:
    """File information model."""
    
    def __init__(self, name: str, path: str, file_type: str, last_updated: datetime, size: int):
        self.name = name
        self.path = path
        self.file_type = file_type
        self.last_updated = last_updated
        self.size = size


async def list_files_from_gcs(
    search_query: Optional[str] = None,
    sort_by: str = "date",
    limit: Optional[int] = None,
    offset: int = 0
) -> List[FileInfo]:
    """List files from Google Cloud Storage uploads directory."""
    try:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.google_application_credentials
        storage_client = storage.Client(project=settings.google_cloud_project_id)
        bucket = storage_client.bucket(settings.storage_bucket_name)
        
        # List all blobs in uploads directory
        blobs = bucket.list_blobs(prefix="uploads/")
        
        files = []
        for blob in blobs:
            # Skip if it's a directory
            if blob.name.endswith('/'):
                continue
                
            # Extract filename from path
            filename = blob.name.split('/')[-1]
            
            # Apply search filter if provided
            if search_query and search_query.lower() not in filename.lower():
                continue
            
            # Get file type from content type or extension
            content_type = blob.content_type or "application/octet-stream"
            if content_type == "application/octet-stream":
                # Try to determine from extension
                file_type, _ = mimetypes.guess_type(filename)
                if file_type:
                    content_type = file_type
            
            # Get file extension for display
            file_extension = os.path.splitext(filename)[1].lower()
            if file_extension:
                file_type = f"{content_type} ({file_extension})"
            else:
                file_type = content_type
            
            file_info = FileInfo(
                name=filename,
                path=blob.name,
                file_type=file_type,
                last_updated=blob.time_created or blob.updated,
                size=blob.size or 0
            )
            files.append(file_info)
        
        # Sort files
        if sort_by == "date":
            files.sort(key=lambda x: x.last_updated, reverse=True)  # Newest first
        elif sort_by == "name":
            files.sort(key=lambda x: x.name.lower())
        elif sort_by == "size":
            files.sort(key=lambda x: x.size, reverse=True)  # Largest first
        
        # Apply pagination
        if limit is not None:
            files = files[offset:offset + limit]
        elif offset > 0:
            files = files[offset:]
        
        return files
        
    except Exception as e:
        raise RAGAPIException(f"Error listing files: {str(e)}")


@router.get("/list")
async def list_files(
    search: Optional[str] = Query(None, description="Search query for filename"),
    sort_by: str = Query("date", description="Sort by: date, name, size"),
    limit: Optional[int] = Query(None, description="Number of files to return"),
    offset: int = Query(0, description="Number of files to skip")
):
    """
    List uploaded files with sorting, pagination, and search.
    
    - **search**: Search for files by filename (case-insensitive)
    - **sort_by**: Sort by 'date' (newest first), 'name' (alphabetical), or 'size' (largest first)
    - **limit**: Maximum number of files to return (default: all)
    - **offset**: Number of files to skip for pagination (default: 0)
    """
    try:
        # Validate sort_by parameter
        valid_sorts = ["date", "name", "size"]
        if sort_by not in valid_sorts:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid sort_by parameter. Must be one of: {', '.join(valid_sorts)}"
            )
        
        # Validate pagination parameters
        if limit is not None and limit < 0:
            raise HTTPException(status_code=400, detail="Limit must be non-negative")
        if offset < 0:
            raise HTTPException(status_code=400, detail="Offset must be non-negative")
        
        files = await list_files_from_gcs(
            search_query=search,
            sort_by=sort_by,
            limit=limit,
            offset=offset
        )
        
        # Convert to response format
        file_list = []
        for file_info in files:
            file_list.append({
                "name": file_info.name,
                "path": file_info.path,
                "file_type": file_info.file_type,
                "last_updated": file_info.last_updated.isoformat(),
                "size": file_info.size
            })
        
        return {
            "success": True,
            "files": file_list,
            "total_count": len(file_list),
            "search_query": search,
            "sort_by": sort_by,
            "limit": limit,
            "offset": offset
        }
        
    except RAGAPIException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/view/{filename}")
async def view_file(filename: str):
    """
    Download a file by its filename.
    
    - **filename**: The name of the file to download (as returned by list API)
    """
    try:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.google_application_credentials
        storage_client = storage.Client(project=settings.google_cloud_project_id)
        bucket = storage_client.bucket(settings.storage_bucket_name)
        
        # Clean filename to match upload format
        clean_filename = filename.replace(" ", "_").replace(":", "-").replace("/", "-")
        file_path = f"uploads/{clean_filename}"
        blob = bucket.blob(file_path)
        
        # Check if file exists
        if not blob.exists():
            raise HTTPException(
                status_code=404,
                detail=f"File '{filename}' not found"
            )
        
        # Get file content
        file_content = blob.download_as_bytes()
        
        # Get file metadata
        blob.reload()
        content_type = blob.content_type or "application/octet-stream"
        
        # Return file as response
        from fastapi.responses import Response
        return Response(
            content=file_content,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Length": str(len(file_content))
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")


@router.get("/embedding-stats/{filename}")
async def get_embedding_stats(filename: str):
    """
    Get embedding statistics for a file.
    
    - **filename**: The name of the file to get stats for (as returned by list API)
    """
    try:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.google_application_credentials
        storage_client = storage.Client(project=settings.google_cloud_project_id)
        bucket = storage_client.bucket(settings.storage_bucket_name)
        
        # Clean filename to match upload format
        clean_filename = filename.replace(" ", "_").replace(":", "-").replace("/", "-")
        file_path = f"uploads/{clean_filename}"
        blob = bucket.blob(file_path)
        
        # Check if file exists
        if not blob.exists():
            raise HTTPException(
                status_code=404,
                detail=f"File '{filename}' not found"
            )
        
        # Get file metadata
        blob.reload()
        content_type = blob.content_type or "application/octet-stream"
        
        # Get file type for display
        file_extension = os.path.splitext(filename)[1].lower()
        if file_extension:
            file_type = f"{content_type} ({file_extension})"
        else:
            file_type = content_type
        
        # Get datapoint IDs from blob metadata
        datapoint_ids = []
        if blob.metadata and "datapoint_ids" in blob.metadata:
            datapoint_ids_str = blob.metadata["datapoint_ids"]
            if datapoint_ids_str:
                datapoint_ids = datapoint_ids_str.split(",")
        
        # Calculate embedding stats
        total_embeddings = len(datapoint_ids)
        
        return {
            "success": True,
            "file_info": {
                "name": filename,
                "path": file_path,
                "file_type": file_type,
                "last_updated": blob.time_created.isoformat() if blob.time_created else None,
                "size": blob.size or 0
            },
            "embedding_stats": {
                "total_embeddings": total_embeddings,
                "datapoint_ids": datapoint_ids,
                "has_embeddings": total_embeddings > 0
            },
            "message": f"File has {total_embeddings} embeddings stored in Vector Search"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting embedding stats: {str(e)}")