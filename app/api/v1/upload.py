"""
Upload API endpoints for the Cymbal RAG API.

This module provides endpoints for uploading files to the knowledge base,
including direct uploads and uploads from temporary storage.
"""

import os
import uuid
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Path, Query
from fastapi.responses import JSONResponse
from google.cloud import storage
import vertexai
from vertexai.language_models import TextEmbeddingModel
from app.core.config import settings, rag_config
from app.core.exceptions import RAGAPIException
from app.models.schemas import FileValidationResponse, ContentAnalysis
from app.services.gemini_document_processor import GeminiDocumentProcessor
from app.utils.chunking import ChunkingService
from app.utils.vector_search import VectorSearchService
import google.generativeai as genai
import json
import asyncio

router = APIRouter()

# Initialize services
vertexai.init(project=settings.google_cloud_project_id, location=settings.google_cloud_region)
# Embedding model will be initialized lazily to avoid auth issues during import
embedding_model = None
document_processor = GeminiDocumentProcessor()
chunking_service = ChunkingService()
vector_search_service = VectorSearchService()


def get_embedding_model():
    """Get embedding model, initializing it lazily."""
    global embedding_model
    if embedding_model is None:
        embedding_model = TextEmbeddingModel.from_pretrained(settings.vertex_ai_embedding_model_name)
    return embedding_model


async def get_file_from_temp_storage(validation_id: str) -> dict:
    """Get file information from temporary storage by validation_id."""
    try:
        # Google Cloud credentials are loaded from .env file via config.py
        storage_client = storage.Client(project=settings.google_cloud_project_id)
        bucket = storage_client.bucket(settings.storage_bucket_name)
        
        # List files in temp directory and find the one with matching validation_id
        blobs = bucket.list_blobs(prefix="tmp/")
        
        for blob in blobs:
            if blob.name.startswith("tmp/"):
                # Reload blob to get metadata
                blob.reload()
                
                # Check if this blob has the matching validation_id in metadata
                if blob.metadata and blob.metadata.get("validation_id") == validation_id:
                    return {
                        "temp_path": blob.name,
                        "filename": blob.name.split("/")[-1],
                        "size": blob.size,
                        "created": blob.time_created,
                        "content_type": blob.content_type,
                        "validation_id": validation_id
                    }
        
        return None
    except Exception as e:
        raise RAGAPIException(f"Error getting file from temp storage: {str(e)}")


async def download_file_from_gcs(gcs_path: str) -> bytes:
    """Download file content from Google Cloud Storage."""
    try:
        # Google Cloud credentials are loaded from .env file via config.py
        storage_client = storage.Client(project=settings.google_cloud_project_id)
        bucket = storage_client.bucket(settings.storage_bucket_name)
        
        blob = bucket.blob(gcs_path)
        return blob.download_as_bytes()
    except Exception as e:
        raise RAGAPIException(f"Error downloading file from GCS: {str(e)}")


async def upload_file_to_uploads(file_content: bytes, filename: str, content_type: str, datapoint_ids: List[str] = None, tags: List[str] = None) -> str:
    """Upload file to the uploads directory in Google Cloud Storage."""
    try:
        # Google Cloud credentials are loaded from .env file via config.py
        storage_client = storage.Client(project=settings.google_cloud_project_id)
        bucket = storage_client.bucket(settings.storage_bucket_name)
        
        # Upload to uploads directory (always override)
        # Clean filename for uploads directory (remove spaces and special chars)
        clean_filename = filename.replace(" ", "_").replace(":", "-").replace("/", "-")
        upload_path = f"uploads/{clean_filename}"
        blob = bucket.blob(upload_path)
        blob.upload_from_string(file_content, content_type=content_type)
        
        # Store datapoint IDs and tags in blob metadata
        metadata = {}
        if datapoint_ids:
            metadata["datapoint_ids"] = ",".join(datapoint_ids)
        if tags:
            metadata["tags"] = ",".join(tags)
        
        if metadata:
            blob.metadata = metadata
            blob.patch()
        
        return upload_path
    except Exception as e:
        raise RAGAPIException(f"Error uploading file to uploads: {str(e)}")


async def delete_file_from_gcs(gcs_path: str) -> bool:
    """Delete file from Google Cloud Storage."""
    try:
        # Google Cloud credentials are loaded from .env file via config.py
        storage_client = storage.Client(project=settings.google_cloud_project_id)
        bucket = storage_client.bucket(settings.storage_bucket_name)
        
        blob = bucket.blob(gcs_path)
        blob.delete()
        return True
    except Exception as e:
        print(f"Warning: Could not delete file {gcs_path}: {e}")
        return False


async def generate_document_title(file_content: bytes, filename: str, content_type: str) -> str:
    """Generate a title for the document using Gemini based on first 1000 characters."""
    try:
        from vertexai.generative_models import GenerativeModel
        import vertexai
        
        # Initialize Vertex AI
        vertexai.init(project=settings.google_cloud_project_id, location=settings.google_cloud_region)
        generative_model = GenerativeModel(settings.vertex_ai_model_name)
        
        # Get first 1000 characters of the document content
        if content_type == "application/pdf":
            # For PDFs, we need to process them first to get text content
            from pdf2image import convert_from_bytes
            import io
            
            images = convert_from_bytes(file_content, first_page=1, last_page=1)
            if images:
                # Convert first page to text using Gemini
                image = images[0]
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                
                from vertexai.generative_models import Part
                image_part = Part.from_data(data=img_byte_arr, mime_type="image/png")
                
                prompt = f"""
                Extract the main title or heading from this document. 
                If there's no clear title, generate a descriptive title based on the content.
                Return only the title, nothing else.
                """
                
                response = generative_model.generate_content([image_part, prompt])
                return response.text.strip()
        else:
            # For text-based files, get first 1000 characters
            try:
                content_text = file_content.decode('utf-8')
            except UnicodeDecodeError:
                # For binary files, use filename as fallback
                return os.path.splitext(filename)[0]
            
            # Limit to first 1000 characters
            sample_content = content_text[:1000]
            
            prompt = f"""
            Based on this document content, generate a concise and descriptive title:
            
            Content: {sample_content}
            
            Return only the title, nothing else.
            """
            
            response = generative_model.generate_content(prompt)
            return response.text.strip()
            
    except Exception as e:
        print(f"Error generating title for {filename}: {e}")
        # Fallback to filename without extension
        return os.path.splitext(filename)[0]


async def process_and_embed_document(file_content: bytes, filename: str, content_type: str, tags: List[str] = None) -> Dict[str, Any]:
    """Process document and create embeddings for Vector Search."""
    try:
        # Generate document title once for all chunks
        document_title = await generate_document_title(file_content, filename, content_type)
        
        # Process document using Gemini processor
        processed_docs = await document_processor.process_document(
            file_content=file_content,
            filename=filename,
            content_type=content_type
        )
        
        # Chunk the processed documents
        chunks = []
        for doc in processed_docs:
            doc_chunks = chunking_service.chunk_document(
                content=doc.content,
                metadata=doc.metadata,
                chunk_size=rag_config.chunk_size,
                chunk_overlap=rag_config.chunk_overlap
            )
            chunks.extend(doc_chunks)
        
        # Clean filename for consistent handling
        clean_filename = filename.replace(" ", "_").replace(":", "-").replace("/", "-")
        
        # Generate embeddings for each chunk
        embeddings = []
        model = get_embedding_model()
        # Processing {len(chunks)} chunks for file: {filename}
        
        for i, chunk in enumerate(chunks):
            # Chunk {i} processing
            
            # Generate embedding using Vertex AI with RETRIEVAL_DOCUMENT task type
            result = genai.embed_content(
                model=settings.vertex_ai_embedding_model_name,  # Use gemini-embedding-001 from .env
                content=chunk.content,
                task_type="RETRIEVAL_DOCUMENT",
                title=document_title  # Add document title to embedding
            )
            embedding = result['embedding']
            # Chunk {i} embedding generated: {len(embedding)} dimensions
            
            # Prepare data for Vector Search
            chunk_data = {
                "id": f"{clean_filename}_{i}_{uuid.uuid4().hex[:8]}",
                "content": chunk.content,
                "metadata": {
                    **chunk.metadata,
                    "filename": clean_filename,
                    "original_filename": filename,  # Keep original for reference
                    "content_type": content_type,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "content": chunk.content,  # Include content in metadata for vector search
                    "tags": ",".join(tags) if tags else "",  # Add tags to metadata
                    "title": document_title  # Add document title to metadata
                },
                "embedding": embedding
            }
            embeddings.append(chunk_data)
        
        return {
            "chunks": chunks,
            "embeddings": embeddings,
            "total_chunks": len(chunks)
        }
        
    except Exception as e:
        raise RAGAPIException(f"Error processing and embedding document: {str(e)}")


async def store_embeddings_in_vector_search(embeddings: List[Dict[str, Any]], filename: str) -> bool:
    """Store embeddings in Vertex AI Vector Search."""
    try:
        # Store embeddings in Vector Search
        success = await vector_search_service.upsert_embeddings(
            embeddings=embeddings,
            index_id=settings.vector_search_index_id,
            endpoint_id=settings.vector_search_index_endpoint_id
        )
        
        return success
    except Exception as e:
        raise RAGAPIException(f"Error storing embeddings in Vector Search: {str(e)}")


async def get_datapoint_ids_from_file(filename: str) -> List[str]:
    """Get datapoint IDs from file metadata in GCS."""
    try:
        # Google Cloud credentials are loaded from .env file via config.py
        storage_client = storage.Client(project=settings.google_cloud_project_id)
        bucket = storage_client.bucket(settings.storage_bucket_name)
        
        # Clean filename to match upload format
        clean_filename = filename.replace(" ", "_").replace(":", "-").replace("/", "-")
        upload_path = f"uploads/{clean_filename}"
        blob = bucket.blob(upload_path)
        
        # Check if file exists and get metadata
        if not blob.exists():
            return []
        
        blob.reload()
        metadata = blob.metadata or {}
        datapoint_ids_str = metadata.get("datapoint_ids", "")
        
        if datapoint_ids_str:
            return datapoint_ids_str.split(",")
        return []
        
    except Exception as e:
        print(f"Warning: Could not get datapoint IDs for {filename}: {e}")
        return []


async def remove_existing_embeddings(filename: str) -> bool:
    """Remove existing embeddings for a file from Vector Search."""
    try:
        # Get datapoint IDs from file metadata
        datapoint_ids = await get_datapoint_ids_from_file(filename)
        
        if not datapoint_ids:
            print(f"No datapoint IDs found for {filename}, trying metadata filter approach")
            # Fallback to metadata filter approach
            success = await vector_search_service.remove_embeddings_by_metadata(
                metadata_filter={"filename": filename},
                index_id=settings.vector_search_index_id,
                endpoint_id=settings.vector_search_index_endpoint_id
            )
            return success
        
        # Remove embeddings by datapoint IDs
        success = await vector_search_service.remove_embeddings_by_ids(
            datapoint_ids=datapoint_ids,
            index_id=settings.vector_search_index_id,
            endpoint_id=settings.vector_search_index_endpoint_id
        )
        
        return success
    except Exception as e:
        print(f"Warning: Could not remove existing embeddings for {filename}: {e}")
        return False


@router.post("/upload/direct")
async def upload_direct_file(
    file: UploadFile = File(...),
    replace_existing: bool = Form(True),
    tags: str = Form("")
):
    """
    Upload a file directly to the knowledge base.
    
    This endpoint:
    1. Processes the uploaded file
    2. Chunks the content based on configuration
    3. Generates embeddings using Vertex AI
    4. Stores embeddings in Vector Search
    5. Uploads file to Google Cloud Storage
    """
    try:
        # Read file content
        file_content = await file.read()
        filename = file.filename
        content_type = file.content_type
        
        # Parse tags (comma-separated string)
        file_tags = [tag.strip() for tag in tags.split(",") if tag.strip()] if tags else []
        
        # Handle content type mapping for files detected as application/octet-stream
        if content_type == "application/octet-stream" or not content_type:
            import mimetypes
            file_extension = os.path.splitext(filename)[1].lower()
            extension_to_content_type = {
                ".pdf": "application/pdf",
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                ".xls": "application/vnd.ms-excel",
                ".csv": "text/csv"
            }
            if file_extension in extension_to_content_type:
                content_type = extension_to_content_type[file_extension]
        
        # Check if file already exists and remove existing embeddings
        if replace_existing:
            await remove_existing_embeddings(filename)
        
        # Process document and create embeddings
        processing_result = await process_and_embed_document(
            file_content=file_content,
            filename=filename,
            content_type=content_type,
            tags=file_tags
        )
        
        # Store embeddings in Vector Search
        embedding_success = await store_embeddings_in_vector_search(
            embeddings=processing_result["embeddings"],
            filename=filename
        )
        
        if not embedding_success:
            raise RAGAPIException("Failed to store embeddings in Vector Search")
        
        # Extract datapoint IDs for storage in metadata
        datapoint_ids = [emb["id"] for emb in processing_result["embeddings"]]
        
        # Upload file to uploads directory
        upload_path = await upload_file_to_uploads(
            file_content=file_content,
            filename=filename,
            content_type=content_type,
            datapoint_ids=datapoint_ids,
            tags=file_tags
        )
        
        clean_filename = filename.replace(" ", "_").replace(":", "-").replace("/", "-")
        
        return {
            "success": True,
            "filename": clean_filename,
            "original_filename": filename,
            "upload_path": upload_path,
            "file_size": len(file_content),
            "content_type": content_type,
            "total_chunks": processing_result["total_chunks"],
            "embeddings_stored": len(processing_result["embeddings"]),
            "message": f"File '{filename}' successfully uploaded and processed for knowledge base"
        }
        
    except RAGAPIException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/upload/from-temp/{validation_id}")
async def upload_from_temp_storage(validation_id: str = Path(...)):
    """
    Upload a file from temporary storage to the knowledge base.
    
    This endpoint:
    1. Retrieves file from temp storage using validation_id
    2. Processes the file and creates embeddings
    3. Stores embeddings in Vector Search
    4. Moves file from temp to uploads directory
    """
    try:
        # Get file from temp storage
        temp_file_info = await get_file_from_temp_storage(validation_id)
        
        if not temp_file_info:
            raise HTTPException(
                status_code=404,
                detail=f"File with validation_id '{validation_id}' not found in temporary storage"
            )
        
        # Download file content from temp storage
        file_content = await download_file_from_gcs(temp_file_info["temp_path"])
        
        # Handle content type mapping for files detected as application/octet-stream
        content_type = temp_file_info["content_type"]
        if content_type == "application/octet-stream" or not content_type:
            file_extension = os.path.splitext(temp_file_info["filename"])[1].lower()
            extension_to_content_type = {
                ".pdf": "application/pdf",
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                ".xls": "application/vnd.ms-excel",
                ".csv": "text/csv"
            }
            if file_extension in extension_to_content_type:
                content_type = extension_to_content_type[file_extension]
        
        # Remove existing embeddings if file exists
        await remove_existing_embeddings(temp_file_info["filename"])
        
        # Process document and create embeddings
        processing_result = await process_and_embed_document(
            file_content=file_content,
            filename=temp_file_info["filename"],
            content_type=content_type,
            tags=[]  # No tags for temp uploads for now
        )
        
        # Store embeddings in Vector Search
        embedding_success = await store_embeddings_in_vector_search(
            embeddings=processing_result["embeddings"],
            filename=temp_file_info["filename"]
        )
        
        if not embedding_success:
            raise RAGAPIException("Failed to store embeddings in Vector Search")
        
        # Clean filename for consistent handling
        clean_filename = temp_file_info["filename"].replace(" ", "_").replace(":", "-").replace("/", "-")
        
        # Extract datapoint IDs for storage in metadata
        datapoint_ids = [emb["id"] for emb in processing_result["embeddings"]]
        
        # Upload file to uploads directory
        upload_path = await upload_file_to_uploads(
            file_content=file_content,
            filename=temp_file_info["filename"],
            content_type=content_type,
            datapoint_ids=datapoint_ids,
            tags=[]  # No tags for temp uploads for now
        )
        
        # Delete file from temp storage
        await delete_file_from_gcs(temp_file_info["temp_path"])
        
        return {
            "success": True,
            "validation_id": validation_id,
            "filename": clean_filename,
            "original_filename": temp_file_info["filename"],
            "upload_path": upload_path,
            "file_size": temp_file_info["size"],
            "content_type": content_type,
            "total_chunks": processing_result["total_chunks"],
            "embeddings_stored": len(processing_result["embeddings"]),
            "message": f"File '{temp_file_info['filename']}' successfully uploaded and processed for knowledge base"
        }
        
    except HTTPException:
        raise
    except RAGAPIException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/upload/status/{filename}")
async def get_upload_status(filename: str = Path(...)):
    """
    Get the upload status and information for a file.
    
    This endpoint provides information about:
    - File existence in uploads directory
    - Number of chunks and embeddings
    - Last modified date
    """
    try:
        # Google Cloud credentials are loaded from .env file via config.py
        storage_client = storage.Client(project=settings.google_cloud_project_id)
        bucket = storage_client.bucket(settings.storage_bucket_name)
        
        # Check if file exists in uploads
        upload_path = f"uploads/{filename}"
        blob = bucket.blob(upload_path)
        
        if not blob.exists():
            return {
                "filename": filename,
                "exists": False,
                "message": "File not found in uploads directory"
            }
        
        # Get file information
        blob.reload()
        
        return {
            "filename": filename,
            "exists": True,
            "upload_path": upload_path,
            "file_size": blob.size,
            "content_type": blob.content_type,
            "created": blob.time_created.isoformat(),
            "updated": blob.updated.isoformat(),
            "message": "File found in uploads directory"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking upload status: {str(e)}")


@router.delete("/upload/delete")
async def delete_uploaded_file(filename: str = Query(..., description="Name of the file to delete")):
    """
    Delete an uploaded file and its embeddings.
    
    This endpoint:
    1. Removes embeddings from Vector Search
    2. Deletes file from uploads directory
    """
    try:
        # Remove embeddings from Vector Search
        embedding_removed = await remove_existing_embeddings(filename)
        
        # Delete file from uploads directory
        upload_path = f"uploads/{filename}"
        file_deleted = await delete_file_from_gcs(upload_path)
        
        if not file_deleted:
            raise HTTPException(
                status_code=404,
                detail=f"File '{filename}' not found in uploads directory"
            )
        
        return {
            "success": True,
            "filename": filename,
            "embeddings_removed": embedding_removed,
            "file_deleted": file_deleted,
            "message": f"File '{filename}' and its embeddings have been removed"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")
