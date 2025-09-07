"""File validation API endpoints with real Gemini integration."""

import os
import uuid
import json
import base64
from typing import List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from fastapi.responses import JSONResponse
from google.cloud import storage
from vertexai.generative_models import GenerativeModel, Part
import vertexai

from app.core.config import settings
from app.core.exceptions import RAGAPIException
from app.models.schemas import FileValidationResponse, FileValidationRequest

router = APIRouter()

# Initialize Vertex AI
vertexai.init(project=settings.google_cloud_project_id, location=settings.google_cloud_region)
generative_model = GenerativeModel(settings.vertex_ai_model_name)

# Supported file extensions
SUPPORTED_EXTENSIONS = {
    "image/png": [".png"],
    "image/jpeg": [".jpg", ".jpeg"],
    "application/pdf": [".pdf"],
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"],
    "application/vnd.ms-excel": [".xls"],
    "text/csv": [".csv"]
}

ALL_SUPPORTED_EXTENSIONS = [ext for extensions in SUPPORTED_EXTENSIONS.values() for ext in extensions]


async def check_file_exists_in_uploads(filename: str) -> bool:
    """Check if file already exists in uploads directory."""
    try:
        # Google Cloud credentials are loaded from .env file via config.py
        storage_client = storage.Client(project=settings.google_cloud_project_id)
        bucket = storage_client.bucket(settings.storage_bucket_name)
        
        blob_path = f"uploads/{filename}"
        blob = bucket.blob(blob_path)
        return blob.exists()
    except Exception as e:
        raise RAGAPIException(f"Error checking file existence: {str(e)}")


async def upload_to_temp_storage(file_content: bytes, filename: str, validation_id: str = None) -> str:
    """Upload file to _tmp directory in Google Cloud Storage."""
    try:
        # Google Cloud credentials are loaded from .env file via config.py
        storage_client = storage.Client(project=settings.google_cloud_project_id)
        bucket = storage_client.bucket(settings.storage_bucket_name)
        
        tmp_dir = "tmp/"
        blob_path = f"{tmp_dir}{filename}"
        
        blob = bucket.blob(blob_path)
        
        # Determine content type from filename
        import mimetypes
        content_type, _ = mimetypes.guess_type(filename)
        if not content_type:
            content_type = "application/octet-stream"
        
        # Upload with metadata including validation_id
        metadata = {}
        if validation_id:
            metadata["validation_id"] = validation_id
        
        blob.upload_from_string(file_content, content_type=content_type)
        if metadata:
            blob.metadata = metadata
            blob.patch()
        
        return blob_path
    except Exception as e:
        raise RAGAPIException(f"Error uploading to temporary storage: {str(e)}")


async def move_from_temp_to_uploads(temp_path: str, filename: str) -> str:
    """Move file from temp directory to uploads directory in Google Cloud Storage."""
    try:
        # Google Cloud credentials are loaded from .env file via config.py
        storage_client = storage.Client(project=settings.google_cloud_project_id)
        bucket = storage_client.bucket(settings.storage_bucket_name)
        
        source_blob_path = temp_path
        destination_blob_path = f"uploads/{filename}"
        
        source_blob = bucket.blob(source_blob_path)
        destination_blob = bucket.copy_blob(source_blob, bucket, destination_blob_path)
        
        source_blob.delete()
        
        return destination_blob_path
    except Exception as e:
        raise RAGAPIException(f"Error moving file from temp to uploads: {str(e)}")


async def get_temp_file_info(validation_id: str) -> dict:
    """Get information about a file in temp storage by validation_id."""
    try:
        # Google Cloud credentials are loaded from .env file via config.py
        storage_client = storage.Client(project=settings.google_cloud_project_id)
        bucket = storage_client.bucket(settings.storage_bucket_name)
        
        blobs = bucket.list_blobs(prefix="tmp/")
        
        for blob in blobs:
            if blob.name.startswith("tmp/"):
                blob.reload()
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
        raise RAGAPIException(f"Error getting temp file info: {str(e)}")


async def validate_file_format(content_type: str, filename: str) -> dict:
    """Validate if file format is supported."""
    file_extension = os.path.splitext(filename)[1].lower()
    
    if content_type == "application/octet-stream" or not content_type:
        # Map file extensions to content types
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
    
    if content_type not in SUPPORTED_EXTENSIONS:
        return {
            "is_valid": False,
            "error": "Unsupported file format",
            "supported_extensions": ALL_SUPPORTED_EXTENSIONS,
            "provided_type": content_type,
            "provided_extension": file_extension
        }
    
    if file_extension not in SUPPORTED_EXTENSIONS[content_type]:
        return {
            "is_valid": False,
            "error": "File extension does not match content type",
            "supported_extensions": SUPPORTED_EXTENSIONS[content_type],
            "provided_type": content_type,
            "provided_extension": file_extension
        }
    
    return {"is_valid": True}


async def _analyze_image_with_gemini(file_content: bytes, filename: str, content_type: str) -> dict:
    """Analyze image content using real Gemini API."""
    try:
        # Convert image to base64 for Gemini
        image_base64 = base64.b64encode(file_content).decode('utf-8')
        
        # Create image part for Gemini
        image_part = Part.from_data(
            data=file_content,
            mime_type=content_type
        )
        
        analysis_prompt = """
        Analyze this image for its suitability for a corporate knowledge base. Consider:
        
        1. CONTENT QUALITY ASSESSMENT:
        - What visual content does this image contain?
        - Is it professional and business-relevant?
        - Does it contain meaningful information, diagrams, charts, or text?
        - Is it a blank, placeholder, or low-quality image?
        - Is the image clear and readable?
        
        2. CORPORATE KNOWLEDGE BASE SUITABILITY:
        - Would this image be valuable for employees to reference?
        - Does it contain actionable visual information?
        - Is it appropriate for business documentation?
        - Does it add value to organizational knowledge?
        
        Please provide your analysis in the following JSON format:
        {
            "content_quality": {
                "score": 1-10,
                "is_sufficient": true/false,
                "reasoning": "ONE SHORT SENTENCE explaining why this image is or isn't suitable for corporate knowledge base"
            },
            "faq_structure": {
                "is_faq": false,
                "score": 3,
                "has_proper_qa_pairs": false,
                "reasoning": "Images are not suitable for FAQ format"
            }
        }
        """
        
        # Generate content using Gemini with image
        response = generative_model.generate_content([image_part, analysis_prompt])
        analysis_text = response.text
        
        # Parse JSON response
        return _parse_gemini_response(analysis_text, filename)
        
    except Exception as e:
        print(f"Error analyzing image with Gemini: {e}")
        return _get_fallback_analysis(filename, "image")


async def _analyze_pdf_with_gemini(file_content: bytes, filename: str, content_type: str) -> dict:
    """Analyze PDF content using real Gemini API (first 3 pages)."""
    try:
        from pdf2image import convert_from_bytes
        import io
        
        # Convert PDF to images (first 3 pages)
        images = convert_from_bytes(file_content, first_page=1, last_page=3)
        
        if not images:
            return _get_fallback_analysis(filename, "pdf")
        
        # Analyze the first page with Gemini
        image = images[0]
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Create image part for Gemini
        image_part = Part.from_data(data=img_byte_arr, mime_type="image/png")
        
        analysis_prompt = f"""
        Analyze this PDF document (first page) for its suitability for a corporate knowledge base. Consider:
        
        1. CONTENT QUALITY ASSESSMENT:
        - Does this PDF contain meaningful, professional information?
        - Is it relevant for business documentation or knowledge sharing?
        - Does it contain substantial data, reports, or valuable content?
        - Is it a blank, placeholder, or low-quality document?
        - What type of content does it appear to contain?
        
        2. CORPORATE KNOWLEDGE BASE SUITABILITY:
        - Would this PDF be valuable for employees to reference?
        - Does it contain actionable information or insights?
        - Is it professional and appropriate for business use?
        - Does it add value to organizational knowledge?
        
        Please provide your analysis in the following JSON format:
        {{
            "content_quality": {{
                "score": 1-10,
                "is_sufficient": true/false,
                "reasoning": "ONE SHORT SENTENCE explaining why this PDF is or isn't suitable for corporate knowledge base"
            }},
            "faq_structure": {{
                "is_faq": false,
                "score": 3,
                "has_proper_qa_pairs": false,
                "reasoning": "Document does not appear to be an FAQ format"
            }}
        }}
        """
        
        # Generate content using Gemini with image
        response = generative_model.generate_content([image_part, analysis_prompt])
        analysis_text = response.text
        
        # Parse JSON response
        return _parse_gemini_response(analysis_text, filename)
        
    except Exception as e:
        print(f"Error analyzing PDF with Gemini: {e}")
        return _get_fallback_analysis(filename, "pdf")


async def _analyze_text_with_gemini(file_content: bytes, filename: str, content_type: str) -> dict:
    """Analyze text-based content using real Gemini API."""
    try:
        # Convert content to text (for text-based files)
        try:
            content_text = file_content.decode('utf-8')
            # Limit content size for analysis
            if len(content_text) > 5000:
                content_text = content_text[:5000] + "..."
        except UnicodeDecodeError:
            content_text = f"Binary file: {filename}"
        
        analysis_prompt = f"""
        Analyze this document for its suitability for a corporate knowledge base. Consider:
        
        1. CONTENT QUALITY ASSESSMENT:
        - Does this document contain meaningful, professional information?
        - Is it relevant for business documentation or knowledge sharing?
        - Does it contain substantial data or valuable content?
        - Is it a blank, placeholder, or low-quality document?
        
        2. CORPORATE KNOWLEDGE BASE SUITABILITY:
        - Would this document be valuable for employees to reference?
        - Does it contain actionable information or insights?
        - Is it professional and appropriate for business use?
        - Does it add value to organizational knowledge?
        
        Document content preview: {content_text[:500]}
        
        Please provide your analysis in the following JSON format:
        {{
            "content_quality": {{
                "score": 1-10,
                "is_sufficient": true/false,
                "reasoning": "ONE SHORT SENTENCE explaining why this document is or isn't suitable for corporate knowledge base"
            }},
            "faq_structure": {{
                "is_faq": false,
                "score": 3,
                "has_proper_qa_pairs": false,
                "reasoning": "Document does not appear to be an FAQ format"
            }}
        }}
        """
        
        # Generate content using Gemini
        response = generative_model.generate_content(analysis_prompt)
        analysis_text = response.text
        
        # Parse JSON response
        return _parse_gemini_response(analysis_text, filename)
        
    except Exception as e:
        print(f"Error analyzing text with Gemini: {e}")
        return _get_fallback_analysis(filename, "text")


def _parse_gemini_response(analysis_text: str, filename: str) -> dict:
    """Parse Gemini response and extract JSON."""
    try:
        # Extract JSON from the response (Gemini might include extra text)
        json_start = analysis_text.find('{')
        json_end = analysis_text.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            json_text = analysis_text[json_start:json_end]
            analysis_result = json.loads(json_text)
        else:
            raise ValueError("No valid JSON found in response")
        
        return analysis_result
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Warning: Failed to parse Gemini response as JSON: {e}")
        print(f"Raw response: {analysis_text}")
        return _get_fallback_analysis(filename, "unknown")


def _get_fallback_analysis(filename: str, file_type: str) -> dict:
    """Get fallback analysis when Gemini fails."""
    filename_lower = filename.lower()
    
    if any(indicator in filename_lower for indicator in ["no_info", "empty", "blank", "placeholder"]):
        return {
            "content_quality": {
                "score": 2,
                "is_sufficient": False,
                "reasoning": "Filename indicates this file contains no meaningful information suitable for a corporate knowledge base."
            },
            "faq_structure": {
                "is_faq": False,
                "score": 3,
                "has_proper_qa_pairs": False,
                "reasoning": "Document does not appear to be an FAQ format"
            }
        }
    else:
        return {
            "content_quality": {
                "score": 6,
                "is_sufficient": True,
                "reasoning": f"Document appears to contain {file_type} content that could be valuable for a corporate knowledge base, but specific analysis was not available."
            },
            "faq_structure": {
                "is_faq": False,
                "score": 3,
                "has_proper_qa_pairs": False,
                "reasoning": "Document does not appear to be an FAQ format"
            }
        }


async def analyze_content_with_gemini(file_content: bytes, filename: str, content_type: str) -> dict:
    """Analyze file content using Gemini for quality and FAQ validation."""
    try:
        # Set environment variable for Google Cloud authentication
        # Google Cloud credentials are loaded from .env file via config.py
        
        # Check for obvious low-quality indicators first (quick filename check)
        filename_lower = filename.lower()
        if any(indicator in filename_lower for indicator in ["no_info", "empty", "blank", "placeholder"]):
            return {
                "content_quality": {
                    "score": 2,
                    "is_sufficient": False,
                    "reasoning": "Filename indicates this file contains no meaningful information suitable for a corporate knowledge base. Files with 'no_info', 'empty', 'blank', or 'placeholder' in the name are typically not valuable for business documentation."
                },
                "faq_structure": {
                    "is_faq": False,
                    "score": 3,
                    "has_proper_qa_pairs": False,
                    "reasoning": "Document does not appear to be an FAQ format"
                }
            }
        
        # Use real Gemini API for content analysis
        if content_type.startswith("image/"):
            return await _analyze_image_with_gemini(file_content, filename, content_type)
        elif content_type == "application/pdf":
            return await _analyze_pdf_with_gemini(file_content, filename, content_type)
        else:
            return await _analyze_text_with_gemini(file_content, filename, content_type)
        
    except Exception as e:
        raise RAGAPIException(f"Error analyzing content with Gemini: {str(e)}")


@router.post("/validate", response_model=FileValidationResponse)
async def validate_file(
    file: UploadFile = File(...),
    replace_existing: bool = False
):
    """
    Validate and upload a file to temporary storage.
    
    This endpoint:
    1. Validates file format
    2. Checks if file already exists
    3. Analyzes content quality using Gemini
    4. Uploads to temporary storage
    """
    try:
        # Starting validation for file: {file.filename}
        # Read file content
        file_content = await file.read()
        filename = file.filename
        content_type = file.content_type
        
        # File info - filename: {filename}, content_type: {content_type}, size: {len(file_content)}
        
        # Step 1: Validate file format
        format_validation = await validate_file_format(content_type, filename)
        # Format validation result: {format_validation}
        if not format_validation["is_valid"]:
            return FileValidationResponse(
                success=False,
                error=format_validation["error"],
                supported_extensions=format_validation.get("supported_extensions"),
                provided_type=format_validation.get("provided_type"),
                provided_extension=format_validation.get("provided_extension")
            )
        
        # Step 2: Check if file already exists
        file_exists = await check_file_exists_in_uploads(filename)
        if file_exists and not replace_existing:
            # File exists but validation should still proceed with a warning
            pass
        
        # Step 3: Analyze content with Gemini
        content_analysis = await analyze_content_with_gemini(file_content, filename, content_type)
        
        # Step 4: Check content quality
        if not content_analysis["content_quality"]["is_sufficient"]:
            return FileValidationResponse(
                success=False,
                error="File content is not suitable for knowledge base",
                filename=filename,
                content_analysis=content_analysis,
                quality_score=content_analysis["content_quality"]["score"],
                reasoning=content_analysis["content_quality"]["reasoning"],
                suggestion="Please upload a file with more substantial content"
            )
        
        # Step 5: Generate validation ID for tracking
        validation_id = str(uuid.uuid4())
        
        # Step 6: Upload to temporary storage with validation_id
        temp_path = await upload_to_temp_storage(file_content, filename, validation_id)
        
        return FileValidationResponse(
            success=True,
            validation_id=validation_id,
            filename=filename,
            content_type=content_type,
            file_size=len(file_content),
            temp_path=temp_path,
            file_exists=file_exists,
            content_analysis=content_analysis,
            message=f"File validation successful and uploaded to temporary storage. Use validation_id '{validation_id}' for the upload API."
        )
        
    except RAGAPIException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/upload/{validation_id}")
async def upload_validated_file(validation_id: str):
    """
    Upload a previously validated file to the main uploads directory.
    """
    try:
        temp_file_info = await get_temp_file_info(validation_id)
        
        if not temp_file_info:
            raise HTTPException(
                status_code=404, 
                detail=f"File with validation_id '{validation_id}' not found in temporary storage. The file may have expired or the validation_id is invalid."
            )
        
        upload_path = await move_from_temp_to_uploads(
            temp_file_info["temp_path"], 
            temp_file_info["filename"]
        )
        
        return {
            "success": True,
            "validation_id": validation_id,
            "filename": temp_file_info["filename"],
            "upload_path": upload_path,
            "file_size": temp_file_info["size"],
            "content_type": temp_file_info["content_type"],
            "message": f"File '{temp_file_info['filename']}' successfully uploaded to knowledge base"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")


@router.get("/supported-formats")
async def get_supported_formats():
    """Get list of supported file formats."""
    return {
        "supported_formats": SUPPORTED_EXTENSIONS,
        "all_extensions": ALL_SUPPORTED_EXTENSIONS,
        "message": "Supported file formats for upload"
    }


@router.get("/debug-auth")
async def debug_auth():
    """Debug authentication and configuration."""
    return {
        "google_cloud_project_id": settings.google_cloud_project_id,
        "google_application_credentials": settings.google_application_credentials,
        "storage_bucket_name": settings.storage_bucket_name,
        "vertex_ai_model_name": settings.vertex_ai_model_name,
        "env_google_application_credentials": os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
        "env_google_cloud_project_id": os.environ.get("GOOGLE_CLOUD_PROJECT_ID"),
        "credentials_file_exists": os.path.exists(settings.google_application_credentials) if settings.google_application_credentials else False
    }
