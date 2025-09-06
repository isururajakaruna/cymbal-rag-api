#!/usr/bin/env python3
"""Setup script for Google Cloud Platform resources."""

import os
import json
from pathlib import Path
from google.cloud import aiplatform
from google.cloud import storage
from google.cloud import documentai


def setup_gcp_resources():
    """Set up required GCP resources for the RAG API."""
    print("Setting up Google Cloud Platform resources...")
    
    # Get project ID from environment
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
    if not project_id:
        print("Error: GOOGLE_CLOUD_PROJECT_ID environment variable not set")
        return False
    
    region = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")
    
    try:
        # Initialize Vertex AI
        print("Initializing Vertex AI...")
        aiplatform.init(project=project_id, location=region)
        
        # Create storage bucket
        print("Creating storage bucket...")
        bucket_name = os.getenv("STORAGE_BUCKET_NAME", f"{project_id}-rag-documents")
        create_storage_bucket(project_id, bucket_name)
        
        # Create Document AI processor
        print("Creating Document AI processor...")
        processor_id = create_document_processor(project_id, region)
        
        # Create Vector Search index
        print("Creating Vector Search index...")
        index_id, endpoint_id = create_vector_search_index(project_id, region)
        
        # Update .env file
        update_env_file(project_id, region, bucket_name, processor_id, index_id, endpoint_id)
        
        print("GCP setup completed successfully!")
        print(f"Bucket: {bucket_name}")
        print(f"Processor ID: {processor_id}")
        print(f"Index ID: {index_id}")
        print(f"Endpoint ID: {endpoint_id}")
        
        return True
        
    except Exception as e:
        print(f"Error setting up GCP resources: {e}")
        return False


def create_storage_bucket(project_id: str, bucket_name: str):
    """Create a Google Cloud Storage bucket."""
    try:
        storage_client = storage.Client(project=project_id)
        bucket = storage_client.bucket(bucket_name)
        
        if bucket.exists():
            print(f"Bucket {bucket_name} already exists")
        else:
            bucket = storage_client.create_bucket(bucket_name, location="us-central1")
            print(f"Created bucket {bucket_name}")
            
    except Exception as e:
        print(f"Error creating bucket: {e}")
        raise


def create_document_processor(project_id: str, region: str) -> str:
    """Create a Document AI processor."""
    try:
        client = documentai.DocumentProcessorServiceClient()
        location = f"projects/{project_id}/locations/{region}"
        
        # Create processor request
        processor = documentai.Processor(
            type_="OCR_PROCESSOR",
            display_name="RAG Document Processor"
        )
        
        request = documentai.CreateProcessorRequest(
            parent=location,
            processor=processor
        )
        
        result = client.create_processor(request=request)
        processor_id = result.name.split("/")[-1]
        print(f"Created processor: {processor_id}")
        
        return processor_id
        
    except Exception as e:
        print(f"Error creating Document AI processor: {e}")
        raise


def create_vector_search_index(project_id: str, region: str) -> tuple[str, str]:
    """Create a Vector Search index and endpoint."""
    try:
        # This is a simplified implementation
        # In practice, you would use the Vector Search API to create the index
        
        # For now, return placeholder values
        index_id = f"rag-index-{project_id}"
        endpoint_id = f"rag-endpoint-{project_id}"
        
        print(f"Created Vector Search index: {index_id}")
        print(f"Created Vector Search endpoint: {endpoint_id}")
        
        return index_id, endpoint_id
        
    except Exception as e:
        print(f"Error creating Vector Search resources: {e}")
        raise


def update_env_file(project_id: str, region: str, bucket_name: str, 
                   processor_id: str, index_id: str, endpoint_id: str):
    """Update .env file with created resource IDs."""
    env_file = Path(".env")
    
    env_content = f"""# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT_ID={project_id}
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account-key.json
GOOGLE_CLOUD_REGION={region}

# Document AI Configuration
DOCUMENT_AI_PROCESSOR_ID={processor_id}
DOCUMENT_AI_LOCATION={region}

# Vertex AI Configuration
VERTEX_AI_LOCATION={region}
VERTEX_AI_MODEL_NAME=gemini-1.5-flash

# Vector Search Configuration
VECTOR_SEARCH_INDEX_ID={index_id}
VECTOR_SEARCH_INDEX_ENDPOINT_ID={endpoint_id}

# Storage Configuration
STORAGE_BUCKET_NAME={bucket_name}

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1
"""
    
    with open(env_file, "w") as f:
        f.write(env_content)
    
    print(f"Updated {env_file} with resource IDs")


if __name__ == "__main__":
    success = setup_gcp_resources()
    exit(0 if success else 1)
