#!/usr/bin/env python3
"""Debug authentication issues."""

import os
import sys
from pathlib import Path

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent))

from app.core.config import settings
from google.cloud import storage

def debug_authentication():
    """Debug Google Cloud authentication."""
    print("=== Authentication Debug ===")
    print(f"GOOGLE_CLOUD_PROJECT_ID: {settings.google_cloud_project_id}")
    print(f"GOOGLE_APPLICATION_CREDENTIALS: {settings.google_application_credentials}")
    print(f"STORAGE_BUCKET_NAME: {settings.storage_bucket_name}")
    print()
    
    # Check if credentials file exists
    creds_path = settings.google_application_credentials
    print(f"Credentials file exists: {os.path.exists(creds_path)}")
    print(f"Credentials file path: {os.path.abspath(creds_path)}")
    print()
    
    # Check environment variables
    print("Environment variables:")
    print(f"  GOOGLE_APPLICATION_CREDENTIALS: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}")
    print(f"  GOOGLE_CLOUD_PROJECT_ID: {os.getenv('GOOGLE_CLOUD_PROJECT_ID')}")
    print()
    
    # Test Google Cloud Storage connection
    try:
        print("Testing Google Cloud Storage connection...")
        storage_client = storage.Client(project=settings.google_cloud_project_id)
        bucket_name = settings.storage_bucket_name
        
        print(f"Attempting to access bucket: {bucket_name}")
        bucket = storage_client.bucket(bucket_name)
        
        # Try to check if bucket exists
        exists = bucket.exists()
        print(f"Bucket exists: {exists}")
        
        if exists:
            print("✅ Successfully connected to Google Cloud Storage!")
        else:
            print("❌ Bucket does not exist or is not accessible")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"Error type: {type(e).__name__}")
        
        # Check if it's a permissions issue
        if "403" in str(e) or "Forbidden" in str(e):
            print("🔍 This is a permissions issue - the service account needs Storage Admin permissions")
        elif "401" in str(e) or "Unauthorized" in str(e):
            print("🔍 This is an authentication issue - check your service account key")
        elif "invalid_grant" in str(e):
            print("🔍 This is an authentication issue - the service account key might be invalid or expired")

if __name__ == "__main__":
    debug_authentication()
