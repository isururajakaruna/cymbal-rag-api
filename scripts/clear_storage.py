#!/usr/bin/env python3
"""Script to clear Google Cloud Storage folders (_tmp and uploads)."""

import os
import sys
from pathlib import Path
from google.cloud import storage
from google.cloud.exceptions import NotFound

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.core.config import settings


def clear_storage_folders():
    """Clear both _tmp and uploads folders in Google Cloud Storage."""
    try:
        # Initialize Google Cloud Storage client
        storage_client = storage.Client(project=settings.google_cloud_project_id)
        bucket_name = settings.storage_bucket_name
        
        print(f"Connecting to Google Cloud Storage bucket: {bucket_name}")
        
        # Get the bucket
        try:
            bucket = storage_client.bucket(bucket_name)
            if not bucket.exists():
                print(f"Error: Bucket '{bucket_name}' does not exist!")
                return False
        except Exception as e:
            print(f"Error accessing bucket: {e}")
            return False
        
        # Folders to clear
        folders_to_clear = ["_tmp", "uploads"]
        
        total_deleted = 0
        
        for folder in folders_to_clear:
            print(f"\nClearing folder: {folder}/")
            
            # List all blobs with the folder prefix
            blobs = bucket.list_blobs(prefix=f"{folder}/")
            blob_count = 0
            
            for blob in blobs:
                try:
                    blob.delete()
                    blob_count += 1
                    total_deleted += 1
                    print(f"  Deleted: {blob.name}")
                except Exception as e:
                    print(f"  Error deleting {blob.name}: {e}")
            
            if blob_count == 0:
                print(f"  No files found in {folder}/")
            else:
                print(f"  Deleted {blob_count} files from {folder}/")
        
        print(f"\n=== Summary ===")
        print(f"Total files deleted: {total_deleted}")
        print(f"Folders cleared: {', '.join(folders_to_clear)}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False


def confirm_clear():
    """Ask for confirmation before clearing storage."""
    print("WARNING: This will permanently delete ALL files in the following folders:")
    print("  - _tmp/ (temporary files)")
    print("  - uploads/ (uploaded files)")
    print()
    
    response = input("Are you sure you want to proceed? (yes/no): ").strip().lower()
    return response in ['yes', 'y']


def main():
    """Main function."""
    print("=== Google Cloud Storage Cleanup Script ===")
    print(f"Project ID: {settings.google_cloud_project_id}")
    print(f"Bucket: {settings.storage_bucket_name}")
    print()
    
    # Check if credentials are set
    if not settings.google_application_credentials:
        print("Error: GOOGLE_APPLICATION_CREDENTIALS not set!")
        print("Please set the environment variable to your service account key file.")
        sys.exit(1)
    
    if not os.path.exists(settings.google_application_credentials):
        print(f"Error: Credentials file not found: {settings.google_application_credentials}")
        sys.exit(1)
    
    # Ask for confirmation
    if not confirm_clear():
        print("Operation cancelled.")
        sys.exit(0)
    
    # Clear the storage
    if clear_storage_folders():
        print("\nStorage cleanup completed successfully!")
    else:
        print("\nStorage cleanup failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
