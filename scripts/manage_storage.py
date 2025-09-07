#!/usr/bin/env python3
"""Advanced script to manage Google Cloud Storage folders."""

import os
import sys
import argparse
from pathlib import Path
from google.cloud import storage
from google.cloud.exceptions import NotFound

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.core.config import settings


def list_files_in_folder(bucket, folder_prefix, show_details=False):
    """List files in a specific folder."""
    blobs = bucket.list_blobs(prefix=folder_prefix)
    files = []
    
    for blob in blobs:
        files.append({
            'name': blob.name,
            'size': blob.size,
            'created': blob.time_created,
            'updated': blob.updated
        })
    
    if show_details:
        print(f"\nFiles in {folder_prefix}:")
        print("-" * 80)
        for file_info in files:
            print(f"Name: {file_info['name']}")
            print(f"Size: {file_info['size']} bytes")
            print(f"Created: {file_info['created']}")
            print(f"Updated: {file_info['updated']}")
            print("-" * 80)
    else:
        print(f"\nFiles in {folder_prefix}:")
        for file_info in files:
            print(f"  {file_info['name']} ({file_info['size']} bytes)")
    
    return files


def clear_folder(bucket, folder_prefix, dry_run=False):
    """Clear all files in a specific folder."""
    blobs = bucket.list_blobs(prefix=folder_prefix)
    files_to_delete = list(blobs)
    
    if not files_to_delete:
        print(f"No files found in {folder_prefix}")
        return 0
    
    print(f"\nFound {len(files_to_delete)} files in {folder_prefix}")
    
    if dry_run:
        print("DRY RUN - Files that would be deleted:")
        for blob in files_to_delete:
            print(f"  {blob.name}")
        return len(files_to_delete)
    
    deleted_count = 0
    for blob in files_to_delete:
        try:
            blob.delete()
            deleted_count += 1
            print(f"  Deleted: {blob.name}")
        except Exception as e:
            print(f"  Error deleting {blob.name}: {e}")
    
    return deleted_count


def get_storage_stats(bucket):
    """Get storage statistics."""
    print("\n=== Storage Statistics ===")
    
    folders = ["_tmp", "uploads"]
    total_files = 0
    total_size = 0
    
    for folder in folders:
        blobs = bucket.list_blobs(prefix=f"{folder}/")
        folder_files = list(blobs)
        folder_size = sum(blob.size for blob in folder_files)
        
        print(f"{folder}/: {len(folder_files)} files, {folder_size} bytes")
        total_files += len(folder_files)
        total_size += folder_size
    
    print(f"Total: {total_files} files, {total_size} bytes")
    return total_files, total_size


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Manage Google Cloud Storage folders")
    parser.add_argument("--list", action="store_true", help="List files in folders")
    parser.add_argument("--list-details", action="store_true", help="List files with detailed information")
    parser.add_argument("--clear-tmp", action="store_true", help="Clear _tmp folder")
    parser.add_argument("--clear-uploads", action="store_true", help="Clear uploads folder")
    parser.add_argument("--clear-all", action="store_true", help="Clear both folders")
    parser.add_argument("--stats", action="store_true", help="Show storage statistics")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without actually deleting")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompts")
    
    args = parser.parse_args()
    
    print("=== Google Cloud Storage Management ===")
    print(f"Project ID: {settings.google_cloud_project_id}")
    print(f"Bucket: {settings.storage_bucket_name}")
    
    # Check if credentials are set
    if not settings.google_application_credentials:
        print("Error: GOOGLE_APPLICATION_CREDENTIALS not set!")
        sys.exit(1)
    
    if not os.path.exists(settings.google_application_credentials):
        print(f"Error: Credentials file not found: {settings.google_application_credentials}")
        sys.exit(1)
    
    try:
        # Initialize Google Cloud Storage client
        storage_client = storage.Client(project=settings.google_cloud_project_id)
        bucket_name = settings.storage_bucket_name
        
        # Get the bucket
        bucket = storage_client.bucket(bucket_name)
        if not bucket.exists():
            print(f"Error: Bucket '{bucket_name}' does not exist!")
            sys.exit(1)
        
        # Handle different operations
        if args.stats:
            get_storage_stats(bucket)
        
        if args.list or args.list_details:
            show_details = args.list_details
            list_files_in_folder(bucket, "_tmp/", show_details)
            list_files_in_folder(bucket, "uploads/", show_details)
        
        if args.clear_tmp or args.clear_uploads or args.clear_all:
            if not args.force and not args.dry_run:
                print("\nWARNING: This will permanently delete files!")
                response = input("Are you sure you want to proceed? (yes/no): ").strip().lower()
                if response not in ['yes', 'y']:
                    print("Operation cancelled.")
                    sys.exit(0)
            
            total_deleted = 0
            
            if args.clear_tmp or args.clear_all:
                deleted = clear_folder(bucket, "_tmp/", args.dry_run)
                total_deleted += deleted
            
            if args.clear_uploads or args.clear_all:
                deleted = clear_folder(bucket, "uploads/", args.dry_run)
                total_deleted += deleted
            
            if args.dry_run:
                print(f"\nDRY RUN: Would delete {total_deleted} files")
            else:
                print(f"\nDeleted {total_deleted} files")
        
        if not any([args.stats, args.list, args.list_details, args.clear_tmp, args.clear_uploads, args.clear_all]):
            print("\nNo operation specified. Use --help for available options.")
            parser.print_help()
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
