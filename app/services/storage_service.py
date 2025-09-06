"""Storage service using Google Cloud Storage."""

import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from google.cloud import storage
from google.cloud.exceptions import NotFound

from app.core.config import settings
from app.core.exceptions import StorageError


class StorageService:
    """Service for file storage operations using Google Cloud Storage."""

    def __init__(self):
        self.client = storage.Client()
        self.bucket_name = settings.storage_bucket_name
        self.bucket = self.client.bucket(self.bucket_name)

    async def upload_file(
        self,
        file_content: bytes,
        filename: str,
        content_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Upload a file to Google Cloud Storage.

        Args:
            file_content: Raw file content
            filename: Original filename
            content_type: MIME type of the file
            metadata: Optional metadata to store with the file

        Returns:
            File ID (blob name) of the uploaded file
        """
        try:
            # Generate unique file ID
            file_id = f"{uuid.uuid4()}_{filename}"

            # Create blob
            blob = self.bucket.blob(file_id)

            # Set metadata
            blob.metadata = {
                "original_filename": filename,
                "content_type": content_type,
                "upload_timestamp": datetime.utcnow().isoformat(),
                **(metadata or {}),
            }

            # Upload file
            blob.upload_from_string(file_content, content_type=content_type)

            return file_id

        except Exception as e:
            raise StorageError(f"Failed to upload file {filename}: {str(e)}")

    async def download_file(self, file_id: str) -> bytes:
        """
        Download a file from Google Cloud Storage.

        Args:
            file_id: ID of the file to download

        Returns:
            File content as bytes
        """
        try:
            blob = self.bucket.blob(file_id)

            if not blob.exists():
                raise StorageError(f"File {file_id} not found")

            return blob.download_as_bytes()

        except NotFound:
            raise StorageError(f"File {file_id} not found")
        except Exception as e:
            raise StorageError(f"Failed to download file {file_id}: {str(e)}")

    async def delete_file(self, file_id: str) -> bool:
        """
        Delete a file from Google Cloud Storage.

        Args:
            file_id: ID of the file to delete

        Returns:
            True if successful
        """
        try:
            blob = self.bucket.blob(file_id)
            blob.delete()
            return True

        except NotFound:
            # File already deleted or doesn't exist
            return True
        except Exception as e:
            raise StorageError(f"Failed to delete file {file_id}: {str(e)}")

    async def get_file_metadata(self, file_id: str) -> Dict[str, Any]:
        """
        Get metadata for a file.

        Args:
            file_id: ID of the file

        Returns:
            File metadata dictionary
        """
        try:
            blob = self.bucket.blob(file_id)

            if not blob.exists():
                raise StorageError(f"File {file_id} not found")

            blob.reload()

            return {
                "file_id": file_id,
                "filename": blob.metadata.get("original_filename", file_id),
                "content_type": blob.content_type,
                "size": blob.size,
                "created": blob.time_created,
                "updated": blob.updated,
                "metadata": blob.metadata,
            }

        except NotFound:
            raise StorageError(f"File {file_id} not found")
        except Exception as e:
            raise StorageError(f"Failed to get metadata for file {file_id}: {str(e)}")

    async def list_files(self, prefix: Optional[str] = None) -> list:
        """
        List files in the storage bucket.

        Args:
            prefix: Optional prefix to filter files

        Returns:
            List of file metadata dictionaries
        """
        try:
            blobs = self.bucket.list_blobs(prefix=prefix)
            files = []

            for blob in blobs:
                blob.reload()
                files.append(
                    {
                        "file_id": blob.name,
                        "filename": blob.metadata.get("original_filename", blob.name),
                        "content_type": blob.content_type,
                        "size": blob.size,
                        "created": blob.time_created,
                        "updated": blob.updated,
                        "metadata": blob.metadata,
                    }
                )

            return files

        except Exception as e:
            raise StorageError(f"Failed to list files: {str(e)}")

    async def file_exists(self, file_id: str) -> bool:
        """
        Check if a file exists in storage.

        Args:
            file_id: ID of the file to check

        Returns:
            True if file exists
        """
        try:
            blob = self.bucket.blob(file_id)
            return blob.exists()
        except Exception:
            return False
