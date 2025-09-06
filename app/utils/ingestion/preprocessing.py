"""Document preprocessing utilities."""

import io
import re
from typing import Any, Dict, List, Optional

import PyPDF2
from PIL import Image


class DocumentPreprocessor:
    """Utilities for preprocessing documents before ingestion."""

    def __init__(self):
        self.text_cleaners = [
            self._remove_extra_whitespace,
            self._normalize_unicode,
            self._remove_control_characters,
            self._fix_line_breaks,
        ]

    async def preprocess_text(self, text: str) -> str:
        """
        Preprocess text content.

        Args:
            text: Raw text content

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        cleaned_text = text

        for cleaner in self.text_cleaners:
            cleaned_text = cleaner(cleaned_text)

        return cleaned_text.strip()

    async def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """
        Extract text from PDF content.

        Args:
            pdf_content: PDF file content as bytes

        Returns:
            Extracted text
        """
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
            text_parts = []

            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text.strip():
                    text_parts.append(page_text)

            return "\n".join(text_parts)

        except Exception as e:
            raise ValueError(f"Failed to extract text from PDF: {str(e)}")

    async def preprocess_image(self, image_content: bytes) -> Optional[bytes]:
        """
        Preprocess image content for better OCR.

        Args:
            image_content: Image file content as bytes

        Returns:
            Preprocessed image content
        """
        try:
            image = Image.open(io.BytesIO(image_content))

            # Convert to RGB if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Resize if too large (OCR works better with reasonable sizes)
            max_size = 2048
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            # Save as optimized JPEG
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=85, optimize=True)
            return output.getvalue()

        except Exception as e:
            raise ValueError(f"Failed to preprocess image: {str(e)}")

    async def detect_content_type(self, content: bytes, filename: str) -> str:
        """
        Detect content type from file content and filename.

        Args:
            content: File content as bytes
            filename: Original filename

        Returns:
            Detected MIME type
        """
        # Check file extension first
        extension = filename.lower().split(".")[-1] if "." in filename else ""

        extension_map = {
            "pdf": "application/pdf",
            "txt": "text/plain",
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
        }

        if extension in extension_map:
            return extension_map[extension]

        # Try to detect from content
        if content.startswith(b"%PDF"):
            return "application/pdf"
        elif content.startswith(b"\x89PNG"):
            return "image/png"
        elif content.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        elif content.startswith(b"PK\x03\x04"):
            return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        else:
            # Default to text if we can't determine
            return "text/plain"

    async def validate_file_size(self, content: bytes, max_size_mb: int = 10) -> bool:
        """
        Validate file size.

        Args:
            content: File content as bytes
            max_size_mb: Maximum allowed size in MB

        Returns:
            True if file size is valid
        """
        max_size_bytes = max_size_mb * 1024 * 1024
        return len(content) <= max_size_bytes

    def _remove_extra_whitespace(self, text: str) -> str:
        """Remove extra whitespace characters."""
        # Replace multiple spaces with single space
        text = re.sub(r" +", " ", text)
        # Replace multiple newlines with double newline
        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
        return text

    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters."""
        import unicodedata

        return unicodedata.normalize("NFKC", text)

    def _remove_control_characters(self, text: str) -> str:
        """Remove control characters except newlines and tabs."""
        return "".join(char for char in text if ord(char) >= 32 or char in "\n\t")

    def _fix_line_breaks(self, text: str) -> str:
        """Fix line break issues."""
        # Replace Windows line endings
        text = text.replace("\r\n", "\n")
        # Replace Mac line endings
        text = text.replace("\r", "\n")
        return text
