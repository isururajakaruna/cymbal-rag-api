"""Image processing utilities for OCR and text extraction."""

import io
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from app.models.schemas import ChunkInfo


class ImageProcessor:
    """Utilities for processing images to improve OCR accuracy."""

    def __init__(self):
        self.supported_formats = ["PNG", "JPEG", "JPG", "BMP", "TIFF"]
        self.max_image_size = (2048, 2048)
        self.min_image_size = (100, 100)

    async def preprocess_for_ocr(self, image_content: bytes) -> bytes:
        """
        Preprocess image to improve OCR accuracy.

        Args:
            image_content: Raw image content

        Returns:
            Preprocessed image content
        """
        try:
            image = Image.open(io.BytesIO(image_content))

            # Convert to RGB if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Resize if too large or too small
            image = await self._resize_image(image)

            # Enhance image quality
            image = await self._enhance_image(image)

            # Convert back to bytes
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=95, optimize=True)
            return output.getvalue()

        except Exception as e:
            raise ValueError(f"Failed to preprocess image: {str(e)}")

    async def extract_text_regions(self, image_content: bytes) -> List[Dict[str, Any]]:
        """
        Extract text regions from image for better OCR processing.

        Args:
            image_content: Raw image content

        Returns:
            List of text region information
        """
        try:
            image = Image.open(io.BytesIO(image_content))

            # Convert to grayscale for analysis
            gray_image = image.convert("L")

            # Find text regions using edge detection
            text_regions = await self._find_text_regions(gray_image)

            return text_regions

        except Exception as e:
            raise ValueError(f"Failed to extract text regions: {str(e)}")

    async def create_image_chunks(
        self, image_content: bytes, ocr_text: str, metadata: Dict[str, Any] = None
    ) -> List[ChunkInfo]:
        """
        Create chunks from image OCR text.

        Args:
            image_content: Raw image content
            ocr_text: Extracted OCR text
            metadata: Additional metadata

        Returns:
            List of ChunkInfo objects
        """
        if not ocr_text.strip():
            return []

        # Split OCR text into logical chunks (by paragraphs or sentences)
        chunks = []
        paragraphs = ocr_text.split("\n\n")

        for i, paragraph in enumerate(paragraphs):
            if paragraph.strip():
                chunks.append(
                    ChunkInfo(
                        chunk_id=f"image_ocr_chunk_{i}",
                        content=paragraph.strip(),
                        chunk_index=i,
                        metadata={
                            "type": "image_ocr",
                            "source": "image",
                            "paragraph_index": i,
                            **(metadata or {}),
                        },
                    )
                )

        return chunks

    async def _resize_image(self, image: Image.Image) -> Image.Image:
        """Resize image to optimal size for OCR."""
        width, height = image.size

        # Check if image is too large
        if width > self.max_image_size[0] or height > self.max_image_size[1]:
            # Calculate scaling factor
            scale_w = self.max_image_size[0] / width
            scale_h = self.max_image_size[1] / height
            scale = min(scale_w, scale_h)

            new_size = (int(width * scale), int(height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        # Check if image is too small
        elif width < self.min_image_size[0] or height < self.min_image_size[1]:
            # Calculate scaling factor
            scale_w = self.min_image_size[0] / width
            scale_h = self.min_image_size[1] / height
            scale = max(scale_w, scale_h)

            new_size = (int(width * scale), int(height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        return image

    async def _enhance_image(self, image: Image.Image) -> Image.Image:
        """Enhance image quality for better OCR."""
        # Convert to grayscale for enhancement
        if image.mode != "L":
            gray_image = image.convert("L")
        else:
            gray_image = image

        # Enhance contrast
        enhancer = ImageEnhance.Contrast(gray_image)
        enhanced = enhancer.enhance(1.5)

        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(2.0)

        # Apply slight blur to reduce noise
        enhanced = enhanced.filter(ImageFilter.MedianFilter(size=3))

        # Convert back to RGB
        return enhanced.convert("RGB")

    async def _find_text_regions(self, gray_image: Image.Image) -> List[Dict[str, Any]]:
        """
        Find text regions in the image using edge detection.

        Args:
            gray_image: Grayscale PIL Image

        Returns:
            List of text region dictionaries
        """
        # Convert PIL image to numpy array
        img_array = np.array(gray_image)

        # Apply edge detection
        edges = await self._detect_edges(img_array)

        # Find contours (simplified approach)
        text_regions = []

        # This is a simplified implementation
        # In practice, you might use OpenCV or other libraries for better contour detection

        # For now, return the whole image as one region
        width, height = gray_image.size
        text_regions.append(
            {"bbox": (0, 0, width, height), "confidence": 0.8, "type": "text_region"}
        )

        return text_regions

    async def _detect_edges(self, img_array: np.ndarray) -> np.ndarray:
        """Detect edges in the image array."""
        # Simple edge detection using gradient
        # This is a basic implementation
        from scipy import ndimage

        # Apply Sobel edge detection
        sobel_x = ndimage.sobel(img_array, axis=1)
        sobel_y = ndimage.sobel(img_array, axis=0)
        edges = np.sqrt(sobel_x**2 + sobel_y**2)

        return edges

    async def validate_image(self, image_content: bytes) -> Tuple[bool, str]:
        """
        Validate image content.

        Args:
            image_content: Raw image content

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            image = Image.open(io.BytesIO(image_content))

            # Check format
            if image.format not in self.supported_formats:
                return False, f"Unsupported image format: {image.format}"

            # Check size
            width, height = image.size
            if width < self.min_image_size[0] or height < self.min_image_size[1]:
                return False, f"Image too small: {width}x{height}"

            if width > self.max_image_size[0] or height > self.max_image_size[1]:
                return False, f"Image too large: {width}x{height}"

            return True, ""

        except Exception as e:
            return False, f"Invalid image: {str(e)}"

    async def get_image_metadata(self, image_content: bytes) -> Dict[str, Any]:
        """
        Extract metadata from image.

        Args:
            image_content: Raw image content

        Returns:
            Image metadata dictionary
        """
        try:
            image = Image.open(io.BytesIO(image_content))

            return {
                "format": image.format,
                "mode": image.mode,
                "size": image.size,
                "width": image.size[0],
                "height": image.size[1],
                "has_transparency": "transparency" in image.info,
                "dpi": image.info.get("dpi", (72, 72)),
            }

        except Exception as e:
            return {"error": str(e)}
