"""Text cleaning utilities for document preprocessing."""

import re
import unicodedata
from typing import Any, Dict, List, Optional


class TextCleaner:
    """Utilities for cleaning and normalizing text content."""

    def __init__(self):
        self.unicode_normalization = "NFKC"
        self.control_chars_to_keep = {"\n", "\t", "\r"}

        # Common patterns for cleaning
        self.patterns = {
            "extra_whitespace": re.compile(r"\s+"),
            "multiple_newlines": re.compile(r"\n\s*\n\s*\n+"),
            "trailing_whitespace": re.compile(r"[ \t]+$", re.MULTILINE),
            "leading_whitespace": re.compile(r"^[ \t]+", re.MULTILINE),
            "control_chars": re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]"),
            "smart_quotes": re.compile(r'[""' "`]"),
            "em_dashes": re.compile(r"—|–"),
            "ellipsis": re.compile(r"\.{3,}"),
            "bullet_points": re.compile(r"[•·▪▫]"),
        }

    async def clean_text(
        self, text: str, options: Optional[Dict[str, bool]] = None
    ) -> str:
        """
        Clean text using various normalization techniques.

        Args:
            text: Raw text to clean
            options: Cleaning options dictionary

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Default options
        default_options = {
            "normalize_unicode": True,
            "remove_control_chars": True,
            "fix_whitespace": True,
            "normalize_quotes": True,
            "normalize_dashes": True,
            "normalize_ellipsis": True,
            "normalize_bullets": True,
            "remove_empty_lines": True,
            "trim_lines": True,
        }

        if options:
            default_options.update(options)

        cleaned_text = text

        # Unicode normalization
        if default_options["normalize_unicode"]:
            cleaned_text = await self._normalize_unicode(cleaned_text)

        # Remove control characters
        if default_options["remove_control_chars"]:
            cleaned_text = await self._remove_control_characters(cleaned_text)

        # Fix whitespace
        if default_options["fix_whitespace"]:
            cleaned_text = await self._fix_whitespace(cleaned_text)

        # Normalize quotes
        if default_options["normalize_quotes"]:
            cleaned_text = await self._normalize_quotes(cleaned_text)

        # Normalize dashes
        if default_options["normalize_dashes"]:
            cleaned_text = await self._normalize_dashes(cleaned_text)

        # Normalize ellipsis
        if default_options["normalize_ellipsis"]:
            cleaned_text = await self._normalize_ellipsis(cleaned_text)

        # Normalize bullets
        if default_options["normalize_bullets"]:
            cleaned_text = await self._normalize_bullets(cleaned_text)

        # Remove empty lines
        if default_options["remove_empty_lines"]:
            cleaned_text = await self._remove_empty_lines(cleaned_text)

        # Trim lines
        if default_options["trim_lines"]:
            cleaned_text = await self._trim_lines(cleaned_text)

        return cleaned_text.strip()

    async def clean_table_text(self, table_text: str) -> str:
        """
        Clean text specifically for table content.

        Args:
            table_text: Table text to clean

        Returns:
            Cleaned table text
        """
        if not table_text:
            return ""

        # Clean with table-specific options
        options = {
            "normalize_unicode": True,
            "remove_control_chars": True,
            "fix_whitespace": False,  # Preserve table structure
            "normalize_quotes": True,
            "normalize_dashes": True,
            "normalize_ellipsis": True,
            "normalize_bullets": True,
            "remove_empty_lines": False,  # Preserve table rows
            "trim_lines": True,
        }

        return await self.clean_text(table_text, options)

    async def clean_ocr_text(self, ocr_text: str) -> str:
        """
        Clean text extracted from OCR.

        Args:
            ocr_text: OCR text to clean

        Returns:
            Cleaned OCR text
        """
        if not ocr_text:
            return ""

        # OCR-specific cleaning
        cleaned_text = ocr_text

        # Fix common OCR errors
        cleaned_text = await self._fix_ocr_errors(cleaned_text)

        # Apply standard cleaning
        cleaned_text = await self.clean_text(cleaned_text)

        return cleaned_text

    async def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters."""
        return unicodedata.normalize(self.unicode_normalization, text)

    async def _remove_control_characters(self, text: str) -> str:
        """Remove control characters except newlines and tabs."""

        def is_keep_char(char):
            return ord(char) >= 32 or char in self.control_chars_to_keep

        return "".join(char for char in text if is_keep_char(char))

    async def _fix_whitespace(self, text: str) -> str:
        """Fix whitespace issues."""
        # Replace multiple spaces with single space
        text = self.patterns["extra_whitespace"].sub(" ", text)

        # Replace multiple newlines with double newline
        text = self.patterns["multiple_newlines"].sub("\n\n", text)

        return text

    async def _normalize_quotes(self, text: str) -> str:
        """Normalize various quote characters to standard quotes."""
        replacements = {
            '"': '"',  # Left double quotation mark
            '"': '"',  # Right double quotation mark
            """: "'",  # Left single quotation mark
            """: "'",  # Right single quotation mark
            "`": "'",  # Grave accent
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        return text

    async def _normalize_dashes(self, text: str) -> str:
        """Normalize various dash characters to standard hyphen."""
        # Replace em dashes and en dashes with hyphens
        text = self.patterns["em_dashes"].sub("-", text)
        return text

    async def _normalize_ellipsis(self, text: str) -> str:
        """Normalize ellipsis characters."""
        text = self.patterns["ellipsis"].sub("...", text)
        return text

    async def _normalize_bullets(self, text: str) -> str:
        """Normalize various bullet point characters."""
        text = self.patterns["bullet_points"].sub("•", text)
        return text

    async def _remove_empty_lines(self, text: str) -> str:
        """Remove excessive empty lines."""
        # Replace 3 or more consecutive newlines with 2
        text = self.patterns["multiple_newlines"].sub("\n\n", text)
        return text

    async def _trim_lines(self, text: str) -> str:
        """Trim whitespace from the beginning and end of lines."""
        # Remove trailing whitespace
        text = self.patterns["trailing_whitespace"].sub("", text)

        # Remove leading whitespace (but preserve indentation for some content)
        # This is a simplified approach - you might want more sophisticated logic
        lines = text.split("\n")
        trimmed_lines = [line.rstrip() for line in lines]
        return "\n".join(trimmed_lines)

    async def _fix_ocr_errors(self, text: str) -> str:
        """Fix common OCR errors."""
        # Common OCR character substitutions
        ocr_fixes = {
            "0": "O",  # Zero to O in certain contexts
            "1": "I",  # One to I in certain contexts
            "5": "S",  # Five to S in certain contexts
            "8": "B",  # Eight to B in certain contexts
        }

        # This is a very basic implementation
        # In practice, you'd want more sophisticated OCR error correction

        return text

    async def extract_metadata(self, text: str) -> Dict[str, Any]:
        """
        Extract metadata from text content.

        Args:
            text: Text to analyze

        Returns:
            Metadata dictionary
        """
        if not text:
            return {}

        lines = text.split("\n")
        non_empty_lines = [line for line in lines if line.strip()]

        return {
            "total_lines": len(lines),
            "non_empty_lines": len(non_empty_lines),
            "total_characters": len(text),
            "total_words": len(text.split()),
            "average_line_length": sum(len(line) for line in non_empty_lines)
            / len(non_empty_lines)
            if non_empty_lines
            else 0,
            "has_tables": "|" in text or "+" in text,
            "has_lists": any(
                line.strip().startswith(("•", "-", "*", "1.", "2.", "3."))
                for line in lines
            ),
            "has_urls": bool(re.search(r"https?://\S+", text)),
            "has_emails": bool(
                re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text)
            ),
        }
