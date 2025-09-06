"""Table extraction utilities for complex table structures."""

import re
from typing import Any, Dict, List, Optional, Tuple

from app.models.schemas import ChunkInfo


class TableExtractor:
    """Utilities for extracting and processing tables from documents."""

    def __init__(self):
        self.table_patterns = [
            r"\|.*\|",  # Pipe-separated tables
            r"\+.*\+",  # Plus-separated tables
            r"^\s*\w+.*\w+\s*$",  # Space-separated tables
        ]

    async def extract_tables_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract table structures from text.

        Args:
            text: Text content to analyze

        Returns:
            List of table dictionaries with structure information
        """
        tables = []
        lines = text.split("\n")

        current_table = []
        in_table = False

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                if in_table and current_table:
                    table_data = await self._process_table_lines(
                        current_table, i - len(current_table)
                    )
                    if table_data:
                        tables.append(table_data)
                    current_table = []
                    in_table = False
                continue

            # Check if line looks like a table row
            if await self._is_table_row(line):
                current_table.append(line)
                in_table = True
            else:
                if in_table and current_table:
                    table_data = await self._process_table_lines(
                        current_table, i - len(current_table)
                    )
                    if table_data:
                        tables.append(table_data)
                    current_table = []
                    in_table = False

        # Process final table if exists
        if in_table and current_table:
            table_data = await self._process_table_lines(
                current_table, len(lines) - len(current_table)
            )
            if table_data:
                tables.append(table_data)

        return tables

    async def extract_nested_tables(self, table_text: str) -> List[Dict[str, Any]]:
        """
        Extract nested table structures.

        Args:
            table_text: Table text content

        Returns:
            List of nested table structures
        """
        # This is a simplified implementation
        # In practice, you might need more sophisticated parsing
        tables = []

        # Split by double separators to find potential nested tables
        sections = re.split(r"\n\s*[-=]+\s*\n", table_text)

        for i, section in enumerate(sections):
            if section.strip():
                table_data = await self._process_table_lines(
                    section.strip().split("\n"), i
                )
                if table_data:
                    table_data["nested_level"] = i
                    tables.append(table_data)

        return tables

    async def create_table_chunks(
        self, tables: List[Dict[str, Any]], metadata: Dict[str, Any] = None
    ) -> List[ChunkInfo]:
        """
        Create chunks from extracted tables.

        Args:
            tables: List of table data
            metadata: Additional metadata

        Returns:
            List of ChunkInfo objects for tables
        """
        chunks = []

        for i, table in enumerate(tables):
            # Create a structured representation of the table
            table_text = await self._format_table_text(table)

            chunks.append(
                ChunkInfo(
                    chunk_id=f"table_{i}",
                    content=table_text,
                    chunk_index=i,
                    metadata={
                        "type": "table",
                        "rows": table.get("rows", 0),
                        "columns": table.get("columns", 0),
                        "nested_level": table.get("nested_level", 0),
                        **(metadata or {}),
                    },
                )
            )

        return chunks

    async def _is_table_row(self, line: str) -> bool:
        """Check if a line looks like a table row."""
        # Check for common table patterns
        for pattern in self.table_patterns:
            if re.search(pattern, line):
                return True

        # Check for multiple columns (words separated by spaces)
        words = line.split()
        if len(words) >= 3:  # At least 3 columns
            # Check if words are roughly aligned (similar lengths)
            lengths = [len(word) for word in words]
            if max(lengths) - min(lengths) <= 5:  # Reasonable length variation
                return True

        return False

    async def _process_table_lines(
        self, lines: List[str], start_line: int
    ) -> Optional[Dict[str, Any]]:
        """Process a list of table lines into structured data."""
        if not lines:
            return None

        # Detect separator row (usually contains dashes, equals, or pipes)
        separator_row = None
        for i, line in enumerate(lines):
            if re.search(r"[-=|]+", line) and len(line.strip()) > 3:
                separator_row = i
                break

        # Extract headers
        if separator_row and separator_row > 0:
            headers = await self._parse_table_row(lines[separator_row - 1])
        else:
            headers = await self._parse_table_row(lines[0])

        # Extract data rows
        data_start = separator_row + 1 if separator_row else 1
        data_rows = []

        for line in lines[data_start:]:
            if line.strip() and not re.search(r"[-=|]+", line):
                row_data = await self._parse_table_row(line)
                if row_data:
                    data_rows.append(row_data)

        return {
            "headers": headers,
            "data_rows": data_rows,
            "rows": len(data_rows) + (1 if headers else 0),
            "columns": len(headers) if headers else 0,
            "start_line": start_line,
            "end_line": start_line + len(lines),
        }

    async def _parse_table_row(self, line: str) -> List[str]:
        """Parse a table row into individual cells."""
        # Try different parsing methods
        if "|" in line:
            # Pipe-separated
            cells = [cell.strip() for cell in line.split("|")]
            return [cell for cell in cells if cell]
        elif "+" in line:
            # Plus-separated
            cells = [cell.strip() for cell in line.split("+")]
            return [cell for cell in cells if cell]
        else:
            # Space-separated (more complex)
            # Split by multiple spaces
            cells = re.split(r"\s{2,}", line.strip())
            return [cell.strip() for cell in cells if cell.strip()]

    async def _format_table_text(self, table: Dict[str, Any]) -> str:
        """Format table data into readable text."""
        if not table.get("headers") and not table.get("data_rows"):
            return ""

        lines = []

        # Add headers
        if table.get("headers"):
            lines.append(" | ".join(table["headers"]))
            lines.append(
                "-"
                * (
                    sum(len(h) for h in table["headers"])
                    + 3 * (len(table["headers"]) - 1)
                )
            )

        # Add data rows
        for row in table.get("data_rows", []):
            lines.append(" | ".join(row))

        return "\n".join(lines)

    async def merge_table_chunks(
        self, table_chunks: List[ChunkInfo]
    ) -> List[ChunkInfo]:
        """
        Merge related table chunks to preserve table structure.

        Args:
            table_chunks: List of table chunks to merge

        Returns:
            Merged table chunks
        """
        if not table_chunks:
            return []

        # Group chunks by nested level
        grouped_chunks = {}
        for chunk in table_chunks:
            nested_level = chunk.metadata.get("nested_level", 0)
            if nested_level not in grouped_chunks:
                grouped_chunks[nested_level] = []
            grouped_chunks[nested_level].append(chunk)

        merged_chunks = []

        for level, chunks in grouped_chunks.items():
            if len(chunks) == 1:
                merged_chunks.append(chunks[0])
            else:
                # Merge chunks at the same level
                combined_content = "\n\n".join(chunk.content for chunk in chunks)
                total_rows = sum(chunk.metadata.get("rows", 0) for chunk in chunks)
                total_columns = max(
                    chunk.metadata.get("columns", 0) for chunk in chunks
                )

                merged_chunk = ChunkInfo(
                    chunk_id=f"merged_table_level_{level}",
                    content=combined_content,
                    chunk_index=chunks[0].chunk_index,
                    metadata={
                        "type": "merged_table",
                        "nested_level": level,
                        "rows": total_rows,
                        "columns": total_columns,
                        "original_chunks": len(chunks),
                    },
                )
                merged_chunks.append(merged_chunk)

        return merged_chunks
