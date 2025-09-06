"""Enhanced document processing using Gemini Flash multimodal capabilities."""

import base64
import io
from typing import Any, Dict, List, Optional

import pandas as pd
import PyPDF2
import vertexai
from pdf2image import convert_from_bytes
from PIL import Image
from vertexai.generative_models import GenerativeModel, Part

from app.core.config import settings
from app.core.exceptions import (DocumentProcessingError,
                                 UnsupportedFileFormatError)
from app.models.schemas import ChunkInfo


class GeminiDocumentProcessor:
    """Enhanced document processor using Gemini Flash multimodal capabilities."""

    def __init__(self):
        # Initialize Vertex AI
        vertexai.init(
            project=settings.google_cloud_project_id,
            location=settings.vertex_ai_location,
        )

        # Initialize Gemini model
        self.model = GenerativeModel("gemini-1.5-flash")

        # Processing prompts for different content types
        self.prompts = {
            "general_text": """
            Analyze this document page and extract all text content. 
            Preserve the structure and formatting as much as possible.
            If there are headers, subheaders, or important formatting, maintain that hierarchy.
            Return only the extracted text content.
            """,
            "table_extraction": """
            Analyze this document page and extract all table content.
            For each table found:
            1. Identify the table structure and column headers
            2. For each row, create a structured format where:
               - Column headers become bullet points
               - Each row's data is described under the relevant column header
            3. If there are nested tables, handle them separately
            4. Preserve the logical relationships between data
            
            Format the output as:
            ## Table: [Table Title if available]
            ### Column Headers:
            - Header 1: [description]
            - Header 2: [description]
            
            ### Data Rows:
            **Row 1:**
            - Header 1: [value]
            - Header 2: [value]
            
            **Row 2:**
            - Header 1: [value]
            - Header 2: [value]
            """,
            "diagram_analysis": """
            Analyze this image/diagram and provide a comprehensive description.
            Include:
            1. What type of diagram/chart it is (flowchart, bar chart, pie chart, etc.)
            2. Main components and their relationships
            3. Key data points, trends, or patterns visible
            4. Any text labels or annotations
            5. Overall purpose or message of the diagram
            
            Format as:
            ## Diagram Analysis
            **Type:** [diagram type]
            **Purpose:** [what it shows]
            **Key Components:**
            - [component 1]: [description]
            - [component 2]: [description]
            **Key Insights:**
            - [insight 1]
            - [insight 2]
            """,
            "mixed_content": """
            Analyze this document page comprehensively and extract all content.
            Handle different content types as follows:
            
            1. **Text Content**: Extract and preserve structure
            2. **Tables**: Use the table extraction format
            3. **Diagrams/Charts**: Use the diagram analysis format
            4. **Images**: Describe what you see and any relevant text
            
            Organize the output clearly with appropriate headers and formatting.
            """,
        }

    async def process_document(
        self, file_content: bytes, filename: str, content_type: str
    ) -> List[ChunkInfo]:
        """
        Process a document using Gemini Flash multimodal capabilities.

        Args:
            file_content: Raw file content
            filename: Name of the file
            content_type: MIME type of the file

        Returns:
            List of ChunkInfo objects
        """
        try:
            if content_type == "application/pdf":
                return await self._process_pdf_with_gemini(file_content, filename)
            elif content_type.startswith("image/"):
                return await self._process_image_with_gemini(file_content, filename)
            elif content_type in [
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "application/vnd.ms-excel",
            ]:
                return await self._process_excel_with_pandas(file_content, filename)
            elif content_type == "text/plain":
                return await self._process_text(file_content, filename)
            else:
                raise UnsupportedFileFormatError(
                    f"Unsupported file format: {content_type}"
                )

        except Exception as e:
            raise DocumentProcessingError(
                f"Failed to process document {filename}: {str(e)}"
            )

    async def _process_pdf_with_gemini(
        self, file_content: bytes, filename: str
    ) -> List[ChunkInfo]:
        """Process PDF by converting pages to images and using Gemini Flash."""
        try:
            # Convert PDF pages to images
            images = convert_from_bytes(
                file_content, dpi=300, first_page=1, last_page=None
            )

            chunks = []
            for page_num, image in enumerate(images, 1):
                # Convert PIL image to bytes
                img_buffer = io.BytesIO()
                image.save(img_buffer, format="PNG")
                img_bytes = img_buffer.getvalue()

                # Encode image to base64
                img_base64 = base64.b64encode(img_bytes).decode("utf-8")

                # Analyze page content with Gemini
                page_content = await self._analyze_page_with_gemini(
                    img_base64, page_num
                )

                if page_content.strip():
                    chunks.append(
                        ChunkInfo(
                            chunk_id=f"{filename}_page_{page_num}",
                            content=page_content,
                            chunk_index=len(chunks),
                            metadata={
                                "page_number": page_num,
                                "type": "pdf_page",
                                "processor": "gemini_multimodal",
                                "filename": filename,
                            },
                        )
                    )

            return chunks

        except Exception as e:
            raise DocumentProcessingError(f"Failed to process PDF {filename}: {str(e)}")

    async def _process_image_with_gemini(
        self, file_content: bytes, filename: str
    ) -> List[ChunkInfo]:
        """Process images using Gemini Flash multimodal capabilities."""
        try:
            # Encode image to base64
            img_base64 = base64.b64encode(file_content).decode("utf-8")

            # Analyze image content
            content = await self._analyze_image_with_gemini(img_base64, filename)

            if content.strip():
                return [
                    ChunkInfo(
                        chunk_id=f"{filename}_image",
                        content=content,
                        chunk_index=0,
                        metadata={
                            "type": "image_analysis",
                            "processor": "gemini_multimodal",
                            "filename": filename,
                        },
                    )
                ]

            return []

        except Exception as e:
            raise DocumentProcessingError(
                f"Failed to process image {filename}: {str(e)}"
            )

    async def _process_excel_with_pandas(
        self, file_content: bytes, filename: str
    ) -> List[ChunkInfo]:
        """Process Excel files using pandas for structured data extraction."""
        try:
            # Read Excel file
            excel_file = io.BytesIO(file_content)
            excel_data = pd.read_excel(excel_file, sheet_name=None)  # Read all sheets

            chunks = []
            for sheet_name, df in excel_data.items():
                if df.empty:
                    continue

                # Convert DataFrame to structured text
                sheet_content = await self._convert_dataframe_to_text(df, sheet_name)

                if sheet_content.strip():
                    chunks.append(
                        ChunkInfo(
                            chunk_id=f"{filename}_sheet_{sheet_name}",
                            content=sheet_content,
                            chunk_index=len(chunks),
                            metadata={
                                "type": "excel_sheet",
                                "processor": "pandas",
                                "filename": filename,
                                "sheet_name": sheet_name,
                                "rows": len(df),
                                "columns": len(df.columns),
                            },
                        )
                    )

            return chunks

        except Exception as e:
            raise DocumentProcessingError(
                f"Failed to process Excel file {filename}: {str(e)}"
            )

    async def _process_text(
        self, file_content: bytes, filename: str
    ) -> List[ChunkInfo]:
        """Process plain text files."""
        try:
            text = file_content.decode("utf-8")

            # Simple chunking for text files
            chunks = []
            lines = text.split("\n")
            current_chunk = []
            chunk_size = 0
            max_chunk_size = 1000

            for line in lines:
                if chunk_size + len(line) > max_chunk_size and current_chunk:
                    chunks.append(
                        ChunkInfo(
                            chunk_id=f"{filename}_chunk{len(chunks)}",
                            content="\n".join(current_chunk),
                            chunk_index=len(chunks),
                            metadata={"type": "text_chunk", "processor": "text"},
                        )
                    )
                    current_chunk = [line]
                    chunk_size = len(line)
                else:
                    current_chunk.append(line)
                    chunk_size += len(line)

            # Add remaining content
            if current_chunk:
                chunks.append(
                    ChunkInfo(
                        chunk_id=f"{filename}_chunk{len(chunks)}",
                        content="\n".join(current_chunk),
                        chunk_index=len(chunks),
                        metadata={"type": "text_chunk", "processor": "text"},
                    )
                )

            return chunks

        except Exception as e:
            raise DocumentProcessingError(
                f"Failed to process text file {filename}: {str(e)}"
            )

    async def _analyze_page_with_gemini(self, img_base64: str, page_num: int) -> str:
        """Analyze a PDF page image using Gemini Flash."""
        try:
            # Create image part
            image_part = Part.from_data(
                data=base64.b64decode(img_base64), mime_type="image/png"
            )

            # Use mixed content prompt for comprehensive analysis
            prompt = f"Page {page_num} Analysis:\n{self.prompts['mixed_content']}"

            # Generate content
            response = self.model.generate_content([prompt, image_part])

            return response.text if response.text else ""

        except Exception as e:
            raise DocumentProcessingError(
                f"Failed to analyze page {page_num}: {str(e)}"
            )

    async def _analyze_image_with_gemini(self, img_base64: str, filename: str) -> str:
        """Analyze an image using Gemini Flash."""
        try:
            # Create image part
            image_part = Part.from_data(
                data=base64.b64decode(img_base64), mime_type="image/png"
            )

            # Use mixed content prompt for comprehensive analysis
            prompt = f"Image Analysis for {filename}:\n{self.prompts['mixed_content']}"

            # Generate content
            response = self.model.generate_content([prompt, image_part])

            return response.text if response.text else ""

        except Exception as e:
            raise DocumentProcessingError(
                f"Failed to analyze image {filename}: {str(e)}"
            )

    async def _convert_dataframe_to_text(
        self, df: pd.DataFrame, sheet_name: str
    ) -> str:
        """Convert pandas DataFrame to structured text format."""
        try:
            content_parts = [f"# Excel Sheet: {sheet_name}"]
            content_parts.append(
                f"**Dimensions:** {len(df)} rows × {len(df.columns)} columns"
            )
            content_parts.append("")

            # Add column information
            content_parts.append("## Column Information:")
            for i, col in enumerate(df.columns):
                dtype = str(df[col].dtype)
                non_null_count = df[col].count()
                content_parts.append(
                    f"- **{col}** ({dtype}): {non_null_count} non-null values"
                )

            content_parts.append("")

            # Add data in structured format
            content_parts.append("## Data Content:")

            # For small datasets, show all data
            if len(df) <= 50:
                for idx, row in df.iterrows():
                    content_parts.append(f"### Row {idx + 1}:")
                    for col in df.columns:
                        value = row[col]
                        if pd.notna(value):
                            content_parts.append(f"- **{col}**: {value}")
                    content_parts.append("")
            else:
                # For large datasets, show sample and summary
                content_parts.append("### Sample Data (first 10 rows):")
                sample_df = df.head(10)
                for idx, row in sample_df.iterrows():
                    content_parts.append(f"### Row {idx + 1}:")
                    for col in df.columns:
                        value = row[col]
                        if pd.notna(value):
                            content_parts.append(f"- **{col}**: {value}")
                    content_parts.append("")

                content_parts.append("### Data Summary:")
                for col in df.columns:
                    if df[col].dtype in ["int64", "float64"]:
                        stats = df[col].describe()
                        content_parts.append(
                            f"- **{col}**: Mean={stats['mean']:.2f}, Min={stats['min']:.2f}, Max={stats['max']:.2f}"
                        )
                    else:
                        unique_count = df[col].nunique()
                        content_parts.append(
                            f"- **{col}**: {unique_count} unique values"
                        )

            return "\n".join(content_parts)

        except Exception as e:
            raise DocumentProcessingError(
                f"Failed to convert DataFrame to text: {str(e)}"
            )

    async def _detect_content_type(self, img_base64: str) -> str:
        """Detect if image contains tables, diagrams, or general text."""
        try:
            image_part = Part.from_data(
                data=base64.b64decode(img_base64), mime_type="image/png"
            )

            prompt = """
            Analyze this image and determine the primary content type:
            - "table" if it contains tables or tabular data
            - "diagram" if it contains charts, graphs, flowcharts, or diagrams
            - "text" if it contains primarily text content
            - "mixed" if it contains multiple content types
            
            Respond with only one word: table, diagram, text, or mixed
            """

            response = self.model.generate_content([prompt, image_part])
            return response.text.strip().lower() if response.text else "text"

        except Exception as e:
            return "text"  # Default fallback
