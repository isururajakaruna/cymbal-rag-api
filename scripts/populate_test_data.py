#!/usr/bin/env python3
"""Script to populate test data for the RAG API."""

import os
import json
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from PIL import Image, ImageDraw, ImageFont
import io


def create_test_data():
    """Create test documents for different content types."""
    test_data_dir = Path("test_data")
    test_data_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    (test_data_dir / "tables").mkdir(exist_ok=True)
    (test_data_dir / "text").mkdir(exist_ok=True)
    (test_data_dir / "images").mkdir(exist_ok=True)
    
    print("Creating test documents...")
    
    # Create simple table PDF
    create_simple_table_pdf(test_data_dir / "tables" / "simple_table.pdf")
    
    # Create nested table PDF
    create_nested_table_pdf(test_data_dir / "tables" / "nested_tables.pdf")
    
    # Create text PDF
    create_text_pdf(test_data_dir / "text" / "sample_text.pdf")
    
    # Create image with text
    create_image_with_text(test_data_dir / "images" / "image_with_text.png")
    
    print("Test data created successfully!")


def create_simple_table_pdf(output_path: Path):
    """Create a PDF with a simple table."""
    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title = Paragraph("Simple Table Document", styles['Title'])
    story.append(title)
    story.append(Paragraph("<br/>", styles['Normal']))
    
    # Create table data
    data = [
        ['Product', 'Price', 'Quantity', 'Total'],
        ['Apple', '$1.50', '10', '$15.00'],
        ['Banana', '$0.75', '20', '$15.00'],
        ['Orange', '$2.00', '5', '$10.00'],
        ['Grape', '$3.00', '8', '$24.00'],
        ['', '', 'Total:', '$64.00']
    ]
    
    # Create table
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(table)
    doc.build(story)


def create_nested_table_pdf(output_path: Path):
    """Create a PDF with nested tables."""
    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title = Paragraph("Nested Tables Document", styles['Title'])
    story.append(title)
    story.append(Paragraph("<br/>", styles['Normal']))
    
    # Main table with nested tables
    main_data = [
        ['Department', 'Q1 Sales', 'Q2 Sales', 'Q3 Sales', 'Q4 Sales'],
        ['Electronics', 'Table 1', 'Table 2', 'Table 3', 'Table 4'],
        ['Clothing', 'Table 5', 'Table 6', 'Table 7', 'Table 8'],
        ['Books', 'Table 9', 'Table 10', 'Table 11', 'Table 12']
    ]
    
    main_table = Table(main_data)
    main_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(main_table)
    story.append(Paragraph("<br/>", styles['Normal']))
    
    # Nested table example
    nested_data = [
        ['Product', 'Jan', 'Feb', 'Mar'],
        ['Laptop', '5', '7', '3'],
        ['Phone', '12', '15', '8'],
        ['Tablet', '3', '4', '2']
    ]
    
    nested_table = Table(nested_data)
    nested_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(Paragraph("Nested Table Example:", styles['Heading2']))
    story.append(nested_table)
    
    doc.build(story)


def create_text_pdf(output_path: Path):
    """Create a PDF with plain text content."""
    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title = Paragraph("Sample Text Document", styles['Title'])
    story.append(title)
    story.append(Paragraph("<br/>", styles['Normal']))
    
    # Sample text content
    text_content = """
    This is a sample text document for testing the RAG API.
    
    The document contains multiple paragraphs with various types of content:
    
    1. Regular text paragraphs
    2. Numbered lists
    3. Bullet points
    4. Technical terms and concepts
    
    The RAG (Retrieval-Augmented Generation) system processes this content
    by first extracting text from the document, then chunking it into
    smaller pieces, and finally creating vector embeddings for semantic search.
    
    This allows users to search for information using natural language queries
    and receive relevant results based on the content of the documents.
    
    The system supports various document formats including PDF, text files,
    images with OCR, and structured documents with tables.
    """
    
    paragraphs = text_content.strip().split('\n\n')
    for para in paragraphs:
        if para.strip():
            story.append(Paragraph(para.strip(), styles['Normal']))
            story.append(Paragraph("<br/>", styles['Normal']))
    
    doc.build(story)


def create_image_with_text(output_path: Path):
    """Create an image with text for OCR testing."""
    # Create image
    width, height = 800, 600
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    # Try to use a default font, fallback to basic if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
    
    # Draw text
    text_lines = [
        "Sample Document with Text",
        "",
        "This is a test image containing text",
        "that will be processed using OCR.",
        "",
        "The text includes:",
        "• Regular paragraphs",
        "• Bullet points",
        "• Numbers: 1, 2, 3",
        "• Special characters: @#$%",
        "",
        "This content should be extracted",
        "and made searchable by the RAG system."
    ]
    
    y_position = 50
    for line in text_lines:
        if line.strip():
            draw.text((50, y_position), line, fill='black', font=font)
        y_position += 30
    
    # Save image
    image.save(output_path, 'PNG')
    print(f"Created image: {output_path}")


if __name__ == "__main__":
    create_test_data()
