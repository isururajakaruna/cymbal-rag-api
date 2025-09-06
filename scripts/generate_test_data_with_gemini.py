#!/usr/bin/env python3
"""Generate test data using Gemini multimodal capabilities."""

import os
import json
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from PIL import Image as PILImage, ImageDraw, ImageFont
import io
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import base64

# Initialize Vertex AI
vertexai.init(project="test-project", location="us-central1")
model = GenerativeModel("gemini-1.5-flash")


def create_test_data():
    """Create test documents using Gemini multimodal capabilities."""
    test_data_dir = Path("test_data")
    test_data_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    (test_data_dir / "tables").mkdir(exist_ok=True)
    (test_data_dir / "text").mkdir(exist_ok=True)
    (test_data_dir / "images").mkdir(exist_ok=True)
    (test_data_dir / "excel").mkdir(exist_ok=True)
    
    print("Creating test documents using Gemini...")
    
    # Generate documents
    create_simple_table_pdf(test_data_dir / "tables" / "simple_table.pdf")
    create_nested_table_pdf(test_data_dir / "tables" / "nested_tables.pdf")
    create_text_document(test_data_dir / "text" / "sample_text.pdf")
    create_image_with_text(test_data_dir / "images" / "image_with_text.png")
    create_diagram_image(test_data_dir / "images" / "flowchart_diagram.png")
    create_excel_sample(test_data_dir / "excel" / "sample_data.xlsx")
    
    print("Test data generated successfully!")


def create_simple_table_pdf(output_path: Path):
    """Create a PDF with a simple table using Gemini-generated content."""
    print("Generating simple table PDF...")
    
    # Generate table content using Gemini
    table_prompt = """
    Create a simple business report table with the following structure:
    - Product Name, Price, Quantity, Total
    - Include 5-6 products with realistic data
    - Make it look professional for a business report
    - Include a total row at the bottom
    """
    
    try:
        response = model.generate_content(table_prompt)
        table_content = response.text
    except Exception as e:
        print(f"Error generating content with Gemini: {e}")
        table_content = "Product Report\nApple, $1.50, 10, $15.00\nBanana, $0.75, 20, $15.00\nOrange, $2.00, 5, $10.00\nGrape, $3.00, 8, $24.00\nTotal: $64.00"
    
    # Create PDF
    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title = Paragraph("Simple Business Report", styles['Title'])
    story.append(title)
    story.append(Paragraph("<br/>", styles['Normal']))
    
    # Parse and create table
    lines = table_content.strip().split('\n')
    data = []
    for line in lines:
        if line.strip() and not line.startswith('#'):
            # Split by comma and clean up
            row = [cell.strip() for cell in line.split(',')]
            data.append(row)
    
    if data:
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
    """Create a PDF with nested tables using Gemini-generated content."""
    print("Generating nested table PDF...")
    
    # Generate nested table content using Gemini
    nested_prompt = """
    Create a complex business report with nested tables showing:
    1. Department-wise sales data for Q1-Q4
    2. Each department should have sub-tables for different product categories
    3. Include summary statistics
    4. Make it realistic for a retail company
    """
    
    try:
        response = model.generate_content(nested_prompt)
        nested_content = response.text
    except Exception as e:
        print(f"Error generating content with Gemini: {e}")
        nested_content = "Department Sales Report\nElectronics Q1: $50,000\nElectronics Q2: $60,000\nClothing Q1: $30,000\nClothing Q2: $35,000"
    
    # Create PDF
    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title = Paragraph("Nested Tables Business Report", styles['Title'])
    story.append(title)
    story.append(Paragraph("<br/>", styles['Normal']))
    
    # Main table structure
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
    
    story.append(Paragraph("Nested Table Example (Electronics Q1):", styles['Heading2']))
    story.append(nested_table)
    
    doc.build(story)


def create_text_document(output_path: Path):
    """Create a text document using Gemini-generated content."""
    print("Generating text document...")
    
    # Generate text content using Gemini
    text_prompt = """
    Write a comprehensive technical document about RAG (Retrieval-Augmented Generation) systems.
    Include:
    1. Introduction to RAG
    2. How it works
    3. Benefits and use cases
    4. Implementation considerations
    5. Future trends
    
    Make it professional and detailed, suitable for a technical audience.
    """
    
    try:
        response = model.generate_content(text_prompt)
        text_content = response.text
    except Exception as e:
        print(f"Error generating content with Gemini: {e}")
        text_content = """
        RAG (Retrieval-Augmented Generation) Systems
        
        Introduction:
        RAG is a powerful approach that combines information retrieval with text generation.
        
        How it works:
        1. Retrieve relevant documents
        2. Generate responses based on retrieved context
        3. Combine both for accurate answers
        
        Benefits:
        - Improved accuracy
        - Reduced hallucinations
        - Better context awareness
        """
    
    # Create PDF
    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Split content into paragraphs
    paragraphs = text_content.split('\n\n')
    for para in paragraphs:
        if para.strip():
            story.append(Paragraph(para.strip(), styles['Normal']))
            story.append(Paragraph("<br/>", styles['Normal']))
    
    doc.build(story)


def create_image_with_text(output_path: Path):
    """Create an image with text for OCR testing using Gemini-generated content."""
    print("Generating image with text...")
    
    # Generate text content using Gemini
    image_text_prompt = """
    Create a professional business card design with the following information:
    - Company: Tech Solutions Inc.
    - Name: John Smith
    - Title: Senior AI Engineer
    - Email: john.smith@techsolutions.com
    - Phone: (555) 123-4567
    - Address: 123 Tech Street, Silicon Valley, CA 94000
    
    Format it nicely for a business card layout.
    """
    
    try:
        response = model.generate_content(image_text_prompt)
        card_content = response.text
    except Exception as e:
        print(f"Error generating content with Gemini: {e}")
        card_content = "Tech Solutions Inc.\nJohn Smith\nSenior AI Engineer\njohn.smith@techsolutions.com\n(555) 123-4567\n123 Tech Street, Silicon Valley, CA 94000"
    
    # Create image
    width, height = 800, 600
    image = PILImage.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    # Try to use a default font
    try:
        font_large = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 32)
        font_medium = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
        font_small = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 18)
    except:
        try:
            font_large = ImageFont.truetype("arial.ttf", 32)
            font_medium = ImageFont.truetype("arial.ttf", 24)
            font_small = ImageFont.truetype("arial.ttf", 18)
        except:
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()
    
    # Draw text
    lines = card_content.strip().split('\n')
    y_position = 50
    
    for i, line in enumerate(lines):
        if line.strip():
            if i == 0:  # Company name
                draw.text((50, y_position), line.strip(), fill='darkblue', font=font_large)
            elif i == 1:  # Name
                draw.text((50, y_position), line.strip(), fill='black', font=font_medium)
            else:  # Other details
                draw.text((50, y_position), line.strip(), fill='darkgreen', font=font_small)
        y_position += 40
    
    # Save image
    image.save(output_path, 'PNG')
    print(f"Created image: {output_path}")


def create_diagram_image(output_path: Path):
    """Create a flowchart diagram using Gemini-generated content."""
    print("Generating flowchart diagram...")
    
    # Generate diagram description using Gemini
    diagram_prompt = """
    Create a flowchart description for a RAG system process that includes:
    1. Document ingestion
    2. Text chunking
    3. Vector embedding
    4. Storage in vector database
    5. Query processing
    6. Retrieval
    7. Response generation
    
    Format it as a step-by-step process with clear connections.
    """
    
    try:
        response = model.generate_content(diagram_prompt)
        diagram_content = response.text
    except Exception as e:
        print(f"Error generating content with Gemini: {e}")
        diagram_content = """
        RAG System Flow:
        1. Document Ingestion
        2. Text Chunking
        3. Vector Embedding
        4. Vector Storage
        5. Query Processing
        6. Document Retrieval
        7. Response Generation
        """
    
    # Create flowchart image
    width, height = 1000, 800
    image = PILImage.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
    
    # Draw flowchart boxes
    boxes = [
        (100, 50, 300, 100, "Document Ingestion"),
        (100, 150, 300, 200, "Text Chunking"),
        (100, 250, 300, 300, "Vector Embedding"),
        (100, 350, 300, 400, "Vector Storage"),
        (500, 100, 700, 150, "Query Processing"),
        (500, 200, 700, 250, "Document Retrieval"),
        (500, 300, 700, 350, "Response Generation")
    ]
    
    for x1, y1, x2, y2, text in boxes:
        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline='black', width=2, fill='lightblue')
        
        # Draw text
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = x1 + (x2 - x1 - text_width) // 2
        text_y = y1 + (y2 - y1 - text_height) // 2
        draw.text((text_x, text_y), text, fill='black', font=font)
    
    # Draw arrows
    arrows = [
        (200, 100, 200, 150),  # Ingestion to Chunking
        (200, 200, 200, 250),  # Chunking to Embedding
        (200, 300, 200, 350),  # Embedding to Storage
        (300, 200, 500, 125),  # Storage to Query
        (600, 150, 600, 200),  # Query to Retrieval
        (600, 250, 600, 300),  # Retrieval to Generation
    ]
    
    for x1, y1, x2, y2 in arrows:
        draw.line([x1, y1, x2, y2], fill='black', width=2)
        # Draw arrowhead
        if y2 > y1:  # Down arrow
            draw.polygon([(x2-5, y2-10), (x2+5, y2-10), (x2, y2)], fill='black')
        else:  # Right arrow
            draw.polygon([(x2-10, y2-5), (x2-10, y2+5), (x2, y2)], fill='black')
    
    # Save image
    image.save(output_path, 'PNG')
    print(f"Created diagram: {output_path}")


def create_excel_sample(output_path: Path):
    """Create an Excel file with sample data using Gemini-generated content."""
    print("Generating Excel sample...")
    
    # Generate Excel content using Gemini
    excel_prompt = """
    Create a comprehensive sales dataset for a retail company with the following columns:
    - Product_ID, Product_Name, Category, Price, Quantity_Sold, Sales_Date, Customer_ID, Region
    
    Include 20-30 rows of realistic data with:
    - Different product categories (Electronics, Clothing, Books, Home)
    - Various price ranges
    - Different regions (North, South, East, West)
    - Sales data spanning 3 months
    - Mix of high and low quantity sales
    """
    
    try:
        response = model.generate_content(excel_prompt)
        excel_content = response.text
    except Exception as e:
        print(f"Error generating content with Gemini: {e}")
        excel_content = """
        Product_ID,Product_Name,Category,Price,Quantity_Sold,Sales_Date,Customer_ID,Region
        1,Laptop,Electronics,999.99,5,2024-01-15,C001,North
        2,T-Shirt,Clothing,19.99,20,2024-01-16,C002,South
        3,Book,Books,12.99,15,2024-01-17,C003,East
        """
    
    # Parse content and create Excel file
    import pandas as pd
    from datetime import datetime, timedelta
    import random
    
    # Create sample data
    data = []
    products = [
        ("Laptop", "Electronics", 999.99),
        ("T-Shirt", "Clothing", 19.99),
        ("Book", "Books", 12.99),
        ("Coffee Maker", "Home", 89.99),
        ("Smartphone", "Electronics", 699.99),
        ("Jeans", "Clothing", 49.99),
        ("Novel", "Books", 15.99),
        ("Blender", "Home", 79.99)
    ]
    
    regions = ["North", "South", "East", "West"]
    start_date = datetime(2024, 1, 1)
    
    for i in range(30):
        product_name, category, base_price = random.choice(products)
        price = base_price + random.uniform(-50, 50)
        quantity = random.randint(1, 25)
        sales_date = start_date + timedelta(days=random.randint(0, 90))
        customer_id = f"C{random.randint(100, 999):03d}"
        region = random.choice(regions)
        
        data.append({
            "Product_ID": i + 1,
            "Product_Name": product_name,
            "Category": category,
            "Price": round(price, 2),
            "Quantity_Sold": quantity,
            "Sales_Date": sales_date.strftime("%Y-%m-%d"),
            "Customer_ID": customer_id,
            "Region": region
        })
    
    # Create DataFrame and save to Excel
    df = pd.DataFrame(data)
    df.to_excel(output_path, index=False, sheet_name="Sales_Data")
    
    # Create additional sheets
    with pd.ExcelWriter(output_path, engine='openpyxl', mode='a') as writer:
        # Summary by category
        category_summary = df.groupby('Category').agg({
            'Quantity_Sold': 'sum',
            'Price': 'mean'
        }).round(2)
        category_summary.to_excel(writer, sheet_name="Category_Summary")
        
        # Summary by region
        region_summary = df.groupby('Region').agg({
            'Quantity_Sold': 'sum',
            'Price': 'mean'
        }).round(2)
        region_summary.to_excel(writer, sheet_name="Region_Summary")
    
    print(f"Created Excel file: {output_path}")


if __name__ == "__main__":
    create_test_data()
