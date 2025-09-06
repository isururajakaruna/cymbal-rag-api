# Cymbal RAG API

A comprehensive RAG (Retrieval-Augmented Generation) API built with FastAPI and Google Vertex AI services.

## Features

- **Document Processing**: Support for PDF, text, images, and structured documents
- **Vector Search**: Semantic search using Google Vertex AI Vector Search
- **RAG Generation**: Response generation using Google's Flash model
- **Table Extraction**: Advanced table processing including nested tables
- **OCR Support**: Image text extraction and processing
- **RESTful API**: Clean FastAPI-based endpoints
- **Comprehensive Testing**: Full test suite with pytest

## Architecture

```
cymbal-rag/
├── app/                    # Main application code
│   ├── api/               # API endpoints
│   ├── core/              # Core configuration and exceptions
│   ├── services/          # Business logic services
│   ├── models/            # Pydantic models
│   └── utils/             # Utility functions
├── tests/                 # Test suite
├── config/                # Configuration files
├── scripts/               # Setup and utility scripts
└── test_data/             # Sample test documents
```

## Prerequisites

- Python 3.11+
- Google Cloud Platform account
- Conda (for environment management)

## Quick Start

### 1. Create Conda Environment

```bash
# Create environment from environment.yml
conda env create -f environment.yml

# Activate environment
conda activate cymbal-rag
```

### 2. Set Up Google Cloud

```bash
# Set up GCP resources
python scripts/setup_gcp.py

# Set up authentication
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service-account-key.json"
```

### 3. Configure Environment

```bash
# Copy environment template
cp config/.env.example .env

# Edit .env with your GCP project details
# Update config/config.json as needed
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the API

```bash
# Development server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or using the main module
python app/main.py
```

### 6. Create Test Data

```bash
# Generate sample test documents
python scripts/populate_test_data.py
```

## API Endpoints

### File Management

- `POST /api/v1/files/upload` - Upload a new file
- `PUT /api/v1/files/{file_id}` - Update an existing file
- `DELETE /api/v1/files/{file_id}` - Delete a file
- `GET /api/v1/files/` - List all files
- `GET /api/v1/files/{file_id}` - Get file information
- `GET /api/v1/files/{file_id}/status` - Get processing status

### Search

- `POST /api/v1/search/` - Search documents (JSON body)
- `GET /api/v1/search/` - Search documents (query parameters)
- `GET /api/v1/search/health` - Search service health check

### System

- `GET /` - Root endpoint with API information
- `GET /health` - Health check

## Configuration

### Environment Variables (.env)

```bash
# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT_ID=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account-key.json
GOOGLE_CLOUD_REGION=us-central1

# Document AI Configuration
DOCUMENT_AI_PROCESSOR_ID=your-processor-id
DOCUMENT_AI_LOCATION=us

# Vertex AI Configuration
VERTEX_AI_LOCATION=us-central1
VERTEX_AI_MODEL_NAME=gemini-1.5-flash

# Vector Search Configuration
VECTOR_SEARCH_INDEX_ID=your-index-id
VECTOR_SEARCH_INDEX_ENDPOINT_ID=your-endpoint-id

# Storage Configuration
STORAGE_BUCKET_NAME=your-bucket-name
```

### RAG Configuration (config.json)

```json
{
  "rag": {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "max_chunks_per_document": 50,
    "similarity_threshold": 0.7,
    "max_results": 10
  },
  "document_processing": {
    "supported_formats": ["pdf", "txt", "docx", "png", "jpg", "jpeg"],
    "max_file_size_mb": 10,
    "ocr_enabled": true,
    "table_extraction_enabled": true
  }
}
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test categories
pytest tests/unit/
pytest tests/integration/

# Run with verbose output
pytest -v
```

## Supported Document Types

### Text Documents
- Plain text files (.txt)
- PDF documents (.pdf)
- Word documents (.docx)

### Images with Text
- PNG images (.png)
- JPEG images (.jpg, .jpeg)

### Tables
- Simple tables
- Nested tables
- Complex table structures

## Development

### Code Quality

```bash
# Format code
black app/ tests/

# Sort imports
isort app/ tests/

# Lint code
flake8 app/ tests/
```

### Adding New Features

1. Create feature branch
2. Implement feature with tests
3. Update documentation
4. Run test suite
5. Submit pull request

## Deployment

The API is designed to run on any platform that supports Python and FastAPI:

- Google Cloud Run
- AWS Lambda
- Azure Functions
- Docker containers
- Traditional servers

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For questions and support, please open an issue in the GitHub repository.
