# Cymbal RAG API Documentation

## Overview

The Cymbal RAG API is a comprehensive Retrieval-Augmented Generation system built with FastAPI and Google Cloud services. It provides intelligent document search and question-answering capabilities through a RESTful API interface.

## Base URL

```
http://localhost:8000
```

## Authentication

All API endpoints (except health checks and supported formats) require authentication using a token passed as a query parameter:

```
?token=your-auth-token
```

## API Endpoints

### System

#### Health Check
```http
GET /health
```

**Description:** Check the overall health of the API and its services.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-09-08T19:46:45.226093",
  "version": "1.0.0",
  "services": {
    "api": "healthy",
    "document_processor": "healthy",
    "vector_store": "healthy",
    "storage": "healthy"
  }
}
```

#### Get Supported File Formats
```http
GET /api/v1/file/supported-formats?token={token}
```

**Description:** Get a list of supported file formats for upload.

**Response:**
```json
{
  "success": true,
  "supported_formats": [
    {
      "extension": "pdf",
      "mime_type": "application/pdf",
      "description": "Portable Document Format"
    },
    {
      "extension": "jpg",
      "mime_type": "image/jpeg",
      "description": "JPEG Image"
    }
  ]
}
```

### File Management

#### Validate File
```http
POST /api/v1/file/validate?token={token}
```

**Description:** Validate a file before uploading to check format, content quality, and handle conflicts.

**Request Body:** `multipart/form-data`
- `file`: File to validate
- `replace_existing`: Boolean (optional, default: false)

**Response:**
```json
{
  "success": true,
  "validation_id": "uuid-string",
  "filename": "document.pdf",
  "file_size": 1024000,
  "content_type": "application/pdf",
  "file_exists": false,
  "content_analysis": {
    "content_quality": {
      "score": 9,
      "is_sufficient": true,
      "reasoning": "Document contains substantial business information"
    },
    "is_faq": false,
    "faq_analysis": null
  }
}
```

#### Upload File Direct
```http
POST /api/v1/upload/direct?token={token}
```

**Description:** Upload a file directly to the knowledge base with processing and embedding.

**Request Body:** `multipart/form-data`
- `file`: File to upload
- `replace_existing`: Boolean (optional, default: true)
- `tags`: Comma-separated tags (optional)

**Response:**
```json
{
  "success": true,
  "filename": "document.pdf",
  "total_chunks": 15,
  "embeddings_stored": 15,
  "processing_time_ms": 2500.5,
  "message": "File uploaded and processed successfully"
}
```

#### Upload from Validation
```http
POST /api/v1/upload/{validation_id}?token={token}
```

**Description:** Upload a file using its validation ID from temporary storage.

**Path Parameters:**
- `validation_id`: UUID from validation response

**Response:**
```json
{
  "success": true,
  "filename": "document.pdf",
  "total_chunks": 15,
  "embeddings_stored": 15,
  "processing_time_ms": 2000.0,
  "message": "File uploaded and processed successfully"
}
```

#### List Files
```http
GET /api/v1/files/list?token={token}
```

**Query Parameters:**
- `search`: Search query for filename (optional)
- `tags`: Comma-separated tags to filter by (optional)
- `sort`: Sort order - "date" or "name" (optional, default: "date")
- `limit`: Number of results per page (optional, default: 50)
- `offset`: Number of results to skip (optional, default: 0)

**Response:**
```json
{
  "success": true,
  "files": [
    {
      "file_id": "uuid-string",
      "filename": "document.pdf",
      "file_size": 1024000,
      "content_type": "application/pdf",
      "upload_timestamp": "2025-09-08T19:46:45.226093",
      "last_modified": "2025-09-08T19:46:45.226093",
      "status": "processed",
      "chunks_count": 15,
      "tags": ["business", "hr"],
      "title": "Employee Handbook 2025"
    }
  ],
  "total_count": 1,
  "page": 0,
  "page_size": 50,
  "tags_filter": ["business", "hr"]
}
```

#### View File
```http
GET /api/v1/files/view?filename={filename}&token={token}
```

**Description:** Download or view a specific file.

**Query Parameters:**
- `filename`: Name of the file to download

**Response:** File content with appropriate Content-Type header

#### Get Embedding Stats
```http
GET /api/v1/files/embedding-stats?filename={filename}&token={token}
```

**Description:** Get embedding statistics for a specific file.

**Query Parameters:**
- `filename`: Name of the file to get stats for

**Response:**
```json
{
  "success": true,
  "filename": "document.pdf",
  "file_size": 1024000,
  "content_type": "application/pdf",
  "last_modified": "2025-09-08T19:46:45.226093",
  "total_chunks": 15,
  "embeddings_stored": 15,
  "processing_time_ms": 2500.5
}
```

#### Delete File
```http
DELETE /api/v1/upload/delete?filename={filename}&token={token}
```

**Description:** Delete a file from the knowledge base.

**Query Parameters:**
- `filename`: Name of the file to delete

**Response:**
```json
{
  "success": true,
  "message": "File deleted successfully",
  "filename": "document.pdf"
}
```

### Search

#### RAG Search
```http
POST /api/v1/search/rag?token={token}
```

**Description:** Perform RAG search with vector similarity and reranking.

**Request Body:**
```json
{
  "query": "What is the company policy on remote work?",
  "ktop": 5,
  "threshold": 0.7,
  "tags": ["hr", "policy"]
}
```

**Response:**
```json
{
  "success": true,
  "query": "What is the company policy on remote work?",
  "files": [
    {
      "name": "employee_handbook.pdf",
      "path": "uploads/employee_handbook.pdf",
      "file_type": "application/pdf",
      "last_updated": "2025-09-08T19:46:45.226093",
      "size": 1024000,
      "tags": ["hr", "policy"],
      "title": "Employee Handbook 2025",
      "matched_chunks": [
        {
          "content": "Remote work policy allows employees to work from home up to 3 days per week...",
          "file_id": "uuid-string",
          "filename": "employee_handbook.pdf",
          "chunk_index": 5,
          "distance": 0.65,
          "metadata": {
            "title": "Employee Handbook 2025",
            "tags": ["hr", "policy"]
          }
        }
      ]
    }
  ],
  "total_files": 1,
  "total_chunks": 1,
  "rag_response": "Based on the employee handbook, the company allows remote work up to 3 days per week with manager approval...",
  "processing_time_ms": 1500.5,
  "search_parameters": {
    "ktop": 5,
    "threshold": 0.7,
    "tags": ["hr", "policy"]
  }
}
```

## Error Responses

All endpoints may return the following error responses:

### 400 Bad Request
```json
{
  "success": false,
  "error": "Validation Error",
  "message": "Invalid request parameters"
}
```

### 401 Unauthorized
```json
{
  "success": false,
  "error": "Authentication required",
  "message": "Missing 'token' parameter in query string"
}
```

### 404 Not Found
```json
{
  "success": false,
  "error": "File Not Found",
  "message": "File 'document.pdf' not found"
}
```

### 500 Internal Server Error
```json
{
  "success": false,
  "error": "Internal Server Error",
  "message": "An unexpected error occurred"
}
```

## Postman Collection

A complete Postman collection is available at `postman_collection.json` with all API endpoints pre-configured for easy testing and integration.

## Support

For technical support or questions about the API, please refer to the main README.md file or open an issue in the project repository.
