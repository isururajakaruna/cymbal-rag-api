# File Validation API Documentation

## Overview

The File Validation API provides endpoints for validating and uploading files to the Cymbal RAG system. It includes comprehensive validation for file formats, content quality, and FAQ structure using Google Gemini AI.

## Base URL

```
http://localhost:8000/api/v1/file
```

## Endpoints

### 1. Validate File

**POST** `/validate`

Upload and validate a file for processing.

#### Request

- **Content-Type**: `multipart/form-data`
- **Body**:
  - `file` (file, required): The file to validate
  - `replace_existing` (boolean, optional): Whether to replace existing files (default: false)

#### Supported File Types

- **Images**: `.jpg`, `.jpeg`, `.png`
- **Documents**: `.pdf`, `.docx`
- **Spreadsheets**: `.xlsx`, `.xls`

#### Response

**Success (200)**:
```json
{
  "success": true,
  "validation_id": "uuid-string",
  "filename": "document.pdf",
  "content_type": "application/pdf",
  "file_size": 1024000,
  "temp_path": "tmp/document.pdf",
  "file_exists": false,
  "content_analysis": {
    "content_quality": {
      "score": 8,
      "is_sufficient": true,
      "reasoning": "Document contains substantial information suitable for knowledge base"
    },
    "faq_structure": {
      "is_faq": false,
      "score": 5,
      "has_proper_qa_pairs": false,
      "reasoning": "Document does not appear to be an FAQ format"
    }
  },
  "message": "File validation successful and uploaded to temporary storage"
}
```

**Error Responses**:

**Unsupported Format (400)**:
```json
{
  "success": false,
  "error": "Unsupported file format",
  "supported_extensions": [".jpg", ".jpeg", ".png", ".pdf", ".docx", ".xlsx", ".xls"],
  "provided_type": "text/plain",
  "provided_extension": ".txt"
}
```

**File Already Exists (409)**:
```json
{
  "success": false,
  "error": "File already exists",
  "filename": "document.pdf",
  "suggestion": "Use replace_existing=true to replace the file, or upload with a different name"
}
```

**Insufficient Content Quality (400)**:
```json
{
  "success": false,
  "error": "Insufficient content quality",
  "quality_score": 3,
  "reasoning": "Document contains very little information",
  "suggestion": "Please upload a document with more substantial content"
}
```

**Incomplete FAQ Structure (400)**:
```json
{
  "success": false,
  "error": "Incomplete FAQ structure",
  "faq_score": 4,
  "reasoning": "FAQ document lacks proper question-answer pairs",
  "suggestion": "Please ensure the FAQ document contains proper question-answer pairs"
}
```

### 2. Get Supported Formats

**GET** `/supported-formats`

Get list of supported file formats and extensions.

#### Response

```json
{
  "supported_formats": {
    "image/jpeg": [".jpg", ".jpeg"],
    "image/png": [".png"],
    "application/pdf": [".pdf"],
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"],
    "application/vnd.ms-excel": [".xls"]
  },
  "all_extensions": [".jpg", ".jpeg", ".png", ".pdf", ".docx", ".xlsx", ".xls"],
  "total_formats": 6
}
```

## Validation Process

The file validation process includes:

1. **Format Validation**: Checks if the file type is supported
2. **Existence Check**: Verifies if the file already exists in the uploads directory
3. **Content Quality Analysis**: Uses Gemini AI to assess if the content is substantial enough
4. **FAQ Structure Validation**: If the document appears to be an FAQ, validates the question-answer structure
5. **Temporary Upload**: Uploads the file to the `_tmp` directory in Google Cloud Storage

## Testing

### Using Postman

1. Import the `postman_collection.json` file
2. Set the `base_url` variable to your API endpoint
3. Run the test requests

### Using Python

```python
import requests

# Test file validation
with open('document.pdf', 'rb') as f:
    files = {'file': f}
    data = {'replace_existing': 'false'}
    
    response = requests.post(
        'http://localhost:8000/api/v1/file/validate',
        files=files,
        data=data
    )
    
print(response.json())
```

### Using cURL

```bash
curl -X POST "http://localhost:8000/api/v1/file/validate" \
  -F "file=@document.pdf" \
  -F "replace_existing=false"
```

## Error Handling

The API provides detailed error messages for different validation failures:

- **400 Bad Request**: Invalid file format, insufficient content quality, or incomplete FAQ structure
- **409 Conflict**: File already exists (when replace_existing=false)
- **500 Internal Server Error**: Server-side errors during processing

## Configuration

The API requires the following environment variables:

- `GOOGLE_CLOUD_PROJECT_ID`: Your Google Cloud project ID
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to your service account key file
- `STORAGE_BUCKET_NAME`: Google Cloud Storage bucket name
- `VERTEX_AI_MODEL_NAME`: Vertex AI model name (default: gemini-pro)

## Next Steps

After successful validation, files are stored in the temporary directory and can be processed further through additional API endpoints (to be implemented).
