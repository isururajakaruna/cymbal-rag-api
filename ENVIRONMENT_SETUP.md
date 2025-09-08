# Environment Setup Guide

This guide explains how to set up the environment variables for the Cymbal RAG API.

## Quick Setup

1. **Copy the template file:**
   ```bash
   cp .env-template .env
   ```

2. **Edit the `.env` file with your actual values:**
   ```bash
   nano .env  # or use your preferred editor
   ```

## Required Environment Variables

### Google Cloud Configuration
- `GOOGLE_CLOUD_PROJECT_ID`: Your Google Cloud project ID
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to your service account key file
- `GOOGLE_CLOUD_REGION`: Google Cloud region (default: us-central1)

### Document AI Configuration
- `DOCUMENT_AI_PROCESSOR_ID`: Your Document AI processor ID
- `DOCUMENT_AI_LOCATION`: Document AI location (default: us)

### Vertex AI Configuration
- `VERTEX_AI_LOCATION`: Vertex AI location (default: us-central1)
- `VERTEX_AI_MODEL_NAME`: Gemini model name (default: gemini-2.5-flash)
- `VERTEX_AI_EMBEDDING_MODEL_NAME`: Embedding model name (default: gemini-embedding-001)

### Vector Search Configuration
- `VECTOR_SEARCH_INDEX_ID`: Your Vector Search index ID
- `VECTOR_SEARCH_INDEX_ENDPOINT_ID`: Your Vector Search endpoint ID
- `VECTOR_SEARCH_DEPLOYED_INDEX_ID`: Your deployed index ID

### Storage Configuration
- `STORAGE_BUCKET_NAME`: Your Google Cloud Storage bucket name

### API Configuration
- `API_HOST`: API host (default: 0.0.0.0)
- `API_PORT`: API port (default: 8000)
- `API_WORKERS`: Number of workers (default: 1)

### Authentication Configuration
- `API_AUTH_TOKEN`: Secure random token for API authentication

## Generating a Secure Token

To generate a secure authentication token, run:
```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

## Security Notes

- **Never commit the `.env` file** - it contains sensitive information
- **Keep your service account key secure** - store it outside the repository
- **Use strong, random tokens** for API authentication
- **Rotate tokens regularly** in production environments

## Testing the Setup

After setting up your environment variables, test the configuration:

```bash
# Test the API
python3 scripts/quick_api_test.py

# Test concurrent requests
python3 scripts/test_concurrent_requests.py
```

## Troubleshooting

### Common Issues

1. **Authentication errors**: Check that your service account key file exists and has the correct permissions
2. **Vector Search errors**: Ensure your Vector Search index and endpoint are properly configured
3. **Storage errors**: Verify your bucket name and permissions
4. **Token errors**: Make sure your API_AUTH_TOKEN is set and matches in your requests

### Getting Help

If you encounter issues:
1. Check the server logs for detailed error messages
2. Verify all environment variables are set correctly
3. Ensure all required Google Cloud APIs are enabled
4. Check that your service account has the necessary permissions
