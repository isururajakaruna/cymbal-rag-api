# Google Cloud Storage Management Scripts

This directory contains scripts to manage Google Cloud Storage folders for the Cymbal RAG API.

## Scripts

### 1. `clear_storage.py` - Simple Cleanup Script

A straightforward script to clear both `_tmp` and `uploads` folders.

**Usage:**
```bash
# Activate conda environment
conda activate cymbal-rag

# Run the script
python scripts/clear_storage.py
```

**Features:**
- Clears both `_tmp/` and `uploads/` folders
- Asks for confirmation before deletion
- Shows detailed progress
- Provides summary of deleted files

### 2. `manage_storage.py` - Advanced Management Script

A comprehensive script with multiple options for storage management.

**Usage:**
```bash
# Activate conda environment
conda activate cymbal-rag

# Show help
python scripts/manage_storage.py --help

# Show storage statistics
python scripts/manage_storage.py --stats

# List files in both folders
python scripts/manage_storage.py --list

# List files with detailed information
python scripts/manage_storage.py --list-details

# Clear only _tmp folder
python scripts/manage_storage.py --clear-tmp

# Clear only uploads folder
python scripts/manage_storage.py --clear-uploads

# Clear both folders
python scripts/manage_storage.py --clear-all

# Dry run (show what would be deleted)
python scripts/manage_storage.py --clear-all --dry-run

# Skip confirmation prompts
python scripts/manage_storage.py --clear-all --force
```

**Features:**
- List files with or without details
- Clear specific folders or all folders
- Dry run mode to preview changes
- Storage statistics
- Force mode to skip confirmations
- Detailed error handling

## Prerequisites

1. **Google Cloud Credentials**: Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable
2. **Conda Environment**: Activate the `cymbal-rag` environment
3. **Bucket Access**: Ensure the service account has access to the specified bucket

## Environment Variables

Make sure these are set in your `.env` file:
```
GOOGLE_CLOUD_PROJECT_ID=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account-key.json
STORAGE_BUCKET_NAME=your-bucket-name
```

## Safety Features

- **Confirmation Prompts**: Both scripts ask for confirmation before deleting files
- **Dry Run Mode**: The advanced script can show what would be deleted without actually deleting
- **Detailed Logging**: Shows exactly which files are being deleted
- **Error Handling**: Gracefully handles errors and continues with remaining files

## Examples

### Quick Cleanup
```bash
# Simple cleanup of both folders
python scripts/clear_storage.py
```

### Check Storage Usage
```bash
# See how much space is being used
python scripts/manage_storage.py --stats
```

### Preview Deletion
```bash
# See what would be deleted without actually deleting
python scripts/manage_storage.py --clear-all --dry-run
```

### Clear Only Temporary Files
```bash
# Clear only the _tmp folder (temporary files)
python scripts/manage_storage.py --clear-tmp
```

### List All Files
```bash
# List all files with detailed information
python scripts/manage_storage.py --list-details
```

## Troubleshooting

### Common Issues

1. **Credentials Error**: Make sure `GOOGLE_APPLICATION_CREDENTIALS` is set correctly
2. **Bucket Not Found**: Verify the bucket name in your `.env` file
3. **Permission Denied**: Ensure the service account has storage admin permissions

### Getting Help

```bash
# Show all available options
python scripts/manage_storage.py --help
```

## Folder Structure

The scripts work with the following folder structure in your GCS bucket:

```
your-bucket/
├── _tmp/           # Temporary files (from file validation)
│   ├── file1.pdf
│   └── file2.xlsx
└── uploads/        # Processed files (from file upload)
    ├── file1.pdf
    └── file2.xlsx
```
