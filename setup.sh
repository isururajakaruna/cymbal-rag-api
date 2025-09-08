#!/bin/bash

# Cymbal RAG API - Environment Setup Script
# Version: 1.0.0
# Description: Creates conda environment and installs all requirements

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENV_NAME="cymbal-rag"
PYTHON_VERSION="3.11"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

# Function to check if conda is available
check_conda() {
    if ! command -v conda &> /dev/null; then
        print_error "Conda is not installed or not in PATH!"
        print_status "Please install Miniconda or Anaconda first:"
        print_status "https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
    print_success "Conda is available"
}

# Function to check if environment already exists
check_environment() {
    if conda info --envs | grep -q "^${ENV_NAME}\s"; then
        print_warning "Environment '${ENV_NAME}' already exists!"
        read -p "Do you want to remove it and recreate? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_status "Removing existing environment..."
            conda env remove -n ${ENV_NAME} -y
            if [ $? -eq 0 ]; then
                print_success "Environment removed successfully"
            else
                print_error "Failed to remove environment"
                exit 1
            fi
        else
            print_status "Using existing environment"
            return 0
        fi
    fi
}

# Function to create conda environment
create_environment() {
    print_status "Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
    conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
    
    if [ $? -eq 0 ]; then
        print_success "Environment created successfully"
    else
        print_error "Failed to create environment"
        exit 1
    fi
}

# Function to activate environment and install requirements
install_requirements() {
    print_status "Activating environment and installing requirements..."
    
    # Activate environment and install requirements
    conda run -n ${ENV_NAME} pip install -r requirements.txt
    
    if [ $? -eq 0 ]; then
        print_success "Requirements installed successfully"
    else
        print_error "Failed to install requirements"
        exit 1
    fi
}

# Function to verify installation
verify_installation() {
    print_status "Verifying installation..."
    
    # Test if key packages can be imported
    conda run -n ${ENV_NAME} python -c "
import fastapi
import uvicorn
import google.cloud.storage
import vertexai
import pydantic
import pandas
import numpy
import PIL
import PyPDF2
import pdf2image
import openpyxl
import reportlab
import aiohttp
import matplotlib
import requests
print('✅ All key packages imported successfully')
"
    
    if [ $? -eq 0 ]; then
        print_success "Installation verification passed"
    else
        print_error "Installation verification failed"
        exit 1
    fi
}

# Function to create .env file if it doesn't exist
setup_env_file() {
    if [ ! -f ".env" ]; then
        if [ -f ".env-template" ]; then
            print_status "Creating .env file from template..."
            cp .env-template .env
            print_warning "Please edit .env file with your actual configuration values"
        else
            print_warning ".env-template not found, creating basic .env file..."
            cat > .env << EOF
# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT_ID=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=service-account-key.json
GOOGLE_CLOUD_REGION=us-central1

# Vertex AI Configuration
VERTEX_AI_LOCATION=us-central1
VERTEX_AI_MODEL_NAME=gemini-2.5-flash
VERTEX_AI_EMBEDDING_MODEL_NAME=gemini-embedding-001

# Vector Search Configuration
VECTOR_SEARCH_INDEX_ID=your-index-id
VECTOR_SEARCH_INDEX_ENDPOINT_ID=your-endpoint-id
VECTOR_SEARCH_DEPLOYED_INDEX_ID=your-deployed-index-id

# Storage Configuration
STORAGE_BUCKET_NAME=your-bucket-name

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Authentication Configuration
API_AUTH_TOKEN=your-secure-random-token-here
EOF
            print_warning "Please edit .env file with your actual configuration values"
        fi
    else
        print_status ".env file already exists"
    fi
}

# Function to display next steps
show_next_steps() {
    print_success "🎉 Setup completed successfully!"
    echo ""
    print_status "Next steps:"
    echo "1. Edit .env file with your actual configuration values"
    echo "2. Place your Google Cloud service account key as 'service-account-key.json'"
    echo "3. Start the server: ./run.sh start"
    echo "4. Or start in development mode: ./run.sh start 8000 1 dev"
    echo ""
    print_status "Available commands:"
    echo "  ./run.sh start [port] [workers] [mode]  - Start the server"
    echo "  ./run.sh stop                          - Stop the server"
    echo "  ./run.sh status                        - Check server status"
    echo "  ./run.sh logs                          - View server logs"
    echo ""
    print_status "API Documentation will be available at:"
    echo "  http://localhost:8000/docs"
    echo "  http://localhost:8000/redoc"
}

# Main execution
main() {
    echo -e "${BLUE}=== Cymbal RAG API Setup ===${NC}"
    echo -e "${YELLOW}This script will create a conda environment and install all requirements${NC}"
    echo ""
    
    # Check prerequisites
    check_conda
    
    # Check if environment exists
    check_environment
    
    # Create environment if needed
    if ! conda info --envs | grep -q "^${ENV_NAME}\s"; then
        create_environment
    fi
    
    # Install requirements
    install_requirements
    
    # Verify installation
    verify_installation
    
    # Setup .env file
    setup_env_file
    
    # Show next steps
    show_next_steps
}

# Run main function
main "$@"
