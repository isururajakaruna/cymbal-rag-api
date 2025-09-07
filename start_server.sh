#!/bin/bash

# Cymbal RAG API Server Startup Script
# Usage: ./start_server.sh [port]
# Default port: 8000

# Set default port
PORT=${1:-8000}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Cymbal RAG API Server ===${NC}"
echo -e "${YELLOW}Starting server on port: ${PORT}${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo ""

# Check if conda environment exists
if ! conda info --envs | grep -q "cymbal-rag"; then
    echo -e "${RED}Error: cymbal-rag conda environment not found!${NC}"
    echo "Please create the environment first:"
    echo "conda env create -f environment.yml"
    exit 1
fi

# Activate conda environment and start server
echo -e "${GREEN}Activating cymbal-rag environment...${NC}"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate cymbal-rag

# Set environment variables for Google Cloud
echo -e "${GREEN}Setting up Google Cloud environment variables...${NC}"
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/service-account-key.json"
export GOOGLE_CLOUD_PROJECT_ID="cymbol-demo"
export STORAGE_BUCKET_NAME="cymbal-rag-store"

# Check if required packages are installed
echo -e "${GREEN}Checking dependencies...${NC}"
python -c "import fastapi, uvicorn, vertexai" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Required packages not installed!${NC}"
    echo "Please install dependencies:"
    echo "pip install -r requirements.txt"
    exit 1
fi

# Start the server
echo -e "${GREEN}Starting FastAPI server...${NC}"
echo -e "${BLUE}Server will be available at: http://localhost:${PORT}${NC}"
echo -e "${BLUE}API Documentation: http://localhost:${PORT}/docs${NC}"
echo -e "${BLUE}ReDoc Documentation: http://localhost:${PORT}/redoc${NC}"
echo ""

uvicorn app.main:app --host 0.0.0.0 --port $PORT --reload
