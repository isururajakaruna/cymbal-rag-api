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

# Kill any existing processes using the port
echo -e "${YELLOW}Checking for existing processes on port ${PORT}...${NC}"
if lsof -ti:${PORT} > /dev/null 2>&1; then
    echo -e "${YELLOW}Found existing processes on port ${PORT}. Killing them...${NC}"
    lsof -ti:${PORT} | xargs kill -9 2>/dev/null
    sleep 2
    echo -e "${GREEN}Port ${PORT} is now free.${NC}"
else
    echo -e "${GREEN}Port ${PORT} is available.${NC}"
fi

# Set environment variables for Google Cloud
echo -e "${GREEN}Setting up Google Cloud environment variables...${NC}"
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/service-account-key.json"
export GOOGLE_CLOUD_PROJECT_ID="cymbol-demo"
export STORAGE_BUCKET_NAME="cymbal-rag-store"

# Start the server with conda environment
echo -e "${GREEN}Starting FastAPI server with cymbal-rag environment...${NC}"
echo -e "${BLUE}Server will be available at: http://localhost:${PORT}${NC}"
echo -e "${BLUE}API Documentation: http://localhost:${PORT}/docs${NC}"
echo -e "${BLUE}ReDoc Documentation: http://localhost:${PORT}/redoc${NC}"
echo ""

# Use conda run to execute uvicorn in the cymbal-rag environment
conda run -n cymbal-rag uvicorn app.main:app --host 0.0.0.0 --port $PORT --reload
