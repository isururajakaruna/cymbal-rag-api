#!/bin/bash

# Cymbal RAG API - Production Run Script
# Version: 1.0.0
# Description: Starts the RAG API server with auto-restart capabilities

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
APP_NAME="cymbal-rag-api"
SERVER_APP="app.main:app"
PID_FILE=".server.pid"
LOG_FILE="server.log"
RESTART_DELAY=2
DEFAULT_PORT=8000
DEFAULT_WORKERS=4

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

# Function to check if server is running
is_running() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0
        else
            rm -f "$PID_FILE"
            return 1
        fi
    fi
    return 1
}

# Function to start the server
start_server() {
    local port=${1:-$DEFAULT_PORT}
    local workers=${2:-$DEFAULT_WORKERS}
    local mode=${3:-"prod"}
    
    print_status "Starting $APP_NAME server on port $port..."
    
    # Check if already running
    if is_running; then
        print_warning "Server is already running (PID: $(cat $PID_FILE))"
        return 1
    fi
    
    # Check if conda environment exists
    if ! conda info --envs | grep -q "cymbal-rag"; then
        print_error "Cymbal RAG conda environment not found!"
        print_status "Please create the environment first:"
        print_status "conda env create -f environment.yml"
        exit 1
    fi
    
    # Check if .env file exists
    if [ ! -f ".env" ]; then
        print_error ".env file not found!"
        print_status "Please copy .env-template to .env and configure it:"
        print_status "cp .env-template .env"
        exit 1
    fi
    
    # Kill any existing processes using the port
    if lsof -ti:$port > /dev/null 2>&1; then
        print_warning "Found existing processes on port $port. Killing them..."
        lsof -ti:$port | xargs kill -9 2>/dev/null
        sleep 2
    fi
    
    # Set environment variables for Google Cloud
    print_status "Setting up Google Cloud environment variables..."
    export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/service-account-key.json"
    export GOOGLE_CLOUD_PROJECT_ID="cymbol-demo"
    export STORAGE_BUCKET_NAME="cymbal-rag-store"
    
    # Start server in background
    if [ "$mode" = "dev" ]; then
        print_status "Starting in development mode with reload..."
        nohup conda run -n cymbal-rag --no-capture-output uvicorn $SERVER_APP --host 0.0.0.0 --port $port --reload --log-level info > "$LOG_FILE" 2>&1 &
    else
        print_status "Starting in production mode with $workers workers..."
        nohup conda run -n cymbal-rag --no-capture-output uvicorn $SERVER_APP --host 0.0.0.0 --port $port --workers $workers --log-level info > "$LOG_FILE" 2>&1 &
    fi
    
    local pid=$!
    echo $pid > "$PID_FILE"
    
    # Wait a moment and check if it started successfully
    sleep 3
    if is_running; then
        print_success "Server started successfully (PID: $pid)"
        print_status "Server URL: http://0.0.0.0:$port"
        print_status "API Docs: http://0.0.0.0:$port/docs"
        print_status "Logs: tail -f $LOG_FILE"
        print_status "Stop: ./stop.sh"
        return 0
    else
        print_error "Failed to start server!"
        print_error "Check logs: cat $LOG_FILE"
        rm -f "$PID_FILE"
        return 1
    fi
}

# Function to stop the server
stop_server() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            print_status "Stopping server (PID: $pid)..."
            kill "$pid"
            sleep 2
            
            # Force kill if still running
            if ps -p "$pid" > /dev/null 2>&1; then
                print_warning "Force killing server..."
                kill -9 "$pid"
            fi
            
            print_success "Server stopped"
        else
            print_warning "Server was not running"
        fi
        rm -f "$PID_FILE"
    else
        print_warning "No PID file found - server may not be running"
    fi
}

# Function to restart the server
restart_server() {
    local port=${1:-$DEFAULT_PORT}
    local workers=${2:-$DEFAULT_WORKERS}
    local mode=${3:-"prod"}
    
    print_status "Restarting server..."
    stop_server
    sleep 1
    start_server $port $workers $mode
}

# Function to monitor and restart server
monitor_server() {
    local port=${1:-$DEFAULT_PORT}
    local workers=${2:-$DEFAULT_WORKERS}
    local mode=${3:-"prod"}
    
    print_status "Starting monitor mode - server will auto-restart if killed"
    print_status "Press Ctrl+C to stop monitoring"
    
    while true; do
        if ! is_running; then
            print_warning "Server is not running, starting..."
            start_server $port $workers $mode
        fi
        
        # Check every 5 seconds
        sleep 5
    done
}

# Main script logic
case "${1:-start}" in
    start)
        start_server $2 $3 $4
        ;;
    stop)
        stop_server
        ;;
    restart)
        restart_server $2 $3 $4
        ;;
    monitor)
        monitor_server $2 $3 $4
        ;;
    status)
        if is_running; then
            print_success "Server is running (PID: $(cat $PID_FILE))"
        else
            print_warning "Server is not running"
        fi
        ;;
    logs)
        if [ -f "$LOG_FILE" ]; then
            tail -f "$LOG_FILE"
        else
            print_error "Log file not found: $LOG_FILE"
        fi
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|monitor|status|logs} [port] [workers] [mode]"
        echo ""
        echo "Commands:"
        echo "  start   - Start the server once"
        echo "  stop    - Stop the server"
        echo "  restart - Restart the server"
        echo "  monitor - Start server with auto-restart monitoring"
        echo "  status  - Check if server is running"
        echo "  logs    - Show live server logs"
        echo ""
        echo "Parameters:"
        echo "  port    - Server port (default: $DEFAULT_PORT)"
        echo "  workers - Number of workers (default: $DEFAULT_WORKERS)"
        echo "  mode    - Server mode: prod|dev (default: prod)"
        echo ""
        echo "Examples:"
        echo "  $0 start                    # Start on port 8000 with 4 workers"
        echo "  $0 start 9000              # Start on port 9000 with 4 workers"
        echo "  $0 start 9000 2            # Start on port 9000 with 2 workers"
        echo "  $0 start 9000 2 dev        # Start on port 9000 with 2 workers in dev mode"
        echo "  $0 monitor 8000 4 prod     # Monitor mode with auto-restart"
        exit 1
        ;;
esac
