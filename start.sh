#!/bin/bash

# WebRTC Real-time Object Detection Demo
# Start script with mode switching support

set -e

MODE=${MODE:-wasm}
NGROK=${NGROK:-false}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --ngrok)
      NGROK=true
      shift
      ;;
    --help)
      echo "Usage: $0 [--mode server|wasm] [--ngrok]"
      echo "  --mode    Set inference mode (server or wasm), default: wasm"
      echo "  --ngrok   Start ngrok tunnel for phone connectivity"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "Starting WebRTC Object Detection Demo..."
echo "Mode: $MODE"

# Check if Docker is available
if command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
    echo "Using Docker Compose..."
    export MODE=$MODE
    docker-compose up --build
else
    echo "Docker not found, starting in development mode..."
    
    # Install Node.js dependencies
    if [ -d "frontend" ]; then
        echo "Installing frontend dependencies..."
        cd frontend
        npm install
        npm run build
        cd ..
    fi
    
    # Install Python dependencies
    echo "Installing Python dependencies..."
    pip install -r requirements.txt
    
    # Start the backend server
    export MODE=$MODE
    python backend/server.py &
    SERVER_PID=$!
    
    # Start ngrok if requested
    if [ "$NGROK" = true ]; then
        if command -v ngrok &> /dev/null; then
            echo "Starting ngrok tunnel..."
            ngrok http 8001 &
            NGROK_PID=$!
            echo "Ngrok tunnel started. Check http://localhost:4040 for the public URL."
        else
            echo "ngrok not found. Please install ngrok for phone connectivity."
        fi
    fi
    
    echo "Demo started!"
    echo "Open http://localhost:8001 in your browser"
    echo "Scan the QR code with your phone to connect"
    
    # Wait for interrupt
    trap 'kill $SERVER_PID; [ ! -z "$NGROK_PID" ] && kill $NGROK_PID; exit' INT
    wait $SERVER_PID
fi
