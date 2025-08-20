#!/bin/bash

# Benchmarking script for WebRTC Object Detection
# Usage: ./bench/run_bench.sh --duration 30 --mode server

set -e

# Default values
DURATION=30
MODE="server"
OUTPUT_FILE="metrics/metrics.json"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --duration)
      DURATION="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --output)
      OUTPUT_FILE="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [--duration SECONDS] [--mode server|wasm] [--output FILE]"
      echo "  --duration  Duration of benchmark run in seconds (default: 30)"
      echo "  --mode      Inference mode (server or wasm, default: server)"
      echo "  --output    Output file for metrics (default: metrics/metrics.json)"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "Starting benchmark run..."
echo "Duration: ${DURATION} seconds"
echo "Mode: ${MODE}"
echo "Output: ${OUTPUT_FILE}"

# Ensure output directory exists
mkdir -p "$(dirname "$OUTPUT_FILE")"

# Start the server if not already running
SERVER_PID=""
if ! curl -s http://localhost:8001/metrics > /dev/null 2>&1; then
    echo "Starting server in ${MODE} mode..."
    export MODE=$MODE
    
    if command -v docker-compose &> /dev/null; then
        docker-compose up -d
        echo "Waiting for server to start..."
        sleep 10
    else
        python backend/server.py &
        SERVER_PID=$!
        echo "Waiting for server to start..."
        sleep 5
    fi
fi

# Wait for server to be ready
echo "Waiting for server to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:8001/metrics > /dev/null 2>&1; then
        echo "Server is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "Server failed to start within 30 seconds"
        exit 1
    fi
    sleep 1
done

# Reset metrics
echo "Resetting metrics..."
curl -s -X POST http://localhost:8001/metrics/reset > /dev/null || true

# Run benchmark for specified duration
echo "Running benchmark for ${DURATION} seconds..."
echo "Connect your phone to http://localhost:8001 and start detection"
echo "Benchmark will automatically collect metrics..."

# Wait for the specified duration
sleep $DURATION

# Export metrics
echo "Exporting metrics..."
curl -s -X POST http://localhost:8001/metrics/export > /dev/null

# Get final metrics
echo "Collecting final metrics..."
METRICS=$(curl -s http://localhost:8001/metrics)

# Create comprehensive benchmark report
cat > "$OUTPUT_FILE" << EOF
{
  "benchmark_info": {
    "duration": $DURATION,
    "mode": "$MODE",
    "timestamp": $(date +%s),
    "date": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  },
  "metrics": $METRICS,
  "system_info": {
    "os": "$(uname -s)",
    "cpu_cores": $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "unknown"),
    "memory_gb": $(free -g 2>/dev/null | awk '/^Mem:/{print $2}' || echo "unknown")
  }
}
EOF

# Parse and display key metrics
echo ""
echo "=== BENCHMARK RESULTS ==="
echo "Duration: ${DURATION} seconds"
echo "Mode: ${MODE}"

if command -v jq &> /dev/null; then
    echo "Processed FPS: $(echo "$METRICS" | jq -r '.processed_fps // "N/A"')"
    echo "Median E2E Latency: $(echo "$METRICS" | jq -r '.median_e2e_latency // "N/A"') ms"
    echo "P95 E2E Latency: $(echo "$METRICS" | jq -r '.p95_e2e_latency // "N/A"') ms"
    echo "Total Frames: $(echo "$METRICS" | jq -r '.total_frames // "N/A"')"
    echo "Detection Rate: $(echo "$METRICS" | jq -r '.detection_rate // "N/A"')"
else
    echo "Raw metrics: $METRICS"
fi

echo ""
echo "Detailed metrics saved to: $OUTPUT_FILE"
echo "========================="

# Cleanup
if [ ! -z "$SERVER_PID" ]; then
    echo "Stopping server..."
    kill $SERVER_PID 2>/dev/null || true
fi

echo "Benchmark complete!"
