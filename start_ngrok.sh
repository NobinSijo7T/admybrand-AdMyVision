#!/bin/bash
# AdMyVision Ngrok Startup Script
# Provides public HTTPS URL to bypass network isolation issues

echo "🌐 AdMyVision Ngrok Startup"
echo "=========================="

# Check if ngrok is installed
if ! command -v ngrok &> /dev/null; then
    echo "❌ ngrok is not installed!"
    echo "📥 Please download and install ngrok from: https://ngrok.com/download"
    echo "💡 Or install via package manager:"
    echo "   - Windows: choco install ngrok"
    echo "   - macOS: brew install ngrok"
    echo "   - Linux: snap install ngrok"
    exit 1
fi

# Check if ngrok is authenticated
if ! ngrok config check &> /dev/null; then
    echo "🔐 Ngrok authentication required"
    echo "1. Sign up at https://dashboard.ngrok.com/signup"
    echo "2. Get your authtoken from https://dashboard.ngrok.com/get-started/your-authtoken"
    echo "3. Run: ngrok config add-authtoken <your_authtoken>"
    exit 1
fi

echo "✅ Ngrok is ready!"
echo "🚀 Starting AdMyVision with public HTTPS access..."

# Start Streamlit in background
echo "📱 Starting Streamlit server..."
python -m streamlit run pages/3_fixed_object_detection.py --server.port=8502 --server.enableCORS=false --server.enableXsrfProtection=false &
STREAMLIT_PID=$!

# Wait for Streamlit to start
echo "⏳ Waiting for Streamlit to initialize..."
sleep 5

# Start ngrok tunnel
echo "🌐 Creating ngrok tunnel..."
ngrok http 8502 &
NGROK_PID=$!

# Wait for ngrok to start
sleep 3

# Get the public URL
echo "🔍 Getting public URL..."
NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | jq -r '.tunnels[0].public_url')

if [ "$NGROK_URL" != "null" ] && [ "$NGROK_URL" != "" ]; then
    echo ""
    echo "🎉 AdMyVision is now accessible worldwide!"
    echo "=================================="
    echo "📱 **Mobile URL (use this on your phone):**"
    echo "   $NGROK_URL"
    echo ""
    echo "💻 **Local URL (use this on laptop):**"
    echo "   http://localhost:8502"
    echo ""
    echo "🔒 **Benefits of using ngrok:**"
    echo "   ✅ HTTPS enabled (fixes camera access issues)"
    echo "   ✅ Works from any network (bypasses WiFi restrictions)"
    echo "   ✅ No firewall configuration needed"
    echo "   ✅ Works with any mobile device"
    echo ""
    echo "📋 **Instructions:**"
    echo "1. Open the ngrok URL on your phone"
    echo "2. Grant camera permissions"
    echo "3. Select 'Phone Camera (WebRTC)' mode"
    echo "4. Your phone camera will stream to laptop"
    echo ""
    echo "🛑 Press Ctrl+C to stop both servers"
else
    echo "❌ Failed to get ngrok URL. Please check your connection and try again."
    kill $STREAMLIT_PID 2>/dev/null
    kill $NGROK_PID 2>/dev/null
    exit 1
fi

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down..."
    kill $STREAMLIT_PID 2>/dev/null
    kill $NGROK_PID 2>/dev/null
    echo "✅ Servers stopped"
    exit 0
}

# Trap Ctrl+C
trap cleanup INT

# Keep script running
echo "⏸️  Servers running... Press Ctrl+C to stop"
wait
