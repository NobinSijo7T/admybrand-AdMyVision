#!/bin/bash

echo "========================================="
echo "    AdMySense - Object Detection App"
echo "========================================="
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 is not installed or not in PATH"
    echo "Please install Python3 from your package manager"
    echo "  Ubuntu/Debian: sudo apt install python3 python3-pip"
    echo "  CentOS/RHEL: sudo yum install python3 python3-pip"
    echo "  macOS: brew install python3"
    exit 1
fi

echo "Python3 found! Installing dependencies..."
echo

# Install required packages
python3 -m pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    echo "Please check your internet connection and try again"
    echo "You may need to run: sudo python3 -m pip install -r requirements.txt"
    exit 1
fi

echo
echo "Dependencies installed successfully!"
echo
echo "Starting AdMySense Object Detection App..."
echo
echo "The app will open in your browser at: http://localhost:8501"
echo
echo "To stop the app, press Ctrl+C in this terminal"
echo

# Run the Streamlit app
python3 -m streamlit run pages/2_fixed_object_detection.py

echo
echo "App stopped."
