# AdMySense Object Detection App - PowerShell Runner
# This script sets up and runs the AdMySense application

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "    AdMySense - Object Detection App" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version 2>$null
    Write-Host "✅ $pythonVersion found!" -ForegroundColor Green
} catch {
    Write-Host "❌ ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python from https://python.org" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
Write-Host ""

# Install required packages
try {
    pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        throw "Installation failed"
    }
    Write-Host "✅ Dependencies installed successfully!" -ForegroundColor Green
} catch {
    Write-Host "❌ ERROR: Failed to install dependencies" -ForegroundColor Red
    Write-Host "Please check your internet connection and try again" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "🚀 Starting AdMySense Object Detection App..." -ForegroundColor Green
Write-Host ""
Write-Host "📱 The app will open in your browser at: http://localhost:8501" -ForegroundColor Cyan
Write-Host ""
Write-Host "⏹️  To stop the app, press Ctrl+C in this window" -ForegroundColor Yellow
Write-Host ""

# Run the Streamlit app
try {
    python -m streamlit run pages/2_fixed_object_detection.py
} catch {
    Write-Host ""
    Write-Host "⚠️  App was interrupted or stopped" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "👋 App stopped. Thanks for using AdMySense!" -ForegroundColor Cyan
Read-Host "Press Enter to exit"
