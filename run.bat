@echo off
echo =========================================
echo     AdMySense - Object Detection App
echo =========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://python.org
    pause
    exit /b 1
)

echo Python found! Installing dependencies...
echo.

REM Install required packages
pip install -r requirements.txt

if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    echo Please check your internet connection and try again
    pause
    exit /b 1
)

echo.
echo Dependencies installed successfully!
echo.
echo Starting AdMySense Object Detection App...
echo.
echo The app will open in your browser at: http://localhost:8501
echo.
echo To stop the app, press Ctrl+C in this window
echo.

REM Run the Streamlit app
python -m streamlit run pages/2_fixed_object_detection.py

echo.
echo App stopped. Press any key to exit...
pause >nul
