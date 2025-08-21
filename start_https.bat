@echo off
echo Starting AdMyVision with HTTPS support...
echo.
echo This script will:
echo 1. Generate a self-signed SSL certificate (if not exists)
echo 2. Start Streamlit with HTTPS enabled
echo 3. This fixes "navigator.mediaDevices is undefined" errors on mobile devices
echo.

REM Check if certificate files exist
if not exist "cert.pem" (
    echo Generating SSL certificate...
    echo WARNING: You'll need OpenSSL installed for this to work
    echo You can download OpenSSL from: https://slproweb.com/products/Win32OpenSSL.html
    echo.
    openssl req -newkey rsa:2048 -nodes -keyout key.pem -x509 -days 365 -out cert.pem -subj "/C=US/ST=Local/L=Local/O=AdMyVision/CN=localhost"
    if errorlevel 1 (
        echo ERROR: Failed to generate certificate. Please install OpenSSL first.
        echo Alternative: Use Chrome flags method mentioned in the app troubleshooting section.
        pause
        exit /b 1
    )
    echo Certificate generated successfully!
    echo.
)

echo Starting Streamlit with HTTPS...
echo Access the app at: https://localhost:8502
echo For phone access: https://YOUR_LAPTOP_IP:8502
echo.
echo Press Ctrl+C to stop the server
echo.

python -m streamlit run pages/3_fixed_object_detection.py --server.port=8502 --server.sslCertFile=cert.pem --server.sslKeyFile=key.pem --server.enableCORS=false --server.enableXsrfProtection=false

pause
