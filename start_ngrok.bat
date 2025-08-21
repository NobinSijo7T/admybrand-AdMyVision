@echo off
REM AdMyVision Ngrok Startup Script for Windows
REM Provides public HTTPS URL to bypass network isolation issues

echo 🌐 AdMyVision Ngrok Startup
echo ==========================

REM Check if ngrok is installed
where ngrok >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ ngrok is not installed!
    echo 📥 Please download and install ngrok from: https://ngrok.com/download
    echo 💡 Or install via: choco install ngrok
    pause
    exit /b 1
)

echo ✅ Ngrok is ready!
echo 🚀 Starting AdMyVision with public HTTPS access...

REM Start Streamlit in background
echo 📱 Starting Streamlit server...
start /B python -m streamlit run pages/3_fixed_object_detection.py --server.port=8502 --server.enableCORS=false --server.enableXsrfProtection=false

REM Wait for Streamlit to start
echo ⏳ Waiting for Streamlit to initialize...
timeout /t 5 /nobreak >nul

REM Start ngrok tunnel
echo 🌐 Creating ngrok tunnel...
start /B ngrok http 8502

REM Wait for ngrok to start
timeout /t 5 /nobreak >nul

REM Try to get the public URL (requires curl or PowerShell)
echo 🔍 Getting public URL...

REM Using PowerShell to get ngrok URL
for /f "delims=" %%i in ('powershell -command "(Invoke-RestMethod http://localhost:4040/api/tunnels).tunnels[0].public_url"') do set NGROK_URL=%%i

if defined NGROK_URL (
    echo.
    echo 🎉 AdMyVision is now accessible worldwide!
    echo ==================================
    echo 📱 **Mobile URL (use this on your phone):**
    echo    %NGROK_URL%
    echo.
    echo 💻 **Local URL (use this on laptop):**
    echo    http://localhost:8502
    echo.
    echo 🔒 **Benefits of using ngrok:**
    echo    ✅ HTTPS enabled (fixes camera access issues)
    echo    ✅ Works from any network (bypasses WiFi restrictions)
    echo    ✅ No firewall configuration needed
    echo    ✅ Works with any mobile device
    echo.
    echo 📋 **Instructions:**
    echo 1. Open the ngrok URL on your phone
    echo 2. Grant camera permissions
    echo 3. Select 'Phone Camera (WebRTC)' mode
    echo 4. Your phone camera will stream to laptop
    echo.
    echo 🌐 Ngrok dashboard: http://localhost:4040
    echo 🛑 Press Ctrl+C to stop both servers
) else (
    echo ❌ Could not retrieve ngrok URL
    echo 💡 Check ngrok dashboard manually at: http://localhost:4040
    echo 📱 Look for the HTTPS URL there
)

echo.
echo ⏸️  Servers running... Press any key to stop
pause >nul

REM Cleanup
echo.
echo 🛑 Shutting down...
taskkill /f /im python.exe >nul 2>&1
taskkill /f /im ngrok.exe >nul 2>&1
echo ✅ Servers stopped
