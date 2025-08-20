@echo off
REM WebRTC Real-time Object Detection Demo
REM Start script with mode switching support for Windows

set MODE=wasm
set NGROK=false

REM Parse command line arguments
:parse_args
if "%1"=="" goto start_setup
if "%1"=="--mode" (
    set MODE=%2
    shift
    shift
    goto parse_args
)
if "%1"=="--ngrok" (
    set NGROK=true
    shift
    goto parse_args
)
if "%1"=="--help" (
    echo Usage: %0 [--mode server^|wasm] [--ngrok]
    echo   --mode    Set inference mode ^(server or wasm^), default: wasm
    echo   --ngrok   Start ngrok tunnel for phone connectivity
    exit /b 0
)
shift
goto parse_args

:start_setup
echo Starting WebRTC Object Detection Demo...
echo Mode: %MODE%

REM Check if Docker is available
docker --version >nul 2>&1
if %errorlevel% == 0 (
    docker-compose --version >nul 2>&1
    if %errorlevel% == 0 (
        echo Using Docker Compose...
        set MODE=%MODE%
        docker-compose up --build
        goto end
    )
)

echo Docker not found, starting in development mode...

REM Install Node.js dependencies
if exist "frontend" (
    echo Installing frontend dependencies...
    cd frontend
    npm install
    npm run build
    cd ..
)

REM Install Python dependencies
echo Installing Python dependencies...
pip install -r requirements.txt

REM Start the backend server
echo Starting backend server...
set MODE=%MODE%
start /b python backend/server.py

REM Start ngrok if requested
if "%NGROK%"=="true" (
    where ngrok >nul 2>&1
    if %errorlevel% == 0 (
        echo Starting ngrok tunnel...
        start /b ngrok http 8001
        echo Ngrok tunnel started. Check http://localhost:4040 for the public URL.
    ) else (
        echo ngrok not found. Please install ngrok for phone connectivity.
    )
)

echo Demo started!
echo Open http://localhost:8001 in your browser
echo Scan the QR code with your phone to connect

pause

:end
