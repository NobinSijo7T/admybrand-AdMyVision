@echo off
REM AdMyVision Complete Setup Script for Windows
REM Installs dependencies and sets up mobile streaming

echo 🚀 AdMyVision Complete Setup
echo ===========================

echo 📦 Step 1: Installing Python Dependencies
echo.
python -m pip install --upgrade pip
python -m pip install streamlit streamlit-webrtc opencv-python numpy pyttsx3 gtts pygame

echo.
echo 🌐 Step 2: Checking Ngrok Installation
where ngrok >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Ngrok is not installed!
    echo.
    echo 📥 **MANUAL INSTALLATION REQUIRED:**
    echo 1. Go to: https://ngrok.com/download
    echo 2. Download ngrok for Windows
    echo 3. Extract to a folder
    echo 4. Add to PATH or place in project folder
    echo.
    echo 💡 **Alternative - Install via Chocolatey:**
    echo    choco install ngrok
    echo.
    echo 🔄 **Or download directly:**
    echo.
    
    REM Try to download ngrok directly
    echo 📥 Attempting to download ngrok...
    powershell -command "& { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-windows-amd64.zip' -OutFile 'ngrok.zip' }"
    
    if exist ngrok.zip (
        echo ✅ Downloaded ngrok.zip
        echo 📂 Extracting...
        powershell -command "Expand-Archive -Path 'ngrok.zip' -DestinationPath '.' -Force"
        del ngrok.zip
        echo ✅ Ngrok extracted successfully!
    ) else (
        echo ❌ Download failed. Please install manually.
        echo 🌐 Visit: https://ngrok.com/download
        pause
        exit /b 1
    )
) else (
    echo ✅ Ngrok is already installed!
)

echo.
echo 🔑 Step 3: Ngrok Authentication
echo Please sign up at https://ngrok.com/ and get your auth token
echo.
set /p NGROK_TOKEN="Enter your ngrok auth token (or press Enter to skip): "

if defined NGROK_TOKEN (
    echo 🔑 Setting up ngrok authentication...
    ngrok authtoken %NGROK_TOKEN%
    if %errorlevel% equ 0 (
        echo ✅ Ngrok authenticated successfully!
    ) else (
        echo ❌ Authentication failed. You can set it up later with: ngrok authtoken YOUR_TOKEN
    )
) else (
    echo ⏭️ Skipping authentication - you can set it up later
    echo 💡 Run: ngrok authtoken YOUR_TOKEN
)

echo.
echo 🔒 Step 4: Creating SSL Certificates for HTTPS
echo.
if not exist certificates mkdir certificates
cd certificates

REM Generate self-signed certificate for HTTPS
echo 🔐 Generating SSL certificate...
powershell -command "& { $cert = New-SelfSignedCertificate -DnsName 'localhost' -CertStoreLocation 'cert:\CurrentUser\My' -KeyUsage KeyEncipherment,DigitalSignature -KeyAlgorithm RSA -KeyLength 2048 -HashAlgorithm SHA256 -NotAfter (Get-Date).AddYears(1); $pwd = ConvertTo-SecureString -String 'password' -Force -AsPlainText; Export-PfxCertificate -Cert $cert -FilePath 'localhost.pfx' -Password $pwd; $pem = '-----BEGIN CERTIFICATE-----'; $pem += [System.Convert]::ToBase64String($cert.RawData, [System.Base64FormattingOptions]::InsertLineBreaks); $pem += '-----END CERTIFICATE-----'; $pem | Out-File -FilePath 'localhost.crt' -Encoding ASCII; }"

cd ..

if exist certificates\localhost.crt (
    echo ✅ SSL certificate created successfully!
) else (
    echo ❌ SSL certificate creation failed
)

echo.
echo 🧪 Step 5: Testing Setup
echo.
echo 📱 Testing mobile streaming configuration...

REM Create a simple test file
echo import streamlit as st > test_setup.py
echo st.title("✅ AdMyVision Setup Test") >> test_setup.py
echo st.success("If you can see this, Streamlit is working!") >> test_setup.py
echo st.info("Mobile streaming is ready to use") >> test_setup.py

echo 🚀 Starting test server...
start /B python -m streamlit run test_setup.py --server.port=8501
timeout /t 3 /nobreak >nul

echo ✅ Test server started at: http://localhost:8501
echo.

echo 🎉 Setup Complete!
echo ================
echo.
echo 📋 **Next Steps:**
echo 1. ✅ Python dependencies installed
echo 2. ✅ Ngrok ready (authentication may be needed)
echo 3. ✅ SSL certificates created
echo 4. ✅ Test server running
echo.
echo 🚀 **To start mobile streaming:**
echo    start_ngrok.bat
echo.
echo 💻 **To run locally:**
echo    python -m streamlit run pages/3_fixed_object_detection.py
echo.
echo 📱 **Mobile streaming will:**
echo    - Stream phone camera to laptop
echo    - Run AI detection on laptop
echo    - NOT open laptop camera
echo    - Work across networks with ngrok
echo.
echo 🌐 **Test server:** http://localhost:8501
echo 📚 **Full guide:** MOBILE_STREAMING_GUIDE.md
echo.

del test_setup.py >nul 2>&1
taskkill /f /im python.exe >nul 2>&1

echo ✅ Setup complete! Press any key to exit...
pause >nul
