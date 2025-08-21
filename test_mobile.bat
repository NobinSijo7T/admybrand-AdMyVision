@echo off
REM Quick Test Launcher for Mobile Streaming
echo 🧪 AdMyVision Mobile Streaming Test
echo ==================================

echo 🚀 Starting test application...
echo.
echo 📱 **Test Features:**
echo    ✅ Device detection
echo    ✅ Conditional WebRTC modes  
echo    ✅ Mobile camera streaming
echo    ✅ Laptop receive-only mode
echo.

python -m streamlit run test_mobile_streaming.py --server.port=8503

echo.
echo 🛑 Test completed!
pause
