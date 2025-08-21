# 📱💻 AdMyVision Mobile Streaming Setup Guide

## 🎯 Goal
Stream mobile phone camera to laptop for AI object detection WITHOUT opening laptop camera.

## 📋 Quick Start (Windows)

### Option 1: Using Ngrok (Recommended - Works on Any Network)

1. **Install Ngrok:**
   ```bash
   # Download from https://ngrok.com/download
   # Or install via Chocolatey:
   choco install ngrok
   ```

2. **Run the startup script:**
   ```bash
   # Double-click or run in terminal:
   start_ngrok.bat
   ```

3. **Get URLs from output:**
   - Mobile URL: `https://xxxxx.ngrok.io` (use this on phone)
   - Laptop URL: `http://localhost:8502` (use this on laptop)

4. **Connect:**
   - Open ngrok URL on phone
   - Grant camera permissions
   - Select "Phone Camera (WebRTC)" mode
   - Your phone camera will stream to laptop

### Option 2: Same WiFi Network

1. **Start the application:**
   ```bash
   python -m streamlit run pages/3_fixed_object_detection.py --server.port=8502
   ```

2. **Get your local IP:**
   - On phone: Open `http://YOUR_LAPTOP_IP:8502`
   - On laptop: Open `http://localhost:8502`

## 🔧 How It Works

### Device Detection
The app automatically detects if you're on mobile or laptop:

**📱 Mobile Phone (Sender):**
- WebRTC mode: `SENDONLY` 
- Sends camera stream to laptop
- Uses back camera (`facingMode: environment`)
- No AI processing (saves battery)

**💻 Laptop (Receiver):**
- WebRTC mode: `RECVONLY`
- Receives stream from phone
- Runs AI object detection
- Does NOT open laptop camera
- Displays detection results

### Camera Access Requirements
- **HTTPS required** for mobile camera access
- **Self-signed certificates** for local HTTPS
- **Ngrok provides** automatic HTTPS tunnel

## 🛠️ Troubleshooting

### Mobile Camera Not Working
1. **Check HTTPS:**
   - URL must be `https://` or `http://localhost`
   - Chrome: Enable "Insecure origins treated as secure"

2. **Grant Permissions:**
   - Allow camera access when prompted
   - Check browser settings if blocked

3. **Try Different Browser:**
   - Chrome Mobile (recommended)
   - Firefox Mobile
   - Safari Mobile

### Connection Issues
1. **Same WiFi Network:**
   - Both devices on same network
   - Check firewall settings
   - Try disabling VPN

2. **Network Isolation:**
   - Use ngrok tunnel instead
   - Bypasses network restrictions
   - Works from anywhere

### Performance Issues
1. **Reduce Quality:**
   - Lower video resolution
   - Reduce frame rate
   - Check CPU usage

2. **Network Speed:**
   - Use 5GHz WiFi if available
   - Close other streaming apps
   - Test with mobile data

## 📂 File Structure
```
admysense/
├── pages/3_fixed_object_detection.py  # Main app with conditional WebRTC
├── start_ngrok.bat                    # Windows ngrok startup
├── start_ngrok.sh                     # Linux/Mac ngrok startup  
├── mobile_streaming_test/             # Isolated test app
├── certificates/                      # SSL certificates
└── MOBILE_STREAMING_GUIDE.md         # This guide
```

## 🔍 Key Features

### Smart WebRTC Configuration
- **Automatic device detection** via JavaScript
- **Conditional streaming modes** based on device
- **Multiple STUN servers** for better connectivity
- **Optimized constraints** for mobile vs laptop

### Camera Constraints
```javascript
// Mobile (Sender)
{
  "video": {
    "facingMode": "environment",  // Back camera
    "width": {"ideal": 640},
    "height": {"ideal": 480},
    "frameRate": {"ideal": 15}
  }
}

// Laptop (Receiver)  
{
  "video": false  // No laptop camera used
}
```

### STUN Server Configuration
```javascript
"iceServers": [
  {"urls": ["stun:stun.l.google.com:19302"]},
  {"urls": ["stun:stun1.l.google.com:19302"]},
  {"urls": ["stun:stun2.l.google.com:19302"]},
  // Additional STUN servers for redundancy
]
```

## 🚀 Advanced Configuration

### Custom Ngrok Domain
```bash
# In start_ngrok.bat, replace:
ngrok http 8502
# With:
ngrok http 8502 --subdomain=your-custom-name
```

### Port Configuration
```bash
# Change port in both scripts:
streamlit run app.py --server.port=8503
ngrok http 8503
```

### Debug Mode
```python
# Add to app for debugging:
st.write("Debug: Device type =", "Mobile" if is_mobile_device else "Desktop")
st.write("Debug: WebRTC mode =", webrtc_ctx.mode)
st.write("Debug: Connection state =", webrtc_ctx.state)
```

## ✅ Success Indicators

### On Mobile Phone:
- ✅ Camera permission granted
- ✅ Video preview visible
- ✅ "Sending camera to laptop" message
- ✅ Connection status: "Playing"

### On Laptop:
- ✅ Receiving mobile camera feed
- ✅ AI detection running on received stream  
- ✅ Object detection boxes visible
- ✅ Voice announcements (if enabled)
- ✅ No laptop camera activated

## 🆘 Support

### Get Help:
1. Check ngrok dashboard: `http://localhost:4040`
2. View browser console for JavaScript errors
3. Check Streamlit logs in terminal
4. Test with mobile_streaming_test app first

### Common Error Solutions:
- `navigator.mediaDevices is undefined` → Use HTTPS or ngrok
- `Permission denied` → Grant camera access in browser
- `Connection failed` → Check firewall/network settings
- `Stream not received` → Verify WebRTC mode configuration

---

🎉 **Enjoy seamless mobile-to-laptop camera streaming with AI object detection!**
