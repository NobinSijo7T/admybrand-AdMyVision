"""
MINIMAL MOBILE CAMERA STREAMING TEST
Simple test to verify mobile camera can stream to laptop
"""

import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import socket

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "localhost"

st.set_page_config(
    page_title="📱 Mobile Camera Test",
    page_icon="📱",
    layout="wide"
)

st.title("📱 Mobile Camera Streaming Test")

# Get network info
local_ip = get_local_ip()
port = 8502

st.markdown(f"""
### 🌐 Connection Info
- **Laptop IP:** `{local_ip}`
- **Port:** `{port}`
- **Mobile URL:** `http://{local_ip}:{port}`

### 📱 Instructions for Phone:
1. Open this URL on your phone: `http://{local_ip}:{port}`
2. **IMPORTANT:** For Chrome on Android, enable insecure origins:
   - Go to: `chrome://flags/#unsafely-treat-insecure-origin-as-secure`
   - Add: `http://{local_ip}:{port}`
   - Set to "Enabled"
   - Restart Chrome
3. Grant camera permissions when prompted
4. You should see your camera feed below
""")

# Simple video frame callback for testing
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # Add simple overlay to confirm processing
    height, width = img.shape[:2]
    cv2.putText(img, "MOBILE CAMERA WORKING!", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, f"Frame: {width}x{height}", 
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# WebRTC configuration optimized for mobile
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
})

# Create WebRTC streamer
st.subheader("📹 Camera Stream")

webrtc_ctx = webrtc_streamer(
    key="mobile_camera_test",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 640, "min": 320, "max": 1280},
            "height": {"ideal": 480, "min": 240, "max": 720},
            "frameRate": {"ideal": 15, "min": 5, "max": 30},
            "facingMode": "environment"  # Use back camera
        },
        "audio": False
    },
    rtc_configuration=RTC_CONFIGURATION,
    async_processing=True,
)

# Status indicators
if webrtc_ctx.state.playing:
    st.success("✅ Camera is streaming successfully!")
    st.info("You should see 'MOBILE CAMERA WORKING!' text overlay if processing is working.")
elif webrtc_ctx.state.signalling:
    st.warning("🔄 Connecting to camera...")
else:
    st.error("❌ Camera not connected. Make sure to:")
    st.markdown("""
    - Grant camera permissions
    - Use the correct URL for your phone
    - Enable Chrome flags for insecure origins
    - Both devices on same WiFi network
    """)

# Add JavaScript diagnostic
st.markdown("### 🔍 Browser Diagnostics")
diagnostic_js = f"""
<div id="diagnostic-output" style="background: #f0f0f0; padding: 15px; border-radius: 5px; margin: 10px 0;">
    Checking browser compatibility...
</div>

<script>
function runDiagnostics() {{
    const output = document.getElementById('diagnostic-output');
    let results = [];
    
    results.push('🌐 URL: ' + window.location.href);
    results.push('📱 User Agent: ' + navigator.userAgent.substring(0, 50) + '...');
    results.push('🔒 Protocol: ' + window.location.protocol);
    results.push('📍 Host: ' + window.location.host);
    
    if (typeof navigator !== 'undefined') {{
        results.push('✅ Navigator: Available');
        
        if (navigator.mediaDevices) {{
            results.push('✅ MediaDevices: Available');
            
            if (navigator.mediaDevices.getUserMedia) {{
                results.push('✅ GetUserMedia: Available');
                
                // Test device enumeration
                navigator.mediaDevices.enumerateDevices()
                .then(devices => {{
                    const cameras = devices.filter(d => d.kind === 'videoinput');
                    results.push('📹 Cameras: ' + cameras.length + ' found');
                    output.innerHTML = results.join('<br>');
                }})
                .catch(err => {{
                    results.push('❌ Device enumeration failed: ' + err.message);
                    output.innerHTML = results.join('<br>');
                }});
            }} else {{
                results.push('❌ GetUserMedia: Not available');
                output.innerHTML = results.join('<br>');
            }}
        }} else {{
            results.push('🚨 MediaDevices: UNDEFINED - Need Chrome flags or HTTPS!');
            results.push('🔧 Fix: chrome://flags/#unsafely-treat-insecure-origin-as-secure');
            results.push('➕ Add: http://{local_ip}:{port}');
            output.innerHTML = results.join('<br>');
        }}
    }} else {{
        results.push('❌ Navigator: Not available');
        output.innerHTML = results.join('<br>');
    }}
}}

// Run diagnostics when page loads
setTimeout(runDiagnostics, 1000);
</script>
"""

st.components.v1.html(diagnostic_js, height=200)

st.markdown("---")
st.markdown("**📞 Quick Debug:**")
st.markdown(f"- 📱 Phone URL: `http://{local_ip}:{port}`")
st.markdown(f"- 🔧 Chrome flags: Add `http://{local_ip}:{port}` to insecure origins")
st.markdown("- 📷 Camera permissions: Must be granted")
st.markdown("- 🌐 WiFi: Both devices on same network")
