"""
ISOLATED MOBILE CAMERA STREAMING TEST
Testing mobile camera streaming to laptop without conflicts
"""

import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import socket
import threading
import time

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "localhost"

# Initialize session state
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'last_frame_time' not in st.session_state:
    st.session_state.last_frame_time = time.time()

st.set_page_config(
    page_title="📱 Mobile Camera Test",
    page_icon="📱",
    layout="wide"
)

st.title("📱 Mobile Camera Streaming Test")

# Get network info
local_ip = get_local_ip()
port = 8503

st.markdown(f"""
### 🌐 Connection Setup
**📱 Access this URL on your phone:** `http://{local_ip}:{port}`

### ⚠️ IMPORTANT: Camera Access Setup for Mobile
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    **🤖 For Android Chrome:**
    1. Copy this URL: `chrome://flags/#unsafely-treat-insecure-origin-as-secure`
    2. Paste in Chrome address bar
    3. Add your laptop URL: `{}`
    4. Set to "Enabled" 
    5. Restart Chrome
    6. Open the app URL again
    """.format(f"http://{local_ip}:{port}"))

with col2:
    st.markdown("""
    **🍎 For iPhone Safari:**
    Safari requires HTTPS for camera access from other devices.
    Use Android Chrome with the flags method above, or
    connect iPhone via USB for localhost access.
    """)

# Video frame callback that processes each frame
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # Update frame count
    st.session_state.frame_count += 1
    current_time = time.time()
    
    # Calculate FPS
    time_diff = current_time - st.session_state.last_frame_time
    if time_diff > 0:
        fps = 1.0 / time_diff
    else:
        fps = 0
    st.session_state.last_frame_time = current_time
    
    # Add visual indicators that processing is working
    height, width = img.shape[:2]
    
    # Green background for status
    cv2.rectangle(img, (10, 10), (width-10, 120), (0, 100, 0), -1)
    
    # Status text
    cv2.putText(img, "📱 MOBILE STREAMING ACTIVE!", 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, f"Frame #{st.session_state.frame_count}", 
                (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(img, f"Size: {width}x{height} | FPS: {fps:.1f}", 
                (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Add timestamp
    timestamp = time.strftime("%H:%M:%S")
    cv2.putText(img, f"Time: {timestamp}", 
                (width-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# WebRTC configuration optimized for mobile streaming
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
    ]
})

st.markdown("---")
st.subheader("📹 Camera Stream Test")

# Create the WebRTC streamer with mobile-optimized settings
webrtc_ctx = webrtc_streamer(
    key="mobile_streaming_test",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 640, "min": 320, "max": 1280},
            "height": {"ideal": 480, "min": 240, "max": 720},
            "frameRate": {"ideal": 15, "min": 5, "max": 25},
            "facingMode": "environment"  # Prefer back camera on mobile
        },
        "audio": False
    },
    rtc_configuration=RTC_CONFIGURATION,
    async_processing=True,
)

# Status display
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if webrtc_ctx.state.playing:
        st.success("✅ Camera Streaming!")
    else:
        st.error("❌ Not Streaming")

with col2:
    if webrtc_ctx.state.signalling:
        st.info("🔄 Connecting...")
    else:
        st.warning("⏸ Idle")

with col3:
    st.metric("Frame Count", st.session_state.frame_count)

# Detailed status and troubleshooting
if webrtc_ctx.state.playing:
    st.balloons()
    st.success("🎉 **SUCCESS!** Your mobile camera is streaming to the laptop!")
    st.info("You should see the green status overlay on the video feed above with frame count and timestamp.")
else:
    st.error("**Camera Not Streaming** - Troubleshooting:")
    
    issues_and_solutions = [
        ("📱 **Camera Permission Denied**", "Grant camera access when browser prompts"),
        ("🚫 **navigator.mediaDevices undefined**", "Enable Chrome flags as shown above"),
        ("🌐 **Wrong Network**", "Ensure both devices on same WiFi"),
        ("🔒 **Security Block**", "Use Chrome with insecure origins enabled"),
        ("📶 **Weak Signal**", "Move devices closer to WiFi router"),
    ]
    
    for issue, solution in issues_and_solutions:
        st.markdown(f"- {issue}: {solution}")

# Add real-time browser diagnostics
st.markdown("---")
st.markdown("### 🔍 Browser Diagnostics (Real-time)")

diagnostic_html = f"""
<div id="diagnostic-results" style="background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #007bff;">
    <strong>🔄 Running diagnostics...</strong>
</div>

<script>
let diagnosticInterval;

function updateDiagnostics() {{
    const resultsDiv = document.getElementById('diagnostic-results');
    let results = [];
    
    // Basic info
    results.push('🌐 Current URL: ' + window.location.href);
    results.push('🔒 Protocol: ' + window.location.protocol + ' (need HTTP for flags method)');
    results.push('📱 Device: ' + (window.innerWidth < 768 ? 'Mobile' : 'Desktop'));
    
    // Navigator check
    if (typeof navigator === 'undefined') {{
        results.push('❌ Navigator: Not available');
    }} else {{
        results.push('✅ Navigator: Available');
        
        // MediaDevices check
        if (!navigator.mediaDevices) {{
            results.push('🚨 MediaDevices: UNDEFINED - This is the main issue!');
            results.push('🔧 Solution: Enable Chrome flags or use HTTPS');
            results.push('🔗 Chrome flags URL: chrome://flags/#unsafely-treat-insecure-origin-as-secure');
            results.push('➕ Add this to flags: http://{local_ip}:{port}');
        }} else {{
            results.push('✅ MediaDevices: Available');
            
            if (!navigator.mediaDevices.getUserMedia) {{
                results.push('❌ GetUserMedia: Not supported');
            }} else {{
                results.push('✅ GetUserMedia: Available');
                
                // Try to enumerate devices
                navigator.mediaDevices.enumerateDevices()
                .then(devices => {{
                    const cameras = devices.filter(d => d.kind === 'videoinput');
                    results.push('📹 Cameras detected: ' + cameras.length);
                    if (cameras.length > 0) {{
                        results.push('🎉 Camera access should work!');
                    }}
                    resultsDiv.innerHTML = '<strong>📊 Diagnostic Results:</strong><br>' + results.join('<br>');
                }})
                .catch(err => {{
                    results.push('⚠️ Device enumeration failed: ' + err.message);
                    resultsDiv.innerHTML = '<strong>📊 Diagnostic Results:</strong><br>' + results.join('<br>');
                }});
                return; // Don't update results div here, wait for async result
            }}
        }}
    }}
    
    resultsDiv.innerHTML = '<strong>📊 Diagnostic Results:</strong><br>' + results.join('<br>');
}}

// Update diagnostics every 3 seconds
updateDiagnostics();
diagnosticInterval = setInterval(updateDiagnostics, 3000);
</script>
"""

st.components.v1.html(diagnostic_html, height=150)

# Quick reference
st.markdown("---")
st.markdown("### 📋 Quick Reference")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown(f"""
    **📱 Phone Setup:**
    - URL: `http://{local_ip}:{port}`
    - Browser: Chrome (recommended)
    - Permissions: Allow camera access
    """)

with col2:
    st.markdown(f"""
    **🔧 If Not Working:**
    - Enable Chrome flags (see above)
    - Check same WiFi network  
    - Restart browser after flag changes
    - Grant camera permissions
    """)

st.info("💡 **Tip:** If you see 'MOBILE STREAMING ACTIVE!' overlay on the video, everything is working correctly!")
