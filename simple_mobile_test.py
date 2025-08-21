"""
Simple Mobile Camera Streaming App - No Complex Dependencies
Tests the core mobile streaming functionality
"""

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import cv2
import numpy as np

st.set_page_config(
    page_title="📱 Mobile Camera Test",
    page_icon="📱",
    layout="wide"
)

st.title("📱💻 Mobile Camera Streaming Test")
st.markdown("---")

# Simple device detection
user_agent = st.context.headers.get("user-agent", "").lower()
is_mobile = any(device in user_agent for device in [
    'android', 'iphone', 'ipad', 'ipod', 'blackberry', 'iemobile', 'opera mini'
])

# Device type display
col1, col2 = st.columns([2, 1])

with col1:
    if is_mobile:
        st.success("📱 **Mobile Device Detected**")
        st.info("🚀 **Mode:** SENDONLY - Sending camera to laptop")
        st.markdown("- Uses back camera (environment)")
        st.markdown("- Optimized for mobile streaming")
        st.markdown("- Battery efficient mode")
    else:
        st.info("💻 **Desktop/Laptop Detected**")
        st.info("📺 **Mode:** RECVONLY - Receiving from mobile")
        st.markdown("- Receives mobile camera stream")
        st.markdown("- Runs AI processing")
        st.markdown("- **NO laptop camera used**")

with col2:
    # Manual override for testing
    st.subheader("🔧 Test Override")
    override = st.radio(
        "Force device type:",
        ["Auto-detect", "Mobile", "Desktop"],
        index=0
    )
    
    if override == "Mobile":
        is_mobile = True
    elif override == "Desktop":
        is_mobile = False

st.markdown("---")

# Simple video processing for desktop
def process_video(frame):
    """Simple video processing to show laptop is receiving and processing"""
    img = frame.to_ndarray(format="bgr24")
    
    # Add visual indicators
    height, width = img.shape[:2]
    
    # Green border to show processing
    cv2.rectangle(img, (5, 5), (width-5, height-5), (0, 255, 0), 3)
    
    # Processing text
    cv2.putText(img, "LAPTOP PROCESSING", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(img, "Receiving from Mobile", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Simple object detection simulation (just rectangles for testing)
    cv2.rectangle(img, (100, 100), (200, 200), (255, 0, 0), 2)
    cv2.putText(img, "Object Detection", (100, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    return img

# WebRTC Streaming Configuration
st.subheader("🎥 Camera Stream")

if is_mobile:
    st.markdown("### 📱 Mobile Camera (Sender)")
    st.info("This will send your mobile camera to the laptop")
    
    webrtc_ctx = webrtc_streamer(
        key="mobile_sender",
        mode=WebRtcMode.SENDONLY,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640, "min": 320, "max": 1280},
                "height": {"ideal": 480, "min": 240, "max": 720},
                "frameRate": {"ideal": 15, "min": 10, "max": 30},
                "facingMode": "environment"  # Back camera
            },
            "audio": False
        },
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]}
            ]
        },
        video_html_attrs={
            "style": {"width": "100%", "border": "4px solid #4CAF50"},
            "autoplay": True,
            "muted": True,
            "playsinline": True
        }
    )
    
else:
    st.markdown("### 💻 Laptop Receiver (with AI Processing)")
    st.info("This will receive and process the mobile camera stream")
    
    webrtc_ctx = webrtc_streamer(
        key="laptop_receiver", 
        mode=WebRtcMode.RECVONLY,
        video_frame_callback=process_video,
        media_stream_constraints={
            "video": False,  # NO laptop camera
            "audio": False
        },
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]}
            ]
        },
        video_html_attrs={
            "style": {"width": "100%", "border": "4px solid #FF5722"},
            "autoplay": True,
            "muted": True,
            "playsinline": True
        }
    )

# Connection Status
st.markdown("---")
st.subheader("📊 Connection Status")

status_col1, status_col2 = st.columns([1, 1])

with status_col1:
    if webrtc_ctx.state.playing:
        if is_mobile:
            st.success("✅ **Mobile camera streaming!**")
            st.info("📤 Sending video to laptop")
            st.info("🔋 Low processing mode (battery efficient)")
        else:
            st.success("✅ **Receiving mobile stream!**")
            st.info("📥 Processing incoming video")
            st.info("🤖 AI detection simulation running")
            st.success("📵 **Laptop camera OFF** (as required)")
            
    elif webrtc_ctx.state.signalling:
        st.warning("🔄 **Connecting...**")
        st.info("Establishing WebRTC connection")
        
    else:
        if is_mobile:
            st.error("❌ **Camera not started**")
            st.warning("🔑 Check camera permissions")
        else:
            st.error("❌ **No stream received**")
            st.warning("📱 Waiting for mobile device")

with status_col2:
    st.subheader("🔍 Technical Details")
    details = {
        "Device": "📱 Mobile" if is_mobile else "💻 Desktop",
        "WebRTC Mode": str(webrtc_ctx.mode),
        "State": str(webrtc_ctx.state),
        "Camera Used": "📱 Mobile back camera" if is_mobile else "📵 NO laptop camera"
    }
    
    for key, value in details.items():
        st.text(f"{key}: {value}")

# Testing Instructions
st.markdown("---")
st.subheader("📋 Testing Instructions")

test_col1, test_col2 = st.columns([1, 1])

with test_col1:
    st.markdown("### 🧪 How to Test")
    steps = [
        "1. **Open this URL on laptop** (should show Desktop mode)",
        "2. **Open same URL on mobile** (should show Mobile mode)", 
        "3. **Grant camera permission** on mobile",
        "4. **Verify mobile camera starts** (green border)",
        "5. **Check laptop receives stream** (red border with processing)",
        "6. **Confirm laptop camera stays OFF**"
    ]
    
    for step in steps:
        st.markdown(step)

with test_col2:
    st.markdown("### 🌐 Network URLs")
    st.code(f"""
Local:   http://localhost:8505
Network: http://192.168.1.8:8505
    """)
    
    st.markdown("### 🔒 HTTPS for Mobile")
    st.warning("""
    **For mobile camera access:**
    - HTTPS required on mobile browsers
    - Use ngrok for automatic HTTPS
    - Or configure Chrome flags for HTTP
    """)

# Error Troubleshooting
st.markdown("---")
with st.expander("🛠️ Troubleshooting Common Issues"):
    st.markdown("""
    ### 📱 Mobile Issues:
    - **Camera permission denied**: Check browser settings, grant camera access
    - **"MediaDevices undefined"**: Use HTTPS or Chrome flags for insecure origins
    - **Camera not starting**: Try different browser (Chrome recommended)
    
    ### 💻 Laptop Issues:
    - **No stream received**: Make sure mobile device is connected and streaming
    - **Laptop camera opens**: Check that WebRTC mode is RECVONLY (this should not happen)
    - **Processing not visible**: Green processing overlay should appear on received stream
    
    ### 🌐 Network Issues:
    - **Connection fails**: Both devices must be on same WiFi network
    - **Can't access URL**: Use correct IP address for your network
    - **Firewall blocking**: Check Windows Firewall settings
    
    ### 🔧 General Fixes:
    - **Refresh both pages** if connection gets stuck
    - **Try incognito/private mode** to avoid cached permissions
    - **Use Chrome or Firefox** for best WebRTC support
    """)

st.success("🎯 **Expected Result:** Mobile sends camera → Laptop receives and processes → Laptop camera stays OFF")
