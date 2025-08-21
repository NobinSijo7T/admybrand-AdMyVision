"""
Standalone Mobile Camera Streaming Test
Isolated from multi-page conflicts - tests core mobile streaming functionality
"""

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import cv2
import numpy as np

# Page config
st.set_page_config(
    page_title="Mobile Streaming Test",
    page_icon="📱",
    layout="wide"
)

st.title("📱💻 Mobile Camera Streaming Test")
st.markdown("---")

# Simple device detection using user agent
user_agent = st.context.headers.get("user-agent", "").lower()
is_mobile = any(mobile in user_agent for mobile in [
    'android', 'iphone', 'ipad', 'ipod', 'blackberry', 'iemobile', 'opera mini'
])

# Display device type
col1, col2 = st.columns([1, 1])

with col1:
    if is_mobile:
        st.success("📱 **Mobile Device Detected**")
        st.info("Mode: SENDONLY - Will send camera to laptop")
    else:
        st.info("💻 **Desktop Device Detected**")
        st.info("Mode: RECVONLY - Will receive from mobile")

with col2:
    # Manual override for testing
    device_override = st.selectbox(
        "Override device type for testing:",
        ["Auto-detect", "Force Mobile (SENDONLY)", "Force Desktop (RECVONLY)"]
    )
    
    if device_override == "Force Mobile (SENDONLY)":
        is_mobile = True
    elif device_override == "Force Desktop (RECVONLY)":
        is_mobile = False

st.markdown("---")

# Simple video callback for desktop processing
def video_callback(frame):
    """Simple video processing"""
    img = frame.to_ndarray(format="bgr24")
    
    # Add processing indicator
    height, width = img.shape[:2]
    cv2.rectangle(img, (10, 10), (width-10, height-10), (0, 255, 0), 3)
    cv2.putText(img, "LAPTOP PROCESSING", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return img

# WebRTC Configuration
st.subheader("🎥 WebRTC Stream")

if is_mobile:
    st.markdown("**📱 Mobile Mode: Sending camera to laptop**")
    
    webrtc_ctx = webrtc_streamer(
        key="mobile_camera_test",
        mode=WebRtcMode.SENDONLY,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640, "min": 320, "max": 1280},
                "height": {"ideal": 480, "min": 240, "max": 720},
                "frameRate": {"ideal": 15, "min": 10, "max": 25},
                "facingMode": "environment"  # Back camera
            },
            "audio": False
        },
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]}
            ]
        },
        video_html_attrs={
            "style": {"width": "100%", "border": "3px solid #4CAF50"},
            "autoplay": True,
            "muted": True,
            "playsinline": True
        }
    )
    
else:
    st.markdown("**💻 Desktop Mode: Receiving camera from mobile**")
    
    webrtc_ctx = webrtc_streamer(
        key="laptop_receiver_test",
        mode=WebRtcMode.RECVONLY,
        video_frame_callback=video_callback,
        media_stream_constraints={
            "video": False,  # Don't use laptop camera
            "audio": False
        },
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]}
            ]
        },
        video_html_attrs={
            "style": {"width": "100%", "border": "3px solid #ff6b6b"},
            "autoplay": True,
            "muted": True,
            "playsinline": True
        }
    )

# Status display
st.subheader("📊 Connection Status")

if webrtc_ctx.state.playing:
    if is_mobile:
        st.success("✅ **Mobile camera streaming successfully!**")
        st.info("📤 Sending video to laptop")
    else:
        st.success("✅ **Receiving mobile camera successfully!**")
        st.info("📥 Processing incoming video stream")
        st.success("🔄 Video processing active")
        
elif webrtc_ctx.state.signalling:
    st.warning("🔄 **Establishing connection...**")
    st.info("Please wait for handshake to complete")
    
else:
    if is_mobile:
        st.error("❌ **Mobile camera not started**")
        st.warning("🔑 **Check camera permissions**")
        st.info("💡 Grant camera access when prompted")
    else:
        st.error("❌ **No stream received**")
        st.warning("📱 **Waiting for mobile connection**")
        st.info("💡 Connect mobile device to start receiving")

# Debug information
st.markdown("---")
with st.expander("🔍 Debug Information"):
    debug_info = {
        "Device Type": "Mobile" if is_mobile else "Desktop",
        "WebRTC Mode": str(webrtc_ctx.mode),
        "Connection State": str(webrtc_ctx.state),
        "User Agent": user_agent[:100] + "..." if len(user_agent) > 100 else user_agent,
        "Override": device_override
    }
    
    for key, value in debug_info.items():
        st.text(f"{key}: {value}")

# Network information
st.markdown("---")
st.subheader("🌐 Network Access")

col1, col2 = st.columns([2, 1])

with col1:
    st.info(f"""
    **🌐 URLs for Testing:**
    - **Local:** http://localhost:8504
    - **Network:** http://192.168.1.8:8504 (same WiFi)
    - **External:** Use ngrok for public access
    
    **📱 Mobile Testing:**
    1. Open network URL on mobile device
    2. Grant camera permissions
    3. Should show "Mobile Mode: SENDONLY"
    4. Camera stream should start
    
    **💻 Laptop Testing:**
    1. Open local URL on laptop
    2. Should show "Desktop Mode: RECVONLY" 
    3. Will receive stream from mobile
    4. Processing overlay should appear
    """)

with col2:
    st.warning("""
    **🔒 HTTPS Requirements:**
    - Mobile browsers need HTTPS for camera
    - Use ngrok for automatic HTTPS
    - Or configure SSL certificates
    
    **🔧 If camera fails:**
    - Check browser permissions
    - Try different browser
    - Use Chrome flags for HTTP
    """)

# Instructions
st.markdown("---")
st.subheader("📋 Test Steps")

steps = [
    "1. 🖥️ **Start this app on laptop** (should show Desktop mode)",
    "2. 📱 **Open same URL on mobile** (should show Mobile mode)",
    "3. 🔑 **Grant camera permissions** on mobile device",
    "4. 📹 **Verify camera starts** on mobile (green border)",
    "5. 📺 **Check laptop receives stream** (red border with processing)",
    "6. ✅ **Confirm laptop camera stays OFF**"
]

for step in steps:
    st.markdown(step)

st.success("🎯 **Expected Result:** Mobile sends camera → Laptop receives and processes → No laptop camera used")
