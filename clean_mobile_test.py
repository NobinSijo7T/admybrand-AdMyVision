"""
Isolated Mobile Camera Streaming Test - No Multi-Page Conflicts
Clean test environment for mobile streaming functionality
"""

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import cv2
import numpy as np

st.set_page_config(
    page_title="📱 Mobile Stream Test",
    page_icon="📱",
    layout="wide"
)

st.title("📱💻 Mobile Camera Streaming Test")
st.markdown("**Testing mobile camera streaming to laptop without laptop camera activation**")
st.markdown("---")

# Device detection
user_agent = st.context.headers.get("user-agent", "").lower()
is_mobile = any(device in user_agent for device in [
    'android', 'iphone', 'ipad', 'ipod', 'blackberry', 'iemobile', 'opera mini'
])

# Display device info
st.subheader("🔍 Device Detection")
col1, col2 = st.columns([2, 1])

with col1:
    if is_mobile:
        st.success("📱 **Mobile Device Detected**")
        st.info("🚀 **WebRTC Mode:** SENDONLY (Send camera to laptop)")
    else:
        st.info("💻 **Desktop/Laptop Detected**")
        st.info("📺 **WebRTC Mode:** RECVONLY (Receive from mobile)")

with col2:
    # Override for testing
    override = st.selectbox("Test Override:", ["Auto", "Mobile", "Desktop"])
    if override == "Mobile":
        is_mobile = True
    elif override == "Desktop":
        is_mobile = False

# Show detected user agent
with st.expander("🔍 User Agent Info"):
    st.text(f"User Agent: {user_agent}")
    st.text(f"Detected as Mobile: {is_mobile}")

st.markdown("---")

# Video processing function for laptop
def laptop_video_processing(frame):
    """Process received video on laptop"""
    img = frame.to_ndarray(format="bgr24")
    height, width = img.shape[:2]
    
    # Add processing indicators
    cv2.rectangle(img, (10, 10), (width-10, height-10), (0, 255, 0), 3)
    cv2.putText(img, "LAPTOP PROCESSING", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, "Mobile Stream Received", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Simulate object detection
    cv2.rectangle(img, (100, 100), (250, 200), (255, 0, 0), 2)
    cv2.putText(img, "Detected Object", (100, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    return img

# WebRTC Configuration
st.subheader("🎥 WebRTC Stream Configuration")

if is_mobile:
    st.markdown("### 📱 Mobile Mode (SENDONLY)")
    st.info("Sending your mobile camera to laptop")
    
    webrtc_ctx = webrtc_streamer(
        key="mobile_camera_sender",
        mode=WebRtcMode.SENDONLY,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640, "min": 320, "max": 1280},
                "height": {"ideal": 480, "min": 240, "max": 720},
                "frameRate": {"ideal": 15, "min": 10, "max": 30},
                "facingMode": "environment"
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
            "style": {"width": "100%", "border": "4px solid #4CAF50"},
            "autoplay": True,
            "muted": True,
            "playsinline": True
        }
    )
    
else:
    st.markdown("### 💻 Desktop Mode (RECVONLY)")
    st.info("Receiving mobile camera stream and processing")
    st.warning("🚫 **Laptop camera will NOT be used**")
    
    webrtc_ctx = webrtc_streamer(
        key="laptop_camera_receiver",
        mode=WebRtcMode.RECVONLY,
        video_frame_callback=laptop_video_processing,
        media_stream_constraints={
            "video": False,  # NO laptop camera
            "audio": False
        },
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]}
            ]
        },
        video_html_attrs={
            "style": {"width": "100%", "border": "4px solid #FF5722"},
            "autoplay": True,
            "muted": True,
            "playsinline": True
        }
    )

# Status monitoring
st.markdown("---")
st.subheader("📊 Connection Status")

status_col1, status_col2 = st.columns([1, 1])

with status_col1:
    if webrtc_ctx.state.playing:
        if is_mobile:
            st.success("✅ **Mobile camera is streaming!**")
            st.info("📤 Video being sent to laptop")
        else:
            st.success("✅ **Receiving mobile camera stream!**")
            st.info("📥 Processing incoming video")
            st.success("🔒 **Laptop camera is OFF** ✓")
            
    elif webrtc_ctx.state.signalling:
        st.warning("🔄 **Establishing connection...**")
        st.info("WebRTC handshake in progress")
        
    else:
        if is_mobile:
            st.error("❌ **Mobile camera not started**")
            st.warning("🔑 Grant camera permissions when prompted")
        else:
            st.error("❌ **No mobile stream received**")
            st.warning("📱 Connect mobile device to begin")

with status_col2:
    st.markdown("**Technical Details:**")
    st.text(f"Device: {'📱 Mobile' if is_mobile else '💻 Desktop'}")
    st.text(f"WebRTC Mode: {webrtc_ctx.mode}")
    st.text(f"Connection: {webrtc_ctx.state}")
    st.text(f"Camera: {'📱 Mobile' if is_mobile else '🚫 None (laptop OFF)'}")

st.markdown("---")
st.success("🎯 **Goal:** Mobile streams camera → Laptop receives and processes → Laptop camera stays OFF")
