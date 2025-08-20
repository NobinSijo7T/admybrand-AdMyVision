"""
AdMyVision - Simple Object Detection App
Minimal version for cloud deployment
"""

import streamlit as st

st.set_page_config(
    page_title="AdMyVision - Object Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    import cv2
    import numpy as np
    from streamlit_webrtc import webrtc_streamer, WebRtcMode
    WEBRTC_AVAILABLE = True
except ImportError as e:
    st.error(f"Required packages not available: {e}")
    WEBRTC_AVAILABLE = False

if WEBRTC_AVAILABLE:
    # Load the original detection app
    with open("pages/2_fixed_object_detection.py", "r", encoding="utf-8") as f:
        exec(f.read())
else:
    st.error("❌ Required packages not available for WebRTC functionality")
    st.info("Please ensure opencv-python-headless and streamlit-webrtc are installed")
    
    st.markdown("""
    ## 🎯 AdMyVision - Object Detection App
    
    This app provides real-time object detection with voice announcements.
    
    ### Features:
    - 📷 PC Camera Detection
    - 📱 Mobile Camera Support 
    - 🔊 Voice Announcements
    - 📊 Live Status Display
    
    ### Requirements:
    - opencv-python-headless
    - streamlit-webrtc
    - numpy
    - Pillow
    """)
