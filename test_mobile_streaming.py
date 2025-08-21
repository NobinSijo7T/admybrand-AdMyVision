"""
Quick Test Script for Mobile Camera Streaming
Tests the conditional WebRTC configuration without full app complexity
"""

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import cv2
import numpy as np

st.set_page_config(page_title="Mobile Streaming Test", page_icon="📱")

st.title("📱💻 Mobile Streaming Test")
st.markdown("---")

# Device detection script (same as main app)
device_detection_script = """
<script>
function detectDevice() {
    const userAgent = navigator.userAgent.toLowerCase();
    const isMobile = /android|webos|iphone|ipad|ipod|blackberry|iemobile|opera mini/i.test(userAgent);
    const isTablet = /ipad|android|tablet/i.test(userAgent) && !/mobile/i.test(userAgent);
    const hasTouchScreen = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
    
    // Additional mobile indicators
    const screenWidth = screen.width;
    const screenHeight = screen.height;
    const smallScreen = Math.min(screenWidth, screenHeight) < 768;
    
    const deviceInfo = {
        isMobile: isMobile || (hasTouchScreen && smallScreen),
        isTablet: isTablet,
        userAgent: userAgent,
        screenWidth: screenWidth,
        screenHeight: screenHeight,
        touchSupport: hasTouchScreen
    };
    
    // Send to Streamlit
    window.parent.postMessage({
        type: 'deviceInfo',
        data: deviceInfo
    }, '*');
    
    return deviceInfo;
}

// Run detection
const device = detectDevice();
console.log('Device Detection:', device);

// Display results on page
document.body.innerHTML += '<div style="background: #f0f0f0; padding: 10px; margin: 10px; border-radius: 5px;"><h3>Device Detection Results:</h3><pre>' + JSON.stringify(device, null, 2) + '</pre></div>';
</script>
"""

# Embed the JavaScript
st.components.v1.html(device_detection_script, height=200)

# Fallback device detection using Streamlit's user agent
user_agent = st.context.headers.get("user-agent", "").lower()
is_mobile_fallback = any(mobile in user_agent for mobile in [
    'android', 'iphone', 'ipad', 'ipod', 'blackberry', 'iemobile', 'opera mini'
])

# Session state for device type
if 'device_type' not in st.session_state:
    st.session_state.device_type = 'mobile' if is_mobile_fallback else 'desktop'

# Manual device selection for testing
st.subheader("🔧 Device Type Selection")
col1, col2 = st.columns(2)

with col1:
    if st.button("📱 Set as Mobile Device"):
        st.session_state.device_type = 'mobile'
        st.rerun()

with col2:
    if st.button("💻 Set as Desktop Device"):
        st.session_state.device_type = 'desktop'
        st.rerun()

# Display current device type
device_type = st.session_state.device_type
if device_type == 'mobile':
    st.success("📱 **Current Mode: Mobile Device (Sender)**")
    st.info("Will send camera stream to laptop")
else:
    st.info("💻 **Current Mode: Desktop Device (Receiver)**")
    st.info("Will receive camera stream from mobile")

st.markdown("---")

# Simple video processing function
def simple_video_callback(frame):
    """Simple video processing - just add a border"""
    img = frame.to_ndarray(format="bgr24")
    
    # Add a colored border to show processing is working
    color = (0, 255, 0) if device_type == 'mobile' else (0, 0, 255)
    cv2.rectangle(img, (10, 10), (img.shape[1]-10, img.shape[0]-10), color, 3)
    
    # Add text overlay
    text = f"Processing on {device_type.upper()}"
    cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return img

# WebRTC Configuration based on device type
st.subheader("🎥 WebRTC Stream Test")

if device_type == 'mobile':
    st.markdown("**📱 Mobile Mode: SENDONLY**")
    st.markdown("- Sends your camera to laptop")
    st.markdown("- Uses back camera (environment)")
    st.markdown("- Minimal processing to save battery")
    
    webrtc_ctx = webrtc_streamer(
        key="mobile_sender_test",
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
    st.markdown("**💻 Desktop Mode: RECVONLY**")
    st.markdown("- Receives camera from mobile")
    st.markdown("- Processes incoming video")
    st.markdown("- Does NOT use laptop camera")
    
    webrtc_ctx = webrtc_streamer(
        key="desktop_receiver_test",
        mode=WebRtcMode.RECVONLY,
        video_frame_callback=simple_video_callback,
        media_stream_constraints={
            "video": False,  # No laptop camera
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

# Connection status
st.subheader("📊 Connection Status")

if webrtc_ctx.state.playing:
    st.success("✅ **Streaming Active!**")
    if device_type == 'mobile':
        st.success("📤 Sending camera to laptop")
    else:
        st.success("📥 Receiving camera from mobile")
        st.success("🔄 Processing video frames")
        
elif webrtc_ctx.state.signalling:
    st.warning("🔄 **Establishing Connection...**")
    st.info("Please wait for handshake to complete")
    
else:
    st.error("❌ **No Connection**")
    if device_type == 'mobile':
        st.error("📱 Mobile not sending camera")
        st.info("💡 Check camera permissions")
    else:
        st.error("💻 Laptop not receiving stream")
        st.info("💡 Make sure mobile device is connected")

# Debug information
with st.expander("🔍 Debug Information"):
    st.json({
        "Device Type": device_type,
        "WebRTC Mode": webrtc_ctx.mode.value if hasattr(webrtc_ctx.mode, 'value') else str(webrtc_ctx.mode),
        "Connection State": str(webrtc_ctx.state),
        "User Agent": user_agent[:100] + "..." if len(user_agent) > 100 else user_agent,
        "Mobile Detected": is_mobile_fallback
    })

# Instructions
st.markdown("---")
st.subheader("📋 Test Instructions")

if device_type == 'mobile':
    st.markdown("""
    **📱 Mobile Device Instructions:**
    1. Grant camera permissions when prompted
    2. Camera should start streaming
    3. Open laptop at same URL to receive stream
    4. Look for green border indicating mobile mode
    """)
else:
    st.markdown("""
    **💻 Laptop Instructions:**
    1. This device will receive mobile camera
    2. No laptop camera should open
    3. Connect mobile device to start receiving
    4. Look for red border indicating desktop mode
    """)

st.info("💡 **Testing Tip:** Open this page on both mobile and laptop with different device modes to test the full streaming pipeline.")

# Network information
st.markdown("---")
st.subheader("🌐 Network Information")

col1, col2 = st.columns(2)

with col1:
    st.info(f"""
    **Current URL:**
    {st.context.host}
    
    **For mobile access:**
    Use ngrok or same WiFi network
    """)

with col2:
    st.warning("""
    **HTTPS Required:**
    Mobile browsers need HTTPS for camera access
    Use ngrok for automatic HTTPS
    """)
