"""
AdMyVision - Real-time Object Detection App
Main entry point for Streamlit deployment
"""

import streamlit as st

st.set_page_config(
    page_title="AdMyVision - Object Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main app
st.title("🎯 AdMyVision - Real-time Object Detection")
st.markdown("---")

# Navigation
st.sidebar.title("🚀 Navigation")
page = st.sidebar.selectbox(
    "Choose Detection App:",
    [
        "🏠 Home",
        "🎯 Object Detection (Fixed)",
        "🎮 Object Detection (Original)",
        "🎙️ Audio Filter",
        "📹 Video Chat"
    ]
)

if page == "🏠 Home":
    st.markdown("""
    ## Welcome to AdMyVision! 👋
    
    ### 🎯 **Real-time Object Detection with Voice Announcements**
    
    #### ✨ **Features:**
    - 📷 **PC Camera Detection**: Real-time object detection using your computer's camera
    - 📱 **Mobile Camera Support**: Front/back camera switching for mobile devices  
    - 🔊 **Voice Announcements**: Audio feedback with distance estimation
    - 📊 **Live Status Display**: Real-time detection metrics and object tracking
    - 🎯 **YOLO Detection**: Advanced object detection with confidence scoring
    
    #### 🚀 **How to Get Started:**
    1. Select **"🎯 Object Detection (Fixed)"** from the sidebar
    2. Choose your camera mode (PC or Mobile)
    3. Enable voice announcements if desired
    4. Click "Start" to begin detection
    
    #### 📱 **Mobile Users:**
    - Use "Phone Camera" mode for best mobile experience
    - Switch between front and back cameras
    - Voice announcements work through your browser
    
    #### 🎮 **Other Apps:**
    - **Audio Filter**: Real-time audio processing
    - **Video Chat**: WebRTC video communication
    
    ---
    Built with ❤️ using Streamlit and OpenCV
    """)

elif page == "🎯 Object Detection (Fixed)":
    # Import and run the fixed object detection app
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    
    # Execute the fixed object detection page
    with open("pages/2_fixed_object_detection.py", "r", encoding="utf-8") as f:
        exec(f.read())

elif page == "🎮 Object Detection (Original)":
    st.markdown("### 🎮 Original Object Detection")
    st.info("This would load the original object detection page")
    
elif page == "🎙️ Audio Filter":
    st.markdown("### 🎙️ Audio Filter")
    st.info("This would load the audio filter page")
    
elif page == "📹 Video Chat":
    st.markdown("### 📹 Video Chat")
    st.info("This would load the video chat page")
