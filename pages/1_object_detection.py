"""Enhanced Object Detection with WebRTC Real-time Streaming
This enhanced demo provides multiple input sources and WebRTC streaming capabilities
as per the PRD requirements:
- PC Camera streaming
- Phone Camera via WebRTC
- Video file upload
- Real-time performance metrics
- QR code generation for phone connectivity

Based on the original MobileNet SSD object detection demo from
https://github.com/robmarkcole/object-detection-app
"""

import asyncio
import base64
import json
import logging
import queue
import tempfile
import time
from io import BytesIO
from pathlib import Path
from typing import List, NamedTuple, Optional
import threading
import socket

import av
import cv2
import numpy as np
import qrcode
import requests
import streamlit as st
from PIL import Image
from streamlit_session_memo import st_session_memo
from streamlit_webrtc import (
    WebRtcMode,
    webrtc_streamer,
    __version__ as st_webrtc_version,
)
import aiortc

from sample_utils.download import download_file

HERE = Path(__file__).parent
ROOT = HERE.parent

logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="🎯 WebRTC Object Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Model URLs and paths
MODEL_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.caffemodel"  # noqa: E501
MODEL_LOCAL_PATH = ROOT / "./models/MobileNetSSD_deploy.caffemodel"
PROTOTXT_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.prototxt.txt"  # noqa: E501
PROTOTXT_LOCAL_PATH = ROOT / "./models/MobileNetSSD_deploy.prototxt.txt"

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

class Detection(NamedTuple):
    class_id: int
    label: str
    score: float
    box: np.ndarray

# Initialize session state for metrics
if 'detection_metrics' not in st.session_state:
    st.session_state.detection_metrics = {
        'total_detections': 0,
        'frames_processed': 0,
        'start_time': time.time(),
        'fps_history': [],
        'latency_history': [],
        'detection_history': []
    }

@st.cache_resource
def generate_label_colors():
    return np.random.uniform(0, 255, size=(len(CLASSES), 3))

COLORS = generate_label_colors()

@st.cache_resource
def download_models():
    """Download model files if they don't exist."""
    try:
        if not MODEL_LOCAL_PATH.exists():
            with st.spinner("Downloading MobileNet-SSD model..."):
                download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=23147564)
        
        if not PROTOTXT_LOCAL_PATH.exists():
            with st.spinner("Downloading model configuration..."):
                download_file(PROTOTXT_URL, PROTOTXT_LOCAL_PATH, expected_size=29353)
        
        return True
    except Exception as e:
        st.error(f"Error downloading models: {e}")
        return False

@st_session_memo
def get_model():
    """Load the object detection model."""
    if not download_models():
        return None
    return cv2.dnn.readNetFromCaffe(str(PROTOTXT_LOCAL_PATH), str(MODEL_LOCAL_PATH))

def get_local_ip():
    """Get the local IP address for QR code generation."""
    try:
        # Connect to a remote server to get local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except:
        return "localhost"

def generate_qr_code(url):
    """Generate QR code for the given URL."""
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(url)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Convert to bytes for Streamlit
    buf = BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()

def perform_detection(image, net, score_threshold):
    """Perform object detection on an image with error handling and optimization."""
    if net is None:
        return [], image
    
    try:
        start_time = time.time()
        
        # Optimize image size for faster processing
        h, w = image.shape[:2]
        if max(h, w) > 640:  # Resize if too large
            scale = 640 / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            image_resized = cv2.resize(image, (new_w, new_h))
        else:
            image_resized = image
            scale = 1.0
        
        # Run inference
        blob = cv2.dnn.blobFromImage(
            image=cv2.resize(image_resized, (300, 300)),
            scalefactor=0.007843,
            size=(300, 300),
            mean=(127.5, 127.5, 127.5),
        )
        net.setInput(blob)
        output = net.forward()
        
        # Convert the output array into a structured form
        output = output.squeeze()
        if len(output.shape) > 1 and output.shape[0] > 0:
            output = output[output[:, 2] >= score_threshold]
            detections = [
                Detection(
                    class_id=int(detection[1]),
                    label=CLASSES[int(detection[1])] if int(detection[1]) < len(CLASSES) else "unknown",
                    score=float(detection[2]),
                    box=(detection[3:7] * np.array([w, h, w, h])),
                )
                for detection in output
                if int(detection[1]) < len(CLASSES)
            ]
        else:
            detections = []
        
        # Render bounding boxes and captions
        result_image = image.copy()
        for detection in detections:
            try:
                caption = f"{detection.label}: {round(detection.score * 100, 2)}%"
                color = COLORS[detection.class_id] if detection.class_id < len(COLORS) else (0, 255, 0)
                xmin, ymin, xmax, ymax = detection.box.astype("int")
                
                # Ensure coordinates are within image bounds
                xmin, ymin = max(0, xmin), max(0, ymin)
                xmax, ymax = min(w-1, xmax), min(h-1, ymax)
                
                cv2.rectangle(result_image, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(
                    result_image,
                    caption,
                    (xmin, ymin - 15 if ymin - 15 > 15 else ymin + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
            except Exception as e:
                logger.warning(f"Error drawing detection: {e}")
                continue
        
        # Update metrics (limit frequency to avoid performance issues)
        if hasattr(st.session_state, 'last_metric_update'):
            if time.time() - st.session_state.last_metric_update > 0.1:  # Update every 100ms
                update_metrics(start_time, len(detections))
                st.session_state.last_metric_update = time.time()
        else:
            update_metrics(start_time, len(detections))
            st.session_state.last_metric_update = time.time()
        
        return detections, result_image
        
    except Exception as e:
        logger.error(f"Error in object detection: {e}")
        return [], image

def update_metrics(start_time, num_detections):
    """Update performance metrics."""
    try:
        processing_time = time.time() - start_time
        st.session_state.detection_metrics['frames_processed'] += 1
        st.session_state.detection_metrics['total_detections'] += num_detections
        st.session_state.detection_metrics['latency_history'].append(processing_time * 1000)  # ms
        st.session_state.detection_metrics['detection_history'].append(num_detections)
        
        # Keep only last 100 measurements for performance
        if len(st.session_state.detection_metrics['latency_history']) > 100:
            st.session_state.detection_metrics['latency_history'] = st.session_state.detection_metrics['latency_history'][-100:]
        if len(st.session_state.detection_metrics['detection_history']) > 100:
            st.session_state.detection_metrics['detection_history'] = st.session_state.detection_metrics['detection_history'][-100:]
    except Exception as e:
        logger.warning(f"Error updating metrics: {e}")

# Header
st.title("🎯 Enhanced WebRTC Object Detection System")
st.markdown("*Real-time object detection with multiple input sources and WebRTC streaming*")

# Auto-refresh for responsive UI
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True

# Add refresh mechanism
refresh_placeholder = st.empty()

st.markdown("---")

# Initialize model
net = get_model()
if net is None:
    st.error("⚠️ Failed to load the MobileNet-SSD model. Please check the model files.")
    st.stop()

# Sidebar Configuration
st.sidebar.title("🔧 Configuration")
score_threshold = st.sidebar.slider("Detection Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# Input source selection
input_source = st.sidebar.selectbox(
    "📹 Select Input Source",
    [
        "PC Camera (Live WebRTC)",
        "Phone Camera (Remote WebRTC)", 
        "Video File Upload",
        "FastAPI Server Integration"
    ]
)

# Performance monitoring toggle
show_metrics = st.sidebar.checkbox("📊 Show Performance Metrics", value=True)
show_detections = st.sidebar.checkbox("🏷️ Show Detection Results", value=True)

# Reset metrics button
if st.sidebar.button("🔄 Reset Metrics"):
    st.session_state.detection_metrics = {
        'total_detections': 0,
        'frames_processed': 0,
        'start_time': time.time(),
        'fps_history': [],
        'latency_history': [],
        'detection_history': []
    }
    st.sidebar.success("Metrics reset!")

# NOTE: The callback will be called in another thread,
#       so use a queue here for thread-safety to pass the data
#       from inside to outside the callback.
result_queue = queue.Queue(maxsize=2)  # Limited size to prevent memory issues

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    """Enhanced video frame callback with metrics tracking and error handling."""
    try:
        image = frame.to_ndarray(format="bgr24")
        
        # Perform detection with metrics
        detections, result_image = perform_detection(image, net, score_threshold)
        
        # Put results in queue (non-blocking)
        try:
            # Clear old results to prevent queue buildup
            while not result_queue.empty():
                try:
                    result_queue.get_nowait()
                except queue.Empty:
                    break
            # Add new result
            result_queue.put_nowait(detections)
        except queue.Full:
            pass  # Skip if queue is full
        
        return av.VideoFrame.from_ndarray(result_image, format="bgr24")
    
    except Exception as e:
        logger.error(f"Error in video frame callback: {e}")
        # Return original frame if processing fails
        return frame

# Main Content Area
if input_source == "PC Camera (Live WebRTC)":
    st.subheader("📹 PC Camera - Real-time Object Detection")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.info("🖥️ **PC Camera Mode**: Using your computer's webcam for real-time object detection")
        
        webrtc_ctx = webrtc_streamer(
            key="pc-camera-detection",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        # Detection results display
        if show_detections and webrtc_ctx.state.playing:
            st.subheader("🔍 Live Detection Results")
            labels_placeholder = st.empty()
            
            # Non-blocking result display using session state
            if 'latest_detections' not in st.session_state:
                st.session_state.latest_detections = []
            
            # Check for new results without blocking
            try:
                while not result_queue.empty():
                    result = result_queue.get_nowait()
                    st.session_state.latest_detections = result
            except queue.Empty:
                pass
            
            # Display latest results
            if st.session_state.latest_detections:
                detection_data = [{
                    'Object': det.label.title(),
                    'Confidence': f"{det.score:.1%}",
                    'Position': f"({int(det.box[0])}, {int(det.box[1])})",
                    'Size': f"{int(det.box[2]-det.box[0])}×{int(det.box[3]-det.box[1])}"
                } for det in st.session_state.latest_detections]
                labels_placeholder.dataframe(detection_data, use_container_width=True)
            else:
                labels_placeholder.info("🔍 No objects detected")
    
    with col2:
        if show_metrics:
            st.subheader("📊 Live Metrics")
            
            # Calculate current metrics
            metrics = st.session_state.detection_metrics
            current_time = time.time()
            elapsed_time = current_time - metrics['start_time']
            
            if elapsed_time > 0:
                avg_fps = metrics['frames_processed'] / elapsed_time
                st.metric("Average FPS", f"{avg_fps:.1f}")
            else:
                st.metric("Average FPS", "0.0")
            
            st.metric("Total Detections", metrics['total_detections'])
            st.metric("Frames Processed", metrics['frames_processed'])
            
            if metrics['latency_history']:
                avg_latency = np.mean(metrics['latency_history'][-10:])  # Last 10 frames
                st.metric("Avg Latency (ms)", f"{avg_latency:.1f}")
            
            # Real-time charts
            if len(metrics['detection_history']) > 1:
                st.subheader("📈 Detection Trends")
                st.line_chart(metrics['detection_history'][-50:])  # Last 50 frames

elif input_source == "Phone Camera (Remote WebRTC)":
    st.subheader("📱 Phone Camera - Remote WebRTC Streaming")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.info("📱 **Phone Camera Mode**: Stream from your phone's camera")
        
        # Generate QR code for this page
        local_ip = get_local_ip()
        current_url = f"http://{local_ip}:8501/1_object_detection"
        
        st.markdown("**📋 Setup Instructions:**")
        st.markdown("""
        1. **Scan the QR code** with your phone's camera app
        2. **Open the link** in your phone's browser
        3. **Allow camera permissions** when prompted
        4. **Point your camera** at objects for detection
        5. **View results** on this PC screen
        """)
        
        # Display QR code
        qr_image = generate_qr_code(current_url)
        st.image(qr_image, caption=f"📱 Scan to open: {current_url}", width=250)
        
        # Alternative access methods
        st.markdown("**🔗 Alternative Access:**")
        st.code(current_url, language="text")
        
        if st.button("📋 Copy URL to Clipboard"):
            st.write(f"URL: {current_url}")
    
    with col2:
        st.info("🌐 **WebRTC Connection Status**")
        
        # Configure WebRTC for better mobile compatibility
        webrtc_ctx = webrtc_streamer(
            key="phone-camera-detection",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={
                "video": {
                    "width": {"min": 320, "ideal": 640, "max": 1280},
                    "height": {"min": 240, "ideal": 480, "max": 720},
                    "frameRate": {"ideal": 15, "max": 30}
                },
                "audio": False
            },
            async_processing=True,
            rtc_configuration={
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {"urls": ["stun:stun1.l.google.com:19302"]},
                ]
            }
        )
        
        if webrtc_ctx.state.playing:
            st.success("✅ Phone camera connected and streaming!")
        elif webrtc_ctx.state.signalling:
            st.warning("🔄 Establishing connection...")
        else:
            st.error("❌ No phone connection detected")
            st.info("💡 **Troubleshooting:**\n- Ensure both devices are on same network\n- Try refreshing the page\n- Check camera permissions")
    
    # Full-width detection results (non-blocking)
    if show_detections:
        st.subheader("📱 Phone Camera Detection Results")
        
        # Initialize session state for phone detections
        if 'phone_detections' not in st.session_state:
            st.session_state.phone_detections = []
        
        # Update detections without blocking
        if webrtc_ctx.state.playing:
            try:
                while not result_queue.empty():
                    result = result_queue.get_nowait()
                    st.session_state.phone_detections = result
            except queue.Empty:
                pass
        
        # Display results
        labels_placeholder = st.empty()
        if st.session_state.phone_detections:
            detection_data = [{
                'Object': det.label.title(),
                'Confidence': f"{det.score:.1%}",
                'Bounding Box': f"[{int(det.box[0])}, {int(det.box[1])}, {int(det.box[2])}, {int(det.box[3])}]"
            } for det in st.session_state.phone_detections]
            labels_placeholder.dataframe(detection_data, use_container_width=True)
        else:
            labels_placeholder.info("🔍 Waiting for objects to be detected...")

elif input_source == "Video File Upload":
    st.subheader("📁 Video File Upload - Batch Analysis")
    
    uploaded_file = st.file_uploader(
        "📤 Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv', 'wmv'],
        help="Upload a video file for offline object detection analysis"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.success(f"✅ Uploaded: **{uploaded_file.name}**")
            
            # Video processing options
            process_every_n_frames = st.slider("Process every N frames (for faster analysis)", 1, 10, 1)
            max_frames = st.slider("Maximum frames to process (0 = all)", 0, 1000, 100)
            
            if st.button("🚀 Start Video Analysis"):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                
                # Process video
                cap = cv2.VideoCapture(tmp_path)
                
                # Get video properties
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = total_frames / fps if fps > 0 else 0
                
                st.info(f"📹 **Video Info**: {total_frames} frames, {fps} FPS, {duration:.1f}s duration")
                
                # Processing UI
                progress_bar = st.progress(0)
                status_text = st.empty()
                frame_placeholder = st.empty()
                
                # Process frames
                frame_count = 0
                processed_count = 0
                all_detections = []
                
                frames_to_process = min(total_frames, max_frames) if max_frames > 0 else total_frames
                
                while frame_count < frames_to_process:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process every N frames
                    if frame_count % process_every_n_frames == 0:
                        detections, result_frame = perform_detection(frame, net, score_threshold)
                        all_detections.extend(detections)
                        processed_count += 1
                        
                        # Display progress
                        progress = frame_count / frames_to_process
                        progress_bar.progress(progress)
                        status_text.text(f"Processing frame {frame_count}/{frames_to_process} (Processed: {processed_count})")
                        
                        # Show frame every 20 processed frames
                        if processed_count % 20 == 0:
                            frame_placeholder.image(
                                cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB),
                                caption=f"Frame {frame_count} - {len(detections)} objects detected",
                                use_container_width=True
                            )
                    
                    frame_count += 1
                
                cap.release()
                
                # Analysis results
                st.success(f"✅ Video analysis complete! Processed {processed_count} frames.")
                
                # Statistics
                object_counts = {}
                confidence_scores = []
                
                for det in all_detections:
                    object_counts[det.label] = object_counts.get(det.label, 0) + 1
                    confidence_scores.append(det.score)
        
        with col2:
            if 'all_detections' in locals():
                st.subheader("📊 Analysis Summary")
                
                col2a, col2b = st.columns(2)
                with col2a:
                    st.metric("Total Objects", len(all_detections))
                    st.metric("Unique Classes", len(object_counts))
                
                with col2b:
                    st.metric("Frames Analyzed", processed_count)
                    if confidence_scores:
                        st.metric("Avg Confidence", f"{np.mean(confidence_scores):.1%}")
                
                # Object distribution
                if object_counts:
                    st.subheader("🏷️ Object Distribution")
                    st.bar_chart(object_counts)
                    
                    # Top detections table
                    st.subheader("🔝 Top Detections")
                    sorted_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)
                    top_objects = [{
                        'Object': obj,
                        'Count': count,
                        'Percentage': f"{count/len(all_detections)*100:.1f}%"
                    } for obj, count in sorted_objects[:5]]
                    st.dataframe(top_objects, use_container_width=True)

elif input_source == "FastAPI Server Integration":
    st.subheader("🌐 FastAPI WebRTC Server Status")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.info("🚀 **FastAPI Server Integration**")
        
        # Check FastAPI server status
        try:
            response = requests.get("http://localhost:8000/qr", timeout=3)
            if response.status_code == 200:
                st.success("✅ FastAPI server is running!")
                
                qr_data = response.json()
                
                # Display server QR code
                if "qr_code" in qr_data:
                    qr_image_data = qr_data["qr_code"]
                    if qr_image_data.startswith("data:image/png;base64,"):
                        qr_base64 = qr_image_data.split(",")[1]
                        qr_image_bytes = base64.b64decode(qr_base64)
                        st.image(qr_image_bytes, caption=f"📱 FastAPI Server: {qr_data.get('url', 'http://localhost:8000')}", width=200)
                
                if st.button("🔗 Open FastAPI Interface"):
                    st.markdown(f"**[➡️ Open FastAPI Server]({qr_data.get('url', 'http://localhost:8000')})**")
                
            else:
                st.error(f"❌ FastAPI server error: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            st.error("❌ FastAPI server is not running")
            
            with st.expander("🔧 How to start FastAPI server"):
                st.code("""
# In a new terminal, run:
python -m uvicorn backend.server:app --host 0.0.0.0 --port 8000 --reload

# Or use the start script:
./start.bat
                """, language="bash")
    
    with col2:
        st.info("📱 **Usage Instructions**")
        st.markdown("""
        **🎯 FastAPI Features:**
        - Advanced WebRTC streaming
        - Real-time performance metrics
        - WebSocket communication
        - Low-latency object detection
        - Multi-device support
        
        **📋 Setup Steps:**
        1. Start the FastAPI server
        2. Scan QR code with phone
        3. Allow camera permissions
        4. Experience real-time detection
        """)

# Performance Metrics Dashboard
if show_metrics:
    st.markdown("---")
    st.subheader("📊 Performance Dashboard")
    
    metrics = st.session_state.detection_metrics
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Detections",
            metrics['total_detections'],
            delta=len(metrics['detection_history'][-1:]) if metrics['detection_history'] else 0
        )
    
    with col2:
        st.metric("Frames Processed", metrics['frames_processed'])
    
    with col3:
        if metrics['frames_processed'] > 0:
            elapsed = time.time() - metrics['start_time']
            fps = metrics['frames_processed'] / elapsed if elapsed > 0 else 0
            st.metric("Average FPS", f"{fps:.1f}")
    
    with col4:
        if metrics['latency_history']:
            avg_latency = np.mean(metrics['latency_history'][-10:])
            st.metric("Processing Latency", f"{avg_latency:.1f}ms")
    
    # Performance charts
    if len(metrics['detection_history']) > 5:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔍 Detections per Frame")
            st.line_chart(metrics['detection_history'][-100:])
        
        with col2:
            st.subheader("⚡ Processing Latency (ms)")
            st.line_chart(metrics['latency_history'][-100:])

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"**🚀 Streamlit**: `{st.__version__}`")

with col2:
    st.markdown(f"**📺 Streamlit-WebRTC**: `{st_webrtc_version}`")

with col3:
    st.markdown(f"**🔗 aiortc**: `{aiortc.__version__}`")

st.markdown("""
---
**🎯 Enhanced WebRTC Object Detection System**  
*This system provides multiple input sources and real-time WebRTC streaming capabilities as per PRD requirements.*  
*Original MobileNet-SSD model from [robmarkcole/object-detection-app](https://github.com/robmarkcole/object-detection-app). Enhanced with WebRTC streaming and performance metrics.*
""")

# Auto-refresh to keep UI responsive
if st.session_state.detection_metrics['frames_processed'] % 30 == 0:
    time.sleep(0.1)  # Small delay to prevent excessive refreshing
