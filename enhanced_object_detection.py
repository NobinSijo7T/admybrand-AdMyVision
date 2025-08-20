"""
Enhanced Object Detection App with Multiple Input Sources
- PC Camera
- Phone Camera (via WebRTC)
- Video File Upload

Based on the existing pages/1_object_detection.py with additional features.
"""

import logging
import queue
import tempfile
from pathlib import Path
from typing import List, NamedTuple, Optional
import time
import requests
import base64
from io import BytesIO

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_session_memo import st_session_memo
from streamlit_webrtc import (
    WebRtcMode,
    webrtc_streamer,
    __version__ as st_webrtc_version,
)
import aiortc
import qrcode
from PIL import Image

from sample_utils.download import download_file

HERE = Path(__file__).parent
ROOT = HERE.parent

logger = logging.getLogger(__name__)

# Model URLs and paths
MODEL_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.caffemodel"
MODEL_LOCAL_PATH = ROOT / "./models/MobileNetSSD_deploy.caffemodel"
PROTOTXT_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.prototxt.txt"
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

@st.cache_resource
def generate_label_colors():
    return np.random.uniform(0, 255, size=(len(CLASSES), 3))

COLORS = generate_label_colors()

# Download models if needed
@st.cache_resource
def download_models():
    try:
        download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=23147564)
        download_file(PROTOTXT_URL, PROTOTXT_LOCAL_PATH, expected_size=29353)
        return True
    except Exception as e:
        st.error(f"Error downloading models: {e}")
        return False

@st_session_memo
def get_model():
    if not MODEL_LOCAL_PATH.exists() or not PROTOTXT_LOCAL_PATH.exists():
        if not download_models():
            return None
    return cv2.dnn.readNetFromCaffe(str(PROTOTXT_LOCAL_PATH), str(MODEL_LOCAL_PATH))

def perform_detection(image, net, score_threshold):
    """Perform object detection on an image."""
    if net is None:
        return [], image
    
    # Run inference
    blob = cv2.dnn.blobFromImage(
        image=cv2.resize(image, (300, 300)),
        scalefactor=0.007843,
        size=(300, 300),
        mean=(127.5, 127.5, 127.5),
    )
    net.setInput(blob)
    output = net.forward()

    h, w = image.shape[:2]

    # Convert the output array into a structured form.
    output = output.squeeze()
    output = output[output[:, 2] >= score_threshold]
    detections = [
        Detection(
            class_id=int(detection[1]),
            label=CLASSES[int(detection[1])],
            score=float(detection[2]),
            box=(detection[3:7] * np.array([w, h, w, h])),
        )
        for detection in output
    ]

    # Render bounding boxes and captions
    result_image = image.copy()
    for detection in detections:
        caption = f"{detection.label}: {round(detection.score * 100, 2)}%"
        color = COLORS[detection.class_id]
        xmin, ymin, xmax, ymax = detection.box.astype("int")

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

    return detections, result_image

def generate_qr_code(url):
    """Generate QR code for the given URL."""
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(url)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    return img

# Streamlit App
st.set_page_config(
    page_title="Enhanced Object Detection",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Enhanced Object Detection System")
st.markdown("---")

# Initialize model
net = get_model()
if net is None:
    st.error("Failed to load the detection model. Please check the model files.")
    st.stop()

# Sidebar controls
st.sidebar.title("🔧 Controls")
score_threshold = st.sidebar.slider("Detection Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
input_source = st.sidebar.selectbox(
    "Choose Input Source",
    ["PC Camera (Live)", "Phone Camera (WebRTC)", "Video File Upload", "FastAPI WebRTC Server"]
)

# Display detection statistics
if 'detection_stats' not in st.session_state:
    st.session_state.detection_stats = {
        'total_detections': 0,
        'frames_processed': 0,
        'start_time': time.time()
    }

# Common result queue for thread-safe detection results
result_queue: "queue.Queue[List[Detection]]" = queue.Queue()

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    """Video frame callback for WebRTC streaming."""
    image = frame.to_ndarray(format="bgr24")
    
    # Perform detection
    detections, result_image = perform_detection(image, net, score_threshold)
    
    # Update statistics
    st.session_state.detection_stats['frames_processed'] += 1
    st.session_state.detection_stats['total_detections'] += len(detections)
    
    # Put results in queue for display
    result_queue.put(detections)
    
    return av.VideoFrame.from_ndarray(result_image, format="bgr24")

# Main content area
if input_source == "PC Camera (Live)":
    st.subheader("📹 PC Camera - Real-time Object Detection")
    
    webrtc_ctx = webrtc_streamer(
        key="pc-camera-detection",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    # Display detection results
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.checkbox("Show Detection Results", value=True):
            if webrtc_ctx.state.playing:
                labels_placeholder = st.empty()
                while True:
                    try:
                        result = result_queue.get(timeout=1.0)
                        if result:
                            df_data = [{
                                'Object': det.label,
                                'Confidence': f"{det.score:.2%}",
                                'Position': f"({int(det.box[0])}, {int(det.box[1])})"
                            } for det in result]
                            labels_placeholder.dataframe(df_data, use_container_width=True)
                        else:
                            labels_placeholder.info("No objects detected")
                    except queue.Empty:
                        continue
                    except:
                        break
    
    with col2:
        st.metric("Total Detections", st.session_state.detection_stats['total_detections'])
        st.metric("Frames Processed", st.session_state.detection_stats['frames_processed'])
        
        if st.session_state.detection_stats['frames_processed'] > 0:
            elapsed_time = time.time() - st.session_state.detection_stats['start_time']
            fps = st.session_state.detection_stats['frames_processed'] / elapsed_time
            st.metric("Average FPS", f"{fps:.1f}")

elif input_source == "Phone Camera (WebRTC)":
    st.subheader("📱 Phone Camera - WebRTC Streaming")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.info("📱 **Instructions:**\n"
                "1. Scan the QR code with your phone\n"
                "2. Allow camera access in your browser\n"
                "3. Point camera at objects for detection")
        
        # Generate QR code for this page
        current_url = "http://localhost:8501"  # Streamlit default port
        qr_img = generate_qr_code(current_url)
        
        # Convert PIL image to display in Streamlit
        buf = BytesIO()
        qr_img.save(buf, format='PNG')
        st.image(buf.getvalue(), caption=f"Scan to open: {current_url}", width=200)
    
    with col2:
        # WebRTC streamer for phone connection
        webrtc_ctx = webrtc_streamer(
            key="phone-camera-detection",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    
    # Detection results display
    if st.checkbox("Show Live Detection Results", value=True):
        if webrtc_ctx.state.playing:
            st.success("📱 Phone camera connected and detecting objects...")
            labels_placeholder = st.empty()
            while True:
                try:
                    result = result_queue.get(timeout=1.0)
                    if result:
                        df_data = [{
                            'Object': det.label,
                            'Confidence': f"{det.score:.2%}",
                            'Bounding Box': f"({int(det.box[0])}, {int(det.box[1])}, {int(det.box[2])}, {int(det.box[3])})"
                        } for det in result]
                        labels_placeholder.dataframe(df_data, use_container_width=True)
                except queue.Empty:
                    continue
                except:
                    break

elif input_source == "Video File Upload":
    st.subheader("📁 Video File Upload - Batch Detection")
    
    uploaded_file = st.file_uploader(
        "Choose a video file", 
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video file for object detection analysis"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        st.success(f"Uploaded: {uploaded_file.name}")
        
        if st.button("🔍 Start Detection Analysis"):
            # Process video file
            cap = cv2.VideoCapture(tmp_path)
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            
            st.info(f"Video info: {total_frames} frames, {fps} FPS, {duration:.1f}s duration")
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process frames
            frame_count = 0
            all_detections = []
            
            # Create columns for results
            col1, col2 = st.columns([2, 1])
            
            with col1:
                frame_placeholder = st.empty()
            
            with col2:
                detection_placeholder = st.empty()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Perform detection
                detections, result_frame = perform_detection(frame, net, score_threshold)
                all_detections.extend(detections)
                
                # Update progress
                frame_count += 1
                progress = frame_count / total_frames
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {frame_count}/{total_frames}")
                
                # Display current frame and detections (every 10th frame to avoid overload)
                if frame_count % 10 == 0:
                    frame_placeholder.image(
                        cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB),
                        caption=f"Frame {frame_count}",
                        use_container_width=True
                    )
                    
                    if detections:
                        det_data = [{
                            'Object': det.label,
                            'Confidence': f"{det.score:.2%}"
                        } for det in detections]
                        detection_placeholder.dataframe(det_data)
            
            cap.release()
            
            # Final results
            st.success("✅ Video processing complete!")
            
            # Statistics
            object_counts = {}
            for det in all_detections:
                object_counts[det.label] = object_counts.get(det.label, 0) + 1
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Detections", len(all_detections))
            with col2:
                st.metric("Unique Objects", len(object_counts))
            with col3:
                st.metric("Frames Processed", frame_count)
            
            # Object distribution chart
            if object_counts:
                st.subheader("📊 Detection Summary")
                st.bar_chart(object_counts)

elif input_source == "FastAPI WebRTC Server":
    st.subheader("🌐 FastAPI WebRTC Server Connection")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.info("🚀 **FastAPI Server Status**")
        
        # Check if FastAPI server is running
        try:
            response = requests.get("http://localhost:8000/qr", timeout=5)
            if response.status_code == 200:
                qr_data = response.json()
                st.success("✅ FastAPI server is running!")
                
                # Display QR code from server
                qr_image_data = qr_data["qr_code"]
                if qr_image_data.startswith("data:image/png;base64,"):
                    qr_base64 = qr_image_data.split(",")[1]
                    qr_image_bytes = base64.b64decode(qr_base64)
                    st.image(qr_image_bytes, caption=f"Scan to connect: {qr_data['url']}", width=200)
                    
                st.code(qr_data["url"], language="text")
                
                if st.button("🔗 Open FastAPI Interface"):
                    st.markdown(f"[Open FastAPI Server]({qr_data['url']})")
                
            else:
                st.error("❌ FastAPI server responded with error")
                
        except requests.exceptions.RequestException:
            st.error("❌ FastAPI server is not running")
            st.info("To start the FastAPI server, run in terminal:\n```\npython -m uvicorn backend.server:app --host 0.0.0.0 --port 8000 --reload\n```")
    
    with col2:
        st.info("📱 **Usage Instructions:**")
        st.markdown("""
        1. Ensure FastAPI server is running
        2. Scan QR code with your phone
        3. Allow camera permissions
        4. Experience real-time WebRTC object detection
        
        **Features:**
        - Real-time WebRTC streaming
        - Low-latency object detection
        - Performance metrics
        - Multi-device support
        """)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"**Streamlit**: {st.__version__}")

with col2:
    st.markdown(f"**Streamlit-WebRTC**: {st_webrtc_version}")

with col3:
    st.markdown(f"**aiortc**: {aiortc.__version__}")

st.markdown(
    "*This enhanced demo extends the original MobileNet-SSD object detection with multiple input sources. "
    "Original model and code from [robmarkcole/object-detection-app](https://github.com/robmarkcole/object-detection-app).*"
)
