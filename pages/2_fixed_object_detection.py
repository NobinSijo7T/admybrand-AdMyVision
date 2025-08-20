"""Simplified and Optimized Object Detection with WebRTC
Fixed version addressing video freezing and mobile connection issues.
"""

import logging
import queue
import time
from pathlib import Path
from typing import List, NamedTuple
import socket

import av
import cv2
import numpy as np
import qrcode
import streamlit as st
from PIL import Image
from io import BytesIO
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
    page_title="🎯 Object Detection",
    page_icon="🎯",
    layout="wide"
)

# Model paths
MODEL_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.caffemodel"
MODEL_LOCAL_PATH = ROOT / "./models/MobileNetSSD_deploy.caffemodel"
PROTOTXT_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.prototxt.txt"
PROTOTXT_LOCAL_PATH = ROOT / "./models/MobileNetSSD_deploy.prototxt.txt"

# Alternative paths in case of naming issues
PROTOTXT_ALT_PATH = ROOT / "./models/MobileNetSSD_deploy.prototxt"

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

class Detection(NamedTuple):
    class_id: int
    label: str
    score: float
    box: np.ndarray

# Initialize session state
if 'detections' not in st.session_state:
    st.session_state.detections = []
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0

@st.cache_resource
def generate_label_colors():
    return np.random.uniform(0, 255, size=(len(CLASSES), 3))

@st.cache_resource
def load_model():
    """Load the object detection model."""
    try:
        # Download models if needed
        if not MODEL_LOCAL_PATH.exists():
            st.info("📥 Downloading MobileNet-SSD model...")
            download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=23147564)
        
        # Check for prototxt file (try both naming conventions)
        prototxt_path = PROTOTXT_LOCAL_PATH
        if not prototxt_path.exists() and PROTOTXT_ALT_PATH.exists():
            prototxt_path = PROTOTXT_ALT_PATH
        elif not prototxt_path.exists():
            st.info("📥 Downloading model configuration...")
            download_file(PROTOTXT_URL, PROTOTXT_LOCAL_PATH, expected_size=29353)
        
        st.success(f"✅ Loading model from: {prototxt_path}")
        net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(MODEL_LOCAL_PATH))
        
        # Test the model with a dummy input
        dummy_blob = np.zeros((1, 3, 300, 300), dtype=np.float32)
        net.setInput(dummy_blob)
        test_output = net.forward()
        
        st.success(f"✅ Model loaded successfully! Output shape: {test_output.shape}")
        return net
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.error(f"Model path: {MODEL_LOCAL_PATH}")
        st.error(f"Prototxt path: {prototxt_path if 'prototxt_path' in locals() else PROTOTXT_LOCAL_PATH}")
        return None

def get_local_ip():
    """Get local IP for mobile connectivity."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except:
        return "localhost"

def generate_qr_code(url):
    """Generate QR code for mobile access."""
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(url)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    buf = BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()

# Load model and colors
COLORS = generate_label_colors()
net = load_model()

if net is None:
    st.error("❌ Failed to load model. Please check your internet connection.")
    st.stop()

# Header
st.title("🎯 Real-time Object Detection")
st.markdown("---")

# Initialize session state
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0
if "detection_results" not in st.session_state:
    st.session_state.detection_results = []
if "total_objects_detected" not in st.session_state:
    st.session_state.total_objects_detected = 0
if "detections" not in st.session_state:
    st.session_state.detections = []
if "last_detection_frame" not in st.session_state:
    st.session_state.last_detection_frame = 0

# Sidebar
st.sidebar.title("🔧 Settings")
score_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.05)  # Lower default threshold

mode = st.sidebar.selectbox(
    "📹 Camera Source",
    ["PC Camera", "Phone Camera (WebRTC)"]
)

# Result queue with limited size
result_queue = queue.Queue(maxsize=2)

# Create a class to handle detection with proper variable access
class ObjectDetector:
    def __init__(self, model_net, classes, colors, confidence_threshold):
        self.net = model_net
        self.classes = classes
        self.colors = colors
        self.confidence_threshold = confidence_threshold
        self.frame_count = 0
        self.last_detection_frame = 0
        self.total_objects = 0
    
    def estimate_distance(self, bbox_area, known_object_sizes):
        """Estimate distance based on bounding box area"""
        # Known approximate real-world widths in cm for common objects
        object_sizes = {
            'person': 50,      # shoulder width
            'car': 180,        # car width 
            'bicycle': 60,     # bike width
            'bottle': 8,       # bottle width
            'chair': 50,       # chair width
            'cat': 25,         # cat width
            'dog': 40,         # dog width
            'bus': 250,        # bus width
            'motorbike': 80,   # motorbike width
            'phone': 7         # phone width
        }
        
        # Camera parameters (approximate for webcam)
        focal_length = 800  # pixels
        
        # Calculate approximate distance
        if bbox_area > 0:
            # Use square root of area as approximate width in pixels
            object_width_pixels = np.sqrt(bbox_area)
            
            # Get known size or use default
            real_width_cm = object_sizes.get('person', 50)  # default to person size
            
            # Distance formula: distance = (real_width * focal_length) / pixel_width
            distance_cm = (real_width_cm * focal_length) / object_width_pixels
            distance_m = distance_cm / 100
            
            return min(distance_m, 10.0)  # Cap at 10 meters for realism
        
        return 0.0
    
    def detect_objects(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Process video frames for object detection."""
        try:
            image = frame.to_ndarray(format="bgr24")
            h, w = image.shape[:2]
            self.frame_count += 1
            
            # Create blob for detection
            blob = cv2.dnn.blobFromImage(
                image, 
                scalefactor=0.017, 
                size=(300, 300), 
                mean=(103.94, 116.78, 123.68),
                swapRB=False
            )
            
            # Run detection
            self.net.setInput(blob)
            detections = self.net.forward()
            
            # Process detections
            detection_list = []
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                class_id = int(detections[0, 0, i, 1])
                
                if confidence > self.confidence_threshold and 0 < class_id < len(self.classes):
                    # Get bounding box coordinates
                    x1 = int(detections[0, 0, i, 3] * w)
                    y1 = int(detections[0, 0, i, 4] * h)
                    x2 = int(detections[0, 0, i, 5] * w)
                    y2 = int(detections[0, 0, i, 6] * h)
                    
                    # Ensure coordinates are within image bounds
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    # Only process valid boxes
                    if x2 > x1 and y2 > y1:
                        # Calculate bounding box area for distance estimation
                        bbox_area = (x2 - x1) * (y2 - y1)
                        distance = self.estimate_distance(bbox_area, self.classes[class_id])
                        
                        detection_list.append(Detection(
                            class_id=class_id,
                            label=self.classes[class_id],
                            score=confidence,
                            box=np.array([x1, y1, x2, y2]),
                        ))
                        
                        # Draw bounding box
                        color = self.colors[class_id % len(self.colors)].tolist()
                        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label with distance
                        label = f"{self.classes[class_id]}: {confidence:.2f}"
                        distance_text = f"~{distance:.1f}m"
                        
                        # Calculate text sizes
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        distance_size = cv2.getTextSize(distance_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                        
                        # Background rectangle for main label
                        cv2.rectangle(image, (x1, y1 - label_size[1] - 25), 
                                     (x1 + max(label_size[0], distance_size[0]), y1), color, -1)
                        
                        # Main label text
                        cv2.putText(image, label, (x1, y1 - 15),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        
                        # Distance text
                        cv2.putText(image, distance_text, (x1, y1 - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Update total count (only count new detection instances)
            if len(detection_list) > 0:
                if self.frame_count - self.last_detection_frame > 30:  # New detection session
                    self.total_objects += len(detection_list)
                    self.last_detection_frame = self.frame_count
            
            # Update session state (non-blocking)
            try:
                # Clear old results
                while not result_queue.empty():
                    result_queue.get_nowait()
                    
                # Store both current detections and counts
                result_data = {
                    'detections': detection_list,
                    'current_count': len(detection_list),
                    'total_count': self.total_objects,
                    'frame_count': self.frame_count
                }
                result_queue.put_nowait(result_data)
            except:
                pass
            
            return av.VideoFrame.from_ndarray(image, format="bgr24")
            
        except Exception as e:
            print(f"Detection error: {e}")
            return frame
    
    def reset_total_count(self):
        """Reset the total object count"""
        self.total_objects = 0
        self.last_detection_frame = 0

# Create detector instance
detector = ObjectDetector(net, CLASSES, COLORS, score_threshold)

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    """Simple callback that uses the detector."""
    return detector.detect_objects(frame)

# Main content
if mode == "PC Camera":
    st.subheader("📹 PC Camera Detection")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        webrtc_ctx = webrtc_streamer(
            key="pc_camera",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={
                "video": {"width": 640, "height": 480, "frameRate": 15},
                "audio": False
            },
            async_processing=True,
        )
    
    with col2:
        st.subheader("📊 Status")
        
        if webrtc_ctx.state.playing:
            st.success("✅ Camera Active")
        else:
            st.error("❌ Camera Inactive")
        
        st.metric("Frames Processed", st.session_state.frame_count)
        
        # Model status
        if net is not None:
            st.success("✅ Model Loaded")
        else:
            st.error("❌ Model Failed")
        
        # Display current threshold
        st.info(f"🎯 Threshold: {score_threshold:.2f}")
        
        # Display latest detections
        st.subheader("🔍 Latest Detections")
        
        # Update detections from queue
        current_detection_count = 0
        total_detected = 0
        try:
            while not result_queue.empty():
                result_data = result_queue.get_nowait()
                if isinstance(result_data, dict):
                    st.session_state.detections = result_data.get('detections', [])
                    current_detection_count = result_data.get('current_count', 0)
                    total_detected = result_data.get('total_count', 0)
                    st.session_state.frame_count = result_data.get('frame_count', 0)
                else:
                    # Fallback for old format
                    st.session_state.detections = result_data
                    current_detection_count = len(result_data) if result_data else 0
        except:
            pass
        
        # Show detection statistics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Objects", current_detection_count)
        with col2:
            st.metric("🎯 Total Objects Found", total_detected)
        
        # Reset button for total count
        if st.button("🔄 Reset Total Count", help="Reset the total objects detected counter"):
            detector.reset_total_count()
            st.rerun()
        
        if st.session_state.detections:
            for det in st.session_state.detections[:5]:  # Show top 5
                confidence_color = "🟢" if det.score > 0.7 else "🟡" if det.score > 0.5 else "🔴"
                st.write(f"{confidence_color} **{det.label}**: {det.score:.1%}")
        else:
            st.info("👀 Point camera at objects like:\n- Person\n- Car\n- Bottle\n- Chair\n- Cat/Dog")
            
        # Debug information
        if st.checkbox("🔧 Debug Info"):
            st.write(f"Model loaded: {net is not None}")
            st.write(f"Threshold: {score_threshold}")
            st.write(f"Frame count: {st.session_state.frame_count}")
            st.write(f"Queue size: {result_queue.qsize()}")

elif mode == "Phone Camera (WebRTC)":
    st.subheader("📱 Phone Camera Detection")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.info("📱 **Connect Your Phone:**")
        
        # Generate QR code
        local_ip = get_local_ip()
        url = f"http://{local_ip}:8501"
        qr_image = generate_qr_code(url)
        
        st.image(qr_image, caption=f"Scan with phone: {url}", width=200)
        
        st.markdown("**📋 Instructions:**")
        st.markdown("""
        1. Scan QR code with phone camera
        2. Open the link in browser
        3. Allow camera permissions
        4. Select this same page
        5. Choose 'PC Camera' mode on phone
        """)
    
    with col2:
        st.info("🌐 **Connection Status:**")
        
        # Simplified WebRTC for phone
        webrtc_ctx = webrtc_streamer(
            key="phone_camera",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640, "max": 1280},
                    "height": {"ideal": 480, "max": 720},
                    "frameRate": {"ideal": 10, "max": 15}
                },
                "audio": False
            },
            async_processing=True,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            }
        )
        
        if webrtc_ctx.state.playing:
            st.success("✅ Phone Connected!")
        elif webrtc_ctx.state.signalling:
            st.warning("🔄 Connecting...")
        else:
            st.error("❌ Not Connected")
    
    # Detection results
    if webrtc_ctx.state.playing:
        st.subheader("🔍 Detection Results")
        
        # Update detections
        current_phone_detections = 0
        total_phone_detected = 0
        try:
            while not result_queue.empty():
                result_data = result_queue.get_nowait()
                if isinstance(result_data, dict):
                    st.session_state.detections = result_data.get('detections', [])
                    current_phone_detections = result_data.get('current_count', 0)
                    total_phone_detected = result_data.get('total_count', 0)
                else:
                    # Fallback for old format
                    st.session_state.detections = result_data
                    current_phone_detections = len(result_data) if result_data else 0
        except:
            pass
        
        # Show detection statistics for phone camera
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Objects", current_phone_detections)
        with col2:
            st.metric("🎯 Total Objects Found", total_phone_detected)
        
        # Reset button for total count
        if st.button("🔄 Reset Total Count", key="phone_reset", help="Reset the total objects detected counter"):
            detector.reset_total_count()
            st.rerun()
        
        if st.session_state.detections:
            detection_data = [{
                'Object': det.label.title(),
                'Confidence': f"{det.score:.1%}",
                'Position': f"({int(det.box[0])}, {int(det.box[1])})"
            } for det in st.session_state.detections]
            st.dataframe(detection_data, use_container_width=True)
        else:
            st.info("🔍 Point camera at objects to detect them")

# Footer
st.markdown("---")
st.markdown(f"**Streamlit-WebRTC**: {st_webrtc_version} | **aiortc**: {aiortc.__version__}")

# Auto-refresh every 2 seconds to keep UI responsive
time.sleep(0.1)
if st.session_state.frame_count % 20 == 0:  # Refresh every 20 frames
    st.rerun()
