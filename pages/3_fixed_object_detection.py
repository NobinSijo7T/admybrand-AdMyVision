"""Simplified and Optimized Object Detection with WebRTC
Fixed version addressing video freezing and mobile connection issues.
"""

import logging
import queue
import time
import threading
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

# Try to import pyttsx3 with proper error handling
try:
    import pyttsx3
    import pythoncom  # For Windows COM initialization
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False

# Try to import Google Text-to-Speech as fallback
try:
    from gtts import gTTS
    import pygame
    import tempfile
    import os
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

from sample_utils.download import download_file

HERE = Path(__file__).parent
ROOT = HERE.parent

logger = logging.getLogger(__name__)

# Voice Manager Class
class VoiceManager:
    def __init__(self):
        self.engine = None
        self.voice_enabled = False
        self.last_announcement = {}
        self.announcement_cooldown = 3  # seconds between same object announcements
        self.use_gtts = False
        self.voice_lock = threading.Lock()  # Lock to prevent overlapping TTS calls
        self.last_announcement_time = 0  # Global cooldown timer
        self.global_cooldown = 0.8  # 0.8 seconds between any announcements (reduced from 1.5)
        self.is_speaking = False
        self.speaking_start_time = 0  # Track when speaking started
        self.init_voice_engine()
    
    def init_voice_engine(self):
        """Initialize the text-to-speech engine with fallback options"""
        # Try pyttsx3 first
        if VOICE_AVAILABLE:
            try:
                # Initialize COM for Windows
                if hasattr(pythoncom, 'CoInitialize'):
                    pythoncom.CoInitialize()
                
                # Try different driver options for better Windows compatibility
                drivers = ['sapi5', 'nsss', 'espeak']
                for driver in drivers:
                    try:
                        self.engine = pyttsx3.init(driver)
                        if self.engine:
                            print(f"Voice engine initialized successfully with {driver} driver")
                            break
                    except:
                        continue
                
                if not self.engine:
                    # Fallback to default initialization
                    self.engine = pyttsx3.init()
                
                if self.engine:
                    # Set voice properties for better audio output
                    voices = self.engine.getProperty('voices')
                    if voices and len(voices) > 0:
                        # Try to find a female voice first, fallback to first voice
                        selected_voice = voices[0]
                        for voice in voices:
                            if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                                selected_voice = voice
                                break
                        self.engine.setProperty('voice', selected_voice.id)
                        print(f"Selected voice: {selected_voice.name}")
                    
                    # Configure speech properties for clear output
                    self.engine.setProperty('rate', 160)  # Slightly faster speech
                    self.engine.setProperty('volume', 1.0)  # Maximum volume
                    
                    print("pyttsx3 voice engine initialized successfully")
                    return
                    
            except Exception as e:
                print(f"pyttsx3 initialization failed: {e}")
                self.engine = None
        
        # Fallback to Google TTS if pyttsx3 fails
        if GTTS_AVAILABLE:
            try:
                # Initialize pygame mixer for audio playback
                pygame.mixer.init()
                self.use_gtts = True
                print("Google Text-to-Speech fallback initialized successfully")
                return
            except Exception as e:
                print(f"Google TTS initialization failed: {e}")
        
        print("No voice engine available - both pyttsx3 and Google TTS failed")
    
    def test_voice_silent(self):
        """Test the voice engine silently without audio output"""
        if self.engine and not self.use_gtts:
            try:
                # Just test if engine can be called without actually speaking
                return True
            except Exception as e:
                print(f"Voice test failed: {e}")
                return False
        elif self.use_gtts:
            return True
        return False
    
    def set_voice_enabled(self, enabled):
        """Enable or disable voice announcements"""
        self.voice_enabled = enabled and self.engine is not None
    
    def announce_detection(self, object_name, distance, confidence=None):
        """Announce detected object with distance using available TTS engine"""
        if not self.voice_enabled or (not self.engine and not self.use_gtts):
            return
        
        current_time = time.time()
        
        # Check if engine is stuck and reset if needed
        self.is_engine_stuck()
        
        # Check if we're currently speaking
        if self.is_speaking:
            print(f"Voice announcement skipped (currently speaking): {object_name}")
            return
        
        # Use only object name for cooldown to reduce announcement frequency
        object_key = object_name
        
        # Check object-specific cooldown to avoid repetitive announcements
        if object_key in self.last_announcement:
            time_since_last = current_time - self.last_announcement[object_key]
            if time_since_last < self.announcement_cooldown:
                print(f"Voice announcement skipped (object cooldown): {object_name}")
                return
        
        # Check global cooldown - but be more lenient for different objects
        if current_time - self.last_announcement_time < self.global_cooldown:
            # Allow announcement if it's a different object and enough time has passed
            last_announced_object = getattr(self, 'last_announced_object', None)
            if last_announced_object == object_name:
                print(f"Voice announcement skipped (global cooldown): {object_name}")
                return
            elif current_time - self.last_announcement_time < 0.5:  # Minimum gap for different objects
                print(f"Voice announcement skipped (minimum gap): {object_name}")
                return
        
        # Update timers
        self.last_announcement[object_key] = current_time
        self.last_announcement_time = current_time
        self.last_announced_object = object_name
        
        # Create announcement text
        if distance < 1.0:
            distance_text = f"{int(distance * 100)} centimeters"
        else:
            distance_text = f"{distance:.1f} meters"
        
        announcement = f"Detected {object_name} at {distance_text}"
        print(f"Voice announcement: {announcement}")  # Debug output
        
        # Choose TTS method based on available engine
        if self.use_gtts:
            self._speak_with_gtts(announcement)
        else:
            self._speak_with_pyttsx3(announcement)
    
    def _speak_with_pyttsx3(self, text):
        """Speak using pyttsx3 engine with simple, reliable approach"""
        def speak():
            try:
                # Set speaking flag
                self.is_speaking = True
                self.speaking_start_time = time.time()  # Record when speaking started
                print(f"Starting pyttsx3 announcement: {text}")
                
                # Initialize COM for this thread on Windows
                if hasattr(pythoncom, 'CoInitialize'):
                    pythoncom.CoInitialize()
                
                # Try to use the main engine first
                success = False
                try:
                    if self.engine:
                        self.engine.say(text)
                        self.engine.runAndWait()
                        success = True
                        print(f"Successfully announced with main pyttsx3: {text}")
                except Exception as e:
                    print(f"Main engine failed: {e}")
                
                # If main engine failed, try creating a new one
                if not success:
                    try:
                        temp_engine = pyttsx3.init()
                        if temp_engine:
                            # Quick configuration
                            temp_engine.setProperty('rate', 150)
                            temp_engine.setProperty('volume', 0.9)
                            
                            temp_engine.say(text)
                            temp_engine.runAndWait()
                            print(f"Successfully announced with temp pyttsx3: {text}")
                            success = True
                    except Exception as e:
                        print(f"Temp engine also failed: {e}")
                
                # If pyttsx3 completely failed, try Google TTS
                if not success and self.gtts_available and GTTS_AVAILABLE:
                    print("Both pyttsx3 engines failed, switching to Google TTS...")
                    self.use_gtts = True
                    # Call Google TTS directly in this thread to avoid flag issues
                    try:
                        from gtts import gTTS
                        import pygame
                        import tempfile
                        import os
                        
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                            temp_filename = temp_file.name
                        
                        tts = gTTS(text=text, lang='en', slow=False)
                        tts.save(temp_filename)
                        
                        pygame.mixer.music.load(temp_filename)
                        pygame.mixer.music.play()
                        
                        while pygame.mixer.music.get_busy():
                            time.sleep(0.1)
                        
                        os.unlink(temp_filename)
                        print(f"Successfully announced with Google TTS: {text}")
                        success = True
                    except Exception as e:
                        print(f"Google TTS also failed: {e}")
                
                if not success:
                    print(f"All TTS methods failed for: {text}")
                
                # Cleanup COM for this thread
                if hasattr(pythoncom, 'CoUninitialize'):
                    pythoncom.CoUninitialize()
                    
            except Exception as e:
                print(f"Voice announcement completely failed: {e}")
            finally:
                # Always clear the speaking flag - this is critical!
                self.is_speaking = False
                print(f"Finished announcement (is_speaking now False): {text}")
        
        # Run in thread to prevent blocking
        thread = threading.Thread(target=speak, daemon=True)
        thread.start()
    
    def _speak_with_gtts(self, text):
        """Speak using Google Text-to-Speech with proper state management"""
        def speak():
            try:
                # Set speaking flag
                self.is_speaking = True
                
                # Create temporary file for audio
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                    temp_filename = temp_file.name
                
                # Generate speech with Google TTS
                tts = gTTS(text=text, lang='en', slow=False)
                tts.save(temp_filename)
                
                # Play the audio file
                pygame.mixer.music.load(temp_filename)
                pygame.mixer.music.play()
                
                # Wait for playback to complete
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                
                # Clean up temporary file
                os.unlink(temp_filename)
                print(f"Successfully announced with Google TTS: {text}")
                
            except Exception as e:
                print(f"Google TTS announcement failed: {e}")
            finally:
                # Always clear the speaking flag
                self.is_speaking = False
        
        # Run in thread to prevent blocking
        thread = threading.Thread(target=speak, daemon=True)
        thread.start()
    
    def reset_speaking_state(self):
        """Force reset the speaking state - useful for debugging"""
        self.is_speaking = False
        print("Voice speaking state forcefully reset")
    
    def is_engine_stuck(self):
        """Check if engine might be stuck (speaking for too long)"""
        current_time = time.time()
        # If we've been "speaking" for more than 5 seconds, something is wrong
        if self.is_speaking:
            speaking_duration = current_time - self.speaking_start_time
            if speaking_duration > 5:
                print(f"Voice engine stuck for {speaking_duration:.1f} seconds, resetting...")
                self.reset_speaking_state()
                return True
        return False

# Initialize voice manager (cached to prevent multiple initializations)
@st.cache_resource
def get_voice_manager():
    """Get a cached voice manager instance"""
    if VOICE_AVAILABLE or GTTS_AVAILABLE:
        return VoiceManager()
    else:
        return None

voice_manager = get_voice_manager()

# Show warning if voice is not available
if not voice_manager or (not voice_manager.engine and not voice_manager.use_gtts):
    st.sidebar.warning("⚠️ Voice functionality disabled - install pyttsx3 or gtts+pygame")
else:
    # Show which voice engine is being used
    if voice_manager.use_gtts:
        st.sidebar.info("🌐 Using Google Text-to-Speech")
    else:
        st.sidebar.info("🔊 Using Windows Voice Engine")

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
    """Get local IP for mobile connectivity with better detection."""
    try:
        # Try multiple methods to get the correct local IP
        import socket
        
        # Method 1: Connect to external server
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            if ip and not ip.startswith("127."):
                return ip
        
        # Method 2: Get hostname IP
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        if ip and not ip.startswith("127."):
            return ip
            
        # Method 3: Check all network interfaces
        import subprocess
        import platform
        
        if platform.system() == "Windows":
            result = subprocess.run(['ipconfig'], capture_output=True, text=True)
            lines = result.stdout.split('\n')
            for line in lines:
                if 'IPv4 Address' in line and '192.168.' in line:
                    ip = line.split(':')[-1].strip()
                    return ip
        
        return "192.168.1.100"  # Fallback IP
        
    except Exception as e:
        print(f"Error getting local IP: {e}")
        return "localhost"

def get_streamlit_port():
    """Get the current Streamlit port."""
    try:
        # Try to get port from Streamlit server config
        import streamlit.web.bootstrap as bootstrap
        if hasattr(bootstrap, '_server') and bootstrap._server:
            return bootstrap._server.port
    except:
        pass
    
    # Try to get port from query params or environment
    try:
        if hasattr(st, 'query_params'):
            port = st.query_params.get("port")
            if port:
                return port
    except:
        pass
    
    # Check common Streamlit ports
    import socket
    common_ports = [8501, 8502, 8503, 8504, 8505]
    for port in common_ports:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex(('localhost', port))
                if result == 0:  # Port is open
                    return str(port)
        except:
            continue
    
    return "8501"  # Default Streamlit port

def generate_qr_code(url):
    """Generate QR code for mobile access with better error handling."""
    try:
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=5
        )
        qr.add_data(url)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        buf = BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()
    except Exception as e:
        print(f"Error generating QR code: {e}")
        # Return a simple text-based QR code as fallback
        return None

def validate_network_connection(ip, port):
    """Validate that the network connection is accessible."""
    try:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(3)
            result = s.connect_ex((ip, int(port)))
            return result == 0
    except:
        return False

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
if "voice_enabled" not in st.session_state:
    st.session_state.voice_enabled = False

# Sidebar
st.sidebar.title("🔧 Settings")
score_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.05)  # Lower default threshold

# Voice toggle button (only show if voice is available)
if voice_manager and (voice_manager.engine or voice_manager.use_gtts):
    voice_enabled = st.sidebar.toggle("🔊 Voice Announcements", value=st.session_state.voice_enabled)
    if voice_enabled != st.session_state.voice_enabled:
        st.session_state.voice_enabled = voice_enabled
        voice_manager.set_voice_enabled(voice_enabled)
    
    # Ensure voice manager is synchronized with session state
    voice_manager.set_voice_enabled(st.session_state.voice_enabled)

    if voice_enabled:
        if voice_manager.use_gtts:
            st.sidebar.success("🎤 Voice ON (Google TTS)")
        else:
            st.sidebar.success("🎤 Voice ON (Windows)")
        # Add test voice button
        if st.sidebar.button("🎯 Test Voice"):
            if voice_manager:
                voice_manager.announce_detection("test object", 1.5)
    else:
        st.sidebar.info("🔇 Voice OFF")
else:
    st.sidebar.warning("🔇 Voice Unavailable")
    st.session_state.voice_enabled = False

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
        self.current_detections = []  # Store current frame detections
    
    def estimate_distance(self, bbox_area, object_name):
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
                        
                        # Voice announcement (if enabled)
                        # Pass voice enabled state directly since session_state is not available in video thread
                        if (voice_manager and (voice_manager.engine or voice_manager.use_gtts) and voice_manager.voice_enabled):
                            print(f"Attempting voice announcement for {self.classes[class_id]} at {distance:.1f}m")
                            voice_manager.announce_detection(self.classes[class_id], distance)
                        else:
                            if voice_manager:
                                engine_available = voice_manager.engine is not None or voice_manager.use_gtts
                                print(f"Voice announcement skipped - voice_enabled: {voice_manager.voice_enabled}, engine_available: {engine_available}")
                            else:
                                print("Voice announcement skipped - no voice_manager")
            
            # Update total count (only count new detection instances)
            if len(detection_list) > 0:
                if self.frame_count - self.last_detection_frame > 30:  # New detection session
                    self.total_objects += len(detection_list)
                    self.last_detection_frame = self.frame_count
            
            # Store current detections
            self.current_detections = detection_list
            
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
                "video": {
                    "width": {"ideal": 640, "max": 1280},
                    "height": {"ideal": 480, "max": 720},
                    "frameRate": {"ideal": 15, "max": 30}
                },
                "audio": False
            },
            async_processing=True,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            }
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
    st.subheader("📱 Phone Camera → 💻 Laptop Detection")
    
    # Detect if accessing from mobile device with fallback
    user_agent = st.context.headers.get("user-agent", "").lower()
    is_mobile_fallback = any(mobile in user_agent for mobile in [
        'android', 'iphone', 'ipad', 'ipod', 'blackberry', 'iemobile', 'opera mini'
    ])
    
    # JavaScript detection with improved reliability
    mobile_detection_js = f"""
    <script>
    function detectMobile() {{
        const userAgent = navigator.userAgent.toLowerCase();
        const isMobile = /android|webos|iphone|ipad|ipod|blackberry|iemobile|opera mini/i.test(userAgent);
        const isTablet = /ipad|android|tablet/i.test(userAgent) && !/mobile/i.test(userAgent);
        const hasTouchScreen = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
        const screenWidth = window.screen.width;
        const screenHeight = window.screen.height;
        const smallScreen = Math.min(screenWidth, screenHeight) < 768;
        
        const isMobileDevice = isMobile || (hasTouchScreen && smallScreen) || isTablet;
        
        // Multiple ways to communicate with Streamlit
        try {{
            window.parent.postMessage({{
                type: 'streamlit:setComponentValue', 
                value: {{is_mobile: isMobileDevice, userAgent: userAgent, screen: {{width: screenWidth, height: screenHeight}}}}
            }}, '*');
            
            // Alternative method
            window.parent.postMessage({{
                type: 'deviceInfo',
                data: {{is_mobile: isMobileDevice, fallback: {str(is_mobile_fallback).lower()}}}
            }}, '*');
        }} catch(e) {{
            console.log('Device detection error:', e);
        }}
        
        // Display detection result
        document.body.innerHTML = '<div style="font-size:12px;color:#666;">Device: ' + (isMobileDevice ? 'Mobile' : 'Desktop') + ' (UA: ' + userAgent.substring(0,50) + '...)</div>';
    }}
    detectMobile();
    </script>
    """
    
    # Get device detection result with fallback
    mobile_result = st.components.v1.html(mobile_detection_js, height=30)
    
    # Safe device detection with fallback
    if mobile_result:
        try:
            is_mobile_device = mobile_result.get('is_mobile', is_mobile_fallback) if isinstance(mobile_result, dict) else is_mobile_fallback
        except:
            is_mobile_device = is_mobile_fallback
    else:
        is_mobile_device = is_mobile_fallback
    
    # Enhanced security warning for mobile camera access
    if not is_mobile_device:
        st.error("🚨 **CAMERA ACCESS FIX REQUIRED FOR MOBILE**")
    
    col_warn1, col_warn2 = st.columns([3, 1])
    with col_warn1:
        if is_mobile_device:
            st.success("📱 **Mobile Device Detected** - Ready to send camera stream to laptop")
        else:
            st.info("💻 **Laptop/Desktop Detected** - Ready to receive camera stream from phone")
            
        st.markdown("""
        **The "navigator.mediaDevices is undefined" error occurs because:**
        - Mobile browsers require HTTPS or special settings for camera access
        - HTTP connections from remote devices are blocked for security
        
        **✅ SIMPLE FIX: Enable Chrome flags (takes 30 seconds)**
        """)
    
    with col_warn2:
        if st.button("📖 Open Setup Guide", key="setup_guide"):
            st.success("Opening Chrome setup guide...")
            # This would ideally open chrome_setup_guide.html
    
    # Get network information
    local_ip = get_local_ip()
    port = get_streamlit_port()
    
    # Create the streaming URL
    base_url = f"http://{local_ip}:{port}"
    
    # Add current page path to ensure phone opens the same page
    current_path = st.query_params.get("page", "3_fixed_object_detection")
    if not current_path.startswith("/"):
        full_url = f"{base_url}/?page={current_path}"
    else:
        full_url = f"{base_url}{current_path}"
    
    # Alternative direct URL (simpler)
    direct_url = base_url
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.info("📱 **Connect Your Phone Camera:**")
        
        # Show network information
        st.markdown(f"**🌐 Network Info:**")
        st.code(f"Laptop IP: {local_ip}")
        st.code(f"Port: {port}")
        st.code(f"Full URL: {direct_url}")
        
        # Validate connection
        is_accessible = validate_network_connection(local_ip, port)
        if is_accessible:
            st.success("✅ Network accessible from other devices")
        else:
            st.warning("⚠️ Network may not be accessible - check firewall settings")
        
        # Generate and display QR code
        qr_image = generate_qr_code(direct_url)
        if qr_image:
            st.image(qr_image, caption=f"Scan to connect: {direct_url}", width=250)
        else:
            st.error("❌ Could not generate QR code")
        
        # Manual connection option
        st.markdown("**📱 Manual Connection:**")
        st.markdown(f"Open this URL on your phone: `{direct_url}`")
        
        # Copy button simulation
        if st.button("📋 Copy URL to Clipboard"):
            st.code(direct_url)
            st.info("📋 Copy the URL above and paste it in your phone browser")
        
        st.markdown("**📋 Connection Steps:**")
        st.markdown("""
        1. 📱 **Scan QR code** or **copy URL manually**
        2. 🌐 **Open in phone browser** (Chrome/Safari recommended)
        3. 📷 **Allow camera permissions** when prompted
        4. 🎯 **Select "Phone Camera (WebRTC)"** mode on phone
        5. ▶️ **Start camera** on phone to begin streaming
        6. 💻 **Detection runs on laptop** with voice announcements
        """)
        
        # Enhanced Chrome flags instructions section
        with st.expander("🔧 🚨 REQUIRED: Enable Camera Access (30 seconds)", expanded=True):
            st.markdown("### 📱 For Android Chrome Users:")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("**Step 1: Copy this URL**")
                chrome_flags_url = "chrome://flags/#unsafely-treat-insecure-origin-as-secure"
                st.code(chrome_flags_url)
                
                st.markdown("**Step 2: Add your laptop URL**")
                st.code(f"http://{local_ip}:{port}")
                
                st.markdown("**Step 3: Set to 'Enabled' and restart Chrome**")
                
                st.success("✅ After this setup, camera will work immediately!")
            
            with col2:
                # Show QR code for quick access to Chrome flags
                qr_image = generate_qr_code(chrome_flags_url)
                if qr_image:
                    st.image(qr_image, caption="Chrome Flags QR", width=150)
                
                # App QR code
                app_qr = generate_qr_code(direct_url)
                if app_qr:
                    st.image(app_qr, caption="AdMyVision App", width=150)
            
            st.markdown("### 🍎 For iPhone/iPad Users:")
            st.warning("Safari requires HTTPS. Use the HTTPS setup button above or try a different solution.")
            
            st.markdown("### 🔄 Alternative: USB Debugging")
            st.code("adb reverse tcp:8502 tcp:8502")
            st.markdown("Then access: `http://localhost:8502`")
        
        # Network troubleshooting
        with st.expander("🔧 Troubleshooting"):
            st.markdown("""
            **If connection fails:**
            - ✅ Ensure both devices are on **same WiFi network**
            - 🔥 **Disable Windows Firewall** temporarily for testing
            - 🔄 **Refresh both pages** if connection drops
            - 📱 Try **different browsers** on phone (Chrome, Safari, Edge)
            - 🌐 **Manually type the URL** if QR code doesn't work
            """)
            
            st.error("🚨 **navigator.mediaDevices is undefined** error fix:")
            st.markdown("""
            This error occurs because camera access requires HTTPS on mobile devices when connecting to remote servers.
            
            **Solution 1: Enable insecure origins in Chrome (RECOMMENDED)**
            1. Open Chrome on your phone
            2. Type: `chrome://flags/#unsafely-treat-insecure-origin-as-secure`
            3. Add your laptop's URL: `http://{}`
            4. Set to "Enabled"
            5. Restart Chrome
            6. Try connecting again
            
            **Solution 2: Use HTTPS (Advanced)**
            Generate SSL certificate and restart Streamlit:
            ```bash
            # Generate self-signed certificate
            openssl req -newkey rsa:2048 -nodes -keyout key.pem -x509 -days 365 -out cert.pem
            
            # Restart Streamlit with HTTPS
            streamlit run pages/3_fixed_object_detection.py --server.sslCertFile=cert.pem --server.sslKeyFile=key.pem
            ```
            
            **Solution 3: Use localhost tunnel**
            1. Connect phone via USB
            2. Enable USB debugging
            3. Use ADB port forwarding: `adb reverse tcp:8502 tcp:8502`
            4. Access via `http://localhost:8502` on phone
            """.format(f"{local_ip}:{port}"))
        
        st.warning("📌 **Critical:** Both devices must be on the same WiFi network!")
    
    with col2:
        st.info("🌐 **Streaming Status:**")
        
        # Conditional WebRTC configuration based on device type
        if is_mobile_device:
            # MOBILE DEVICE: Send camera to laptop (SENDONLY)
            st.success("📱 **Mobile Mode:** Sending your camera to laptop")
            webrtc_ctx = webrtc_streamer(
                key="mobile_camera_sender",
                mode=WebRtcMode.SENDONLY,  # SEND ONLY - mobile sends its camera
                video_frame_callback=None,  # No processing on mobile side
                media_stream_constraints={
                    "video": {
                        "width": {"ideal": 640, "min": 320, "max": 1280},
                        "height": {"ideal": 480, "min": 240, "max": 720},
                        "frameRate": {"ideal": 15, "min": 10, "max": 25},
                        "facingMode": "environment"  # Use back camera on mobile
                    },
                    "audio": False
                },
                async_processing=True,
                rtc_configuration={
                    "iceServers": [
                        {"urls": ["stun:stun.l.google.com:19302"]},
                        {"urls": ["stun:stun1.l.google.com:19302"]},
                        {"urls": ["stun:stun2.l.google.com:19302"]},
                        {"urls": ["stun:stun3.l.google.com:19302"]},
                        {"urls": ["stun:stun4.l.google.com:19302"]}
                    ],
                    "iceCandidatePoolSize": 20,
                    "iceTransportPolicy": "all"
                },
                video_html_attrs={
                    "style": {"width": "100%", "margin": "0 auto", "border": "2px solid #4CAF50"},
                    "controls": False,
                    "autoplay": True,
                    "muted": True
                }
            )
        else:
            # LAPTOP/DESKTOP: Receive camera from mobile (RECVONLY)
            st.info("💻 **Laptop Mode:** Receiving camera from phone")
            webrtc_ctx = webrtc_streamer(
                key="laptop_camera_receiver", 
                mode=WebRtcMode.RECVONLY,  # RECEIVE ONLY - laptop doesn't send its camera
                video_frame_callback=video_frame_callback,  # Process received frames
                media_stream_constraints={
                    "video": False,  # Laptop doesn't use its camera
                    "audio": False   # No audio needed
                },
                async_processing=True,
                rtc_configuration={
                    "iceServers": [
                        {"urls": ["stun:stun.l.google.com:19302"]},
                        {"urls": ["stun:stun1.l.google.com:19302"]},
                        {"urls": ["stun:stun2.l.google.com:19302"]},
                        {"urls": ["stun:stun3.l.google.com:19302"]},
                        {"urls": ["stun:stun4.l.google.com:19302"]}
                    ],
                    "iceCandidatePoolSize": 20,
                    "iceTransportPolicy": "all"
                },
                video_html_attrs={
                    "style": {"width": "100%", "margin": "0 auto", "border": "2px solid #ff6b6b"},
                    "controls": False,
                    "autoplay": True,
                    "muted": True,
                    "playsinline": True  # Important for mobile browsers
                }
            )
        
        # Enhanced connection status with more detailed feedback
        if webrtc_ctx.state.playing:
            st.success("✅ Phone camera streaming to laptop!")
            st.success("🎯 AI detection running on laptop")
            if voice_manager and st.session_state.voice_enabled:
                st.success("🔊 Voice announcements enabled")
            
            # Show connection quality info
            st.info("📊 **Stream Quality:** Good")
            
        elif webrtc_ctx.state.signalling:
            st.warning("🔄 Establishing connection with phone...")
            st.info("💡 Please wait, handshake in progress...")
            
            # Show troubleshooting hints during connection
            with st.expander("Connection taking too long?"):
                st.markdown("""
                - 🔄 **Refresh both devices**
                - 🌐 **Check WiFi connection**
                - 📱 **Allow camera permissions** on phone
                - 🔥 **Check firewall settings** on laptop
                """)
                
        else:
            st.error("❌ No phone connection")
            st.info("📱 Scan QR code with phone to start streaming")
            
            # Show detailed connection instructions
            st.markdown("**🔍 Connection Status:**")
            st.markdown(f"- 📡 Laptop ready at: `{local_ip}:{port}`")
            st.markdown(f"- 🌐 Waiting for phone at: `{direct_url}`")
            
            if not is_accessible:
                st.error("🚫 **Network Issue Detected!**")
                st.markdown("""
                **Quick Fixes:**
                1. 🔥 Turn off Windows Firewall temporarily
                2. 🌐 Check WiFi network (same for both devices)
                3. 🔄 Restart Streamlit app
                4. 📱 Try manual URL entry on phone
                """)
        
        # Network diagnostics
        st.markdown("**🔍 Network Diagnostics:**")
        st.caption(f"📡 Laptop IP: {local_ip}")
        st.caption(f"🌐 Port: {port}")
        st.caption(f"🔗 Accessibility: {'✅ Good' if is_accessible else '❌ Issues'}")
    
    # Detection results - only show when streaming
    if webrtc_ctx.state.playing:
        st.markdown("---")
        st.subheader("🔍 Live Detection Results (Processing on Laptop)")
        
        # Create three columns for better layout
        col1, col2, col3 = st.columns([1, 1, 1])
        
        # Update detections from phone stream
        current_phone_detections = 0
        total_phone_detected = 0
        try:
            while not result_queue.empty():
                result_data = result_queue.get_nowait()
                if isinstance(result_data, dict):
                    st.session_state.detections = result_data.get('detections', [])
                    current_phone_detections = result_data.get('current_count', 0)
                    total_phone_detected = result_data.get('total_count', 0)
                    st.session_state.frame_count = result_data.get('frame_count', 0)
                else:
                    # Fallback for old format
                    st.session_state.detections = result_data
                    current_phone_detections = len(result_data) if result_data else 0
        except:
            pass
        
        # Show detection statistics in columns
        with col1:
            st.metric("📊 Current Objects", current_phone_detections)
            st.metric("🎯 Total Detected", total_phone_detected)
        
        with col2:
            st.metric("📹 Frames Processed", st.session_state.frame_count)
            st.metric("🎚️ Confidence Threshold", f"{score_threshold:.2f}")
        
        with col3:
            # Voice status
            if voice_manager and st.session_state.voice_enabled:
                if voice_manager.use_gtts:
                    st.success("🔊 Google TTS Active")
                else:
                    st.success("🔊 Windows Voice Active")
            else:
                st.info("🔇 Voice Disabled")
            
            # Reset button for total count
            if st.button("🔄 Reset Count", key="phone_reset", help="Reset the total objects detected counter"):
                detector.reset_total_count()
                st.rerun()
        
        # Detailed detection list
        if st.session_state.detections:
            st.markdown("### 📋 Detected Objects")
            
            # Create a more detailed table
            detection_data = []
            for i, det in enumerate(st.session_state.detections, 1):
                # Calculate approximate distance based on box size
                box_area = (det.box[2] - det.box[0]) * (det.box[3] - det.box[1])
                distance = detector.estimate_distance(box_area, det.label)
                
                # Confidence level indicator
                if det.score > 0.8:
                    confidence_indicator = "🟢 High"
                elif det.score > 0.6:
                    confidence_indicator = "🟡 Medium"
                else:
                    confidence_indicator = "🔴 Low"
                
                detection_data.append({
                    '#': i,
                    'Object': det.label.title(),
                    'Confidence': f"{det.score:.1%}",
                    'Level': confidence_indicator,
                    'Distance': f"~{distance:.1f}m",
                    'Position': f"({int(det.box[0])}, {int(det.box[1])})"
                })
            
            st.dataframe(detection_data, use_container_width=True, hide_index=True)
            
            # Show top detection
            if detection_data:
                top_detection = max(st.session_state.detections, key=lambda x: x.score)
                st.info(f"🎯 **Best Detection:** {top_detection.label.title()} ({top_detection.score:.1%} confidence)")
        else:
            st.info("👀 **Point your phone camera at objects to detect:**")
            st.markdown("""
            - 👤 People
            - 🚗 Vehicles (car, bus, bicycle)
            - 🪑 Furniture (chair, table)
            - 🍼 Common items (bottle, cup)
            - 🐕 Animals (dog, cat, bird)
            """)
    
    else:
        # Instructions when not connected
        st.markdown("---")
        st.info("📱 **Waiting for phone connection...**")
        st.markdown("""
        **Troubleshooting Tips:**
        - Ensure both devices are on the same WiFi network
        - Allow camera permissions in your phone browser
        - Try refreshing both pages if connection fails
        - Use Chrome or Safari on your phone for best compatibility
        """)

# Footer
st.markdown("---")

# Add JavaScript for real-time camera access detection (especially for mobile users)
if mode == "Phone Camera (WebRTC)":
    st.markdown("### 🔍 Real-time Camera Access Check")
    
    # Enhanced JavaScript with polyfills and multiple detection methods
    camera_check_js = """
    <div id="camera-status" style="padding: 15px; margin: 10px 0; border-radius: 8px; background-color: #fff3e0; color: #ef6c00; border-left: 4px solid #ff9800;">
        🔄 Checking camera access...
    </div>
    
    <div id="detailed-status" style="margin-top: 10px; padding: 10px; border-radius: 5px; background-color: #f5f5f5; font-family: monospace; font-size: 12px; display: none;">
    </div>
    
    <script>
    // Enhanced camera access detection with polyfills
    function updateStatus(message, type, details = '') {
        const statusDiv = document.getElementById('camera-status');
        const detailsDiv = document.getElementById('detailed-status');
        
        const styles = {
            error: { bg: '#ffebee', color: '#c62828', border: '#f44336' },
            success: { bg: '#e8f5e8', color: '#2e7d32', border: '#4caf50' },
            warning: { bg: '#fff3e0', color: '#ef6c00', border: '#ff9800' },
            info: { bg: '#e3f2fd', color: '#1565c0', border: '#2196f3' }
        };
        
        const style = styles[type] || styles.warning;
        statusDiv.innerHTML = message;
        statusDiv.style.backgroundColor = style.bg;
        statusDiv.style.color = style.color;
        statusDiv.style.borderLeftColor = style.border;
        
        if (details) {
            detailsDiv.innerHTML = details;
            detailsDiv.style.display = 'block';
        } else {
            detailsDiv.style.display = 'none';
        }
    }

    function addPolyfills() {
        // Add navigator polyfill if missing
        if (typeof navigator === 'undefined') {
            window.navigator = {};
        }
        
        // Add mediaDevices polyfill for older browsers
        if (!navigator.mediaDevices) {
            navigator.mediaDevices = {};
        }
        
        // Add getUserMedia polyfill
        if (!navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia = function(constraints) {
                // Try legacy getUserMedia methods
                const getUserMedia = navigator.getUserMedia || 
                                   navigator.webkitGetUserMedia || 
                                   navigator.mozGetUserMedia ||
                                   navigator.msGetUserMedia;
                
                if (!getUserMedia) {
                    return Promise.reject(new Error('getUserMedia is not implemented in this browser'));
                }
                
                return new Promise(function(resolve, reject) {
                    getUserMedia.call(navigator, constraints, resolve, reject);
                });
            };
        }
        
        // Add enumerateDevices polyfill
        if (!navigator.mediaDevices.enumerateDevices) {
            navigator.mediaDevices.enumerateDevices = function() {
                return Promise.resolve([{
                    deviceId: 'default',
                    kind: 'videoinput',
                    label: 'Default Camera',
                    groupId: 'default'
                }]);
            };
        }
    }

    async function checkCameraAccess() {
        let details = '';
        
        try {
            // Add polyfills first
            addPolyfills();
            
            details += 'Navigator: ' + (typeof navigator !== 'undefined' ? '✅' : '❌') + '\\n';
            details += 'MediaDevices: ' + (navigator.mediaDevices ? '✅' : '❌') + '\\n';
            details += 'GetUserMedia: ' + (navigator.mediaDevices && navigator.mediaDevices.getUserMedia ? '✅' : '❌') + '\\n';
            details += 'HTTPS: ' + (location.protocol === 'https:' ? '✅' : '❌') + '\\n';
            details += 'Localhost: ' + (location.hostname === 'localhost' || location.hostname === '127.0.0.1' ? '✅' : '❌') + '\\n';
            details += 'User Agent: ' + navigator.userAgent.substring(0, 50) + '...\\n';
            
            // Check basic availability
            if (typeof navigator === 'undefined') {
                updateStatus('❌ Navigator not available - this might be a server-side render', 'error', details);
                return;
            }
            
            if (!navigator.mediaDevices) {
                updateStatus(`
                    🚨 <strong>CAMERA ACCESS BLOCKED!</strong><br><br>
                    � <strong>Quick Fix for Android Chrome:</strong><br>
                    1️⃣ Copy: <code>chrome://flags/#unsafely-treat-insecure-origin-as-secure</code><br>
                    2️⃣ Paste in Chrome address bar<br>
                    3️⃣ Add: <code>${window.location.origin}</code><br>
                    4️⃣ Set to "Enabled" and restart Chrome<br><br>
                    🍎 <strong>iPhone users:</strong> This requires HTTPS setup<br>
                    📍 <strong>Current URL:</strong> ${window.location.href}
                `, 'error', details);
                return;
            }
            
            if (!navigator.mediaDevices.getUserMedia) {
                updateStatus('❌ getUserMedia not available in this browser', 'error', details);
                return;
            }
            
            // Test device enumeration first
            updateStatus('🔍 Checking for available cameras...', 'info', details);
            
            try {
                const devices = await navigator.mediaDevices.enumerateDevices();
                const cameras = devices.filter(device => device.kind === 'videoinput');
                details += 'Cameras found: ' + cameras.length + '\\n';
                
                if (cameras.length === 0) {
                    updateStatus('⚠️ No cameras detected on this device', 'warning', details);
                    return;
                }
                
                // Test actual camera access without requesting permission yet
                updateStatus('✅ Camera devices detected! MediaDevices API is working.', 'success', details);
                details += 'Camera test: Successful\\n';
                
            } catch (error) {
                details += 'Device enumeration error: ' + error.message + '\\n';
                updateStatus(`⚠️ Camera check partial: ${error.message}`, 'warning', details);
            }
            
        } catch (error) {
            details += 'Check error: ' + error.message + '\\n';
            updateStatus(`❌ Camera access check failed: ${error.message}`, 'error', details);
        }
    }
    
    // Check immediately and also after page loads
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', checkCameraAccess);
    } else {
        checkCameraAccess();
    }
    
    // Re-check every 3 seconds in case user fixes the issue
    setInterval(checkCameraAccess, 3000);
    </script>
    """
    
    st.components.v1.html(camera_check_js, height=200)

st.markdown(f"**Streamlit-WebRTC**: {st_webrtc_version} | **aiortc**: {aiortc.__version__}")
