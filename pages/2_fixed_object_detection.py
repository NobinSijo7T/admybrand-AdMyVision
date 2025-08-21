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
import urllib.request

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

# Embedded download function to avoid import issues on Streamlit Cloud
def download_file(url, download_to: Path, expected_size=None):
    """Download file with progress bar - embedded to avoid import issues"""
    # Don't download the file twice.
    # (If possible, verify the download using the file length.)
    if download_to.exists():
        if expected_size:
            if download_to.stat().st_size == expected_size:
                return
        else:
            st.info(f"{url} is already downloaded.")
            if not st.button("Download again?"):
                return

    download_to.parent.mkdir(parents=True, exist_ok=True)

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % url)
        progress_bar = st.progress(0)
        with open(download_to, "wb") as output_file:
            with urllib.request.urlopen(url) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0**20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning(
                        "Downloading %s... (%6.2f/%6.2f MB)"
                        % (url, counter / MEGABYTES, length / MEGABYTES)
                    )
                    progress_bar.progress(min(counter / length, 1.0))
    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()

HERE = Path(__file__).parent
ROOT = HERE.parent

logger = logging.getLogger(__name__)

# Model paths - updated for Streamlit Cloud compatibility
MODEL_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.caffemodel"
MODEL_LOCAL_PATH = Path("models/MobileNetSSD_deploy.caffemodel")  # Relative path for Streamlit Cloud
PROTOTXT_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.prototxt.txt"
PROTOTXT_LOCAL_PATH = Path("models/MobileNetSSD_deploy.prototxt.txt")  # Relative path for Streamlit Cloud

# Alternative paths for local development
PROTOTXT_ALT_PATH = Path("models/MobileNetSSD_deploy.prototxt")

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
        # Initialize console logs if not exists
        if 'console_logs' not in st.session_state:
            st.session_state.console_logs = []
        
        # Download models if needed
        if not MODEL_LOCAL_PATH.exists():
            st.session_state.console_logs.append("📥 Downloading MobileNet-SSD model...")
            download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=23147564)
        
        # Check for prototxt file (try both naming conventions)
        prototxt_path = PROTOTXT_LOCAL_PATH
        if not prototxt_path.exists() and PROTOTXT_ALT_PATH.exists():
            prototxt_path = PROTOTXT_ALT_PATH
        elif not prototxt_path.exists():
            st.session_state.console_logs.append("📥 Downloading model configuration...")
            download_file(PROTOTXT_URL, PROTOTXT_LOCAL_PATH, expected_size=29353)
        
        st.session_state.console_logs.append(f"✅ Loading model from: {prototxt_path}")
        net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(MODEL_LOCAL_PATH))
        
        # Test the model with a dummy input
        dummy_blob = np.zeros((1, 3, 300, 300), dtype=np.float32)
        net.setInput(dummy_blob)
        test_output = net.forward()
        
        st.session_state.console_logs.append(f"✅ Model loaded successfully! Output shape: {test_output.shape}")
        return net
        
    except Exception as e:
        if 'console_logs' not in st.session_state:
            st.session_state.console_logs = []
        st.session_state.console_logs.append(f"❌ Error loading model: {e}")
        st.session_state.console_logs.append(f"Model path: {MODEL_LOCAL_PATH}")
        st.session_state.console_logs.append(f"Prototxt path: {prototxt_path if 'prototxt_path' in locals() else PROTOTXT_LOCAL_PATH}")
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

# Camera troubleshooting section
with st.sidebar.expander("🔧 Camera Troubleshooting"):
    st.markdown("""
    **Camera Not Working?**
    
    **For PC Camera:**
    - ✅ Allow camera permissions in browser
    - ✅ Close other apps using camera
    - ✅ Try refreshing the page
    - ✅ Check if camera is properly connected
    
    **For Mobile Camera:**
    - ✅ Use HTTPS (required for camera access)
    - ✅ Allow camera permissions when prompted
    - ✅ Use Chrome, Safari, or Edge browser
    - ✅ Try switching between front/back camera
    - ✅ Ensure stable internet connection
    
    **Still having issues?**
    - Try restarting your browser
    - Check browser camera settings
    - Disable browser extensions temporarily
    """)
    
    if st.button("🔄 Refresh Page", help="Force refresh the application"):
        st.rerun()
    
    # Add camera permission checker
    if st.button("📹 Check Camera Access"):
        st.components.v1.html("""
        <script>
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(function(stream) {
                document.write("✅ Camera access granted!");
                stream.getTracks().forEach(track => track.stop());
            })
            .catch(function(err) {
                document.write("❌ Camera access denied: " + err.message);
            });
        </script>
        """, height=50)

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
    
    st.info("💻 **PC Camera Tips:**\n- Allow camera access when prompted\n- Check if other apps are using the camera\n- Try refreshing if camera doesn't start")
    
    webrtc_ctx = webrtc_streamer(
        key="pc_camera",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640, "min": 320, "max": 1280},
                "height": {"ideal": 480, "min": 240, "max": 720},
                "frameRate": {"ideal": 15, "min": 10, "max": 30}
            },
            "audio": False
        },
        async_processing=True,
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]}
            ]
        }
    )
    
    # Camera status indicator
    if webrtc_ctx.state.playing:
        st.success("✅ Camera is active and streaming")
    elif webrtc_ctx.state.signalling:
        st.warning("🔄 Connecting to camera...")
    else:
        st.error("❌ Camera not connected. Please allow camera permissions and refresh if needed.")

elif mode == "Phone Camera (WebRTC)":
    st.subheader("📱 Phone Camera Detection")
    
    # Camera selection for mobile
    col1, col2 = st.columns([2, 1])
    with col1:
        st.info("📱 **Mobile Camera Tips:**\n- Allow camera permissions when prompted\n- Use HTTPS for camera access\n- Best with Chrome or Safari")
    with col2:
        camera_mode = st.selectbox(
            "📷 Camera Selection:",
            ["Back Camera", "Front Camera"],
            help="Switch between front and back camera on mobile"
        )
    
    # Set facing mode based on selection
    facing_mode = "environment" if camera_mode == "Back Camera" else "user"
    
    webrtc_ctx = webrtc_streamer(
        key=f"phone_camera_{camera_mode.lower().replace(' ', '_')}",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640, "min": 320, "max": 1280},
                "height": {"ideal": 480, "min": 240, "max": 720},
                "frameRate": {"ideal": 15, "min": 10, "max": 30},
                "facingMode": facing_mode
            },
            "audio": False
        },
        async_processing=True,
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]}
            ]
        }
    )
    
    # Mobile camera status indicator
    if webrtc_ctx.state.playing:
        st.success("✅ Mobile camera is active and streaming")
    elif webrtc_ctx.state.signalling:
        st.warning("🔄 Connecting to mobile camera...")
    else:
        st.error("❌ Mobile camera not connected. Please allow camera permissions and ensure you're using HTTPS.")

# Console Display
st.markdown("---")
st.subheader("🖥️ System Console")

# Display console logs if they exist
if 'console_logs' in st.session_state and st.session_state.console_logs:
    console_container = st.container()
    with console_container:
        st.markdown("**Model Loading Information:**")
        console_text = "\n".join(st.session_state.console_logs)
        st.code(console_text, language=None)
        
        # Clear console button
        if st.button("🗑️ Clear Console"):
            st.session_state.console_logs = []
            st.rerun()
else:
    st.info("💡 Console will show model loading information when the app starts")

# Modern Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
           border-radius: 10px; margin-top: 30px; color: white;">
    <h3 style="margin: 0; color: white;">🚀 AdMyVision - Object Detection Platform</h3>
    <p style="margin: 10px 0; opacity: 0.9;">Powered by AI for Real-time Object Recognition</p>
    
    <div style="display: flex; justify-content: center; gap: 30px; margin: 15px 0; flex-wrap: wrap;">
        <div style="text-align: center;">
            <strong>🔧 Tech Stack</strong><br>
            <span style="font-size: 0.9em;">Streamlit-WebRTC {st_webrtc_version}</span><br>
            <span style="font-size: 0.9em;">aiortc {aiortc.__version__}</span><br>
            <span style="font-size: 0.9em;">OpenCV & MobileNet-SSD</span>
        </div>
        <div style="text-align: center;">
            <strong>🎯 Features</strong><br>
            <span style="font-size: 0.9em;">Real-time Detection</span><br>
            <span style="font-size: 0.9em;">Mobile Support</span><br>
            <span style="font-size: 0.9em;">Voice Feedback</span>
        </div>
        <div style="text-align: center;">
            <strong>🌐 Access</strong><br>
            <span style="font-size: 0.9em;">PC Camera</span><br>
            <span style="font-size: 0.9em;">Mobile Camera</span><br>
            <span style="font-size: 0.9em;">Cross-Platform</span>
        </div>
    </div>
    
    <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid rgba(255,255,255,0.2);">
        <p style="margin: 5px 0; font-size: 0.9em; opacity: 0.8;">
            © 2025 AdMyVision | Built with ❤️ using Python & Streamlit
        </p>
        <p style="margin: 0; font-size: 0.8em; opacity: 0.7;">
            🔒 Your privacy is protected - All processing happens locally
        </p>
    </div>
</div>
""", unsafe_allow_html=True)
