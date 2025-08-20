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

# Try to import download_file, if not available, define it inline
try:
    from sample_utils.download import download_file
except ImportError:
    # Define download_file inline if the module is not available
    import urllib.request
    from pathlib import Path
    
    def download_file(url, download_to: Path, expected_size=None):
        """Download a file from URL to local path."""
        # Don't download the file twice.
        if download_to.exists():
            if expected_size:
                if download_to.stat().st_size == expected_size:
                    return
            else:
                st.info(f"{url} is already downloaded.")
                if not st.button("Download again?"):
                    return
        
        download_to.parent.mkdir(parents=True, exist_ok=True)
        
        # Download the file
        with st.spinner(f"Downloading {url}..."):
            urllib.request.urlretrieve(url, download_to)
        
        st.success(f"Downloaded {url} to {download_to}")

HERE = Path(__file__).parent
ROOT = HERE.parent

logger = logging.getLogger(__name__)

# Voice Manager Class
class VoiceManager:
    def __init__(self):
        self.engine = None
        self.voice_enabled = True  # Default to enabled when engines are available
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
        """Initialize the text-to-speech engine with simple cloud-first approach"""
        print(f"Initializing voice engine - VOICE_AVAILABLE: {VOICE_AVAILABLE}, GTTS_AVAILABLE: {GTTS_AVAILABLE}")
        
        # Detect cloud deployment environment
        import platform
        is_cloud = (platform.system() == "Linux" or 
                   "STREAMLIT_SHARING" in os.environ or 
                   "STREAMLIT_CLOUD" in os.environ)
        
        print(f"Environment: {'Cloud' if is_cloud else 'Local'} ({platform.system()})")
        
        # For cloud: prioritize Google TTS (no audio playback needed for Streamlit Cloud)
        if GTTS_AVAILABLE:
            try:
                print("🌐 Initializing Google TTS (cloud-optimized)...")
                
                # Simple test to ensure gtts works
                from gtts import gTTS
                test_tts = gTTS(text="test", lang='en', slow=False)
                
                # For Streamlit Cloud, we don't need pygame - just generate the audio
                self.use_gtts = True
                self.engine = None
                print("✅ Google Text-to-Speech initialized successfully")
                return
                
            except Exception as e:
                print(f"❌ Google TTS initialization failed: {e}")
        
        # Fallback to pyttsx3 for local development
        if VOICE_AVAILABLE and not is_cloud:
            try:
                print("🖥️ Local environment: Attempting pyttsx3...")
                
                # Initialize COM for Windows
                if platform.system() == "Windows" and hasattr(pythoncom, 'CoInitialize'):
                    pythoncom.CoInitialize()
                
                # Simple initialization
                self.engine = pyttsx3.init()
                if self.engine:
                    print("✅ pyttsx3 initialized successfully")
                    return
                    
            except Exception as e:
                print(f"❌ pyttsx3 initialization failed: {e}")
        
        # If everything fails
        print("❌ No voice engines available")
        self.engine = None
        self.use_gtts = False
    
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
        self.voice_enabled = enabled and (self.engine is not None or self.use_gtts)
        print(f"🔊 Voice enabled set to: {self.voice_enabled} (enabled={enabled}, engine={self.engine is not None}, gtts={self.use_gtts})")
    
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
        """Speak using Google Text-to-Speech with cloud-optimized approach"""
        def speak():
            try:
                # Set speaking flag
                self.is_speaking = True
                self.speaking_start_time = time.time()
                print(f"Starting Google TTS announcement: {text}")
                
                # Import required modules
                from gtts import gTTS
                import tempfile
                import platform
                
                # Generate speech with Google TTS
                tts = gTTS(text=text, lang='en', slow=False, timeout=5)
                
                # For cloud deployment, just generate and save the audio
                # Streamlit Cloud doesn't support audio playback anyway
                is_cloud = platform.system() == "Linux"
                
                if is_cloud:
                    # Cloud mode: just generate the audio file as confirmation
                    with tempfile.NamedTemporaryFile(delete=True, suffix='.mp3') as temp_file:
                        tts.save(temp_file.name)
                        print(f"✅ Google TTS audio generated successfully (cloud mode): {text}")
                else:
                    # Local mode: try to play the audio
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                            temp_filename = temp_file.name
                        
                        tts.save(temp_filename)
                        
                        # Try pygame playback for local
                        if pygame and hasattr(pygame, 'mixer'):
                            try:
                                if not pygame.mixer.get_init():
                                    pygame.mixer.init()
                                pygame.mixer.music.load(temp_filename)
                                pygame.mixer.music.play()
                                
                                # Wait for playback with timeout
                                max_wait = 10
                                start_time = time.time()
                                while pygame.mixer.music.get_busy() and (time.time() - start_time) < max_wait:
                                    time.sleep(0.1)
                                    
                                print(f"✅ Google TTS played locally: {text}")
                            except Exception as play_error:
                                print(f"Local playback failed: {play_error}")
                        
                        # Cleanup
                        try:
                            os.unlink(temp_filename)
                        except:
                            pass
                
            except Exception as e:
                print(f"❌ Google TTS announcement failed: {e}")
            finally:
                # Always clear the speaking flag
                self.is_speaking = False
                print(f"Google TTS announcement completed: {text}")
        
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

def test_voice_engines():
    """Test which voice engines are available for cloud deployment"""
    results = {
        'pyttsx3': False,
        'gtts': False, 
        'pygame': False,
        'system_audio': False
    }
    
    # Test pyttsx3
    if VOICE_AVAILABLE:
        try:
            import platform
            if platform.system() == "Linux":
                # Try espeak driver for Linux
                test_engine = pyttsx3.init('espeak')
            else:
                test_engine = pyttsx3.init()
            if test_engine:
                results['pyttsx3'] = True
                del test_engine
        except Exception as e:
            print(f"pyttsx3 test failed: {e}")
    
    # Test Google TTS
    if GTTS_AVAILABLE:
        try:
            from gtts import gTTS
            # Test with a simple word
            tts = gTTS(text="test", lang='en', slow=False)
            results['gtts'] = True
        except Exception as e:
            print(f"Google TTS test failed: {e}")
    
    # Test pygame
    try:
        import pygame
        pygame.mixer.init()
        results['pygame'] = True
        pygame.mixer.quit()
    except Exception as e:
        print(f"Pygame test failed: {e}")
    
    # Test system audio
    try:
        import subprocess
        import platform
        if platform.system() == "Linux":
            # Test if aplay is available
            result = subprocess.run(['which', 'aplay'], capture_output=True)
            if result.returncode == 0:
                results['system_audio'] = True
    except Exception as e:
        print(f"System audio test failed: {e}")
    
    return results

# Initialize voice manager (cached to prevent multiple initializations)
@st.cache_resource
def get_voice_manager():
    """Get a cached voice manager instance"""
    print(f"Creating voice manager - VOICE_AVAILABLE: {VOICE_AVAILABLE}, GTTS_AVAILABLE: {GTTS_AVAILABLE}")
    if VOICE_AVAILABLE or GTTS_AVAILABLE:
        vm = VoiceManager()
        print(f"Voice manager created - voice_enabled: {vm.voice_enabled}, use_gtts: {vm.use_gtts}")
        return vm
    else:
        print("No voice engines available")
        return None

voice_manager = get_voice_manager()

# Show warning if voice is not available
if not voice_manager:
    st.sidebar.warning("⚠️ Voice functionality disabled - voice engines not available")
    st.sidebar.info("💡 This might be due to cloud platform limitations")
elif not voice_manager.engine and not voice_manager.use_gtts:
    st.sidebar.warning("⚠️ Voice functionality disabled - engine initialization failed")
    st.sidebar.info("💡 Try refreshing the page or check system dependencies")
else:
    # Show which voice engine is being used
    if voice_manager.use_gtts:
        st.sidebar.success("🌐 Using Google Text-to-Speech")
    else:
        st.sidebar.success("🔊 Using Windows Voice Engine")

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
if "voice_enabled" not in st.session_state:
    st.session_state.voice_enabled = True  # Default to enabled when voice is available

# Sidebar
st.sidebar.title("🔧 Settings")
score_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.05)  # Lower default threshold

# Add voice engine diagnostics section for cloud deployment debugging
import platform
if platform.system() == "Linux" or "STREAMLIT_SHARING" in os.environ or "STREAMLIT_CLOUD" in os.environ:
    with st.sidebar.expander("🔍 Voice Engine Diagnostics", expanded=True):
        st.write("**Environment:** Cloud/Linux")
        
        # Test voice engines
        test_results = test_voice_engines()
        
        for engine, available in test_results.items():
            if available:
                st.success(f"✅ {engine.upper()}")
            else:
                st.error(f"❌ {engine.upper()}")
        
        st.write(f"**Voice Manager Status:**")
        if voice_manager:
            st.write(f"- Engine: {'Available' if voice_manager.engine else 'None'}")
            st.write(f"- Google TTS: {'Active' if voice_manager.use_gtts else 'Inactive'}")
            st.write(f"- Voice Enabled: {voice_manager.voice_enabled}")
        else:
            st.error("Voice Manager: Not initialized")
        
        # Show environment variables
        st.write("**Environment Variables:**")
        for env_var in ["STREAMLIT_SHARING", "STREAMLIT_CLOUD", "HOSTNAME"]:
            value = os.environ.get(env_var, "Not set")
            st.text(f"{env_var}: {value}")

# Voice toggle button (only show if voice is available)
if voice_manager and (voice_manager.engine or voice_manager.use_gtts):
    voice_enabled = st.sidebar.toggle("🔊 Voice Announcements", value=st.session_state.voice_enabled)
    if voice_enabled != st.session_state.voice_enabled:
        st.session_state.voice_enabled = voice_enabled
        voice_manager.set_voice_enabled(voice_enabled)
    
    # Ensure voice manager is synchronized with session state
    voice_manager.set_voice_enabled(st.session_state.voice_enabled)

    if voice_enabled:
        import platform
        is_cloud = platform.system() == "Linux"
        
        if voice_manager.use_gtts:
            if is_cloud:
                st.sidebar.success("🎤 Voice ON (Google TTS - Cloud)")
                st.sidebar.info("� Voice announcements are generated (no audio playback in cloud)")
            else:
                st.sidebar.success("🎤 Voice ON (Google TTS)")
                st.sidebar.info("�🔊 Audio will play through your device speakers")
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

# Final safety check: Ensure voice manager is properly enabled
if voice_manager and (voice_manager.engine or voice_manager.use_gtts):
    voice_manager.set_voice_enabled(st.session_state.voice_enabled)
    print(f"🚨 FINAL SAFETY CHECK: Forcing voice_enabled to {st.session_state.voice_enabled} after creation")

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
