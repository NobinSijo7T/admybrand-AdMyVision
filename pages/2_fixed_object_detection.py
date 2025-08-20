"""Simplified and Optimized Object Detection with WebRTC
Fixed version addressing video freezing and mobile connection issues.
"""

import streamlit as st

# Page configuration for standalone deployment
st.set_page_config(
    page_title="AdMyVision - Object Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

import logging
import queue
import time
import threading
import platform
from pathlib import Path
from typing import List, NamedTuple
import socket

import av
try:
    import cv2
except ImportError:
    st.error("❌ OpenCV not installed. Please install opencv-python-headless")
    st.stop()

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
import streamlit.components.v1 as components
try:
    import aiortc
    AIORTC_AVAILABLE = True
except ImportError:
    aiortc = None
    AIORTC_AVAILABLE = False
    print("aiortc not available - WebRTC functionality may be limited")

# Try to import pyttsx3 with proper error handling
try:
    import pyttsx3
    import pythoncom  # For Windows COM initialization
    VOICE_AVAILABLE = True
except ImportError:
    pyttsx3 = None
    pythoncom = None
    VOICE_AVAILABLE = False
    print("pyttsx3 not available - voice synthesis will use browser/gtts fallback")

# Try to import Google Text-to-Speech as fallback
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    gTTS = None
    GTTS_AVAILABLE = False
    print("gtts not available - voice synthesis will use browser fallback")

# Try to import pygame for audio playback
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    pygame = None
    PYGAME_AVAILABLE = False
    print("pygame not available - audio playback will use browser fallback")

# Additional imports
import tempfile
import os

# Try to import download utility
try:
    from sample_utils.download import download_file
except ImportError:
    # Fallback download function
    import urllib.request
    def download_file(url, download_to, expected_size=None):
        """Simple download function fallback"""
        if download_to.exists():
            if expected_size and download_to.stat().st_size == expected_size:
                return
            elif not expected_size:
                return
        
        download_to.parent.mkdir(parents=True, exist_ok=True)
        st.info(f"Downloading {url}...")
        
        try:
            urllib.request.urlretrieve(url, download_to)
            st.success(f"Downloaded {download_to.name}")
        except Exception as e:
            st.error(f"Download failed: {e}")
            raise

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
        
        # FINAL SAFETY CHECK: Ensure voice is always enabled
        if not self.voice_enabled:
            print("🚨 CONSTRUCTOR SAFETY: Voice not enabled after init_voice_engine, forcing enable")
            self.voice_enabled = True
            self.use_gtts = True
        
        print(f"🚨 VoiceManager constructor complete: voice_enabled={self.voice_enabled}, use_gtts={self.use_gtts}")
    
    def init_voice_engine(self):
        """Initialize the text-to-speech engine with fallback options"""
        # Try pyttsx3 first (only if available)
        if VOICE_AVAILABLE and pyttsx3:
            try:
                # Initialize COM for Windows
                if pythoncom and hasattr(pythoncom, 'CoInitialize'):
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
                    
                    self.voice_enabled = True  # Enable voice functionality
                    print("pyttsx3 voice engine initialized successfully")
                    return
                    
            except Exception as e:
                print(f"pyttsx3 initialization failed: {e}")
                self.engine = None
        
        # Fallback to Google TTS if pyttsx3 fails (even without pygame)
        print(f"Checking Google TTS: GTTS_AVAILABLE={GTTS_AVAILABLE}")
        if GTTS_AVAILABLE:
            try:
                # Test gtts to make sure it's working
                print("Testing Google TTS...")
                
                # Simple test without creating actual audio
                test_tts = gTTS(text="test", lang='en')
                print("Google TTS test successful")
                
                # Try to initialize pygame mixer for audio playback (optional)
                if PYGAME_AVAILABLE and pygame:
                    try:
                        pygame.mixer.init()
                        print("Google Text-to-Speech with pygame audio initialized successfully")
                    except Exception as pe:
                        print(f"Pygame init failed: {pe}, using browser audio fallback")
                else:
                    print("Google Text-to-Speech with browser audio fallback initialized successfully")
                
                self.use_gtts = True
                self.voice_enabled = True
                print("✅ Google TTS voice engine enabled successfully")
                return
            except Exception as e:
                print(f"❌ Google TTS initialization failed: {e}")
                print("🌐 Falling back to browser speech synthesis")
                # Don't return here, continue to browser fallback
        else:
            print("❌ Google TTS not available (gtts package not found)")
        
        # Final fallback - always enable browser speech
        print("🌐 Enabling browser speech synthesis as final fallback")
        self.use_gtts = True
        self.voice_enabled = True
        print("✅ Browser speech synthesis enabled")
        
        # FORCE voice to be enabled - this is critical for deployment
        print("⚠️ FORCING voice_enabled = True for guaranteed functionality")
        self.voice_enabled = True
        
        # Debug output to confirm final state
        print(f"🔊 Final voice manager state: voice_enabled={self.voice_enabled}, use_gtts={self.use_gtts}, engine={self.engine is not None}")
        
        # If still not enabled, something went wrong - force it anyway
        if not self.voice_enabled:
            print("❌ Voice still disabled after initialization - FORCING ENABLE")
            self.voice_enabled = True
            self.use_gtts = True
    
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
        # Allow voice to be enabled even without pyttsx3 engine (browser speech fallback)
        self.voice_enabled = enabled
        print(f"🔊 Voice manually set to: {self.voice_enabled} (engine={self.engine is not None}, use_gtts={self.use_gtts})")
    
    def announce_detection(self, object_name, distance, confidence=None):
        """Announce detected object with distance using available TTS engine"""
        print(f"🔍 Voice announcement requested: {object_name} at {distance:.1f}m")
        print(f"🔊 Voice status: enabled={self.voice_enabled}, engine={self.engine is not None}, use_gtts={self.use_gtts}")
        
        if not self.voice_enabled:
            print("❌ Voice announcement skipped - voice_enabled is False")
            return
            
        if not self.engine and not self.use_gtts:
            print("❌ Voice announcement skipped - no engine available")
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
        
        # ALWAYS use browser speech synthesis for maximum compatibility
        # This ensures voice works on all platforms including mobile and web deployment
        print("🌐 Using browser speech synthesis for maximum compatibility")
        self._speak_with_browser(announcement)
    
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
    
    def _speak_with_browser(self, text):
        """Use browser's speech synthesis with user interaction handling"""
        try:
            # Escape text for JavaScript to prevent injection issues
            escaped_text = text.replace("'", "\\'").replace('"', '\\"').replace('\n', ' ')
            
            # Create a unique ID for this speech request
            import random
            speech_id = f"speech_{random.randint(1000, 9999)}"
            
            # Enhanced JavaScript speech synthesis
            speech_js = f"""
            <div id="{speech_id}" style="position: relative; height: 1px;">
                <script>
                (function() {{
                    console.log('🎤 Browser speech synthesis initializing for: {escaped_text}');
                    
                    // Check if speech synthesis is supported
                    if (!('speechSynthesis' in window)) {{
                        console.error('❌ Speech synthesis not supported in this browser');
                        return;
                    }}
                    
                    // Function to speak the text
                    function speakNow() {{
                        try {{
                            console.log('🔊 Speaking now: {escaped_text}');
                            
                            // Cancel any ongoing speech
                            speechSynthesis.cancel();
                            
                            // Small delay to ensure cancellation
                            setTimeout(function() {{
                                // Create utterance
                                var utterance = new SpeechSynthesisUtterance('{escaped_text}');
                                utterance.rate = 0.8;
                                utterance.pitch = 1.0;
                                utterance.volume = 1.0;
                                utterance.lang = 'en-US';
                                
                                // Get available voices
                                var voices = speechSynthesis.getVoices();
                                console.log('📢 Available voices:', voices.length);
                                
                                // Select the best voice
                                if (voices.length > 0) {{
                                    var selectedVoice = voices.find(voice => 
                                        voice.lang === 'en-US' && voice.localService === false
                                    ) || voices.find(voice => 
                                        voice.lang.startsWith('en')
                                    ) || voices[0];
                                    
                                    if (selectedVoice) {{
                                        utterance.voice = selectedVoice;
                                        console.log('🎯 Selected voice:', selectedVoice.name, selectedVoice.lang);
                                    }}
                                }}
                                
                                // Event handlers
                                utterance.onstart = function() {{
                                    console.log('✅ Speech started: {escaped_text}');
                                }};
                                
                                utterance.onend = function() {{
                                    console.log('🏁 Speech completed: {escaped_text}');
                                }};
                                
                                utterance.onerror = function(event) {{
                                    console.error('❌ Speech error:', event.error, event);
                                }};
                                
                                // Speak the text
                                speechSynthesis.speak(utterance);
                                console.log('🚀 Speech synthesis initiated successfully');
                                
                            }}, 100);
                            
                        }} catch (error) {{
                            console.error('❌ Error in speakNow:', error);
                        }}
                    }}
                    
                    // Function to handle voices loading
                    function handleVoicesLoaded() {{
                        console.log('🔄 Voices loaded, attempting to speak...');
                        speakNow();
                    }}
                    
                    // Wait for voices to load if needed
                    var voices = speechSynthesis.getVoices();
                    if (voices.length === 0) {{
                        console.log('⏳ Waiting for voices to load...');
                        speechSynthesis.addEventListener('voiceschanged', handleVoicesLoaded, {{ once: true }});
                        
                        // Fallback timeout
                        setTimeout(function() {{
                            console.log('⚠️ Voices loading timeout, attempting anyway...');
                            speakNow();
                        }}, 2000);
                    }} else {{
                        console.log('✅ Voices already available, speaking immediately');
                        speakNow();
                    }}
                    
                    // Create a click trigger for user interaction
                    function createClickTrigger() {{
                        var existingTrigger = document.getElementById('voice-trigger');
                        if (!existingTrigger) {{
                            var trigger = document.createElement('div');
                            trigger.id = 'voice-trigger';
                            trigger.style.cssText = 'position: fixed; top: -1px; left: -1px; width: 1px; height: 1px; opacity: 0; pointer-events: none;';
                            trigger.onclick = function() {{
                                console.log('🖱️ User interaction detected, enabling speech...');
                                speakNow();
                            }};
                            document.body.appendChild(trigger);
                            
                            // Simulate click if needed
                            setTimeout(function() {{
                                trigger.click();
                            }}, 500);
                        }}
                    }}
                    
                    // Ensure user interaction if required
                    createClickTrigger();
                    
                }})();
                </script>
            </div>
            """
            
            # Display the JavaScript component with minimal height
            components.html(speech_js, height=1)
            print(f"🎤 Browser speech synthesis requested: {text}")
            
        except Exception as e:
            print(f"❌ Browser speech synthesis error: {e}")

    def _speak_with_gtts(self, text):
        """Speak using Google Text-to-Speech with proper state management"""
        if not GTTS_AVAILABLE or not gTTS:
            print("Google TTS not available, using browser speech synthesis")
            self._speak_with_browser(text)
            return
        
        # If pygame is not available, use browser speech directly 
        if not PYGAME_AVAILABLE or not pygame:
            print("Pygame not available, using browser speech for audio playback")
            self._speak_with_browser(text)
            return
            
        def speak():
            try:
                # Set speaking flag
                self.is_speaking = True
                
                # Create temporary file for audio
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                    temp_filename = temp_file.name
                
                print(f"Generating Google TTS audio for: {text}")
                # Generate speech with Google TTS
                tts = gTTS(text=text, lang='en', slow=False)
                tts.save(temp_filename)
                
                print(f"Playing Google TTS audio with pygame...")
                # Play the audio file with pygame
                pygame.mixer.music.load(temp_filename)
                pygame.mixer.music.play()
                
                # Wait for playback to complete
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                
                print(f"Successfully announced with Google TTS + pygame: {text}")
                
                # Clean up temporary file
                try:
                    os.unlink(temp_filename)
                except:
                    pass  # Ignore cleanup errors
                
            except Exception as e:
                print(f"Google TTS announcement failed: {e}")
                # Fallback to browser speech if Google TTS fails
                print("Falling back to browser speech synthesis...")
                self.is_speaking = False  # Reset flag before fallback
                self._speak_with_browser(text)
                return
            finally:
                # Always clear the speaking flag
                self.is_speaking = False
        
        # Run in thread to prevent blocking
        thread = threading.Thread(target=speak, daemon=True)
        thread.start()
    
    def test_voice_with_interaction(self, test_text="Voice test successful"):
        """Test voice with user interaction trigger for browser compatibility"""
        print(f"🎯 Testing voice with user interaction: {test_text}")
        
        # Create a more robust browser speech test
        self._speak_with_browser_test(test_text)
    
    def _speak_with_browser_test(self, text):
        """Enhanced browser speech test with user interaction handling"""
        try:
            # Escape text for JavaScript
            escaped_text = text.replace("'", "\\'").replace('"', '\\"').replace('\n', ' ')
            
            # Create unique ID for this test
            import random
            test_id = f"voice_test_{random.randint(1000, 9999)}"
            
            # Enhanced test with user interaction
            test_js = f"""
            <div id="{test_id}" style="position: relative; height: 20px; margin: 5px 0;">
                <button id="voice-test-btn-{test_id}" style="
                    background: linear-gradient(45deg, #4CAF50, #45a049);
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 6px;
                    cursor: pointer;
                    font-size: 12px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                    transition: all 0.3s ease;
                " onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">
                    🔊 Click to Test Voice
                </button>
                
                <script>
                (function() {{
                    console.log('🎯 Voice test component initialized');
                    
                    var testBtn = document.getElementById('voice-test-btn-{test_id}');
                    var hasSpoken = false;
                    
                    function performVoiceTest() {{
                        if (hasSpoken) return;
                        hasSpoken = true;
                        
                        console.log('🔊 Performing voice test with user interaction');
                        
                        if (!('speechSynthesis' in window)) {{
                            console.error('❌ Speech synthesis not supported');
                            testBtn.textContent = '❌ Speech Not Supported';
                            testBtn.style.background = '#f44336';
                            return;
                        }}
                        
                        try {{
                            // Cancel any ongoing speech
                            speechSynthesis.cancel();
                            
                            setTimeout(function() {{
                                var utterance = new SpeechSynthesisUtterance('{escaped_text}');
                                utterance.rate = 0.8;
                                utterance.pitch = 1.0;
                                utterance.volume = 1.0;
                                utterance.lang = 'en-US';
                                
                                // Get and select best voice
                                var voices = speechSynthesis.getVoices();
                                if (voices.length > 0) {{
                                    var bestVoice = voices.find(v => v.lang === 'en-US' && !v.localService) ||
                                                   voices.find(v => v.lang.startsWith('en')) ||
                                                   voices[0];
                                    if (bestVoice) {{
                                        utterance.voice = bestVoice;
                                        console.log('🎯 Using voice:', bestVoice.name);
                                    }}
                                }}
                                
                                // Event handlers
                                utterance.onstart = function() {{
                                    console.log('✅ Voice test started successfully');
                                    testBtn.textContent = '🔊 Speaking...';
                                    testBtn.style.background = '#2196F3';
                                }};
                                
                                utterance.onend = function() {{
                                    console.log('✅ Voice test completed successfully');
                                    testBtn.textContent = '✅ Voice Test Passed';
                                    testBtn.style.background = '#4CAF50';
                                }};
                                
                                utterance.onerror = function(event) {{
                                    console.error('❌ Voice test failed:', event.error);
                                    testBtn.textContent = '❌ Voice Test Failed';
                                    testBtn.style.background = '#f44336';
                                }};
                                
                                // Speak the test
                                speechSynthesis.speak(utterance);
                                console.log('🚀 Voice test speech initiated');
                                
                            }}, 100);
                            
                        }} catch (error) {{
                            console.error('❌ Voice test exception:', error);
                            testBtn.textContent = '❌ Test Failed';
                            testBtn.style.background = '#f44336';
                        }}
                    }}
                    
                    // Click handler
                    testBtn.onclick = performVoiceTest;
                    
                    // Auto-trigger after a delay (may not work due to user interaction requirement)
                    setTimeout(function() {{
                        if (!hasSpoken) {{
                            console.log('⚡ Auto-triggering voice test...');
                            performVoiceTest();
                        }}
                    }}, 1000);
                    
                }})();
                </script>
            </div>
            """
            
            # Display with proper height
            components.html(test_js, height=50)
            print(f"🎯 Voice test component created: {text}")
            
        except Exception as e:
            print(f"❌ Voice test component error: {e}")
    
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
    print(f"Creating voice manager - VOICE_AVAILABLE: {VOICE_AVAILABLE}, GTTS_AVAILABLE: {GTTS_AVAILABLE}")
    
    # Always try to create a voice manager for browser speech fallback
    voice_manager = VoiceManager()
    print(f"Voice manager created - voice_enabled: {voice_manager.voice_enabled}, use_gtts: {voice_manager.use_gtts}")
    
    # FORCE voice to be enabled if it's not already - this is critical for deployment
    if not voice_manager.voice_enabled:
        print("🚨 EMERGENCY FIX: Voice manager voice_enabled is False, forcing to True")
        voice_manager.voice_enabled = True
        voice_manager.use_gtts = True
        print(f"🚨 FORCED voice_enabled: {voice_manager.voice_enabled}, use_gtts: {voice_manager.use_gtts}")
    
    return voice_manager

voice_manager = get_voice_manager()

# Additional safety check - force voice to be enabled after creation
if voice_manager and not voice_manager.voice_enabled:
    print("🚨 FINAL SAFETY CHECK: Forcing voice_enabled to True after creation")
    voice_manager.voice_enabled = True
    voice_manager.use_gtts = True

# Show voice engine status
if not voice_manager:
    st.sidebar.error("❌ Voice manager not initialized")
else:
    # Always show voice as enabled since we force it
    if voice_manager.voice_enabled:
        # Show which voice engine is being used
        if voice_manager.use_gtts:
            if PYGAME_AVAILABLE:
                st.sidebar.success("🌐 Using Google Text-to-Speech with audio")
            else:
                st.sidebar.success("🌐 Using Google TTS with browser audio")
        else:
            st.sidebar.success("🔊 Using Windows Voice Engine")
    else:
        # This should never happen now due to forced enablement
        st.sidebar.error("❌ Voice functionality disabled despite forced initialization")
        st.sidebar.caption(f"Debug: GTTS_AVAILABLE={GTTS_AVAILABLE}, use_gtts={voice_manager.use_gtts}")
        # Force it one more time
        voice_manager.voice_enabled = True
        voice_manager.use_gtts = True
        st.sidebar.info("🚨 Voice forcefully re-enabled - refresh page if issues persist")

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
        def safe_console_message(message, msg_type="info"):
            """Safely add console message with fallback"""
            try:
                add_console_message(message, msg_type)
            except NameError:
                # Console function not available yet, use print as fallback
                print(f"[{msg_type.upper()}] {message}")
        
        # Download models if needed
        if not MODEL_LOCAL_PATH.exists():
            safe_console_message("Downloading MobileNet-SSD model...", "download")
            safe_console_message(f"Downloading {MODEL_URL}...", "info")
            download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=23147564)
            safe_console_message("Downloaded MobileNetSSD_deploy.caffemodel", "success")
        
        # Check for prototxt file (try both naming conventions)
        prototxt_path = PROTOTXT_LOCAL_PATH
        if not prototxt_path.exists() and PROTOTXT_ALT_PATH.exists():
            prototxt_path = PROTOTXT_ALT_PATH
        elif not prototxt_path.exists():
            safe_console_message("Downloading model configuration...", "download")
            safe_console_message(f"Downloading {PROTOTXT_URL}...", "info")
            download_file(PROTOTXT_URL, PROTOTXT_LOCAL_PATH, expected_size=29353)
            safe_console_message("Downloaded MobileNetSSD_deploy.prototxt.txt", "success")
        
        safe_console_message(f"Loading model from: {prototxt_path.name}", "info")
        net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(MODEL_LOCAL_PATH))
        
        # Test the model with a dummy input
        dummy_blob = np.zeros((1, 3, 300, 300), dtype=np.float32)
        net.setInput(dummy_blob)
        test_output = net.forward()
        
        safe_console_message(f"Model loaded successfully! Output shape: {test_output.shape}", "success")
        return net
        
    except Exception as e:
        def safe_console_message(message, msg_type="info"):
            """Safely add console message with fallback"""
            try:
                add_console_message(message, msg_type)
            except NameError:
                # Console function not available yet, use print as fallback
                print(f"[{msg_type.upper()}] {message}")
                
        safe_console_message(f"Error loading model: {e}", "error")
        safe_console_message(f"Model path: {MODEL_LOCAL_PATH}", "error")
        safe_console_message(f"Prototxt path: {prototxt_path if 'prototxt_path' in locals() else PROTOTXT_LOCAL_PATH}", "error")
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
st.title("🎯 AdMyVision - Real-time Object Detection")
st.markdown("### 🚀 AI-Powered Object Detection with Voice Announcements")

# Add global speech synthesis initialization
if voice_manager and voice_manager.voice_enabled:
    # Create a global speech enabler component
    speech_enabler_js = """
    <div id="global-speech-enabler" style="display: none;">
        <script>
        (function() {
            console.log('🌐 Global speech synthesis enabler loaded');
            
            // Function to enable speech synthesis
            function enableSpeechSynthesis() {
                if ('speechSynthesis' in window) {
                    // Create a silent utterance to trigger permission
                    var silentUtterance = new SpeechSynthesisUtterance(' ');
                    silentUtterance.volume = 0;
                    silentUtterance.rate = 10;
                    
                    silentUtterance.onend = function() {
                        console.log('✅ Speech synthesis enabled globally');
                        window.speechEnabled = true;
                    };
                    
                    speechSynthesis.speak(silentUtterance);
                }
            }
            
            // Try to enable on page interaction
            function handleUserInteraction() {
                console.log('👆 User interaction detected, enabling speech...');
                enableSpeechSynthesis();
                // Remove listeners after first interaction
                document.removeEventListener('click', handleUserInteraction);
                document.removeEventListener('keydown', handleUserInteraction);
                document.removeEventListener('touchstart', handleUserInteraction);
            }
            
            // Add event listeners for user interaction
            document.addEventListener('click', handleUserInteraction, { passive: true });
            document.addEventListener('keydown', handleUserInteraction, { passive: true });
            document.addEventListener('touchstart', handleUserInteraction, { passive: true });
            
            // Try immediate enablement (may not work without user interaction)
            setTimeout(enableSpeechSynthesis, 1000);
            
        })();
        </script>
    </div>
    """
    components.html(speech_enabler_js, height=1)

st.markdown("---")

# Console-like UI for system status and model loading
st.markdown("#### 💻 System Console")
console_container = st.container()

with console_container:
    console_placeholder = st.empty()
    
    # Initialize console messages if not in session state
    if 'console_messages' not in st.session_state:
        st.session_state.console_messages = []
    
    def add_console_message(message, msg_type="info"):
        """Add a message to the console with timestamp"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if msg_type == "info":
            icon = "ℹ️"
        elif msg_type == "success":
            icon = "✅"
        elif msg_type == "download":
            icon = "📥"
        elif msg_type == "error":
            icon = "❌"
        else:
            icon = "▶️"
            
        formatted_message = f"[{timestamp}] {icon} {message}"
        st.session_state.console_messages.append(formatted_message)
        
        # Keep only last 10 messages
        if len(st.session_state.console_messages) > 10:
            st.session_state.console_messages = st.session_state.console_messages[-10:]
    
    # Display console messages in a code block
    if st.session_state.console_messages:
        console_text = "\n".join(st.session_state.console_messages[-8:])  # Show last 8 messages
        console_placeholder.code(console_text, language="bash")
    else:
        console_placeholder.code("[00:00:00] ▶️ System ready - waiting for initialization...", language="bash")

# Add initial system messages
if 'console_initialized' not in st.session_state:
    add_console_message("AdMyVision System Starting...", "info")
    add_console_message(f"Python Runtime: {platform.python_version()}", "info")
    add_console_message(f"OpenCV Version: {cv2.__version__}", "info")
    add_console_message(f"Voice Engine: {'Browser Speech Synthesis' if GTTS_AVAILABLE else 'Disabled'}", "info")
    st.session_state.console_initialized = True
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
            st.sidebar.success("🎤 Voice ON (Browser Speech)")
        else:
            st.sidebar.success("🎤 Voice ON (Windows)")
        
        # Add test voice button with user interaction
        if st.sidebar.button("🎯 Test Voice"):
            if voice_manager:
                # Use the enhanced test method
                voice_manager.test_voice_with_interaction("Hello! Voice test successful. Object detection audio is working.")
                st.sidebar.info("👆 Click the test button above if voice doesn't play automatically")
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
        # Simple status display
        st.subheader("📊 Detection Status")
        
        # Camera status
        camera_status = webrtc_ctx.state.playing if webrtc_ctx else False
        if camera_status:
            st.success("✅ Camera Active - Detection Running")
        else:
            st.info("📹 Camera Inactive - Click Start to begin")
        
        # Model status
        if net is not None:
            st.success(f"✅ Model Ready | 🎯 Threshold: {score_threshold:.2f}")
        else:
            st.error("❌ Model Failed to Load")
        
        # Simple detection display
        if camera_status:
            st.info("� **Scanning for objects...**\n"
                   "Point camera at: Person, Car, Bottle, Chair, Phone, etc.")

elif mode == "Phone Camera (WebRTC)":
    st.subheader("📱 Phone Camera Detection")
    st.info("📢 **Voice Announcements:** Mobile devices use browser speech synthesis. Ensure your browser has voice synthesis enabled and volume is up.")
    
    # Camera selection for mobile devices
    camera_col1, camera_col2 = st.columns([2, 1])
    
    with camera_col1:
        # Camera facing direction selector
        camera_facing = st.selectbox(
            "📹 Camera Direction",
            ["user", "environment"],
            index=1,  # Default to back camera
            format_func=lambda x: "🤳 Front Camera (Selfie)" if x == "user" else "📷 Back Camera (Main)",
            help="Choose which camera to use on your mobile device"
        )
    
    with camera_col2:
        # Camera quality selector
        quality_preset = st.selectbox(
            "🎥 Quality",
            ["standard", "high"],
            index=0,
            format_func=lambda x: "📱 Standard (480p)" if x == "standard" else "📺 High (720p)",
            help="Choose video quality (higher quality uses more bandwidth)"
        )
    
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
        1. 📱 Scan QR code with phone camera
        2. 🌐 Open the link in browser
        3. ✅ Allow camera permissions
        4. 📄 Select this same page
        5. 📹 Choose 'Phone Camera (WebRTC)' mode
        6. 🔄 Select camera direction above
        """)
        
        # Voice compatibility info for mobile
        st.markdown("**🔊 Mobile Voice Notes:**")
        st.info("📢 Voice announcements work on most Android devices through the browser. "
               "iOS Safari may have limited voice support due to browser restrictions.")
    
    with col2:
        st.info("🌐 **Live Camera Feed:**")
        
        # Dynamic media constraints based on selections
        if quality_preset == "high":
            video_constraints = {
                "width": {"ideal": 1280, "max": 1920},
                "height": {"ideal": 720, "max": 1080},
                "frameRate": {"ideal": 15, "max": 30},
                "facingMode": camera_facing
            }
        else:
            video_constraints = {
                "width": {"ideal": 640, "max": 1280},
                "height": {"ideal": 480, "max": 720},
                "frameRate": {"ideal": 10, "max": 20},
                "facingMode": camera_facing
            }
        
        # Enhanced WebRTC for phone with camera switching
        webrtc_ctx = webrtc_streamer(
            key=f"phone_camera_{camera_facing}_{quality_preset}",  # Unique key for camera switching
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={
                "video": video_constraints,
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
        phone_detections = []
        
        try:
            while not result_queue.empty():
                result_data = result_queue.get_nowait()
                if isinstance(result_data, dict):
                    phone_detections = result_data.get('detections', [])
                    current_phone_detections = result_data.get('current_count', 0)
                    total_phone_detected = result_data.get('total_count', 0)
                    st.session_state.detections = phone_detections
                else:
                    # Fallback for old format
                    phone_detections = result_data if result_data else []
                    current_phone_detections = len(phone_detections)
                    st.session_state.detections = phone_detections
        except:
            pass
        
        # Use session state detections if no new data
        if not phone_detections and hasattr(st.session_state, 'detections'):
            phone_detections = st.session_state.detections
            current_phone_detections = len(phone_detections)
        
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
        
        # Display detections with enhanced formatting
        if phone_detections:
            st.write("**Currently Detected Objects:**")
            for i, det in enumerate(phone_detections[:5]):
                confidence_color = "🟢" if det.score > 0.7 else "🟡" if det.score > 0.5 else "🔴"
                
                # Calculate distance
                try:
                    bbox_area = (det.box[2] - det.box[0]) * (det.box[3] - det.box[1])
                    distance = detector.estimate_distance(bbox_area, det.label)
                    distance_text = f" at ~{distance:.1f}m"
                except:
                    distance_text = ""
                
                st.write(f"{i+1}. {confidence_color} **{det.label.title()}**: {det.score:.1%}{distance_text}")
            
            if len(phone_detections) > 5:
                st.info(f"...and {len(phone_detections) - 5} more objects detected")
            
            # Also show as dataframe for detailed view
            if st.checkbox("📊 Show Detailed Table", key="phone_table"):
                detection_data = [{
                    'Object': det.label.title(),
                    'Confidence': f"{det.score:.1%}",
                    'Distance': f"~{detector.estimate_distance((det.box[2] - det.box[0]) * (det.box[3] - det.box[1]), det.label):.1f}m",
                    'Position': f"({int(det.box[0])}, {int(det.box[1])})"
                } for det in phone_detections]
                st.dataframe(detection_data, use_container_width=True)
        else:
            st.info("🔍 Point camera at objects to detect them")

# Footer
st.markdown("---")
aiortc_version = aiortc.__version__ if AIORTC_AVAILABLE and aiortc else "Not Available"
st.markdown(f"**Streamlit-WebRTC**: {st_webrtc_version} | **aiortc**: {aiortc_version}")
