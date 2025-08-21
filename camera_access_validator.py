#!/usr/bin/env python3
"""
Camera Access Validator for AdMyVision
Tests network connectivity and provides solutions for navigator.mediaDevices issues.
"""

import socket
import subprocess
import sys
import webbrowser
from pathlib import Path

def get_local_ip():
    """Get the local IP address"""
    try:
        # Connect to a remote address to determine local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "localhost"

def check_port_accessible(ip, port):
    """Check if a port is accessible"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(3)
            result = s.connect_ex((ip, port))
            return result == 0
    except Exception:
        return False

def generate_chrome_flags_url(laptop_ip, port):
    """Generate Chrome flags URL for enabling insecure origins"""
    return f"chrome://flags/#unsafely-treat-insecure-origin-as-secure"

def create_simple_test_page():
    """Create a simple HTML page to test camera access"""
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Camera Access Test</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { font-family: Arial, sans-serif; padding: 20px; }
        .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .error { background-color: #ffebee; color: #c62828; }
        .success { background-color: #e8f5e8; color: #2e7d32; }
        .warning { background-color: #fff3e0; color: #ef6c00; }
        video { max-width: 100%; height: auto; }
    </style>
</head>
<body>
    <h1>📱 Camera Access Test</h1>
    <div id="status" class="status warning">Testing camera access...</div>
    <video id="video" autoplay playsinline style="display: none;"></video>
    <button id="testBtn" onclick="testCamera()">Test Camera Access</button>
    
    <div id="instructions" style="margin-top: 20px;">
        <h3>If camera test fails with "navigator.mediaDevices is undefined":</h3>
        <ol>
            <li>Open Chrome settings</li>
            <li>Type: <code>chrome://flags/#unsafely-treat-insecure-origin-as-secure</code></li>
            <li>Add this page's URL to the list</li>
            <li>Set to "Enabled"</li>
            <li>Restart Chrome</li>
        </ol>
    </div>

    <script>
        function updateStatus(message, type) {
            const status = document.getElementById('status');
            status.textContent = message;
            status.className = 'status ' + type;
        }

        async function testCamera() {
            try {
                updateStatus('Checking navigator.mediaDevices...', 'warning');
                
                if (!navigator.mediaDevices) {
                    updateStatus('❌ navigator.mediaDevices is undefined! This page needs HTTPS or localhost.', 'error');
                    return;
                }
                
                if (!navigator.mediaDevices.getUserMedia) {
                    updateStatus('❌ getUserMedia is not available in this browser.', 'error');
                    return;
                }
                
                updateStatus('✅ navigator.mediaDevices is available! Testing camera access...', 'success');
                
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    video: { facingMode: 'environment' }, 
                    audio: false 
                });
                
                const video = document.getElementById('video');
                video.srcObject = stream;
                video.style.display = 'block';
                
                updateStatus('🎉 Camera access successful! You can now use AdMyVision.', 'success');
                
            } catch (error) {
                updateStatus(`❌ Camera access failed: ${error.message}`, 'error');
                console.error('Camera test error:', error);
            }
        }

        // Auto-test on page load
        document.addEventListener('DOMContentLoaded', testCamera);
    </script>
</body>
</html>
"""
    
    test_file = Path("camera_test.html")
    test_file.write_text(html_content, encoding='utf-8')
    return test_file

def main():
    print("🎯 AdMyVision Camera Access Validator")
    print("=" * 50)
    
    # Get network information
    local_ip = get_local_ip()
    port = 8502
    
    print(f"📡 Local IP: {local_ip}")
    print(f"🌐 Port: {port}")
    print(f"🔗 App URL: http://{local_ip}:{port}")
    
    # Check if Streamlit is running
    if check_port_accessible(local_ip, port):
        print("✅ Streamlit app is running and accessible")
    else:
        print("❌ Streamlit app is not running on the expected port")
        print("   Start it with: python -m streamlit run pages/3_fixed_object_detection.py --server.port=8502")
    
    print("\n🔧 Camera Access Solutions:")
    print("-" * 30)
    
    print("\n1. 📱 Chrome Flags Method (RECOMMENDED):")
    print(f"   • Open Chrome on your phone")
    print(f"   • Go to: chrome://flags/#unsafely-treat-insecure-origin-as-secure")
    print(f"   • Add: http://{local_ip}:{port}")
    print(f"   • Set to 'Enabled' and restart Chrome")
    
    print("\n2. 🔒 HTTPS Method:")
    print(f"   • Run: start_https.bat (Windows) or use OpenSSL commands")
    print(f"   • Access via: https://{local_ip}:{port}")
    
    print("\n3. 🔌 USB Debugging Method:")
    print(f"   • Connect phone via USB")
    print(f"   • Enable USB debugging")
    print(f"   • Run: adb reverse tcp:{port} tcp:{port}")
    print(f"   • Access via: http://localhost:{port}")
    
    # Create and open test page
    print("\n🧪 Creating camera test page...")
    test_file = create_simple_test_page()
    test_url = f"http://{local_ip}:{port}/camera_test.html"
    
    print(f"📄 Test file created: {test_file}")
    print(f"🌐 Test this URL on your phone: file://{test_file.absolute()}")
    
    # Open test page in default browser
    try:
        webbrowser.open(f"file://{test_file.absolute()}")
        print("🌐 Opening test page in default browser...")
    except Exception as e:
        print(f"❌ Could not open browser: {e}")
    
    print("\nℹ️  Use the test page to verify camera access before using AdMyVision!")

if __name__ == "__main__":
    main()
