"""
Simple HTTP server to serve the Chrome setup guide
Run this if Streamlit is having issues
"""

import http.server
import socketserver
import webbrowser
import socket
from pathlib import Path

def get_local_ip():
    """Get the local IP address"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "localhost"

def serve_setup_guide():
    """Serve the Chrome setup guide on a simple HTTP server"""
    
    PORT = 8503  # Use a different port from Streamlit
    local_ip = get_local_ip()
    
    # Change to the directory containing the HTML files
    web_dir = Path(__file__).parent
    
    class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=web_dir, **kwargs)
    
    with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        print(f"🌐 Chrome Setup Guide Server Running!")
        print(f"📱 Access on phone: http://{local_ip}:{PORT}/chrome_setup_guide.html")
        print(f"💻 Access on laptop: http://localhost:{PORT}/chrome_setup_guide.html")
        print(f"🧪 Camera test page: http://{local_ip}:{PORT}/mobile_camera_test.html")
        print(f"\n🔧 Instructions:")
        print(f"1. Open http://{local_ip}:{PORT}/chrome_setup_guide.html on your phone")
        print(f"2. Follow the Chrome flags setup instructions")
        print(f"3. Then access the main AdMyVision app at http://{local_ip}:8502")
        print(f"\nPress Ctrl+C to stop the server")
        
        # Try to open the setup guide in the default browser
        try:
            webbrowser.open(f"http://localhost:{PORT}/chrome_setup_guide.html")
        except:
            pass
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Setup guide server stopped")

if __name__ == "__main__":
    serve_setup_guide()
