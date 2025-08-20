import asyncio
import json
import logging
import os
import time
from typing import Dict, List, Optional
import uuid
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import qrcode
from io import BytesIO
import base64

from .webrtc_handler import WebRTCHandler
from .object_detector import ObjectDetector
from .metrics_collector import MetricsCollector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global components
webrtc_handler = WebRTCHandler()
object_detector = ObjectDetector()
metrics_collector = MetricsCollector()

# Store active connections
active_connections: Dict[str, WebSocket] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await object_detector.initialize()
    logger.info("WebRTC Object Detection Server started")
    yield
    # Shutdown
    logger.info("WebRTC Object Detection Server shutting down")

app = FastAPI(title="WebRTC Object Detection API", lifespan=lifespan)

@app.get("/")
async def get_index():
    """Serve the main application page"""
    with open("frontend/dist/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.get("/qr")
async def get_qr_code():
    """Generate QR code for phone connection"""
    # Get the server URL (in production, this would be the public URL)
    server_url = os.getenv("SERVER_URL", "http://localhost:8000")
    
    # Generate QR code
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(server_url)
    qr.make(fit=True)
    
    # Create QR code image
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Convert to base64
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    qr_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    return JSONResponse({
        "qr_code": f"data:image/png;base64,{qr_base64}",
        "url": server_url
    })

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time communication"""
    await websocket.accept()
    active_connections[client_id] = websocket
    logger.info(f"Client {client_id} connected")
    
    try:
        while True:
            # Receive frame data from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "frame":
                await handle_frame(client_id, message, websocket)
            elif message["type"] == "offer":
                await handle_webrtc_offer(client_id, message, websocket)
            elif message["type"] == "answer":
                await handle_webrtc_answer(client_id, message, websocket)
            elif message["type"] == "ice_candidate":
                await handle_ice_candidate(client_id, message, websocket)
                
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
    finally:
        if client_id in active_connections:
            del active_connections[client_id]

async def handle_frame(client_id: str, message: dict, websocket: WebSocket):
    """Process incoming video frame for object detection"""
    try:
        recv_ts = int(time.time() * 1000)
        frame_id = message.get("frame_id")
        capture_ts = message.get("capture_ts")
        frame_data = message.get("frame_data")
        
        # Decode frame data (base64 encoded image)
        frame_bytes = base64.b64decode(frame_data.split(',')[1])
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Run object detection
        inference_start = int(time.time() * 1000)
        detections = await object_detector.detect(frame)
        inference_ts = int(time.time() * 1000)
        
        # Prepare response
        response = {
            "frame_id": frame_id,
            "capture_ts": capture_ts,
            "recv_ts": recv_ts,
            "inference_ts": inference_ts,
            "detections": detections
        }
        
        # Send detection results back to client
        await websocket.send_text(json.dumps({
            "type": "detection_results",
            "data": response
        }))
        
        # Collect metrics
        metrics_collector.add_frame_metrics(
            frame_id=frame_id,
            capture_ts=capture_ts,
            recv_ts=recv_ts,
            inference_ts=inference_ts,
            detections_count=len(detections)
        )
        
    except Exception as e:
        logger.error(f"Error processing frame: {e}")

async def handle_webrtc_offer(client_id: str, message: dict, websocket: WebSocket):
    """Handle WebRTC offer for peer connection setup"""
    try:
        offer = message.get("offer")
        answer = await webrtc_handler.handle_offer(client_id, offer)
        
        await websocket.send_text(json.dumps({
            "type": "answer",
            "answer": answer
        }))
    except Exception as e:
        logger.error(f"Error handling WebRTC offer: {e}")

async def handle_webrtc_answer(client_id: str, message: dict, websocket: WebSocket):
    """Handle WebRTC answer"""
    try:
        answer = message.get("answer")
        await webrtc_handler.handle_answer(client_id, answer)
    except Exception as e:
        logger.error(f"Error handling WebRTC answer: {e}")

async def handle_ice_candidate(client_id: str, message: dict, websocket: WebSocket):
    """Handle ICE candidate for WebRTC connection"""
    try:
        candidate = message.get("candidate")
        await webrtc_handler.handle_ice_candidate(client_id, candidate)
    except Exception as e:
        logger.error(f"Error handling ICE candidate: {e}")

@app.get("/metrics")
async def get_metrics():
    """Get current performance metrics"""
    return metrics_collector.get_metrics()

@app.post("/metrics/export")
async def export_metrics():
    """Export metrics to JSON file"""
    metrics = metrics_collector.export_metrics()
    
    # Save to file
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    return JSONResponse({"message": "Metrics exported to metrics/metrics.json"})

# Mount static files
app.mount("/static", StaticFiles(directory="frontend/dist"), name="static")

if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment
    mode = os.getenv("MODE", "server")
    host = "0.0.0.0"
    port = 8001
    
    logger.info(f"Starting server in {mode} mode on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
