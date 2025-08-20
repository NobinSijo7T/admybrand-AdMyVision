
<div align="center" style="margin-bottom: 24px;">
  <a href="https://admy-sense-web.vercel.app/" style="text-decoration: none; margin: 0 12px;">
    <img src="https://img.shields.io/badge/🌐%20Visit%20Website-AdMySense-1a73e8?style=for-the-badge&logo=vercel&logoColor=white&labelColor=22223b&color=1a73e8" alt="Visit Website"/>
  </a>
  <a href="https://nobinsijo7t-admybrand-admy-pages2-fixed-object-detection-dqerbk.streamlit.app/" style="text-decoration: none; margin: 0 12px;">
    <img src="https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit-e74c3c?style=for-the-badge&logo=streamlit&logoColor=white&labelColor=22223b&color=e74c3c" alt="Deployed App"/>
  </a>
</div>

# AdMySense (previously AdMyVision)
### *Advanced Detection for My Senses - Real-Time WebRTC Multi-Object Detection*

<div align="center">
  <img src="src/logo.png" alt="AdMySense Logo" width="200"/>
</div>

## Description

A cutting-edge real-time multi-object detection system that streams live video from your phone to browser via WebRTC, performs AI-powered inference, and overlays detection results in near real-time. This application supports both server-side and WASM on-device inference modes, providing flexible deployment options for different hardware constraints while maintaining low latency performance.

## 🚀 Quick Start (One Command)

### Default Mode (WASM - Low Resource)
```bash
./start.sh
```

### Server Mode (Higher Performance)
```bash
MODE=server ./start.sh
```

### Docker Compose
```bash
docker-compose up --build
```

## 📱 Phone Connection

1. Open `http://localhost:3000` on your laptop
2. Scan the displayed QR code with your phone
3. Allow camera permissions
4. View live detection overlays on both devices

**Connection Issues?** Use ngrok for remote access:
```bash
./start.sh --ngrok
```

## 🎯 Live Demo

 <a href="https://share.streamlit.io/whitphx/streamlit-webrtc-example/main/app.py">
    <img src="https://global.discourse-cdn.com/streamlit/original/2X/a/af111a7393c77cb69d7712ac8e71ca862feaeb24.gif" />
  </a>

*Demo showing real-time object detection with WebRTC streaming*

## 📊 Benchmarking

Run performance metrics collection:
```bash
./bench/run_bench.sh --duration 30 --mode wasm
./bench/run_bench.sh --duration 30 --mode server
```

Results will be saved to `metrics.json` with median & P95 latency, FPS, and bandwidth metrics.

## 🛠️ Technologies Used

| Category | Technology | Purpose |
|----------|------------|---------|
| **Web Framework** | Streamlit | Faster development, cross-platform compatibility, low resource usage |
| **Frontend** | HTML5, CSS3, JavaScript | Browser interface and WebRTC client |
| **Real-time Communication** | WebRTC, WebSocket | Live video streaming and data channels |
| **AI/ML Inference** | ONNX Runtime, TensorFlow.js | Object detection models |
| **Models** | MobileNet-SSD, YOLOv5n | Quantized detection models |
| **Backend** | Python 3.9+, Node.js | Server-side processing |
| **Containerization** | Docker, Docker Compose | Reproducible deployment |
| **Libraries** | aiortc, onnxruntime-web | WebRTC gateway and WASM inference |
| **Performance** | WASM, Quantization | Low-resource optimization |

## 📦 Installation Guide

### Linux Installation
```bash
chmod +x install.sh
./install.sh
```

### Windows Installation
```batch
start.bat
```

### Manual Installation
```bash
pip install -r requirements.txt
```

## 🔧 Build and Run Guide

To run the main object detection application:

```bash
python -m streamlit run pages/2_fixed_object_detection.py --server.headless true --server.port 8502
```

### Voice Announcement Feature
⚠️ **Note:** There is a bug that the voice announcement feature may not be available in the deployed link. This issue is fixed in the local running version.

To run the version with working voice announcements:
```bash
python -m streamlit run pages/3_fixed_object_detection.py --server.headless true --server.port 8502
```

## 🏗️ Architecture & Design

### Low-Resource Mode (WASM)
- **On-device inference** using ONNX Runtime Web or TensorFlow.js WASM
- **Input resolution**: 320×240 for optimal performance
- **Frame processing**: 10-15 FPS with adaptive sampling
- **Backpressure handling**: Frame queue with drop-old policy
- **Target hardware**: Intel i5, 8GB RAM (modest laptops)

### Server Mode
- **Server-side inference** with full-resolution processing
- **GPU acceleration** support (when available)
- **Higher accuracy** with larger models
- **WebRTC gateway** for video stream handling

## 🔧 API Contract

Detection results are sent via DataChannel/WebSocket in this format:

```json
{
  "frame_id": "string_or_int",
  "capture_ts": 1690000000000,
  "recv_ts": 1690000000100,
  "inference_ts": 1690000000120,
  "detections": [
    {
      "label": "person",
      "score": 0.93,
      "xmin": 0.12,
      "ymin": 0.08,
      "xmax": 0.34,
      "ymax": 0.67
    }
  ]
}
```

*Coordinates are normalized [0..1] for resolution independence*

## 📈 Performance Metrics

The system tracks and reports:
- **End-to-end latency**: `overlay_display_ts - capture_ts`
- **Server latency**: `inference_ts - recv_ts`
- **Network latency**: `recv_ts - capture_ts`
- **Processed FPS**: Detection frames per second
- **Bandwidth**: Uplink/downlink kbps

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| Phone won't connect | Ensure same network OR use `./start.sh --ngrok` |
| Misaligned overlays | Check timestamp units (ms) and frame alignment |
| High CPU usage | Reduce to 320×240 resolution or switch to WASM mode |
| Connection drops | Use Chrome `webrtc-internals` to inspect stats |



## 📋 Requirements

### Minimum System Requirements
- **CPU**: Intel i5 or equivalent
- **RAM**: 8GB
- **Browser**: Chrome (Android) or Safari (iOS)
- **Network**: Wi-Fi or mobile data

### Dependencies
- Docker & Docker Compose
- Node.js >=16
- Python 3.9+
- ONNX Runtime / TensorFlow.js

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

---

<div align="center">
  <p>Made with ❤️ by <a href="https://github.com/NobinSijo7T">NobinSijo7T</a></p>
</div>
