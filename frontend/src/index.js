// WebRTC Real-time Object Detection Frontend
class WebRTCObjectDetection {
    constructor() {
        this.ws = null;
        this.pc = null;
        this.localStream = null;
        this.isDetectionActive = false;
        this.frameCount = 0;
        this.detectionCount = 0;
        this.clientId = this.generateClientId();
        
        // Metrics tracking
        this.metrics = {
            totalFrames: 0,
            totalDetections: 0,
            latencies: [],
            startTime: Date.now()
        };
        
        this.initializeElements();
        this.setupEventListeners();
        this.loadQRCode();
    }
    
    generateClientId() {
        return 'client_' + Math.random().toString(36).substr(2, 9);
    }
    
    initializeElements() {
        this.elements = {
            startBtn: document.getElementById('startBtn'),
            stopBtn: document.getElementById('stopBtn'),
            exportMetricsBtn: document.getElementById('exportMetricsBtn'),
            status: document.getElementById('status'),
            remoteVideo: document.getElementById('remoteVideo'),
            overlay: document.getElementById('overlay'),
            qrCode: document.getElementById('qrCode'),
            connectionUrl: document.getElementById('connectionUrl'),
            fpsValue: document.getElementById('fpsValue'),
            latencyValue: document.getElementById('latencyValue'),
            detectionsValue: document.getElementById('detectionsValue'),
            framesValue: document.getElementById('framesValue'),
            detectionList: document.getElementById('detectionList')
        };
    }
    
    setupEventListeners() {
        this.elements.startBtn.addEventListener('click', () => this.startDetection());
        this.elements.stopBtn.addEventListener('click', () => this.stopDetection());
        this.elements.exportMetricsBtn.addEventListener('click', () => this.exportMetrics());
        
        // Update metrics every second
        setInterval(() => this.updateMetricsDisplay(), 1000);
    }
    
    async loadQRCode() {
        try {
            const response = await fetch('/qr');
            const data = await response.json();
            
            this.elements.qrCode.innerHTML = `<img src="${data.qr_code}" alt="QR Code" style="max-width: 200px;">`;
            this.elements.connectionUrl.innerHTML = `<p><strong>URL:</strong> <a href="${data.url}" target="_blank">${data.url}</a></p>`;
        } catch (error) {
            console.error('Error loading QR code:', error);
            this.elements.qrCode.innerHTML = 'Error loading QR code';
        }
    }
    
    async startDetection() {
        try {
            this.updateStatus('Connecting...', 'disconnected');
            
            // Connect WebSocket
            await this.connectWebSocket();
            
            // Setup WebRTC if on phone
            if (this.isMobileDevice()) {
                await this.setupWebRTC();
            }
            
            this.isDetectionActive = true;
            this.elements.startBtn.disabled = true;
            this.elements.stopBtn.disabled = false;
            
            this.updateStatus('Connected - Detection Active', 'connected');
            
        } catch (error) {
            console.error('Error starting detection:', error);
            this.updateStatus('Connection Failed', 'disconnected');
        }
    }
    
    async stopDetection() {
        this.isDetectionActive = false;
        
        if (this.ws) {
            this.ws.close();
        }
        
        if (this.pc) {
            this.pc.close();
        }
        
        if (this.localStream) {
            this.localStream.getTracks().forEach(track => track.stop());
        }
        
        this.elements.startBtn.disabled = false;
        this.elements.stopBtn.disabled = true;
        this.updateStatus('Disconnected', 'disconnected');
    }
    
    async connectWebSocket() {
        return new Promise((resolve, reject) => {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/${this.clientId}`;
            
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                console.log('WebSocket connected');
                resolve();
            };
            
            this.ws.onmessage = (event) => {
                this.handleWebSocketMessage(JSON.parse(event.data));
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                reject(error);
            };
            
            this.ws.onclose = () => {
                console.log('WebSocket disconnected');
                this.updateStatus('Disconnected', 'disconnected');
            };
        });
    }
    
    async setupWebRTC() {
        try {
            // Get user media (camera)
            this.localStream = await navigator.mediaDevices.getUserMedia({
                video: { 
                    width: { ideal: 640 }, 
                    height: { ideal: 480 },
                    facingMode: 'environment' // Use back camera on mobile
                },
                audio: false
            });
            
            // Setup peer connection
            this.pc = new RTCPeerConnection({
                iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
            });
            
            // Add local stream to peer connection
            this.localStream.getTracks().forEach(track => {
                this.pc.addTrack(track, this.localStream);
            });
            
            // Handle ICE candidates
            this.pc.onicecandidate = (event) => {
                if (event.candidate) {
                    this.sendWebSocketMessage({
                        type: 'ice_candidate',
                        candidate: event.candidate
                    });
                }
            };
            
            // Handle remote stream
            this.pc.ontrack = (event) => {
                this.elements.remoteVideo.srcObject = event.streams[0];
            };
            
            // Create offer
            const offer = await this.pc.createOffer();
            await this.pc.setLocalDescription(offer);
            
            // Send offer to server
            this.sendWebSocketMessage({
                type: 'offer',
                offer: {
                    type: offer.type,
                    sdp: offer.sdp
                }
            });
            
            // Start sending frames for detection
            this.startFrameCapture();
            
        } catch (error) {
            console.error('Error setting up WebRTC:', error);
            throw error;
        }
    }
    
    startFrameCapture() {
        if (!this.localStream) return;
        
        const video = document.createElement('video');
        video.srcObject = this.localStream;
        video.play();
        
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            
            // Capture frames at 15 FPS
            const captureInterval = setInterval(() => {
                if (!this.isDetectionActive) {
                    clearInterval(captureInterval);
                    return;
                }
                
                // Draw video frame to canvas
                ctx.drawImage(video, 0, 0);
                
                // Convert to base64
                const frameData = canvas.toDataURL('image/jpeg', 0.8);
                const frameId = `frame_${Date.now()}_${this.frameCount++}`;
                const captureTs = Date.now();
                
                // Send frame for detection
                this.sendWebSocketMessage({
                    type: 'frame',
                    frame_id: frameId,
                    capture_ts: captureTs,
                    frame_data: frameData
                });
                
                this.metrics.totalFrames++;
                
            }, 1000 / 15); // 15 FPS
        };
    }
    
    handleWebSocketMessage(message) {
        switch (message.type) {
            case 'answer':
                this.handleWebRTCAnswer(message.answer);
                break;
                
            case 'detection_results':
                this.handleDetectionResults(message.data);
                break;
                
            default:
                console.log('Unknown message type:', message.type);
        }
    }
    
    async handleWebRTCAnswer(answer) {
        if (this.pc) {
            await this.pc.setRemoteDescription(new RTCSessionDescription(answer));
        }
    }
    
    handleDetectionResults(data) {
        try {
            // Calculate latency
            const displayTs = Date.now();
            const e2eLatency = displayTs - data.capture_ts;
            
            // Update metrics
            this.metrics.latencies.push(e2eLatency);
            this.metrics.totalDetections += data.detections.length;
            
            // Keep only last 100 latency measurements
            if (this.metrics.latencies.length > 100) {
                this.metrics.latencies = this.metrics.latencies.slice(-100);
            }
            
            // Display detections
            this.displayDetections(data.detections);
            this.updateDetectionList(data.detections);
            
        } catch (error) {
            console.error('Error handling detection results:', error);
        }
    }
    
    displayDetections(detections) {
        // Clear previous overlays
        this.elements.overlay.innerHTML = '';
        
        const video = this.elements.remoteVideo;
        const videoRect = video.getBoundingClientRect();
        
        // Set overlay dimensions to match video
        this.elements.overlay.style.width = videoRect.width + 'px';
        this.elements.overlay.style.height = videoRect.height + 'px';
        
        detections.forEach((detection, index) => {
            const color = this.getDetectionColor(detection.label);
            
            // Convert normalized coordinates to pixel coordinates
            const x = detection.xmin * videoRect.width;
            const y = detection.ymin * videoRect.height;
            const width = (detection.xmax - detection.xmin) * videoRect.width;
            const height = (detection.ymax - detection.ymin) * videoRect.height;
            
            // Create bounding box
            const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            rect.setAttribute('x', x);
            rect.setAttribute('y', y);
            rect.setAttribute('width', width);
            rect.setAttribute('height', height);
            rect.setAttribute('fill', 'none');
            rect.setAttribute('stroke', color);
            rect.setAttribute('stroke-width', '2');
            
            // Create label
            const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            text.setAttribute('x', x);
            text.setAttribute('y', y - 5);
            text.setAttribute('fill', color);
            text.setAttribute('font-size', '14');
            text.setAttribute('font-weight', 'bold');
            text.textContent = `${detection.label} (${Math.round(detection.score * 100)}%)`;
            
            this.elements.overlay.appendChild(rect);
            this.elements.overlay.appendChild(text);
        });
    }
    
    updateDetectionList(detections) {
        const timestamp = new Date().toLocaleTimeString();
        const listItem = document.createElement('div');
        listItem.innerHTML = `
            <strong>${timestamp}:</strong> 
            ${detections.map(d => `${d.label} (${Math.round(d.score * 100)}%)`).join(', ') || 'No detections'}
        `;
        
        this.elements.detectionList.insertBefore(listItem, this.elements.detectionList.firstChild);
        
        // Keep only last 10 detection results
        while (this.elements.detectionList.children.length > 10) {
            this.elements.detectionList.removeChild(this.elements.detectionList.lastChild);
        }
    }
    
    updateMetricsDisplay() {
        const elapsed = (Date.now() - this.metrics.startTime) / 1000;
        const fps = this.metrics.totalFrames / elapsed;
        const avgLatency = this.metrics.latencies.length > 0 
            ? this.metrics.latencies.reduce((a, b) => a + b, 0) / this.metrics.latencies.length 
            : 0;
        
        this.elements.fpsValue.textContent = fps.toFixed(1);
        this.elements.latencyValue.textContent = Math.round(avgLatency);
        this.elements.detectionsValue.textContent = this.metrics.totalDetections;
        this.elements.framesValue.textContent = this.metrics.totalFrames;
    }
    
    async exportMetrics() {
        try {
            const response = await fetch('/metrics/export', { method: 'POST' });
            const result = await response.json();
            alert('Metrics exported to metrics/metrics.json');
        } catch (error) {
            console.error('Error exporting metrics:', error);
            alert('Error exporting metrics');
        }
    }
    
    sendWebSocketMessage(message) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(message));
        }
    }
    
    updateStatus(message, type) {
        this.elements.status.textContent = message;
        this.elements.status.className = `status ${type}`;
    }
    
    getDetectionColor(label) {
        // Simple hash function to generate consistent colors for labels
        let hash = 0;
        for (let i = 0; i < label.length; i++) {
            hash = label.charCodeAt(i) + ((hash << 5) - hash);
        }
        const hue = Math.abs(hash) % 360;
        return `hsl(${hue}, 70%, 50%)`;
    }
    
    isMobileDevice() {
        return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.webrtcApp = new WebRTCObjectDetection();
});
