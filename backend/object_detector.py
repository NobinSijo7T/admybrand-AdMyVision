import cv2
import numpy as np
import os
import logging
from typing import List, Dict, Any
import asyncio

logger = logging.getLogger(__name__)

class ObjectDetector:
    """Object detection using OpenCV DNN with MobileNet-SSD"""
    
    def __init__(self):
        self.net = None
        self.classes = [
            "background", "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person",
            "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        ]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.mode = os.getenv("MODE", "server")
        self.confidence_threshold = 0.5
    
    async def initialize(self):
        """Initialize the object detection model"""
        try:
            model_path = "models/MobileNetSSD_deploy.caffemodel"
            config_path = "models/MobileNetSSD_deploy.prototxt.txt"
            
            if os.path.exists(model_path) and os.path.exists(config_path):
                self.net = cv2.dnn.readNetFromCaffe(config_path, model_path)
                logger.info("MobileNet-SSD model loaded successfully")
            else:
                logger.error("Model files not found. Please ensure model files are in the models/ directory")
                # For demo purposes, we'll create a dummy detector
                self.net = None
                logger.warning("Running in dummy mode - no actual detection will be performed")
                
        except Exception as e:
            logger.error(f"Error initializing object detector: {e}")
            self.net = None
    
    async def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Perform object detection on a frame"""
        try:
            if self.net is None:
                # Dummy detection for testing
                return self._dummy_detection(frame)
            
            # Resize frame for inference (320x240 for low-resource mode)
            if self.mode == "wasm":
                target_size = (320, 240)
            else:
                target_size = (300, 300)
            
            h, w = frame.shape[:2]
            
            # Create blob from image
            blob = cv2.dnn.blobFromImage(
                frame, 
                scalefactor=0.007843,
                size=target_size,
                mean=(127.5, 127.5, 127.5)
            )
            
            # Set input to the network
            self.net.setInput(blob)
            
            # Run inference
            detections = self.net.forward()
            
            # Process detections
            results = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                if confidence > self.confidence_threshold:
                    class_id = int(detections[0, 0, i, 1])
                    
                    # Get bounding box coordinates (normalized)
                    box = detections[0, 0, i, 3:7]
                    
                    # Ensure coordinates are in [0, 1] range
                    xmin = max(0, min(1, float(box[0])))
                    ymin = max(0, min(1, float(box[1])))
                    xmax = max(0, min(1, float(box[2])))
                    ymax = max(0, min(1, float(box[3])))
                    
                    # Create detection result
                    detection = {
                        "label": self.classes[class_id] if class_id < len(self.classes) else "unknown",
                        "score": float(confidence),
                        "xmin": xmin,
                        "ymin": ymin,
                        "xmax": xmax,
                        "ymax": ymax
                    }
                    results.append(detection)
            
            return results
            
        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return []
    
    def _dummy_detection(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Create dummy detections for testing when model is not available"""
        h, w = frame.shape[:2]
        
        # Create some fake detections for demonstration
        detections = [
            {
                "label": "person",
                "score": 0.85,
                "xmin": 0.1,
                "ymin": 0.1,
                "xmax": 0.4,
                "ymax": 0.8
            },
            {
                "label": "car",
                "score": 0.72,
                "xmin": 0.5,
                "ymin": 0.3,
                "xmax": 0.9,
                "ymax": 0.7
            }
        ]
        
        return detections
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "model_loaded": self.net is not None,
            "mode": self.mode,
            "classes_count": len(self.classes),
            "confidence_threshold": self.confidence_threshold
        }
