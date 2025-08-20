"""Test script to verify MobileNet-SSD object detection is working."""

import cv2
import numpy as np
from pathlib import Path

# Paths
ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models/MobileNetSSD_deploy.caffemodel"
PROTOTXT_PATH = ROOT / "models/MobileNetSSD_deploy.prototxt.txt"
PROTOTXT_ALT_PATH = ROOT / "models/MobileNetSSD_deploy.prototxt"

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

def test_model():
    """Test the object detection model."""
    print("🔍 Testing MobileNet-SSD Object Detection Model")
    print("=" * 50)
    
    # Check file existence
    print(f"📁 Model file exists: {MODEL_PATH.exists()}")
    print(f"📁 Prototxt file exists: {PROTOTXT_PATH.exists()}")
    print(f"📁 Prototxt alt exists: {PROTOTXT_ALT_PATH.exists()}")
    
    # Choose prototxt file
    prototxt_path = PROTOTXT_PATH if PROTOTXT_PATH.exists() else PROTOTXT_ALT_PATH
    print(f"📁 Using prototxt: {prototxt_path}")
    
    try:
        # Load model
        print("\n🚀 Loading model...")
        net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(MODEL_PATH))
        print("✅ Model loaded successfully!")
        
        # Create test image (simple pattern)
        print("\n🖼️ Creating test image...")
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw some simple shapes that might be detected
        cv2.rectangle(test_image, (100, 100), (200, 200), (255, 255, 255), -1)  # White square
        cv2.circle(test_image, (400, 200), 50, (0, 255, 0), -1)  # Green circle
        cv2.rectangle(test_image, (300, 300), (500, 400), (0, 0, 255), -1)  # Red rectangle
        
        print("✅ Test image created!")
        
        # Prepare input
        print("\n🔧 Preparing model input...")
        blob = cv2.dnn.blobFromImage(
            image=test_image,
            scalefactor=0.007843,
            size=(300, 300),
            mean=(127.5, 127.5, 127.5),
            swapRB=False,
            crop=False
        )
        print(f"✅ Blob shape: {blob.shape}")
        
        # Run inference
        print("\n🎯 Running inference...")
        net.setInput(blob)
        output = net.forward()
        print(f"✅ Output shape: {output.shape}")
        
        # Process results
        print("\n📊 Processing results...")
        output = output.squeeze()
        print(f"Output after squeeze: {output.shape}")
        
        detection_count = 0
        if len(output.shape) == 2 and output.shape[1] >= 7:
            for i, detection in enumerate(output):
                confidence = float(detection[2])
                class_id = int(detection[1])
                
                if confidence > 0.1 and class_id < len(CLASSES) and class_id > 0:
                    detection_count += 1
                    class_name = CLASSES[class_id]
                    print(f"  🔍 Detection {i}: {class_name} ({confidence:.3f})")
        
        print(f"\n📈 Total valid detections: {detection_count}")
        
        # Test with webcam frame if available
        print("\n📷 Testing with webcam...")
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("✅ Webcam frame captured!")
                
                # Process webcam frame
                blob = cv2.dnn.blobFromImage(
                    image=frame,
                    scalefactor=0.007843,
                    size=(300, 300),
                    mean=(127.5, 127.5, 127.5),
                    swapRB=False,
                    crop=False
                )
                net.setInput(blob)
                output = net.forward()
                output = output.squeeze()
                
                webcam_detections = 0
                if len(output.shape) == 2:
                    for detection in output:
                        confidence = float(detection[2])
                        class_id = int(detection[1])
                        if confidence > 0.3 and class_id < len(CLASSES) and class_id > 0:
                            webcam_detections += 1
                            print(f"  📷 Webcam detection: {CLASSES[class_id]} ({confidence:.3f})")
                
                print(f"📈 Webcam detections: {webcam_detections}")
            cap.release()
        else:
            print("❌ Could not access webcam")
        
        print("\n🎉 Model test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        return False

if __name__ == "__main__":
    test_model()
