#!/usr/bin/env python3
"""
Standalone Object Detection Model Test
Tests MobileNet-SSD model functionality without Streamlit dependencies
"""

import cv2
import numpy as np
import os
import sys

def test_model_loading():
    """Test if the model files can be loaded correctly"""
    print("Testing model loading...")
    
    # Try different possible paths for model files
    possible_paths = [
        "MobileNetSSD_deploy.caffemodel",
        "models/MobileNetSSD_deploy.caffemodel",
        "./MobileNetSSD_deploy.caffemodel"
    ]
    
    prototxt_paths = [
        "MobileNetSSD_deploy.prototxt.txt",
        "models/MobileNetSSD_deploy.prototxt.txt", 
        "./MobileNetSSD_deploy.prototxt.txt"
    ]
    
    model_path = None
    prototxt_path = None
    
    # Find model file
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            print(f"✓ Found model file: {path}")
            break
    
    # Find prototxt file
    for path in prototxt_paths:
        if os.path.exists(path):
            prototxt_path = path
            print(f"✓ Found prototxt file: {path}")
            break
    
    if not model_path:
        print("✗ Model file not found!")
        return None, None
        
    if not prototxt_path:
        print("✗ Prototxt file not found!")
        return None, None
    
    try:
        # Load the DNN model
        net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        print("✓ Model loaded successfully!")
        return net, (model_path, prototxt_path)
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None, None

def test_detection_on_sample():
    """Test detection on a sample image"""
    print("\nTesting object detection...")
    
    net, paths = test_model_loading()
    if net is None:
        return False
    
    # Create a sample test image (colored rectangles)
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some colored rectangles to simulate objects
    cv2.rectangle(test_image, (100, 100), (200, 200), (0, 255, 0), -1)  # Green rectangle
    cv2.rectangle(test_image, (300, 200), (400, 350), (255, 0, 0), -1)  # Blue rectangle
    cv2.rectangle(test_image, (450, 50), (600, 150), (0, 0, 255), -1)   # Red rectangle
    
    try:
        # Prepare the image for detection
        blob = cv2.dnn.blobFromImage(
            test_image, 
            scalefactor=0.017, 
            size=(300, 300), 
            mean=(103.94, 116.78, 123.68),
            swapRB=False
        )
        
        net.setInput(blob)
        detections = net.forward()
        
        print(f"✓ Detection completed!")
        print(f"  Detection shape: {detections.shape}")
        print(f"  Number of detections: {detections.shape[2]}")
        
        # Analyze detections
        confidence_threshold = 0.3
        valid_detections = 0
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_threshold:
                valid_detections += 1
                class_id = int(detections[0, 0, i, 1])
                print(f"  Detection {i}: Class {class_id}, Confidence {confidence:.3f}")
        
        print(f"✓ Found {valid_detections} valid detections (confidence > {confidence_threshold})")
        return True
        
    except Exception as e:
        print(f"✗ Error during detection: {e}")
        return False

def test_webcam_detection():
    """Test detection with webcam (optional)"""
    print("\nTesting webcam detection (optional)...")
    
    net, paths = test_model_loading()
    if net is None:
        return False
    
    try:
        # Try to open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("✗ Could not open webcam")
            return False
        
        print("✓ Webcam opened successfully")
        
        # Test a few frames
        for frame_num in range(3):
            ret, frame = cap.read()
            if not ret:
                print(f"✗ Could not read frame {frame_num}")
                break
            
            # Prepare frame for detection
            blob = cv2.dnn.blobFromImage(
                frame, 
                scalefactor=0.017, 
                size=(300, 300), 
                mean=(103.94, 116.78, 123.68),
                swapRB=False
            )
            
            net.setInput(blob)
            detections = net.forward()
            
            # Count valid detections
            valid_detections = 0
            for i in range(detections.shape[2]):
                if detections[0, 0, i, 2] > 0.3:
                    valid_detections += 1
            
            print(f"  Frame {frame_num}: {valid_detections} detections")
        
        cap.release()
        print("✓ Webcam test completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Error during webcam test: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 50)
    print("MOBILENET-SSD MODEL TEST")
    print("=" * 50)
    
    # Test 1: Model Loading
    success1 = test_model_loading()[0] is not None
    
    # Test 2: Sample Detection
    success2 = test_detection_on_sample()
    
    # Test 3: Webcam (optional)
    print("\n" + "=" * 30)
    print("OPTIONAL: Webcam Test")
    print("Press Ctrl+C to skip webcam test")
    print("=" * 30)
    
    try:
        success3 = test_webcam_detection()
    except KeyboardInterrupt:
        print("\nWebcam test skipped by user")
        success3 = True  # Don't fail overall test for skipped webcam
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Model Loading: {'✓ PASS' if success1 else '✗ FAIL'}")
    print(f"Sample Detection: {'✓ PASS' if success2 else '✗ FAIL'}")
    print(f"Webcam Test: {'✓ PASS' if success3 else '✗ FAIL (optional)'}")
    
    overall_success = success1 and success2
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if overall_success else '✗ SOME TESTS FAILED'}")
    
    return overall_success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)
