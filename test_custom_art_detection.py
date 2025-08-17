#!/usr/bin/env python3
"""
Museum Art Detection Test Script

Tests the custom trained model for detecting museum art pieces:
- Liberty Leading the People
- Mona Lisa
- The Scream
- The Starry Night
- Sunflowers
"""

import cv2
import sys
import os
from pathlib import Path

# Add current directory to path for imports (same as scripts approach)
sys.path.append(os.path.dirname(__file__))

from src.detection.detector import ArtworkDetector

def test_custom_art_detection():
    """Test custom model for art detection"""
    print("🎨 Starting Museum Art Detection Test")
    print("=" * 50)
    
    # Check if custom model exists
    model_path = Path("models/best.pt")
    if not model_path.exists():
        print("❌ Custom model not found!")
        print(f"Please copy your custom model to: {model_path}")
        print("\nExpected art classes:")
        print("  - Liberty Leading the People")
        print("  - Mona Lisa") 
        print("  - The Scream")
        print("  - The Starry Night")
        print("  - Sunflowers")
        return False
    
    try:
        # Initialize detector with custom model
        print("📥 Loading custom art detection model...")
        detector = ArtworkDetector(use_custom_model=True, confidence_threshold=0.1)  # Lower threshold for testing
        print("✅ Custom model loaded successfully!")
        
        # Initialize camera
        print("📷 Initializing camera...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return False
            
        print("✅ Camera initialized")
        print("\n🎯 Starting real-time art detection...")
        print("Show famous artworks to the camera!")
        print("Expected detections: Liberty, Mona Lisa, Scream, Starry Night, Sunflowers")
        print("Press 'q' to quit")
        print("=" * 50)
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error reading frame")
                break
                
            frame_count += 1
            
            # Run detection
            results = detector.detect_artworks(frame)
            
            if results and len(results) > 0:
                # Draw detections
                annotated_frame = detector.draw_detections(frame, results)
                
                # Print detections every 30 frames
                if frame_count % 30 == 0:
                    print(f"🎨 Frame {frame_count}: Detected {len(results)} artworks:")
                    for i, detection in enumerate(results):
                        label = detection['label']
                        confidence = detection['confidence']
                        distance = detection.get('distance_cm', 'N/A')
                        print(f"   {i+1}. {label} ({confidence:.2f}) - Distance: {distance}cm")
            else:
                annotated_frame = frame
                
            # Display frame
            cv2.imshow('Museum Art Detection', annotated_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"❌ Error during detection: {e}")
        return False
        
    finally:
        # Cleanup
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        print("\n🛑 Art detection stopped")
        print("=" * 50)
        
    return True

if __name__ == "__main__":
    print("🎨 Museum Art Detection for LexiBot")
    print("Custom model trained for famous artwork detection")
    print()
    
    success = test_custom_art_detection()
    
    if success:
        print("✅ Custom art detection test completed successfully!")
    else:
        print("❌ Custom art detection test failed!")
        print("\nTroubleshooting:")
        print("1. Make sure your custom model (.pt file) is in the models/ folder")
        print("2. Ensure the model was trained with the expected art classes")
        print("3. Check that your camera is working")
