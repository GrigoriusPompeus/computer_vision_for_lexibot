#!/usr/bin/env python3
"""
Camera test script for LexiBot Computer Vision System
Tests camera functionality and basic setup
"""

import cv2
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.config import CAMERA_INDEX, DEFAULT_FPS

def test_camera():
    """Test camera functionality"""
    print("🎥 LexiBot Camera Test")
    print("=" * 30)
    
    print(f"📷 Attempting to open camera {CAMERA_INDEX}...")
    
    # Initialize camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    if not cap.isOpened():
        print(f"❌ Failed to open camera {CAMERA_INDEX}")
        print("💡 Try adjusting CAMERA_INDEX in src/utils/config.py")
        return False
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"✅ Camera opened successfully!")
    print(f"📊 Resolution: {width}x{height}")
    print(f"🎬 FPS: {fps}")
    
    print(f"\n🔄 Testing frame capture...")
    
    frame_count = 0
    test_frames = 10
    
    for i in range(test_frames):
        ret, frame = cap.read()
        if ret:
            frame_count += 1
        else:
            print(f"⚠️  Frame {i+1} failed to capture")
    
    print(f"📈 Successfully captured {frame_count}/{test_frames} frames")
    
    if frame_count > 0:
        print(f"\n🖼️  Last frame info:")
        print(f"   Shape: {frame.shape}")
        print(f"   Data type: {frame.dtype}")
    
    # Test live preview
    print(f"\n👀 Starting live preview...")
    print("   Press ESC or 'q' to exit")
    print("   Press SPACE to capture a test frame")
    
    captured_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Failed to read frame")
            break
        
        # Add overlay text
        cv2.putText(frame, "LexiBot Camera Test", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Resolution: {width}x{height}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, "Press ESC or 'q' to exit, SPACE to capture", (10, height-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('LexiBot Camera Test', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or 'q'
            break
        elif key == ord(' '):  # SPACE
            captured_frames += 1
            filename = f"test_capture_{captured_frames}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Captured frame saved as {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ Camera test completed successfully!")
    print(f"📸 Total frames captured: {captured_frames}")
    
    return True

if __name__ == "__main__":
    try:
        success = test_camera()
        if success:
            print(f"\n🎉 Camera is ready for LexiBot Computer Vision System!")
        else:
            print(f"\n❌ Camera test failed. Please check your camera setup.")
            sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
