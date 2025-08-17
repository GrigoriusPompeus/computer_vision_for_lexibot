#!/usr/bin/env python3
"""
Real-time artwork detection script for LexiBot Computer Vision System
Enhanced version of the original test.py with improved structure and features
"""

import cv2
import os
import sys
import time
import logging
from typing import Tuple

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: YOLO not available. Install with: pip install ultralytics")

from src.utils.config import (
    CAMERA_INDEX, MODEL_PATH, CUSTOM_MODEL_PATH, USE_CUSTOM_MODEL, CONFIDENCE_THRESHOLD, 
    OUTPUT_DIR, TEMP_OUTPUT_FILE, EXIT_BUTTON_CONFIG,
    KNOWN_WIDTHS, FOCAL_LENGTH
)
from src.detection.distance_estimator import DistanceEstimator


class RealTimeArtworkDetection:
    """
    Real-time artwork detection system with enhanced UI and features.
    """
    
    def __init__(self):
        """Initialize the detection system."""
        self.model = None
        self.cap = None
        self.out = None
        self.distance_estimator = DistanceEstimator()
        
        # UI elements
        self.exit_button = None
        self.frame_width = 0
        self.frame_height = 0
        self.exit_requested = False  # Add exit flag
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load YOLO model
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model for detection."""
        if not YOLO_AVAILABLE:
            self.logger.error("YOLO not available - detection disabled")
            return
        
        try:
            # Use custom model if available and configured
            model_path = CUSTOM_MODEL_PATH if USE_CUSTOM_MODEL else MODEL_PATH
            self.model = YOLO(model_path)
            self.logger.info(f"Loaded YOLO model: {model_path}")
            if USE_CUSTOM_MODEL:
                self.logger.info("Using custom art detection model!")
        except Exception as e:
            self.logger.error(f"Error loading YOLO model: {e}")
            sys.exit(1)
    
    def _setup_camera(self) -> bool:
        """
        Setup camera capture.
        
        Returns:
            True if camera setup successful, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(CAMERA_INDEX)
            if not self.cap.isOpened():
                self.logger.error(f"Could not open camera {CAMERA_INDEX}")
                return False
            
            # Get camera properties
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 20.0
            
            self.logger.info(f"Camera setup: {self.frame_width}x{self.frame_height} @ {fps}fps")
            
            # Setup video writer
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            output_path = os.path.join(OUTPUT_DIR, TEMP_OUTPUT_FILE)
            
            self.out = cv2.VideoWriter(
                output_path, 
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps, 
                (self.frame_width, self.frame_height)
            )
            
            # Setup exit button coordinates
            self._setup_exit_button()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting up camera: {e}")
            return False
    
    def _setup_exit_button(self):
        """Setup exit button coordinates."""
        self.exit_button = {
            'x1': self.frame_width - EXIT_BUTTON_CONFIG['width'] - EXIT_BUTTON_CONFIG['margin_x'],
            'y1': EXIT_BUTTON_CONFIG['margin_y'],
            'x2': self.frame_width - EXIT_BUTTON_CONFIG['margin_x'],
            'y2': EXIT_BUTTON_CONFIG['margin_y'] + EXIT_BUTTON_CONFIG['height']
        }
    
    def _check_exit_button(self, x: int, y: int) -> bool:
        """
        Check if coordinates are within exit button bounds.
        
        Args:
            x, y: Coordinate to check
            
        Returns:
            True if coordinates are within exit button
        """
        return (self.exit_button['x1'] <= x <= self.exit_button['x2'] and 
                self.exit_button['y1'] <= y <= self.exit_button['y2'])
    
    def _on_mouse(self, event, x: int, y: int, flags, param):
        """
        Mouse callback function.
        
        Args:
            event: Mouse event type
            x, y: Mouse coordinates
            flags: Event flags
            param: Additional parameters
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.logger.info(f"Mouse clicked at ({x}, {y})")
            if self._check_exit_button(x, y):
                self.logger.info("EXIT BUTTON CLICKED!")
                self.exit_requested = True
    
    def _draw_ui_elements(self, frame):
        """
        Draw UI elements on frame.
        
        Args:
            frame: Input frame to draw on
            
        Returns:
            Frame with UI elements drawn
        """
        # Draw exit button
        cv2.rectangle(frame, 
                     (self.exit_button['x1'], self.exit_button['y1']),
                     (self.exit_button['x2'], self.exit_button['y2']),
                     EXIT_BUTTON_CONFIG['color'], -1)
        
        # Draw exit button text
        cv2.putText(frame, "EXIT",
                   (self.exit_button['x1'] + 25, self.exit_button['y2'] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   EXIT_BUTTON_CONFIG['font_scale'],
                   EXIT_BUTTON_CONFIG['text_color'],
                   EXIT_BUTTON_CONFIG['thickness'])
        
        # Draw title
        cv2.putText(frame, "LexiBot Artwork Detection",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, (255, 255, 255), 2)
        
        # Draw instructions
        cv2.putText(frame, "Press ESC to exit",
                   (10, self.frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (200, 200, 200), 1)
        
        return frame
    
    def _process_detections(self, frame):
        """
        Process detections on frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with detections drawn
        """
        if self.model is None:
            return frame
        
        try:
            # Run detection
            results = self.model(frame, conf=CONFIDENCE_THRESHOLD)
            
            # Process each detection
            for box in results[0].boxes:
                # Extract box information
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                label = self.model.names[class_id]
                
                # Calculate distance
                bbox = (x1, y1, x2, y2)
                distance = self.distance_estimator.estimate_distance(bbox, label)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label with distance
                text = f"{label} {confidence:.2f} {distance:.1f}cm"
                cv2.putText(frame, text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
        except Exception as e:
            self.logger.error(f"Error processing detections: {e}")
        
        return frame
    
    def run(self):
        """Main detection loop."""
        if not self._setup_camera():
            self.logger.error("Failed to setup camera")
            return
        
        # Setup window and mouse callback
        cv2.namedWindow("LexiBot Artwork Detection")
        cv2.setMouseCallback("LexiBot Artwork Detection", self._on_mouse)
        
        self.logger.info("Starting real-time detection...")
        self.logger.info("Press ESC to exit or click EXIT button")
        
        try:
            frame_count = 0
            start_time = time.time()
            
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.warning("Failed to read frame from camera")
                    break
                
                # Process detections
                frame = self._process_detections(frame)
                
                # Draw UI elements
                frame = self._draw_ui_elements(frame)
                
                # Show frame
                cv2.imshow("LexiBot Artwork Detection", frame)
                
                # Save frame to video
                if self.out:
                    self.out.write(frame)
                
                # Update frame counter
                frame_count += 1
                
                # Display FPS every 30 frames
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    self.logger.info(f"Processing at {fps:.1f} FPS")
                
                # Check for exit
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    self.logger.info("ESC key pressed - exiting")
                    break
                
                # Check if exit button was clicked
                if self.exit_requested:
                    self.logger.info("Exit button clicked - exiting")
                    break
                    
        except KeyboardInterrupt:
            self.logger.info("Detection stopped by user (Ctrl+C)")
        except Exception as e:
            self.logger.error(f"Unexpected error in detection loop: {e}")
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Cleanup resources."""
        if self.cap:
            self.cap.release()
        if self.out:
            self.out.release()
        cv2.destroyAllWindows()
        self.logger.info("Cleanup completed")


def main():
    """Main function."""
    print("LexiBot Computer Vision System - Real-Time Artwork Detection")
    print("=" * 60)
    
    # Check dependencies
    if not YOLO_AVAILABLE:
        print("Error: YOLOv11 not available. Install with: pip install ultralytics")
        sys.exit(1)
    
    # Create and run detector
    detector = RealTimeArtworkDetection()
    detector.run()


if __name__ == "__main__":
    main()
