#!/usr/bin/env python3
"""
MQTT-enabled artwork detection script for LexiBot robot integration
Enhanced version of the original model_with_mqtt.py
"""

import cv2
import os
import sys
import json
import time
import logging

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: YOLO not available. Install with: pip install ultralytics")

from src.detection.mqtt_client import MQTTClient
from src.detection.distance_estimator import DistanceEstimator
from src.utils.config import (
    CAMERA_INDEX, CUSTOM_MODEL_PATH, CONFIDENCE_THRESHOLD,
    MQTT_BROKER, MQTT_PORT, MQTT_TOPIC
)


class MQTTArtworkDetection:
    """
    MQTT-enabled artwork detection system for robot integration.
    """
    
    def __init__(self, use_custom_model: bool = True):
        """
        Initialize the MQTT detection system.
        
        Args:
            use_custom_model: Whether to use custom trained model
        """
        self.model = None
        self.cap = None
        self.mqtt_client = MQTTClient()
        self.distance_estimator = DistanceEstimator()
        self.use_custom_model = use_custom_model
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model for detection."""
        if not YOLO_AVAILABLE:
            self.logger.error("YOLO not available - detection disabled")
            return
        
        try:
            if self.use_custom_model and os.path.exists(CUSTOM_MODEL_PATH):
                self.model = YOLO(CUSTOM_MODEL_PATH)
                self.logger.info(f"Loaded custom model: {CUSTOM_MODEL_PATH}")
            else:
                # Fallback to standard YOLOv11 model
                self.model = YOLO('yolo11l.pt')
                self.logger.info("Loaded standard YOLOv11 model")
                
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            sys.exit(1)
    
    def _setup_camera(self) -> bool:
        """
        Setup camera capture.
        
        Returns:
            True if setup successful, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(CAMERA_INDEX)
            if not self.cap.isOpened():
                self.logger.error(f"Could not open camera {CAMERA_INDEX}")
                return False
            
            # Get camera properties
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 20.0
            
            self.logger.info(f"Camera setup: {width}x{height} @ {fps}fps")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting up camera: {e}")
            return False
    
    def _setup_mqtt(self) -> bool:
        """
        Setup MQTT connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            if self.mqtt_client.connect():
                self.logger.info(f"Connected to MQTT broker: {MQTT_BROKER}:{MQTT_PORT}")
                self.logger.info(f"Publishing to topic: {MQTT_TOPIC}")
                return True
            else:
                self.logger.warning("MQTT connection failed - continuing without MQTT")
                return False
        except Exception as e:
            self.logger.error(f"Error setting up MQTT: {e}")
            return False
    
    def _process_frame(self, frame):
        """
        Process single frame for detections.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (processed_frame, detections_list)
        """
        detections = []
        
        if self.model is None:
            return frame, detections
        
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
                
                # Draw detection on frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}",
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                           0.6, (0, 255, 0), 2)
                
                # Create detection record
                detection = {
                    "label": label,
                    "confidence": round(confidence, 3),
                    "bbox": [x1, y1, x2, y2],
                    "distance_cm": distance,
                    "class_id": class_id,
                    "timestamp": time.time()
                }
                
                detections.append(detection)
                
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
        
        return frame, detections
    
    def _publish_detections(self, detections):
        """
        Publish detections to MQTT.
        
        Args:
            detections: List of detection dictionaries
        """
        if detections and self.mqtt_client.is_connected():
            try:
                success = self.mqtt_client.publish_detections(detections)
                if success:
                    self.logger.debug(f"Published {len(detections)} detections to MQTT")
                else:
                    self.logger.warning("Failed to publish detections to MQTT")
            except Exception as e:
                self.logger.error(f"Error publishing to MQTT: {e}")
    
    def run(self):
        """Main detection and MQTT publishing loop."""
        # Setup camera
        if not self._setup_camera():
            self.logger.error("Failed to setup camera")
            return
        
        # Setup MQTT (optional)
        mqtt_connected = self._setup_mqtt()
        if not mqtt_connected:
            self.logger.warning("Continuing without MQTT - detections will not be published")
        
        self.logger.info("Starting MQTT artwork detection...")
        self.logger.info("Press ESC to exit")
        
        try:
            frame_count = 0
            detection_count = 0
            start_time = time.time()
            
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.warning("Failed to read frame from camera")
                    break
                
                # Process frame for detections
                processed_frame, detections = self._process_frame(frame)
                
                # Publish detections to MQTT
                if detections:
                    self._publish_detections(detections)
                    detection_count += len(detections)
                
                # Display frame
                cv2.imshow("LexiBot MQTT Detection", processed_frame)
                
                frame_count += 1
                
                # Log statistics every 100 frames
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    det_rate = detection_count / elapsed
                    self.logger.info(f"Stats: {fps:.1f} FPS, {det_rate:.1f} detections/sec")
                
                # Check for exit
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    self.logger.info("ESC key pressed - exiting")
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
        cv2.destroyAllWindows()
        if self.mqtt_client:
            self.mqtt_client.disconnect()
        self.logger.info("Cleanup completed")


def main():
    """Main function."""
    print("LexiBot Computer Vision System - MQTT Artwork Detection")
    print("=" * 60)
    
    # Check dependencies
    if not YOLO_AVAILABLE:
        print("Error: YOLOv11 not available. Install with: pip install ultralytics")
        sys.exit(1)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='MQTT Artwork Detection for LexiBot')
    parser.add_argument('--custom-model', action='store_true', 
                       help='Use custom trained model instead of standard YOLO')
    parser.add_argument('--broker', type=str, default=MQTT_BROKER,
                       help='MQTT broker address')
    parser.add_argument('--port', type=int, default=MQTT_PORT,
                       help='MQTT broker port')
    parser.add_argument('--topic', type=str, default=MQTT_TOPIC,
                       help='MQTT topic for publishing detections')
    
    args = parser.parse_args()
    
    # Update configuration if provided
    if args.broker != MQTT_BROKER:
        print(f"Using custom MQTT broker: {args.broker}:{args.port}")
    
    # Create and run detector
    detector = MQTTArtworkDetection(use_custom_model=args.custom_model)
    detector.run()


if __name__ == "__main__":
    main()
