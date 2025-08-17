"""
Main detection module for the LexiBot Computer Vision System
"""

import cv2
import logging
import time
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("YOLO not available. Install with: pip install ultralytics")

from src.utils.config import (
    MODEL_PATH, CUSTOM_MODEL_PATH, USE_CUSTOM_MODEL, CONFIDENCE_THRESHOLD, 
    DETECTION_BOX_COLOR, DETECTION_TEXT_COLOR, 
    DETECTION_FONT_SCALE, DETECTION_THICKNESS, CLASS_NAMES
)
from src.detection.distance_estimator import DistanceEstimator
from src.detection.mqtt_client import MQTTClient


class ArtworkDetector:
    """
    Main detector class for artwork recognition and analysis.
    """
    
    def __init__(self, model_path: str = CUSTOM_MODEL_PATH if USE_CUSTOM_MODEL else MODEL_PATH, 
                 use_custom_model: bool = USE_CUSTOM_MODEL,
                 confidence_threshold: float = CONFIDENCE_THRESHOLD):
        """
        Initialize the artwork detector.
        
        Args:
            model_path: Path to YOLO model file
            use_custom_model: Whether to use custom trained model
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self.distance_estimator = DistanceEstimator()
        self.model = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        if not YOLO_AVAILABLE:
            self.logger.error("YOLO not available - detection disabled")
            return
        
        # Load model
        try:
            if use_custom_model:
                self.model = YOLO(CUSTOM_MODEL_PATH)
                self.logger.info(f"Loaded custom model: {CUSTOM_MODEL_PATH}")
            else:
                self.model = YOLO(model_path)
                self.logger.info(f"Loaded model: {model_path}")
        except Exception as e:
            self.logger.error(f"Error loading YOLO model: {e}")
            raise
    
    def detect_artworks(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect artworks in a single frame.
        
        Args:
            frame: Input image frame
            
        Returns:
            List of detection dictionaries
        """
        if self.model is None or not YOLO_AVAILABLE:
            return []
        
        detections = []
        
        try:
            # Run inference
            results = self.model(frame, conf=self.confidence_threshold)
            
            # Process results
            for box in results[0].boxes:
                # Extract box information
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                
                # Get label name
                if hasattr(self.model, 'names'):
                    label = self.model.names[class_id]
                else:
                    label = CLASS_NAMES.get(class_id, f"class_{class_id}")
                
                # Estimate distance
                bbox = (x1, y1, x2, y2)
                distance = self.distance_estimator.estimate_distance(bbox, label)
                
                # Create detection dictionary
                detection = {
                    "label": label,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                    "distance_cm": distance,
                    "class_id": class_id
                }
                
                detections.append(detection)
                
        except Exception as e:
            self.logger.error(f"Error during detection: {e}")
        
        return detections
    
    def draw_detections(self, frame: np.ndarray, 
                       detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw detection results on frame.
        
        Args:
            frame: Input image frame
            detections: List of detections to draw
            
        Returns:
            Frame with drawn detections
        """
        for detection in detections:
            label = detection["label"]
            confidence = detection["confidence"]
            bbox = detection["bbox"]
            distance = detection.get("distance_cm", 0)
            
            # Ensure coordinates are integers
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), 
                         DETECTION_BOX_COLOR, DETECTION_THICKNESS)
            
            # Create label text
            if distance > 0:
                text = f"{label} {confidence:.2f} {distance:.1f}cm"
            else:
                text = f"{label} {confidence:.2f}"
            
            # Draw label text (simpler version like the original)
            cv2.putText(frame, text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       (0, 255, 0), 2)
        
        return frame
    
    def get_artwork_info(self, label: str) -> Dict[str, str]:
        """
        Get detailed information about detected artwork.
        
        Args:
            label: Artwork label
            
        Returns:
            Dictionary with artwork information
        """
        artwork_info = {
            "monalisa": {
                "title": "Mona Lisa",
                "artist": "Leonardo da Vinci",
                "year": "1503-1519",
                "location": "Louvre Museum, Paris"
            },
            "starrynight": {
                "title": "The Starry Night",
                "artist": "Vincent van Gogh",
                "year": "1889",
                "location": "Museum of Modern Art, New York"
            },
            "libre": {
                "title": "Liberty Leading the People",
                "artist": "Eugène Delacroix",
                "year": "1830",
                "location": "Louvre Museum, Paris"
            },
            "skrik": {
                "title": "The Scream",
                "artist": "Edvard Munch",
                "year": "1893",
                "location": "National Gallery, Oslo"
            },
            "sunflower": {
                "title": "Sunflowers",
                "artist": "Vincent van Gogh",
                "year": "1888",
                "location": "Various museums worldwide"
            }
        }
        
        return artwork_info.get(label, {
            "title": "Unknown Artwork",
            "artist": "Unknown",
            "year": "Unknown",
            "location": "Unknown"
        })
    
    def is_available(self) -> bool:
        """
        Check if detector is available and ready.
        
        Returns:
            True if detector is ready, False otherwise
        """
        return YOLO_AVAILABLE and self.model is not None


class RealTimeDetector:
    """
    Real-time detection system for camera input.
    """
    
    def __init__(self, camera_index: int = 0, 
                 enable_mqtt: bool = False,
                 enable_recording: bool = False):
        """
        Initialize real-time detector.
        
        Args:
            camera_index: Camera device index
            enable_mqtt: Enable MQTT communication
            enable_recording: Enable video recording
        """
        self.camera_index = camera_index
        self.enable_mqtt = enable_mqtt
        self.enable_recording = enable_recording
        
        # Initialize components
        self.detector = ArtworkDetector()
        self.mqtt_client = MQTTClient() if enable_mqtt else None
        
        # Camera setup
        self.cap = None
        self.out = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def start_camera(self) -> bool:
        """
        Start camera capture.
        
        Returns:
            True if camera started successfully, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                self.logger.error(f"Could not open camera {self.camera_index}")
                return False
            
            self.logger.info(f"Camera {self.camera_index} started successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error starting camera: {e}")
            return False
    
    def stop_camera(self):
        """Stop camera capture and cleanup."""
        if self.cap:
            self.cap.release()
        if self.out:
            self.out.release()
        cv2.destroyAllWindows()
        if self.mqtt_client:
            self.mqtt_client.disconnect()
    
    def run_detection_loop(self):
        """Main detection loop for real-time processing."""
        if not self.start_camera():
            return
        
        # Connect MQTT if enabled
        if self.mqtt_client and not self.mqtt_client.connect():
            self.logger.warning("MQTT connection failed - continuing without MQTT")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Detect artworks
                detections = self.detector.detect_artworks(frame)
                
                # Draw detections on frame
                frame = self.detector.draw_detections(frame, detections)
                
                # Publish to MQTT if enabled and detections found
                if self.mqtt_client and detections:
                    self.mqtt_client.publish_detections(detections)
                
                # Display frame
                cv2.imshow("LexiBot Artwork Detection", frame)
                
                # Check for exit
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    break
                    
        except KeyboardInterrupt:
            self.logger.info("Detection stopped by user")
        except Exception as e:
            self.logger.error(f"Error in detection loop: {e}")
        finally:
            self.stop_camera()
