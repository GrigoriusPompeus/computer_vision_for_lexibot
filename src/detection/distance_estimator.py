"""
Distance estimation utilities for the LexiBot Computer Vision System
"""

import math
from typing import Tuple, Optional
from src.utils.config import KNOWN_WIDTHS, FOCAL_LENGTH


class DistanceEstimator:
    """
    Estimates distance to detected objects using computer vision techniques.
    
    Uses the relationship: Distance = (Real_Width * Focal_Length) / Pixel_Width
    """
    
    def __init__(self, focal_length: float = FOCAL_LENGTH):
        """
        Initialize the distance estimator.
        
        Args:
            focal_length: Camera focal length in pixels (calibrated value)
        """
        self.focal_length = focal_length
        self.known_widths = KNOWN_WIDTHS.copy()
    
    def estimate_distance(self, bbox: Tuple[int, int, int, int], 
                         object_class: str) -> float:
        """
        Estimate distance to an object based on its bounding box and class.
        
        Args:
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            object_class: Detected object class name
            
        Returns:
            Estimated distance in centimeters
        """
        x1, y1, x2, y2 = bbox
        pixel_width = x2 - x1
        
        if pixel_width <= 0:
            return 0.0
        
        # Get known real-world width for this object class
        real_width = self.known_widths.get(object_class, 50.0)
        
        # Calculate distance using similar triangles
        distance_cm = (real_width * self.focal_length) / pixel_width
        
        return round(distance_cm, 1)
    
    def estimate_distance_with_height(self, bbox: Tuple[int, int, int, int], 
                                    object_class: str, 
                                    use_height: bool = False) -> float:
        """
        Enhanced distance estimation using both width and height information.
        
        Args:
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            object_class: Detected object class name
            use_height: Whether to use height for estimation (for tall objects)
            
        Returns:
            Estimated distance in centimeters
        """
        x1, y1, x2, y2 = bbox
        pixel_width = x2 - x1
        pixel_height = y2 - y1
        
        if pixel_width <= 0 and pixel_height <= 0:
            return 0.0
        
        real_width = self.known_widths.get(object_class, 50.0)
        
        if use_height and object_class in ['person']:
            # Use height for people (average height ~170cm)
            real_height = 170.0
            distance_cm = (real_height * self.focal_length) / pixel_height
        else:
            # Use width for artworks and most objects
            distance_cm = (real_width * self.focal_length) / pixel_width
        
        return round(distance_cm, 1)
    
    def calibrate_focal_length(self, known_distance: float, 
                             known_width: float, 
                             pixel_width: float) -> float:
        """
        Calibrate focal length using a known object at known distance.
        
        Args:
            known_distance: Actual distance to object in cm
            known_width: Actual width of object in cm
            pixel_width: Pixel width of object in image
            
        Returns:
            Calculated focal length
        """
        focal_length = (pixel_width * known_distance) / known_width
        return round(focal_length, 2)
    
    def add_known_object(self, object_class: str, width_cm: float):
        """
        Add a new object type with known dimensions.
        
        Args:
            object_class: Name of the object class
            width_cm: Real-world width in centimeters
        """
        self.known_widths[object_class] = width_cm
    
    def get_distance_accuracy_estimate(self, distance: float) -> str:
        """
        Provide an accuracy estimate for the distance measurement.
        
        Args:
            distance: Estimated distance in cm
            
        Returns:
            Accuracy description string
        """
        if distance < 50:
            return "High accuracy"
        elif distance < 200:
            return "Good accuracy"
        elif distance < 500:
            return "Moderate accuracy"
        else:
            return "Low accuracy"
