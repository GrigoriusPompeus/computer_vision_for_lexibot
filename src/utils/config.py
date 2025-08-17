"""
Configuration settings for the LexiBot Computer Vision System
"""

import os

# Model Configuration
MODEL_PATH = "yolo11l.pt"  # Will download automatically if not present
CUSTOM_MODEL_PATH = "models/best.pt"  # Path to custom trained model (NOW AVAILABLE!)
USE_CUSTOM_MODEL = True  # Set to True to use your custom art detection model
CONFIDENCE_THRESHOLD = 0.4  # Increased for more accurate detections

# Camera Configuration
CAMERA_INDEX = 0
FOCAL_LENGTH = 800  # Calibrated for standard webcam
DEFAULT_FPS = 20.0

# Known object dimensions (in cm) for distance estimation
KNOWN_WIDTHS = {
    "monalisa": 77.0,      # Actual Mona Lisa width
    "starrynight": 92.1,   # Actual Starry Night width  
    "libre": 260.0,        # Liberty Leading the People width
    "skrik": 91.0,         # The Scream width
    "sunflower": 92.1,     # Sunflowers width
    # Fallback for unrecognized objects
    "person": 50.0,
    "bottle": 7.0,
    "book": 15.0,
    "cell phone": 7.0,
    "laptop": 33.0,
    "tv": 100.0
}

# MQTT Configuration
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "yolo/detections"
MQTT_QOS = 0

# Output Configuration
OUTPUT_DIR = os.path.expanduser("~/Desktop/Video_Output")
TEMP_OUTPUT_FILE = "temp_output.mp4"

# UI Configuration
EXIT_BUTTON_CONFIG = {
    'width': 140,
    'height': 40,
    'margin_x': 20,
    'margin_y': 20,
    'color': (0, 0, 255),  # Red
    'text_color': (255, 255, 255),  # White
    'font': 'cv2.FONT_HERSHEY_SIMPLEX',
    'font_scale': 0.7,
    'thickness': 2
}

# Detection Display Configuration
DETECTION_BOX_COLOR = (0, 255, 0)  # Green
DETECTION_TEXT_COLOR = (0, 255, 0)  # Green
DETECTION_FONT_SCALE = 0.6
DETECTION_THICKNESS = 2

# Class mapping for custom model
CLASS_NAMES = {
    0: "libre",        # Liberty Leading the People
    1: "monalisa",     # Mona Lisa
    2: "background",   # Background/other
    3: "skrik",        # The Scream
    4: "starrynight",  # The Starry Night
    5: "sunflower"     # Sunflowers
}
