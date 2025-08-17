# LexiBot Computer Vision System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org)
[![YOLOv11](https://img.shields.io/badge/YOLO-v11-red.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A sophisticated computer vision system for autonomous art detection and cataloging, featuring custom-trained YOLO models for museum environments.

## Overview

This project implements a real-time computer vision system designed for the LexiBot autonomous museum guide robot. The system uses a custom-trained YOLOv11 model to detect and classify artwork in museum environments, enabling autonomous navigation and interactive experiences.

### Key Features

- **Custom Art Detection**: YOLOv11 model trained on museum artwork dataset
- **Real-time Processing**: Live camera feed analysis with object detection
- **MQTT Integration**: Robot communication and command streaming
- **Modular Architecture**: Clean separation of detection, communication, and utilities
- **Easy Setup**: Single-command installation and configuration

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam or USB camera
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/GrigoriusPompeus/computer_vision_for_lexibot.git
   cd computer_vision_for_lexibot
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Test your camera**
   ```bash
   python test_camera.py
   ```

5. **Run art detection**
   ```bash
   python test_custom_art_detection.py
   ```

## Project Structure

```
computer_vision_for_lexibot/
├── src/
│   ├── detection/
│   │   ├── detector.py          # Main detection engine
│   │   └── __init__.py
│   ├── communication/
│   │   ├── mqtt_client.py       # MQTT communication handler
│   │   └── __init__.py
│   └── utils/
│       ├── camera.py            # Camera utilities
│       └── __init__.py
├── scripts/
│   ├── real_time_detection.py   # Main application script
│   └── mqtt_demo.py            # MQTT demonstration
├── config/
│   └── data.yaml               # Model configuration
├── models/
│   └── best.pt                 # Custom-trained YOLO weights
├── training/                   # Training scripts and data
├── test_camera.py              # Camera functionality test
├── test_custom_art_detection.py # Art detection test
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Usage

### Basic Art Detection

Test the custom art detection model:

```bash
python test_custom_art_detection.py
```

This will open your camera and display real-time detection results with bounding boxes around detected artwork.

### Real-time Detection with MQTT

Run the full system with MQTT communication:

```bash
python scripts/real_time_detection.py
```

### MQTT Demo

Test MQTT communication features:

```bash
python scripts/mqtt_demo.py
```

## Configuration

### Camera Settings

Edit camera parameters in `src/utils/camera.py`:

```python
# Camera resolution
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Detection confidence threshold
CONFIDENCE_THRESHOLD = 0.5
```

### MQTT Configuration

Configure MQTT settings in `src/communication/mqtt_client.py`:

```python
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
TOPICS = {
    "detections": "lexibot/art_detections",
    "commands": "lexibot/robot_commands",
    "status": "lexibot/system_status"
}
```

## Model Information

The custom YOLO model (`models/best.pt`) was trained specifically for museum artwork detection:

- **Architecture**: YOLOv11
- **Training Dataset**: Custom museum artwork collection
- **Classes**: Various art piece types and museum objects
- **Performance**: Optimized for real-time detection on standard hardware

## Development

### Adding New Features

1. Create feature branch: `git checkout -b feature/new-feature`
2. Implement changes in appropriate modules
3. Add tests for new functionality
4. Update documentation
5. Submit pull request

### Testing

Run individual tests:

```bash
# Test camera functionality
python test_camera.py

# Test art detection
python test_custom_art_detection.py

# Test MQTT communication
python scripts/mqtt_demo.py
```

## Troubleshooting

### Common Issues

**Camera not detected:**
- Ensure camera is connected and not used by another application
- Try different camera indices in `test_camera.py`
- Check camera permissions

**Model loading errors:**
- Verify `models/best.pt` exists
- Check file permissions
- Ensure sufficient disk space

**MQTT connection issues:**
- Verify MQTT broker is running
- Check network connectivity
- Confirm broker address and port

**Import errors:**
- Activate virtual environment
- Reinstall requirements: `pip install -r requirements.txt`
- Check Python version compatibility

### Performance Optimization

For better performance:
- Use GPU acceleration (install `torch` with CUDA support)
- Adjust detection confidence threshold
- Optimize camera resolution based on hardware capabilities

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **DECO3801 Course**: University of Queensland
- **Ultralytics**: YOLOv11 implementation
- **OpenCV Community**: Computer vision tools
- **Python Community**: Core libraries and frameworks

## Contact

For questions or collaboration opportunities, please open an issue on GitHub.

---

*Developed as part of DECO3801 coursework at the University of Queensland*
