#!/usr/bin/env python3
"""
Setup script for LexiBot Computer Vision System
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("Error: Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    return True


def install_requirements():
    """Install required packages."""
    print("Installing Python dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing dependencies: {e}")
        return False


def create_directories():
    """Create necessary directories."""
    directories = [
        "models",
        "dataset/train/images",
        "dataset/train/labels", 
        "dataset/valid/images",
        "dataset/valid/labels",
        "dataset/test/images",
        "dataset/test/labels",
        "output",
        "logs"
    ]
    
    print("Creating project directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {directory}")


def download_yolo_model():
    """Download YOLOv11 model if not present."""
    model_path = "yolo11l.pt"
    if not os.path.exists(model_path):
        print("Downloading YOLOv11 model...")
        try:
            # This will be downloaded automatically when first used
            print("✓ Model will be downloaded automatically on first run")
            return True
        except Exception as e:
            print(f"✗ Error preparing model: {e}")
            return False
    else:
        print("✓ YOLOv11 model already present")
        return True


def test_imports():
    """Test if critical imports work."""
    print("Testing critical imports...")
    
    try:
        import cv2
        print("✓ OpenCV imported successfully")
    except ImportError:
        print("✗ OpenCV import failed")
        return False
    
    try:
        from ultralytics import YOLO
        print("✓ Ultralytics YOLO imported successfully")
    except ImportError:
        print("✗ Ultralytics import failed")
        return False
    
    try:
        import paho.mqtt.client as mqtt
        print("✓ MQTT client imported successfully")
    except ImportError:
        print("⚠ MQTT client not available (optional)")
    
    return True


def run_basic_test():
    """Run a basic functionality test."""
    print("Running basic functionality test...")
    
    try:
        # Test detector initialization
        sys.path.append('src')
        from detection.detector import ArtworkDetector
        
        detector = ArtworkDetector()
        if detector.is_available():
            print("✓ Artwork detector initialized successfully")
            return True
        else:
            print("✗ Artwork detector failed to initialize")
            return False
            
    except Exception as e:
        print(f"✗ Basic test failed: {e}")
        return False


def print_system_info():
    """Print system information."""
    print("\nSystem Information:")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print(f"Architecture: {platform.machine()}")


def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*60)
    print("🎉 LexiBot Computer Vision System Setup Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. To run real-time detection:")
    print("   python scripts/real_time_detection.py")
    print("\n2. To run MQTT-enabled detection:")
    print("   python scripts/mqtt_detection.py")
    print("\n3. To process batch files:")
    print("   python scripts/batch_inference.py <input_path>")
    print("\n4. To train a custom model:")
    print("   python src/training/train_model.py")
    print("\n5. Add your dataset to the dataset/ directory")
    print("   and update config/data.yaml if needed")
    print("\nFor more information, see README.md")


def main():
    """Main setup function."""
    print("LexiBot Computer Vision System - Setup Script")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Print system info
    print_system_info()
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("\nSetup failed during dependency installation")
        sys.exit(1)
    
    # Download/prepare model
    download_yolo_model()
    
    # Test imports
    if not test_imports():
        print("\nSetup completed with warnings - some features may not work")
        print("Please check the installation instructions in README.md")
        sys.exit(1)
    
    # Run basic test
    if not run_basic_test():
        print("\nSetup completed but basic test failed")
        print("The system may still work, but please check the configuration")
    
    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    main()
