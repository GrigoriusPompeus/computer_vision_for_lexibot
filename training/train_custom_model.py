"""
Quick Custom Art Model Training Script

This script will help you train a custom YOLO model for art detection
if you have your training data available.
"""

import os
import sys
from pathlib import Path

def check_training_data():
    """Check if training data is available"""
    print("🔍 Checking for training data...")
    
    # Check common locations for training data
    data_locations = [
        "train",
        "valid", 
        "test",
        "data/train",
        "data/valid",
        "data/test",
        "../train",
        "../valid",
        "../test"
    ]
    
    found_data = []
    for location in data_locations:
        if os.path.exists(location):
            found_data.append(location)
            print(f"✅ Found: {location}")
    
    if not found_data:
        print("❌ No training data found in common locations")
        return False
    
    return True

def create_data_yaml():
    """Create data.yaml file for training"""
    print("📝 Creating data.yaml configuration...")
    
    # Check if we have the art training data structure
    if os.path.exists("train") and os.path.exists("valid"):
        data_yaml = """
# Art Detection Dataset Configuration
path: .  # dataset root dir
train: train/images  # train images (relative to 'path')
val: valid/images   # val images (relative to 'path')
test: test/images   # test images (optional)

# Classes
names:
  0: libre        # Liberty Leading the People
  1: monalisa     # Mona Lisa
  2: background   # Background/other
  3: skrik        # The Scream
  4: starrynight  # The Starry Night
  5: sunflower    # Sunflowers
"""
        
        with open("data.yaml", "w") as f:
            f.write(data_yaml)
        
        print("✅ Created data.yaml")
        return True
    else:
        print("❌ Expected training data structure not found")
        return False

def train_custom_model():
    """Train the custom art detection model"""
    try:
        from ultralytics import YOLO
        
        print("🚀 Starting custom art model training...")
        print("This may take a while depending on your hardware...")
        
        # Load a YOLOv11 model
        model = YOLO('yolo11n.pt')  # Start with nano model for faster training
        
        # Train the model
        results = model.train(
            data='data.yaml',
            epochs=50,  # Reduced for faster training
            imgsz=640,
            batch=8,   # Reduced batch size
            workers=2,
            project='runs/detect',
            name='art_detection'
        )
        
        print("✅ Training completed!")
        print(f"Model saved to: runs/detect/art_detection/weights/best.pt")
        
        return True
        
    except ImportError:
        print("❌ Ultralytics not available. Install with: pip install ultralytics")
        return False
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def main():
    print("🎨 Custom Art Detection Model Training")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("config"):
        print("❌ Please run this from the project root directory")
        return
    
    # Check for training data
    if not check_training_data():
        print("\n📥 To train a custom model, you need:")
        print("1. Training images in train/images/")
        print("2. Training labels in train/labels/")
        print("3. Validation images in valid/images/")
        print("4. Validation labels in valid/labels/")
        print("\nYou can:")
        print("- Copy your training data from the original repo")
        print("- Download from Roboflow: https://universe.roboflow.com/testing-images/paintings-mk265")
        print("- Use the existing model if available")
        return
    
    # Create data.yaml
    if not create_data_yaml():
        return
    
    # Train the model
    if train_custom_model():
        print("\n🎉 Custom art detection model ready!")
        print("Copy the trained model to models/best.pt to use it")
        
        # Copy the model automatically
        best_model = Path("runs/detect/art_detection/weights/best.pt")
        if best_model.exists():
            import shutil
            os.makedirs("models", exist_ok=True)
            shutil.copy(best_model, "models/best.pt")
            print("✅ Model copied to models/best.pt")
    else:
        print("❌ Training failed")

if __name__ == "__main__":
    main()
