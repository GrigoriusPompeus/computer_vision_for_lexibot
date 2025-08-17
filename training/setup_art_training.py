"""
Manual Art Dataset Setup and Training

Since Roboflow requires an API key, this script helps you set up 
the training after manual download.
"""

import os
import sys
import zipfile
from pathlib import Path

def manual_download_instructions():
    """Provide instructions for manual dataset download"""
    print("📥 Manual Dataset Download Instructions")
    print("=" * 50)
    print()
    print("1. 🌐 Open: https://universe.roboflow.com/testing-images/paintings-mk265")
    print("2. 📋 Click 'Download Dataset' button")
    print("3. 🎯 Select 'YOLOv11' format")
    print("4. 💾 Download the ZIP file")
    print("5. 📁 Extract it to this directory:")
    print(f"   {os.getcwd()}")
    print()
    print("Expected folder structure after extraction:")
    print("  📁 paintings-mk265-1/")
    print("    📁 train/")
    print("      📁 images/")
    print("      📁 labels/")
    print("    📁 valid/")
    print("      📁 images/")
    print("      📁 labels/")
    print("    📁 test/")
    print("      📁 images/")
    print("      📁 labels/")
    print("    📄 data.yaml")
    print()
    
    # Check if user has downloaded it
    response = input("🤔 Have you downloaded and extracted the dataset? (y/n): ").lower()
    return response == 'y'

def find_dataset():
    """Find the downloaded dataset"""
    print("🔍 Searching for dataset...")
    
    # Common dataset folder names
    possible_names = [
        "paintings-mk265-1",
        "paintings-1", 
        "paintings-mk265",
        "paintings",
        "dataset"
    ]
    
    found_dataset = None
    for name in possible_names:
        if os.path.exists(name):
            # Check if it has the right structure
            if (os.path.exists(os.path.join(name, "train")) and 
                os.path.exists(os.path.join(name, "valid")) and
                os.path.exists(os.path.join(name, "data.yaml"))):
                found_dataset = name
                print(f"✅ Found dataset: {name}")
                break
    
    if not found_dataset:
        print("❌ Dataset not found. Please ensure you've:")
        print("  1. Downloaded the dataset from Roboflow")
        print("  2. Extracted it to this directory") 
        print("  3. The folder contains train/, valid/, and data.yaml")
        return None
    
    return found_dataset

def check_dataset_structure(dataset_path):
    """Verify the dataset has the correct structure"""
    print(f"🔍 Checking dataset structure in {dataset_path}...")
    
    required_paths = [
        "train/images",
        "train/labels", 
        "valid/images",
        "valid/labels",
        "data.yaml"
    ]
    
    missing = []
    for path in required_paths:
        full_path = os.path.join(dataset_path, path)
        if os.path.exists(full_path):
            if path.endswith('images') or path.endswith('labels'):
                count = len(os.listdir(full_path))
                print(f"✅ {path}: {count} files")
            else:
                print(f"✅ {path}: exists")
        else:
            missing.append(path)
            print(f"❌ {path}: missing")
    
    if missing:
        print(f"❌ Missing required paths: {missing}")
        return False
    
    return True

def create_art_detection_config(dataset_path):
    """Create or update the data.yaml for art detection"""
    print("📝 Setting up art detection configuration...")
    
    data_yaml_path = os.path.join(dataset_path, "data.yaml")
    
    # Our custom configuration for art detection
    config = f"""# Custom Art Detection Dataset
# Path to dataset root
path: {os.path.abspath(dataset_path)}

# Relative paths from path
train: train/images
val: valid/images
test: test/images

# Number of classes
nc: 6

# Class names (based on your original training)
names:
  0: libre        # Liberty Leading the People  
  1: monalisa     # Mona Lisa
  2: background   # Background/other
  3: skrik        # The Scream
  4: starrynight  # The Starry Night
  5: sunflower    # Sunflowers
"""
    
    with open(data_yaml_path, 'w') as f:
        f.write(config)
    
    print("✅ Updated data.yaml for art detection")
    return data_yaml_path

def train_custom_art_model(data_yaml_path):
    """Train the custom art detection model"""
    print("🚀 Starting Custom Art Model Training")
    print("=" * 50)
    print("🎨 Artworks to detect:")
    print("  • Liberty Leading the People (libre)")
    print("  • Mona Lisa (monalisa)")
    print("  • The Scream (skrik)")
    print("  • The Starry Night (starrynight)")
    print("  • Sunflowers (sunflower)")
    print("  • Background (background)")
    print()
    print("⏰ Estimated training time: 30-60 minutes")
    print("🔥 Starting training...")
    print()
    
    try:
        from ultralytics import YOLO
        
        # Load YOLOv11 model (start with nano for faster training)
        print("📥 Loading YOLOv11 nano model...")
        model = YOLO('yolo11n.pt')
        
        # Train the model
        print("🔥 Training started...")
        results = model.train(
            data=data_yaml_path,
            epochs=100,          # Comprehensive training
            imgsz=640,           # Standard image size
            batch=16,            # Adjust based on GPU memory
            workers=4,           # Parallel workers
            project='runs/detect',
            name='art_detection_custom',
            patience=15,         # Early stopping
            save=True,          # Save checkpoints
            plots=True,         # Generate training plots
            verbose=True,       # Show detailed output
            cache=True,         # Cache images for faster training
            device='auto'       # Auto-detect GPU/CPU
        )
        
        print("🎉 Training completed successfully!")
        
        # Locate the best model
        best_model_path = Path("runs/detect/art_detection_custom/weights/best.pt")
        
        if best_model_path.exists():
            print(f"✅ Best model saved to: {best_model_path}")
            
            # Copy to models directory for easy access
            os.makedirs("models", exist_ok=True)
            import shutil
            shutil.copy(best_model_path, "models/best.pt")
            print("✅ Model copied to models/best.pt")
            
            # Show training results
            print("\n📊 Training Results:")
            print(f"   Model file: {best_model_path}")
            print(f"   Size: {best_model_path.stat().st_size / (1024*1024):.1f} MB")
            
            return True
        else:
            print("❌ Training completed but best.pt not found")
            return False
            
    except ImportError:
        print("❌ Ultralytics not installed")
        print("Installing ultralytics...")
        os.system("pip install ultralytics")
        return train_custom_art_model(data_yaml_path)
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("\n💡 Troubleshooting tips:")
        print("  • Reduce batch size if you get memory errors")
        print("  • Ensure your GPU drivers are updated")
        print("  • Try training with fewer epochs first")
        return False

def test_trained_model():
    """Test the newly trained model"""
    print("🧪 Testing trained model...")
    
    if not os.path.exists("models/best.pt"):
        print("❌ Trained model not found")
        return False
    
    try:
        from ultralytics import YOLO
        
        model = YOLO("models/best.pt")
        print("✅ Model loaded successfully!")
        
        print(f"📋 Model classes: {list(model.names.values())}")
        print("🎯 Ready for art detection!")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def main():
    print("🎨 Custom Art Detection Training Setup")
    print("=" * 60)
    
    # Step 1: Manual download instructions
    if not manual_download_instructions():
        print("📥 Please download the dataset first, then run this script again.")
        return
    
    # Step 2: Find the dataset
    dataset_path = find_dataset()
    if not dataset_path:
        return
    
    # Step 3: Check dataset structure
    if not check_dataset_structure(dataset_path):
        print("❌ Dataset structure is incomplete")
        return
    
    # Step 4: Create configuration
    data_yaml_path = create_art_detection_config(dataset_path)
    
    # Step 5: Train the model
    print("\n🚀 Ready to start training!")
    start_training = input("Start training now? This will take 30-60 minutes (y/n): ").lower()
    
    if start_training != 'y':
        print("⏸️ Training postponed. Run this script again when ready.")
        print(f"💾 Dataset ready at: {dataset_path}")
        print(f"⚙️ Config file: {data_yaml_path}")
        return
    
    if not train_custom_art_model(data_yaml_path):
        print("❌ Training failed")
        return
    
    # Step 6: Test the model
    if not test_trained_model():
        print("❌ Model testing failed")
        return
    
    print("\n🎉 SUCCESS! Your custom art detection model is ready!")
    print("=" * 60)
    print("✅ Model saved to: models/best.pt")
    print("🎮 Test commands:")
    print("   python test_custom_art_detection.py")
    print("   .\\test_art_detection.ps1")
    print()
    print("🎨 Show these artworks to your camera:")
    print("   • Liberty Leading the People")
    print("   • Mona Lisa") 
    print("   • The Scream")
    print("   • The Starry Night")
    print("   • Sunflowers")
    print()
    print("📱 Try showing them on your phone/computer screen or printed copies!")

if __name__ == "__main__":
    main()
