"""
Model training module for custom artwork detection
"""

import os
import logging
from typing import Dict, Any, Optional

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("YOLO not available. Install with: pip install ultralytics")

from src.utils.config import MODEL_PATH


class ModelTrainer:
    """
    Handles training of custom YOLO models for artwork detection.
    """
    
    def __init__(self, base_model_path: str = MODEL_PATH):
        """
        Initialize model trainer.
        
        Args:
            base_model_path: Path to base YOLO model for transfer learning
        """
        self.base_model_path = base_model_path
        self.model = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        if not YOLO_AVAILABLE:
            self.logger.error("YOLO not available - training disabled")
            return
        
        # Load base model
        try:
            # Use YOLOv11 nano for faster training (can change to 'yolo11s.pt', 'yolo11m.pt', etc.)
            self.model = YOLO('yolo11n.pt')
            self.logger.info(f"Loaded base model for training")
        except Exception as e:
            self.logger.error(f"Error loading base model: {e}")
    
    def train_model(self, 
                   data_yaml_path: str = 'config/data.yaml',
                   epochs: int = 100,
                   img_size: int = 640,
                   batch_size: int = 16,
                   workers: int = 4,
                   project: str = 'runs/detect',
                   name: str = 'train',
                   **kwargs) -> Optional[str]:
        """
        Train the artwork detection model.
        
        Args:
            data_yaml_path: Path to dataset configuration file
            epochs: Number of training epochs
            img_size: Training image size
            batch_size: Training batch size
            workers: Number of data loader workers
            project: Project directory for saving results
            name: Experiment name
            **kwargs: Additional training parameters
            
        Returns:
            Path to trained model or None if training failed
        """
        if self.model is None or not YOLO_AVAILABLE:
            self.logger.error("Model not available for training")
            return None
        
        if not os.path.exists(data_yaml_path):
            self.logger.error(f"Dataset configuration not found: {data_yaml_path}")
            return None
        
        try:
            self.logger.info("Starting model training...")
            self.logger.info(f"Dataset: {data_yaml_path}")
            self.logger.info(f"Epochs: {epochs}, Batch size: {batch_size}")
            self.logger.info(f"Image size: {img_size}, Workers: {workers}")
            
            # Train the model
            results = self.model.train(
                data=data_yaml_path,
                epochs=epochs,
                imgsz=img_size,
                batch=batch_size,
                workers=workers,
                project=project,
                name=name,
                **kwargs
            )
            
            # Get path to best model
            best_model_path = os.path.join(project, name, 'weights', 'best.pt')
            
            if os.path.exists(best_model_path):
                self.logger.info(f"Training completed successfully!")
                self.logger.info(f"Best model saved to: {best_model_path}")
                return best_model_path
            else:
                self.logger.error("Training completed but best model not found")
                return None
                
        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            return None
    
    def validate_model(self, model_path: str, data_yaml_path: str = 'config/data.yaml') -> Dict[str, Any]:
        """
        Validate trained model performance.
        
        Args:
            model_path: Path to trained model
            data_yaml_path: Path to dataset configuration
            
        Returns:
            Validation results dictionary
        """
        if not YOLO_AVAILABLE:
            return {"error": "YOLO not available"}
        
        try:
            # Load trained model
            model = YOLO(model_path)
            
            # Run validation
            results = model.val(data=data_yaml_path)
            
            return {
                "mAP50": results.box.map50,
                "mAP50-95": results.box.map,
                "precision": results.box.mp,
                "recall": results.box.mr,
                "model_path": model_path
            }
            
        except Exception as e:
            self.logger.error(f"Error during validation: {e}")
            return {"error": str(e)}
    
    def export_model(self, model_path: str, format: str = 'onnx') -> Optional[str]:
        """
        Export trained model to different formats for deployment.
        
        Args:
            model_path: Path to trained model
            format: Export format ('onnx', 'torchscript', 'coreml', etc.)
            
        Returns:
            Path to exported model or None if export failed
        """
        if not YOLO_AVAILABLE:
            return None
        
        try:
            model = YOLO(model_path)
            exported_path = model.export(format=format)
            
            self.logger.info(f"Model exported to {format} format: {exported_path}")
            return exported_path
            
        except Exception as e:
            self.logger.error(f"Error exporting model: {e}")
            return None
    
    def create_training_config(self, 
                             train_path: str = 'dataset/train',
                             val_path: str = 'dataset/valid',
                             test_path: str = 'dataset/test',
                             output_path: str = 'config/data.yaml') -> bool:
        """
        Create YOLO dataset configuration file.
        
        Args:
            train_path: Path to training images
            val_path: Path to validation images  
            test_path: Path to test images
            output_path: Output path for data.yaml file
            
        Returns:
            True if config created successfully, False otherwise
        """
        try:
            # Define class names based on artwork dataset
            class_names = [
                'libre',        # Liberty Leading the People
                'monalisa',     # Mona Lisa
                'background',   # Background/other
                'skrik',        # The Scream
                'starrynight',  # The Starry Night
                'sunflower'     # Sunflowers
            ]
            
            # Create configuration content
            config_content = f"""# LexiBot Artwork Detection Dataset Configuration
# Generated automatically by ModelTrainer

# Dataset paths (relative to this file)
train: {train_path}
val: {val_path}
test: {test_path}

# Number of classes
nc: {len(class_names)}

# Class names
names: {class_names}

# Optional: Additional dataset info
dataset_name: "LexiBot Artwork Detection"
description: "Custom dataset for detecting famous artworks in museum/gallery settings"
version: "1.0"
"""
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Write configuration file
            with open(output_path, 'w') as f:
                f.write(config_content)
            
            self.logger.info(f"Training configuration created: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating training config: {e}")
            return False


def train_artwork_model(epochs: int = 100, 
                       batch_size: int = 16,
                       img_size: int = 640) -> Optional[str]:
    """
    Convenience function to train artwork detection model with default settings.
    
    Args:
        epochs: Number of training epochs
        batch_size: Training batch size
        img_size: Training image size
        
    Returns:
        Path to trained model or None if training failed
    """
    trainer = ModelTrainer()
    
    # Create training configuration if it doesn't exist
    if not os.path.exists('config/data.yaml'):
        trainer.create_training_config()
    
    # Train the model
    return trainer.train_model(
        epochs=epochs,
        batch_size=batch_size,
        img_size=img_size
    )


if __name__ == "__main__":
    # Example usage: train model with custom parameters
    import argparse
    
    parser = argparse.ArgumentParser(description='Train LexiBot artwork detection model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Training image size')
    parser.add_argument('--data', type=str, default='config/data.yaml', help='Dataset config path')
    
    args = parser.parse_args()
    
    # Train model
    model_path = train_artwork_model(
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size
    )
    
    if model_path:
        print(f"Training completed! Model saved to: {model_path}")
    else:
        print("Training failed!")
