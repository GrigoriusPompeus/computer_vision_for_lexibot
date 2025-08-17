# Training Scripts

This directory contains tools for retraining and customizing the LexiBot Computer Vision System.

## Files

### `setup_art_training.py`
- Manual art dataset setup script
- Helps organize training data after manual download
- Use when you have new artwork images to train on

### `train_custom_model.py`
- Custom YOLO model training script
- Trains new models on your artwork dataset
- Use to improve detection accuracy or add new art classes

### `setup.py`
- Environment setup and dependency installation
- Development tool for setting up the training environment

## Usage

To retrain the model with new data:

1. Organize your training images using `setup_art_training.py`
2. Run `train_custom_model.py` to train a new model
3. Replace `models/best.pt` with your new trained model

## Note

These are development tools. The main application in the parent directory uses the pre-trained model in `models/best.pt`.
