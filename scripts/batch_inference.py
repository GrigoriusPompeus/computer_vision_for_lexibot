#!/usr/bin/env python3
"""
Batch inference script for processing multiple images or videos
"""

import os
import sys
import argparse
import cv2
import glob
from pathlib import Path

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: YOLO not available. Install with: pip install ultralytics")

from src.detection.detector import ArtworkDetector
from src.utils.config import MODEL_PATH, CUSTOM_MODEL_PATH


def process_image(detector, image_path, output_dir):
    """
    Process a single image for artwork detection.
    
    Args:
        detector: ArtworkDetector instance
        image_path: Path to input image
        output_dir: Directory to save results
    """
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return
        
        # Detect artworks
        detections = detector.detect_artworks(image)
        
        # Draw detections
        result_image = detector.draw_detections(image, detections)
        
        # Save result
        filename = Path(image_path).stem
        output_path = os.path.join(output_dir, f"{filename}_detected.jpg")
        cv2.imwrite(output_path, result_image)
        
        # Print results
        print(f"Processed {image_path}: {len(detections)} detections")
        for det in detections:
            print(f"  - {det['label']}: {det['confidence']:.3f} ({det['distance_cm']:.1f}cm)")
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")


def process_video(detector, video_path, output_dir):
    """
    Process a video file for artwork detection.
    
    Args:
        detector: ArtworkDetector instance
        video_path: Path to input video
        output_dir: Directory to save results
    """
    try:
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup output video
        filename = Path(video_path).stem
        output_path = os.path.join(output_dir, f"{filename}_detected.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Processing video: {video_path}")
        print(f"Properties: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        frame_count = 0
        detection_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect artworks
            detections = detector.detect_artworks(frame)
            detection_count += len(detections)
            
            # Draw detections
            result_frame = detector.draw_detections(frame, detections)
            
            # Write frame
            out.write(result_frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        # Cleanup
        cap.release()
        out.release()
        
        print(f"Video processing complete: {output_path}")
        print(f"Total detections: {detection_count}")
        
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")


def main():
    """Main function for batch processing."""
    parser = argparse.ArgumentParser(description='Batch inference for LexiBot artwork detection')
    parser.add_argument('input', help='Input file/directory path')
    parser.add_argument('-o', '--output', default='output', 
                       help='Output directory (default: output)')
    parser.add_argument('--custom-model', action='store_true',
                       help='Use custom trained model')
    parser.add_argument('--confidence', type=float, default=0.1,
                       help='Confidence threshold (default: 0.1)')
    parser.add_argument('--extensions', nargs='+', 
                       default=['jpg', 'jpeg', 'png', 'bmp', 'mp4', 'avi', 'mov'],
                       help='File extensions to process')
    
    args = parser.parse_args()
    
    # Check dependencies
    if not YOLO_AVAILABLE:
        print("Error: YOLOv11 not available. Install with: pip install ultralytics")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize detector
    try:
        if args.custom_model and os.path.exists(CUSTOM_MODEL_PATH):
            detector = ArtworkDetector(CUSTOM_MODEL_PATH, use_custom_model=True,
                                     confidence_threshold=args.confidence)
            print(f"Using custom model: {CUSTOM_MODEL_PATH}")
        else:
            detector = ArtworkDetector(MODEL_PATH, confidence_threshold=args.confidence)
            print(f"Using standard model: {MODEL_PATH}")
    except Exception as e:
        print(f"Error initializing detector: {e}")
        sys.exit(1)
    
    # Process input
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single file
        extension = input_path.suffix[1:].lower()
        if extension in ['jpg', 'jpeg', 'png', 'bmp']:
            process_image(detector, str(input_path), args.output)
        elif extension in ['mp4', 'avi', 'mov', 'mkv']:
            process_video(detector, str(input_path), args.output)
        else:
            print(f"Unsupported file format: {extension}")
    
    elif input_path.is_dir():
        # Directory of files
        files_processed = 0
        
        for ext in args.extensions:
            pattern = os.path.join(str(input_path), f"*.{ext}")
            files = glob.glob(pattern)
            
            for file_path in files:
                extension = Path(file_path).suffix[1:].lower()
                
                if extension in ['jpg', 'jpeg', 'png', 'bmp']:
                    process_image(detector, file_path, args.output)
                elif extension in ['mp4', 'avi', 'mov', 'mkv']:
                    process_video(detector, file_path, args.output)
                
                files_processed += 1
        
        print(f"Batch processing complete: {files_processed} files processed")
    
    else:
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
