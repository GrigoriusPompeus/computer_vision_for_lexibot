"""
LexiBot Computer Vision System

A sophisticated computer vision system for detecting and identifying famous artworks
in real-time, designed for LexiBot tour guide robot.
"""

__version__ = "1.0.0"
__author__ = "Grigor Crandon"
__description__ = "Computer Vision System for LexiBot Tour Guide Robot"

from src.detection.detector import ArtworkDetector, RealTimeDetector
from src.detection.distance_estimator import DistanceEstimator
from src.detection.mqtt_client import MQTTClient
from src.training.train_model import ModelTrainer

__all__ = [
    'ArtworkDetector',
    'RealTimeDetector', 
    'DistanceEstimator',
    'MQTTClient',
    'ModelTrainer'
]
