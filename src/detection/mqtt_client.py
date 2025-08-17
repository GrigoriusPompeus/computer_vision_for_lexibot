"""
MQTT client for real-time communication with LexiBot robot system
"""

import json
import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    logging.warning("MQTT library not available. Install with: pip install paho-mqtt")

from src.utils.config import MQTT_BROKER, MQTT_PORT, MQTT_TOPIC, MQTT_QOS


class MQTTClient:
    """
    MQTT client for publishing detection results to LexiBot robot system.
    """
    
    def __init__(self, broker: str = MQTT_BROKER, port: int = MQTT_PORT, 
                 topic: str = MQTT_TOPIC, client_id: Optional[str] = None):
        """
        Initialize MQTT client.
        
        Args:
            broker: MQTT broker address
            port: MQTT broker port
            topic: Topic to publish detection results
            client_id: Unique client identifier
        """
        self.broker = broker
        self.port = port
        self.topic = topic
        self.connected = False
        self.client = None
        
        if not MQTT_AVAILABLE:
            logging.error("MQTT not available - communication disabled")
            return
        
        # Create MQTT client
        if client_id is None:
            client_id = f"lexibot_vision_{int(time.time())}"
            
        self.client = mqtt.Client(client_id)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_publish = self._on_publish
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _on_connect(self, client, userdata, flags, rc):
        """Callback for when client connects to broker."""
        if rc == 0:
            self.connected = True
            self.logger.info(f"Connected to MQTT broker {self.broker}:{self.port}")
        else:
            self.logger.error(f"Failed to connect to MQTT broker. Code: {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """Callback for when client disconnects from broker."""
        self.connected = False
        self.logger.info("Disconnected from MQTT broker")
    
    def _on_publish(self, client, userdata, mid):
        """Callback for when message is published."""
        self.logger.debug(f"Message {mid} published successfully")
    
    def connect(self) -> bool:
        """
        Connect to MQTT broker.
        
        Returns:
            True if connection successful, False otherwise
        """
        if not MQTT_AVAILABLE or self.client is None:
            return False
        
        try:
            self.client.connect(self.broker, self.port, 60)
            self.client.loop_start()  # Start background thread
            
            # Wait for connection
            timeout = 5  # seconds
            start_time = time.time()
            while not self.connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            return self.connected
        except Exception as e:
            self.logger.error(f"Error connecting to MQTT broker: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from MQTT broker."""
        if self.client and self.connected:
            self.client.loop_stop()
            self.client.disconnect()
    
    def publish_detections(self, detections: List[Dict[str, Any]]) -> bool:
        """
        Publish detection results to MQTT topic.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            True if published successfully, False otherwise
        """
        if not self.connected or not detections:
            return False
        
        # Create message payload
        payload = {
            "timestamp": datetime.now().isoformat(),
            "detections": detections,
            "count": len(detections)
        }
        
        try:
            # Convert to JSON
            json_payload = json.dumps(payload, indent=2)
            
            # Publish message
            result = self.client.publish(self.topic, json_payload, qos=MQTT_QOS)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                self.logger.info(f"📤 MQTT Published: {len(detections)} detections")
                self.logger.debug(f"Payload: {json_payload}")
                return True
            else:
                self.logger.error(f"Failed to publish MQTT message. RC: {result.rc}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error publishing MQTT message: {e}")
            return False
    
    def publish_detection(self, label: str, confidence: float, 
                         bbox: List[int], distance: Optional[float] = None) -> bool:
        """
        Publish a single detection result.
        
        Args:
            label: Object class label
            confidence: Detection confidence score
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            distance: Estimated distance in cm (optional)
            
        Returns:
            True if published successfully, False otherwise
        """
        detection = {
            "label": label,
            "confidence": round(confidence, 3),
            "bbox": bbox
        }
        
        if distance is not None:
            detection["distance_cm"] = distance
        
        return self.publish_detections([detection])
    
    def is_connected(self) -> bool:
        """
        Check if client is connected to broker.
        
        Returns:
            True if connected, False otherwise
        """
        return self.connected and MQTT_AVAILABLE
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current MQTT client status.
        
        Returns:
            Status dictionary
        """
        return {
            "mqtt_available": MQTT_AVAILABLE,
            "connected": self.connected,
            "broker": self.broker,
            "port": self.port,
            "topic": self.topic
        }
