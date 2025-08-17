#!/usr/bin/env python3
"""
Art Detection with MQTT Demo

Demonstrates how your art detection would work with MQTT streaming for LexiBot.
Shows the MQTT messages that would be sent to the robot.
"""

import cv2
import json
import time
from datetime import datetime

def simulate_mqtt_art_detection():
    """Demonstrate art detection with simulated MQTT messages"""
    print("🤖 LexiBot Art Detection MQTT Demo")
    print("=" * 50)
    
    try:
        from ultralytics import YOLO
        
        # Load custom art model
        model = YOLO("models/best.pt")
        print("✅ Custom art detection model loaded")
        
        # Art information for robot tour guide
        art_info = {
            'libre': {
                'name': 'Liberty Leading the People',
                'artist': 'Eugène Delacroix',
                'year': '1830',
                'description': 'This painting commemorates the July Revolution of 1830 in France',
                'tour_script': 'Welcome! This is Liberty Leading the People by Delacroix...'
            },
            'monalisa': {
                'name': 'Mona Lisa',
                'artist': 'Leonardo da Vinci',
                'year': '1503-1519',
                'description': 'The most famous portrait in the world, known for her enigmatic smile',
                'tour_script': 'Behold the famous Mona Lisa! Notice her mysterious smile...'
            },
            'skrik': {
                'name': 'The Scream',
                'artist': 'Edvard Munch',
                'year': '1893',
                'description': 'An expressionist painting representing human anxiety',
                'tour_script': 'This is The Scream, expressing the anxiety of modern life...'
            },
            'starrynight': {
                'name': 'The Starry Night',
                'artist': 'Vincent van Gogh',
                'year': '1889',
                'description': 'A swirling night sky painted during van Gogh\'s time in an asylum',
                'tour_script': 'Observe the swirling patterns in this masterpiece by van Gogh...'
            },
            'sunflower': {
                'name': 'Sunflowers',
                'artist': 'Vincent van Gogh',
                'year': '1888-1889',
                'description': 'A series of still life paintings featuring sunflowers',
                'tour_script': 'These vibrant sunflowers showcase van Gogh\'s unique style...'
            }
        }
        
        print("\n📡 Simulating MQTT Topics:")
        print("   • lexibot/art_detections - Detection data")
        print("   • lexibot/robot_commands - Robot navigation")
        print("   • lexibot/tour_guide - Tourist information")
        
        print("\n📷 Starting art detection...")
        print("🎨 Show artworks to camera - MQTT messages will be displayed")
        print("Press ESC or 'q' to stop")
        
        # Camera setup
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Could not open camera")
            return False
        
        frame_count = 0
        last_mqtt_time = 0
        mqtt_interval = 2.0  # Send updates every 2 seconds
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            current_time = time.time()
            
            # Run art detection
            results = model(frame, conf=0.3)
            
            detected_artworks = []
            
            # Process detections
            if results and len(results) > 0:
                detections = results[0].boxes
                if detections is not None and len(detections) > 0:
                    for detection in detections:
                        # Get detection info
                        box = detection.xyxy[0].cpu().numpy()
                        class_id = int(detection.cls.cpu().numpy()[0])
                        confidence = float(detection.conf.cpu().numpy()[0])
                        class_name = model.names[class_id]
                        
                        if class_name != 'background' and confidence > 0.5:
                            # Get artwork info
                            artwork_data = art_info.get(class_name, {})
                            
                            detected_artworks.append({
                                'class_name': class_name,
                                'confidence': round(confidence, 3),
                                'bbox': [int(x) for x in box],
                                'artwork_info': artwork_data
                            })
                            
                            # Draw detection
                            x1, y1, x2, y2 = map(int, box)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            
                            # Draw label
                            if artwork_data:
                                label = f"{artwork_data['name']} {confidence:.2f}"
                                cv2.putText(frame, label, (x1, y1-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display simulated MQTT messages
            if detected_artworks and (current_time - last_mqtt_time > mqtt_interval):
                print(f"\n📡 MQTT Messages (Frame {frame_count}):")
                print("=" * 40)
                
                # 1. Art Detection Topic
                detection_msg = {
                    'timestamp': datetime.now().isoformat(),
                    'frame_count': frame_count,
                    'detections': len(detected_artworks),
                    'artworks': detected_artworks
                }
                
                print("📤 Topic: lexibot/art_detections")
                print(json.dumps(detection_msg, indent=2))
                
                # 2. Robot Command Topic
                if detected_artworks:
                    artwork = detected_artworks[0]  # Focus on first detection
                    robot_cmd = {
                        'timestamp': datetime.now().isoformat(),
                        'command': 'STOP_AND_GUIDE',
                        'artwork_detected': artwork['artwork_info'].get('name', 'Unknown'),
                        'confidence': artwork['confidence'],
                        'action': 'approach_and_describe'
                    }
                    
                    print(f"\n📤 Topic: lexibot/robot_commands")
                    print(json.dumps(robot_cmd, indent=2))
                    
                    # 3. Tour Guide Topic
                    tour_msg = {
                        'timestamp': datetime.now().isoformat(),
                        'artwork': artwork['artwork_info'].get('name', 'Unknown'),
                        'artist': artwork['artwork_info'].get('artist', 'Unknown'),
                        'year': artwork['artwork_info'].get('year', 'Unknown'),
                        'description': artwork['artwork_info'].get('description', ''),
                        'tour_script': artwork['artwork_info'].get('tour_script', ''),
                        'action': 'start_tour_narration'
                    }
                    
                    print(f"\n📤 Topic: lexibot/tour_guide")
                    print(json.dumps(tour_msg, indent=2))
                    
                    print(f"\n🎨 Detected: {artwork['artwork_info'].get('name', 'Unknown Artwork')}")
                    print(f"🎭 Artist: {artwork['artwork_info'].get('artist', 'Unknown')}")
                    print(f"📅 Year: {artwork['artwork_info'].get('year', 'Unknown')}")
                
                last_mqtt_time = current_time
            
            # Display frame
            try:
                cv2.imshow('🎨 Art Detection MQTT Demo - LexiBot Tour Guide', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
            except cv2.error:
                # No GUI available
                if frame_count % 60 == 0:
                    print(f"📸 Frame {frame_count}: {len(detected_artworks)} artworks detected")
                if frame_count > 300:  # Stop after ~10 seconds
                    break
        
        # Cleanup
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        
        print("\n🛑 MQTT demo stopped")
        print("\n📋 Summary:")
        print("✅ Custom art detection working")
        print("✅ MQTT message format demonstrated") 
        print("✅ Tour guide integration ready")
        print("\n🚀 Next steps:")
        print("   1. Set up MQTT broker on robot")
        print("   2. Integrate with robot navigation")
        print("   3. Add audio tour narration")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    simulate_mqtt_art_detection()
