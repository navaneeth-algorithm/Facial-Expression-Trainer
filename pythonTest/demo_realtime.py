#!/usr/bin/env python3
"""
Real-time Emotion Detection Demo (No Model Required)

This script demonstrates real-time emotion detection using OpenCV
without requiring a trained model. Perfect for testing the setup.

Requirements:
    pip install opencv-python mediapipe

Usage:
    python demo_realtime.py
"""

import cv2
import numpy as np
from datetime import datetime

# Try to import MediaPipe, fallback if not available
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("✅ MediaPipe available - Full face detection support")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️  MediaPipe not available - Using basic face detection")
    print("💡 Install MediaPipe for better accuracy: pip install mediapipe")

def demo_realtime_detection():
    """
    Demo real-time face detection with emotion simulation.
    """
    print("🎥 Initializing webcam...")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return
    
    # Set webcam properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Initialize MediaPipe Face Mesh if available
    if MEDIAPIPE_AVAILABLE:
        mp_face_mesh = mp.solutions.face_mesh
        mp_drawing = mp.solutions.drawing_utils
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("✅ MediaPipe Face Mesh initialized")
    else:
        # Fallback to OpenCV face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        print("✅ OpenCV face detection initialized")
    
    # Mock emotions for demonstration
    mock_emotions = ['😀 Happy', '😐 Neutral', '😮 Surprised', '😞 Sad', '😠 Angry', '😊 Smile']
    current_emotion = '😐 Neutral'
    emotion_index = 0
    
    print("\n🎬 Starting real-time face detection demo...")
    print("📝 Controls:")
    print("   - Press 'q' to quit")
    print("   - Press 's' to save screenshot")
    print("   - Press 'e' to cycle through emotions")
    print("   - Press 'r' to reset")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Could not read frame from webcam")
            break
        
        frame_count += 1
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_detected = False
        
        if MEDIAPIPE_AVAILABLE:
            # Use MediaPipe for face detection
            results = face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                face_detected = True
                # Draw face mesh
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                        None, mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                    )
        else:
            # Use OpenCV for basic face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                face_detected = True
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Display emotion information on frame
        display_demo_info(frame, current_emotion, face_detected, frame_count)
        
        # Show frame
        cv2.imshow('Real-time Face Detection Demo', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save screenshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"demo_screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Screenshot saved: {filename}")
        elif key == ord('e'):
            # Cycle through emotions
            emotion_index = (emotion_index + 1) % len(mock_emotions)
            current_emotion = mock_emotions[emotion_index]
            print(f"🎭 Emotion changed to: {current_emotion}")
        elif key == ord('r'):
            # Reset
            current_emotion = '😐 Neutral'
            emotion_index = 0
            print("🔄 Demo reset")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("🎬 Demo ended")

def display_demo_info(frame, emotion, face_detected, frame_count):
    """
    Display demo information on the video frame.
    """
    # Prepare text
    status_text = "Face Detected ✅" if face_detected else "No Face Detected ❌"
    emotion_text = f"Emotion: {emotion}"
    frame_text = f"Frame: {frame_count}"
    
    # Set text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    # Draw background rectangle
    cv2.rectangle(frame, (10, 10), (400, 120), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (400, 120), (255, 255, 255), 2)
    
    # Draw text
    cv2.putText(frame, status_text, (20, 35), font, font_scale, (0, 255, 0), thickness)
    cv2.putText(frame, emotion_text, (20, 60), font, font_scale, (255, 255, 0), thickness)
    cv2.putText(frame, frame_text, (20, 85), font, font_scale, (255, 255, 255), thickness)
    
    # Add status indicator
    status_color = (0, 255, 0) if face_detected else (0, 0, 255)
    cv2.circle(frame, (frame.shape[1] - 30, 30), 10, status_color, -1)
    
    # Add instructions
    instructions = [
        "Press 'q' to quit",
        "Press 's' to save screenshot", 
        "Press 'e' to cycle emotions",
        "Press 'r' to reset"
    ]
    
    y_offset = frame.shape[0] - 100
    for i, instruction in enumerate(instructions):
        cv2.putText(frame, instruction, (10, y_offset + i * 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

def main():
    """
    Main function for the demo.
    """
    print("🎭 Real-time Face Detection Demo")
    print("=" * 40)
    print("This demo shows face detection without requiring a trained model.")
    print("Perfect for testing your OpenCV and MediaPipe setup!")
    print()
    
    demo_realtime_detection()
    
    print("\n💡 Next steps:")
    print("1. Train a model in the web application")
    print("2. Export the model")
    print("3. Run: python main.py path/to/exported_model.json")
    print("4. Enjoy real-time emotion detection!")

if __name__ == "__main__":
    main()
