#!/usr/bin/env python3
"""
Real-time Emotion Detection with OpenCV and MediaPipe

This script demonstrates how to load and use the emotion classifier
exported from the TensorFlow.js web application in a real-time video feed.

Requirements:
    pip install numpy scikit-learn opencv-python mediapipe

Usage:
    python main.py path/to/emotion_classifier_YYYY-MM-DD.json
"""

import json
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import sys
from datetime import datetime
import cv2

# Try to import MediaPipe, fallback if not available
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("✅ MediaPipe available - Full face detection support")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️  MediaPipe not available - Using basic face detection")
    print("💡 Install MediaPipe for better accuracy: pip install mediapipe")

def load_exported_model(json_file_path):
    """
    Load the exported emotion classifier model from JSON file.
    
    Args:
        json_file_path (str): Path to the exported JSON model file
        
    Returns:
        tuple: (classifier, metadata, classes)
    """
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        print(f"📁 Loading model: {json_file_path}")
        print(f"📊 Model Info:")
        print(f"   - Version: {data['metadata']['version']}")
        print(f"   - Export Date: {data['metadata']['exportDate']}")
        print(f"   - Total Classes: {data['metadata']['totalClasses']}")
        print(f"   - Total Samples: {data['metadata']['totalSamples']}")
        print(f"   - Feature Dimensions: {data['metadata']['featureDimensions']}")
        print(f"   - Classes: {data['classes']}")
        print(f"   - Sample Counts: {data['metadata']['sampleCounts']}")
        
        # Prepare data for scikit-learn
        X = []  # Features (52-dimensional blend shape vectors)
        y = []  # Labels (emotion names)
        
        for emotion_class, samples in data['dataset'].items():
            for sample in samples:
                X.append(sample)
                y.append(emotion_class)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"\n🔢 Data Shape: {X.shape}")
        print(f"🏷️  Labels: {np.unique(y)}")
        
        # Create and train KNN classifier
        knn = KNeighborsClassifier(n_neighbors=10, weights='distance')
        knn.fit(X, y)
        
        print("✅ Model loaded and trained successfully!")
        
        return knn, data['metadata'], data['classes']
        
    except FileNotFoundError:
        print(f"❌ Error: File '{json_file_path}' not found.")
        return None, None, None
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON file '{json_file_path}'.")
        return None, None, None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None, None

def predict_emotion(classifier, blend_shapes_vector):
    """
    Predict emotion from a 52-dimensional blend shapes vector.
    
    Args:
        classifier: Trained KNN classifier
        blend_shapes_vector (list): 52-dimensional vector of blend shape scores
        
    Returns:
        tuple: (predicted_emotion, confidence_scores)
    """
    if len(blend_shapes_vector) != 52:
        raise ValueError(f"Expected 52 features, got {len(blend_shapes_vector)}")
    
    # Reshape for single prediction
    X_pred = np.array(blend_shapes_vector).reshape(1, -1)
    
    # Get prediction and probabilities
    prediction = classifier.predict(X_pred)[0]
    probabilities = classifier.predict_proba(X_pred)[0]
    
    # Get confidence scores for all classes
    classes = classifier.classes_
    confidences = dict(zip(classes, probabilities))
    
    return prediction, confidences

def demo_prediction(classifier, classes):
    """
    Demo function showing how to use the model with sample data.
    """
    print("\n🎭 Demo Prediction:")
    print("=" * 50)
    
    # Create a sample blend shapes vector (52 features)
    # In real usage, this would come from MediaPipe Face Mesh
    np.random.seed(42)  # For reproducible demo
    sample_blend_shapes = np.random.random(52).tolist()
    
    try:
        prediction, confidences = predict_emotion(classifier, sample_blend_shapes)
        
        print(f"🎯 Predicted Emotion: {prediction}")
        print(f"📊 Confidence Scores:")
        
        # Sort by confidence (highest first)
        sorted_confidences = sorted(confidences.items(), key=lambda x: x[1], reverse=True)
        
        for emotion, confidence in sorted_confidences[:5]:  # Top 5
            print(f"   {emotion}: {confidence:.3f} ({confidence*100:.1f}%)")
            
    except Exception as e:
        print(f"❌ Prediction error: {e}")

def real_time_emotion_detection(classifier, metadata, classes):
    """
    Real-time emotion detection using webcam and OpenCV.
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
    
    # Emotion emoji mapping
    emotion_emojis = {
        'Happy': '😀', 'Laughing': '🤣', 'Joy': '😂', 'Nervous': '😅',
        'Relief': '🥲', 'Emotional': '🥹', 'Wink': '😉', 'Smile': '😊',
        'Love': '🥰', 'Heart': '😍', 'Kiss': '😘', 'Neutral': '😐',
        'Expressionless': '😑', 'Unamused': '😒', 'EyeRoll': '🙄',
        'Sigh': '😮‍💨', 'Grimace': '😬', 'Lying': '🤥', 'Calm': '😌',
        'Sad': '😔', 'Sleepy': '😪', 'Drooling': '🤤', 'Sleeping': '😴',
        'Angry': '😠', 'Surprised': '😮', 'Confused': '😕', 'Worried': '😟',
        'Fear': '😨', 'Cry': '😢', 'Disappointed': '😞'
    }
    
    print("\n🎬 Starting real-time emotion detection...")
    print("📝 Controls:")
    print("   - Press 'q' to quit")
    print("   - Press 's' to save screenshot")
    print("   - Press 'r' to reset detection")
    
    frame_count = 0
    last_prediction = "Neutral"
    last_confidence = 0.0
    last_all_confidences = {}
    
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
        
        # Detect faces and extract blend shapes
        blend_shapes = None
        
        if MEDIAPIPE_AVAILABLE:
            # Use MediaPipe for face detection
            results = face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                # Draw face mesh
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                        None, mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                    )
                
                # Extract blend shapes (simplified - you'll need to implement this properly)
                # For now, we'll use a placeholder
                blend_shapes = extract_blend_shapes_mediapipe(results, frame_count)
        else:
            # Use OpenCV for basic face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            if len(faces) > 0:
                # Generate mock blend shapes for demonstration
                blend_shapes = generate_mock_blend_shapes(frame_count)
        
        # Predict emotion if blend shapes are available
        if blend_shapes is not None:
            try:
                prediction, confidences = predict_emotion(classifier, blend_shapes)
                confidence = confidences[prediction]
                
                # Update display values
                last_prediction = prediction
                last_confidence = confidence
                last_all_confidences = confidences
                
                # Debug output every 30 frames to show changing percentages
                if frame_count % 30 == 0:
                    print(f"🎯 Frame {frame_count}: {prediction} ({confidence:.1%})")
                    for emotion, conf in sorted(confidences.items(), key=lambda x: x[1], reverse=True)[:3]:
                        print(f"   {emotion}: {conf:.1%}")
                
            except Exception as e:
                print(f"⚠️  Prediction error: {e}")
        
        # Display emotion information on frame
        display_emotion_info(frame, last_prediction, last_confidence, emotion_emojis, last_all_confidences)
        
        # Show frame
        cv2.imshow('Real-time Emotion Detection', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save screenshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"emotion_screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Screenshot saved: {filename}")
        elif key == ord('r'):
            # Reset detection
            last_prediction = "Neutral"
            last_confidence = 0.0
            last_all_confidences = {}
            print("🔄 Detection reset")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("🎬 Real-time detection ended")

def extract_blend_shapes_mediapipe(results, frame_count=None):
    """
    Extract blend shapes from MediaPipe results.
    This is a simplified implementation - you'll need to implement proper blend shape extraction.
    """
    # For now, return mock blend shapes
    # In a real implementation, you would extract the actual 52 blend shape scores
    return generate_mock_blend_shapes(frame_count)

def generate_mock_blend_shapes(frame_count=None):
    """
    Generate mock blend shapes for demonstration purposes.
    In real usage, these would come from MediaPipe Face Mesh.
    """
    # Generate dynamic blend shapes that change over time
    if frame_count is not None:
        # Use frame count to create time-based variation
        np.random.seed(42 + (frame_count // 10))  # Change every 10 frames
    else:
        np.random.seed(None)  # Use current time
    
    # Generate random blend shapes with some structure
    blend_shapes = np.random.random(52) * 0.5  # Scale down for more realistic values
    
    # Add some periodic variation to make it more interesting
    if frame_count is not None:
        # Add sine wave variation to some blend shapes
        for i in range(0, 52, 8):  # Every 8th blend shape gets variation
            variation = 0.2 * np.sin(frame_count * 0.1 + i * 0.5)
            blend_shapes[i] = np.clip(blend_shapes[i] + variation, 0, 1)
    
    return blend_shapes.tolist()

def display_emotion_info(frame, emotion, confidence, emotion_emojis, all_confidences=None):
    """
    Display emotion information on the video frame with all emotion percentages.
    """
    # Get emoji for the emotion
    emoji = emotion_emojis.get(emotion, '❓')
    
    # Set text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    small_font_scale = 0.5
    medium_font_scale = 0.7
    large_font_scale = 1.0
    thickness = 1
    
    # Calculate panel dimensions
    panel_width = 300
    panel_height = 200
    
    # Draw main background rectangle
    cv2.rectangle(frame, (10, 10), (10 + panel_width, 10 + panel_height), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (10 + panel_width, 10 + panel_height), (255, 255, 255), 2)
    
    # Draw primary emotion (larger)
    primary_text = f"{emoji} {emotion}"
    cv2.putText(frame, primary_text, (20, 40), font, large_font_scale, (0, 255, 0), thickness + 1)
    
    # Draw primary confidence
    primary_conf_text = f"{confidence:.1%}"
    cv2.putText(frame, primary_conf_text, (20, 70), font, medium_font_scale, (255, 255, 0), thickness)
    
    # Draw separator line
    cv2.line(frame, (20, 85), (20 + panel_width - 40, 85), (255, 255, 255), 1)
    
    # Draw all emotion percentages if available
    if all_confidences:
        y_offset = 105
        cv2.putText(frame, "All Emotions:", (20, y_offset), font, small_font_scale, (200, 200, 200), thickness)
        y_offset += 20
        
        # Sort emotions by confidence (highest first)
        sorted_emotions = sorted(all_confidences.items(), key=lambda x: x[1], reverse=True)
        
        for i, (emotion_name, conf) in enumerate(sorted_emotions[:5]):  # Show top 5
            if y_offset > panel_height - 10:  # Don't exceed panel height
                break
                
            # Get emoji for this emotion
            emoji_char = emotion_emojis.get(emotion_name, '❓')
            
            # Color based on confidence
            if conf > 0.7:
                color = (0, 255, 0)  # Green for high confidence
            elif conf > 0.5:
                color = (0, 255, 255)  # Yellow for medium confidence
            else:
                color = (200, 200, 200)  # Gray for low confidence
            
            # Draw emotion line
            emotion_line = f"{emoji_char} {emotion_name}: {conf:.1%}"
            cv2.putText(frame, emotion_line, (20, y_offset), font, small_font_scale, color, thickness)
            y_offset += 15
    
    # Add status indicator
    status_color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255) if confidence > 0.5 else (0, 0, 255)
    cv2.circle(frame, (frame.shape[1] - 30, 30), 10, status_color, -1)

def main():
    """
    Main function to demonstrate model usage.
    """
    if len(sys.argv) != 2:
        print("Usage: python main.py <path_to_exported_model.json>")
        print("\nExample:")
        print("python main.py emotion_classifier_2024-01-15.json")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    print("🤖 Real-time Emotion Detection with OpenCV")
    print("=" * 50)
    
    # Load the model
    classifier, metadata, classes = load_exported_model(model_path)
    
    if classifier is None:
        sys.exit(1)
    
    # Ask user what they want to do
    print("\n🎯 What would you like to do?")
    print("1. Real-time emotion detection (webcam)")
    print("2. Demo prediction with sample data")
    print("3. Both")
    
    choice = input("\nEnter your choice (1/2/3): ").strip()
    
    if choice in ['1', '3']:
        print("\n🎥 Starting real-time detection...")
        real_time_emotion_detection(classifier, metadata, classes)
    
    if choice in ['2', '3']:
        print("\n🧪 Running demo prediction...")
        demo_prediction(classifier, classes)
    
    print("\n💡 Usage Tips:")
    print("- The model expects 52-dimensional blend shape vectors")
    print("- Real-time detection works best with good lighting")
    print("- Higher confidence scores indicate more reliable predictions")
    print("- Press 'q' to quit, 's' to save screenshot during video detection")
    
    print("\n🔗 Integration with MediaPipe:")
    print("1. Use MediaPipe Face Mesh to detect facial landmarks")
    print("2. Extract the 52 blend shape scores")
    print("3. Pass the scores to predict_emotion() function")
    print("4. Use the predicted emotion for your application")

if __name__ == "__main__":
    main()
