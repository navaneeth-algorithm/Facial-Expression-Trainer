#!/usr/bin/env python3
"""
Python Usage Example for Exported Emotion Classifier Model

This script demonstrates how to load and use the emotion classifier
exported from the TensorFlow.js web application.

Requirements:
    pip install numpy scikit-learn

Usage:
    python python_usage_example.py path/to/emotion_classifier_YYYY-MM-DD.json
"""

import json
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import sys
from datetime import datetime

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

def main():
    """
    Main function to demonstrate model usage.
    """
    if len(sys.argv) != 2:
        print("Usage: python python_usage_example.py <path_to_exported_model.json>")
        print("\nExample:")
        print("python python_usage_example.py emotion_classifier_2024-01-15.json")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    print("🤖 Emotion Classifier - Python Usage Example")
    print("=" * 50)
    
    # Load the model
    classifier, metadata, classes = load_exported_model(model_path)
    
    if classifier is None:
        sys.exit(1)
    
    # Run demo prediction
    demo_prediction(classifier, classes)
    
    print("\n💡 Usage Tips:")
    print("- The model expects 52-dimensional blend shape vectors")
    print("- Use MediaPipe Face Mesh to extract blend shapes from images/video")
    print("- Higher confidence scores indicate more reliable predictions")
    print("- You can adjust k-neighbors parameter for different behavior")
    
    print("\n🔗 Integration with MediaPipe:")
    print("1. Use MediaPipe Face Mesh to detect facial landmarks")
    print("2. Extract the 52 blend shape scores")
    print("3. Pass the scores to predict_emotion() function")
    print("4. Use the predicted emotion for your application")

if __name__ == "__main__":
    main()
