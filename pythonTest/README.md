# Python Emotion Classifier Usage

This folder contains Python scripts to load and use emotion classifier models exported from the web application.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install core requirements
pip install -r requirements.txt

# Or install minimal requirements for basic usage
pip install numpy scikit-learn
```

### 2. Export Model from Web App

1. Open the web application in your browser
2. Train your emotion classifier with different emotions
3. Click "💾 Export Model" button
4. Download the JSON file (e.g., `emotion_classifier_2024-01-15.json`)

### 3. Use the Model in Python

```bash
# Basic usage
python main.py path/to/emotion_classifier_2024-01-15.json

# Example
python main.py ../emotion_classifier_2024-01-15.json
```

## 📁 Files

- **`main.py`** - Main script demonstrating model loading and usage
- **`requirements.txt`** - Python dependencies
- **`README.md`** - This documentation

## 🔧 Usage Examples

### Basic Model Loading

```python
import json
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Load exported model
with open('emotion_classifier_2024-01-15.json', 'r') as f:
    data = json.load(f)

# Prepare training data
X, y = [], []
for emotion_class, samples in data['dataset'].items():
    for sample in samples:
        X.append(sample)
        y.append(emotion_class)

# Train classifier
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X, y)

# Predict emotion
blend_shapes = [0.1, 0.2, ...]  # 52-dimensional vector
prediction = knn.predict([blend_shapes])[0]
print(f"Predicted emotion: {prediction}")
```

### Integration with MediaPipe

```python
import mediapipe as mp
import cv2
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Load your trained model (from main.py)
# ... (load model code) ...

def extract_blend_shapes(image):
    """Extract 52 blend shape scores from image using MediaPipe."""
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if results.multi_face_landmarks:
        # Extract blend shapes (simplified - you'll need to implement this)
        # This is a placeholder - actual implementation depends on MediaPipe version
        blend_shapes = np.random.random(52)  # Replace with actual extraction
        return blend_shapes
    
    return None

# Use with webcam
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    blend_shapes = extract_blend_shapes(frame)
    
    if blend_shapes is not None:
        prediction = knn.predict([blend_shapes])[0]
        cv2.putText(frame, f"Emotion: {prediction}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 📊 Model Format

The exported JSON model contains:

```json
{
  "metadata": {
    "version": "1.0",
    "exportDate": "2024-01-15T10:30:00.000Z",
    "totalClasses": 5,
    "totalSamples": 150,
    "sampleCounts": {"Happy": 30, "Sad": 25, ...},
    "featureDimensions": 52,
    "description": "KNN Classifier trained on MediaPipe Face Blend Shapes"
  },
  "classes": ["Happy", "Sad", "Angry", "Surprised", "Neutral"],
  "dataset": {
    "Happy": [[0.1, 0.2, ...], [0.3, 0.4, ...], ...],
    "Sad": [[0.5, 0.6, ...], [0.7, 0.8, ...], ...],
    ...
  }
}
```

## 🎯 Key Features

- **52-dimensional features** - MediaPipe Face Blend Shapes
- **K-Nearest Neighbors** - Simple, interpretable classification
- **Cross-platform** - Works with any Python ML framework
- **Complete data** - All training samples and metadata included
- **Easy integration** - Compatible with scikit-learn, MediaPipe, OpenCV

## 🔧 Advanced Usage

### Custom KNN Parameters

```python
# Adjust KNN parameters
knn = KNeighborsClassifier(
    n_neighbors=15,        # Number of neighbors
    weights='distance',    # Weight by distance
    algorithm='auto',      # Algorithm to use
    metric='euclidean'    # Distance metric
)
```

### Model Evaluation

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Split data for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train and evaluate
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

### Batch Prediction

```python
def predict_batch(classifier, blend_shapes_list):
    """Predict emotions for multiple samples."""
    predictions = classifier.predict(blend_shapes_list)
    probabilities = classifier.predict_proba(blend_shapes_list)
    
    results = []
    for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
        results.append({
            'prediction': pred,
            'confidence': max(probs),
            'all_probabilities': dict(zip(classifier.classes_, probs))
        })
    
    return results
```

## 🐛 Troubleshooting

### Common Issues

1. **File not found**: Make sure the JSON file path is correct
2. **Invalid JSON**: Ensure the file was exported properly from the web app
3. **Dimension mismatch**: Verify you're using 52-dimensional feature vectors
4. **Import errors**: Install required packages with `pip install -r requirements.txt`

### Debug Mode

```python
# Enable debug output
import logging
logging.basicConfig(level=logging.DEBUG)

# Check model metadata
print(f"Model classes: {data['classes']}")
print(f"Total samples: {data['metadata']['totalSamples']}")
print(f"Feature dimensions: {data['metadata']['featureDimensions']}")
```

## 📚 Additional Resources

- [scikit-learn KNN Documentation](https://scikit-learn.org/stable/modules/neighbors.html)
- [MediaPipe Face Mesh Documentation](https://google.github.io/mediapipe/solutions/face_mesh.html)
- [NumPy Documentation](https://numpy.org/doc/stable/)

## 🤝 Contributing

Feel free to extend this example with:
- Real-time webcam emotion detection
- Model evaluation and visualization
- Integration with other ML frameworks
- Web API endpoints for emotion prediction
