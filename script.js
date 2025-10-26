// Copyright 2023 The MediaPipe Authors.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//      http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";
const { FaceLandmarker, FilesetResolver, DrawingUtils } = vision;
const demosSection = document.getElementById("demos");
const videoBlendShapes = document.getElementById("video-blend-shapes");

// NEW: Elements for the Emoji Display
const bigEmojiOutput = document.getElementById("big-emoji-output");
const emotionLabel = document.getElementById("emotion-label");

// NEW: Globals for Custom Classifier
const classifier = knnClassifier.create();
let isTraining = false;
let currentEmotionClass = null;
let sampleCount = {}; // e.g., { 'Grinning': 0, 'Tired': 0 }

// NEW: UI Elements for Training
const trainingStatusEl = document.getElementById("training-status");
const trainButton = document.getElementById("train-button");
const emojiDropdown = document.getElementById("emoji-dropdown");
const collectButton = document.getElementById("collect-button");

let faceLandmarker;
let runningMode = "IMAGE";
let enableWebcamButton;
let webcamRunning = false;
const videoWidth = 480;

// Function to update the training status UI
function updateTrainingStatus() {
  const totalSamples = Object.values(sampleCount).reduce((sum, count) => sum + count, 0);
  const statusText = isTraining ? `Collecting: ${currentEmotionClass} (${sampleCount[currentEmotionClass] || 0})` : 'Ready to collect';
  trainingStatusEl.innerHTML = `<p>Status: <strong>${statusText}</strong> | Total Samples: <strong>${totalSamples}</strong></p>`;

  if (totalSamples > 0) {
    trainButton.disabled = false;
  }
}

// Before we can use FaceLandmarker class we must wait for it to finish
// loading. Machine Learning models can be large and take a moment to
// get everything needed to run.
async function createFaceLandmarker() {
  try {
    console.log("Initializing FaceLandmarker...");
    const filesetResolver = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
    );
    
    // Try GPU first, fallback to CPU if GPU fails
    let delegate = "GPU";
    try {
      faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
        baseOptions: {
          modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
          delegate: "GPU"
        },
        outputFaceBlendshapes: true,
        runningMode,
        numFaces: 1
      });
      console.log("✅ FaceLandmarker initialized with GPU delegate");
    } catch (gpuError) {
      console.warn("⚠️ GPU delegate failed, falling back to CPU:", gpuError.message);
      delegate = "CPU";
      faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
        baseOptions: {
          modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
          delegate: "CPU"
        },
        outputFaceBlendshapes: true,
        runningMode,
        numFaces: 1
      });
      console.log("✅ FaceLandmarker initialized with CPU delegate");
    }
    
    demosSection?.classList.remove("invisible");
    console.log("🎉 FaceLandmarker ready!");
  } catch (error) {
    console.error("❌ Failed to initialize FaceLandmarker:", error);
    alert("Failed to initialize face detection. Please refresh the page and try again.");
  }
}
createFaceLandmarker();

// Function to handle dropdown selection
function handleDropdownChange() {
  const selectedOption = emojiDropdown.options[emojiDropdown.selectedIndex];
  
  if (selectedOption.value) {
    const emotion = selectedOption.dataset.emotion;
    const emoji = selectedOption.value;
    
    // Enable collect button and update its text
    collectButton.disabled = false;
    collectButton.innerHTML = `<span class="mdc-button__label">📊 Collect ${emoji} ${emotion}</span>`;
    
    // Update current emotion class
    currentEmotionClass = emotion;
    
    console.log(`Selected emotion: ${emotion} (${emoji})`);
  } else {
    // No selection, disable collect button
    collectButton.disabled = true;
    collectButton.innerHTML = `<span class="mdc-button__label">📊 Start Collecting</span>`;
    currentEmotionClass = null;
  }
}

// Function to handle collect button click
function handleCollectButtonClick() {
  if (!currentEmotionClass) {
    alert("Please select an emotion from the dropdown first.");
    return;
  }

  if (isTraining && currentEmotionClass) {
    // Stop collecting
    isTraining = false;
    currentEmotionClass = null;
    collectButton.classList.remove('mdc-button--collecting');
    collectButton.innerHTML = `<span class="mdc-button__label">📊 Start Collecting</span>`;
    collectButton.disabled = true;
    emojiDropdown.disabled = false;
  } else {
    // Start collecting
    isTraining = true;
    collectButton.classList.add('mdc-button--collecting');
    collectButton.innerHTML = `<span class="mdc-button__label">⏹️ Stop Collecting</span>`;
    emojiDropdown.disabled = true;
  }
  
  updateTrainingStatus();
}

// Function to handle class button clicks
function handleClassButtonClick(event) {
  const emotion = event.currentTarget.dataset.emotion;

  // Start training/collection mode for the selected emotion
  if (isTraining && currentEmotionClass === emotion) {
    // Stop collecting if the same button is pressed again
    isTraining = false;
    currentEmotionClass = null;
    event.currentTarget.classList.remove('mdc-button--collecting');
  } else {
    // Start collecting for a new emotion
    isTraining = true;
    currentEmotionClass = emotion;

    // Reset all buttons' state and set the new one
    event.currentTarget.classList.add('mdc-button--collecting');
  }
  updateTrainingStatus();
}

// Function to handle the Train button click
function handleTrainButtonClick() {
  if (classifier.getNumClasses() > 0) {
    isTraining = false;
    currentEmotionClass = null;
    collectButton.disabled = true;
    emojiDropdown.disabled = true;
    trainButton.disabled = true;
    trainButton.textContent = '✅ Prediction Mode Active';
    
    updateTrainingStatus();
  }
}

// Attach listeners once the document is ready
window.addEventListener('DOMContentLoaded', () => {
  // Dropdown and collect button listeners
  emojiDropdown.addEventListener('change', handleDropdownChange);
  collectButton.addEventListener('click', handleCollectButtonClick);
  
  // Train button listener
  trainButton.addEventListener('click', handleTrainButtonClick);
});

/********************************************************************
// Demo: Continuously grab image from webcam stream and detect it.
********************************************************************/

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");

// Add debugging for video element
console.log("📹 Video element:", video);
console.log("🎨 Canvas element:", canvasElement);

const canvasCtx = canvasElement.getContext("2d");
console.log("🎨 Canvas context:", canvasCtx);

// Check if webcam access is supported.
function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

// If webcam supported, add event listener to button for when user
// wants to activate it.
if (hasGetUserMedia()) {
  enableWebcamButton = document.getElementById("webcamButton");
  enableWebcamButton.addEventListener("click", enableCam);
} else {
  console.warn("getUserMedia() is not supported by your browser");
}

// Enable the live webcam view and start detection.
function enableCam(event) {
  console.log("🎥 EnableCam called");
  
  if (!faceLandmarker) {
    console.log("Wait! faceLandmarker not loaded yet.");
    alert("Face detection model is still loading. Please wait a moment and try again.");
    return;
  }

  if (webcamRunning === true) {
    webcamRunning = false;
    enableWebcamButton.innerText = "🚀 ENABLE WEBCAM";
    console.log("📹 Webcam stopped");
  } else {
    webcamRunning = true;
    enableWebcamButton.innerText = "⏹️ DISABLE WEBCAM";
    console.log("📹 Starting webcam...");
  }

  // getUsermedia parameters.
  const constraints = {
    video: {
      width: { ideal: 640 },
      height: { ideal: 480 },
      facingMode: "user"
    }
  };

  // Activate the webcam stream.
  navigator.mediaDevices.getUserMedia(constraints)
    .then((stream) => {
      console.log("✅ Webcam stream obtained");
      video.srcObject = stream;
      video.addEventListener("loadeddata", () => {
        console.log("📹 Video loaded, starting prediction");
        console.log("📹 Video dimensions:", video.videoWidth, "x", video.videoHeight);
        predictWebcam();
      });
      
      // Add event listeners for debugging
      video.addEventListener("loadedmetadata", () => {
        console.log("📹 Video metadata loaded");
      });
      
      video.addEventListener("canplay", () => {
        console.log("📹 Video can play");
      });
      
      // Also add error handling for video
      video.addEventListener("error", (e) => {
        console.error("❌ Video error:", e);
      });
    })
    .catch((error) => {
      console.error("❌ Error accessing webcam:", error);
      alert("Could not access webcam. Please check permissions and try again.");
      webcamRunning = false;
      enableWebcamButton.innerText = "🚀 ENABLE WEBCAM";
    });
}

let lastVideoTime = -1;
let results = undefined;
const drawingUtils = new DrawingUtils(canvasCtx);
async function predictWebcam() {
  try {
    // Clear canvas before drawing
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    // Set canvas to overlay the video
    canvasElement.style.position = "absolute";
    canvasElement.style.top = "0";
    canvasElement.style.left = "0";
    canvasElement.style.width = "100%";
    canvasElement.style.height = "100%";
    canvasElement.width = video.videoWidth;
    canvasElement.height = video.videoHeight;
    
    // Now let's start detecting the stream.
    if (runningMode === "IMAGE") {
      runningMode = "VIDEO";
      await faceLandmarker.setOptions({ runningMode: runningMode });
      console.log("🔄 Switched to VIDEO mode");
    }
    
    let startTimeMs = performance.now();
    if (lastVideoTime !== video.currentTime) {
      lastVideoTime = video.currentTime;
      results = faceLandmarker.detectForVideo(video, startTimeMs);
    }
    
    if (results?.faceLandmarks && results.faceLandmarks.length > 0) {
      console.log(`🎯 Detected ${results.faceLandmarks.length} face(s)`);
      
      for (const landmarks of results.faceLandmarks) {
        drawingUtils.drawConnectors(
          landmarks,
          FaceLandmarker.FACE_LANDMARKS_TESSELATION,
          { color: "#C0C0C070", lineWidth: 1 }
        );
        drawingUtils.drawConnectors(
          landmarks,
          FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
          { color: "#FF3030" }
        );
        drawingUtils.drawConnectors(
          landmarks,
          FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW,
          { color: "#FF3030" }
        );
        drawingUtils.drawConnectors(
          landmarks,
          FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
          { color: "#30FF30" }
        );
        drawingUtils.drawConnectors(
          landmarks,
          FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW,
          { color: "#30FF30" }
        );
        drawingUtils.drawConnectors(
          landmarks,
          FaceLandmarker.FACE_LANDMARKS_FACE_OVAL,
          { color: "#E0E0E0" }
        );
        drawingUtils.drawConnectors(
          landmarks,
          FaceLandmarker.FACE_LANDMARKS_LIPS,
          { color: "#E0E0E0" }
        );
        drawingUtils.drawConnectors(
          landmarks,
          FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS,
          { color: "#FF3030" }
        );
        drawingUtils.drawConnectors(
          landmarks,
          FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS,
          { color: "#30FF30" }
        );
      }
    }
    
    drawBlendShapes(videoBlendShapes, results?.faceBlendshapes);

    // --- ML TRAINING AND PREDICTION LOGIC ---
    if (results?.faceBlendshapes && results.faceBlendshapes.length > 0) {
      // 1. Get the 52-dimensional feature vector (the blend shape scores)
      const features = getBlendShapeFeatures(results.faceBlendshapes);

      if (features) {
        // --- DATA COLLECTION MODE ---
        if (isTraining && currentEmotionClass) {
          // Add the current 52 scores (features) with the current label (currentEmotionClass)
          classifier.addExample(features, currentEmotionClass);
          sampleCount[currentEmotionClass] = (sampleCount[currentEmotionClass] || 0) + 1;
          updateTrainingStatus();
          
          // Show collecting feedback
          bigEmojiOutput.innerHTML = `<span class="emoji">📊</span>`;
          emotionLabel.textContent = `Collecting ${currentEmotionClass}...`;
        }
        
        // --- PREDICTION MODE ---
        else if (!isTraining && classifier.getNumClasses() > 0) {
          // Only try to predict if the model has been trained (has examples)
          const kValue = 10; // K-Nearest Neighbors parameter
          classifier.predictClass(features, kValue).then(({ label, confidences }) => {
            // Find the confidence score for the predicted label
            const score = confidences[label] || 0;
            
            // Get the emoji (comprehensive map for all emotions)
            const emojiMap = {
              'Happy': '😀',
              'Laughing': '🤣',
              'Joy': '😂',
              'Nervous': '😅',
              'Relief': '🥲',
              'Emotional': '🥹',
              'Wink': '😉',
              'Smile': '😊',
              'Love': '🥰',
              'Heart': '😍',
              'Kiss': '😘',
              'Neutral': '😐',
              'Expressionless': '😑',
              'Unamused': '😒',
              'EyeRoll': '🙄',
              'Sigh': '😮‍💨',
              'Grimace': '😬',
              'Lying': '🤥',
              'Calm': '😌',
              'Sad': '😔',
              'Sleepy': '😪',
              'Drooling': '🤤',
              'Sleeping': '😴',
              'Angry': '😠',
              'Surprised': '😮',
              'Confused': '😕',
              'Worried': '😟',
              'Fear': '😨',
              'Cry': '😢',
              'Disappointed': '😞',
              // Legacy mappings for backward compatibility
              'Grinning': '😀',
              'Tired': '😫',
              'Pouting': '😒'
            };

            // The new primary emotion result
            const mlPrediction = { 
              name: label, 
              emoji: emojiMap[label] || '❓', 
              score: score * 100 // Convert to percentage
            };

            // Update UI elements with the ML prediction result
            bigEmojiOutput.innerHTML = `<span class="emoji">${mlPrediction.emoji}</span>`;
            emotionLabel.textContent = `${mlPrediction.name} (${mlPrediction.score.toFixed(1)}%)`;
            
            console.log(`🤖 ML Prediction: ${mlPrediction.name} (${mlPrediction.score.toFixed(1)}%)`);
          });
        }
        
        // --- DEFAULT STATE ---
        else {
          bigEmojiOutput.innerHTML = `<span class="emoji">😐</span>`;
          emotionLabel.textContent = "Neutral";
        }
        
        // To avoid memory leaks, explicitly dispose of the tensor
        features.dispose();
      }
    } else if (webcamRunning === true) {
      // If no face is detected, display a prompt
      bigEmojiOutput.innerHTML = `<span class="emoji">👤</span>`;
      emotionLabel.textContent = "No Face Detected";
    }
    // --- END ML LOGIC ---

    // Call this function again to keep predicting when the browser is ready.
    if (webcamRunning === true) {
      window.requestAnimationFrame(predictWebcam);
    }
  } catch (error) {
    console.error("❌ Error in predictWebcam:", error);
  }
}

/**
 * Extracts the 52 blend shape scores from MediaPipe results into a tf.Tensor1D.
 * @param {object} faceBlendshapes - The raw MediaPipe blend shape results.
 * @returns {tf.Tensor1D | null} A 1D tensor of 52 floating-point blend shape scores.
 */
function getBlendShapeFeatures(faceBlendshapes) {
  if (!faceBlendshapes || !faceBlendshapes.length) {
    return null;
  }

  const scores = faceBlendshapes[0].categories;
  const featuresArray = scores.map(shape => shape.score);

  // Ensure we have exactly 52 features
  if (featuresArray.length !== 52) {
    console.error("Expected 52 blend shapes, found:", featuresArray.length);
    return null;
  }

  // Convert the array of 52 scores into a 1D TensorFlow Tensor
  return tf.tensor1d(featuresArray);
}

function drawBlendShapes(el, blendShapes) {
  if (!blendShapes.length) {
    return;
  }

  console.log(blendShapes[0]);

  let htmlMaker = "";
  blendShapes[0].categories.map((shape) => {
    htmlMaker += `
      <li class="blend-shapes-item">
        <span class="blend-shapes-label">${
          shape.displayName || shape.categoryName
        }</span>
        <span class="blend-shapes-value" style="width: calc(${
          +shape.score * 100
        }% - 120px)">${(+shape.score).toFixed(4)}</span>
      </li>
    `;
  });

  el.innerHTML = htmlMaker;
}