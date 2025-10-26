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
const imageBlendShapes = document.getElementById("image-blend-shapes");
const videoBlendShapes = document.getElementById("video-blend-shapes");

// NEW: Elements for the Emoji Display
const bigEmojiOutput = document.getElementById("big-emoji-output");
const emotionLabel = document.getElementById("emotion-label");

let faceLandmarker;
let runningMode = "IMAGE";
let enableWebcamButton;
let webcamRunning = false;
const videoWidth = 480;

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

/********************************************************************
// Demo 1: Grab a bunch of images from the page and detection them
// upon click.
********************************************************************/

// In this demo, we have put all our clickable images in divs with the
// CSS class 'detectionOnClick'. Lets get all the elements that have
// this class.
const imageContainers = document.getElementsByClassName("detectOnClick");

// Now let's go through all of these and add a click event listener.
for (let imageContainer of imageContainers) {
  // Add event listener to the child element whichis the img element.
  imageContainer.children[0].addEventListener("click", handleClick);
}

// When an image is clicked, let's detect it and display results!
async function handleClick(event) {
  if (!faceLandmarker) {
    console.log("Wait for faceLandmarker to load before clicking!");
    return;
  }

  if (runningMode === "VIDEO") {
    runningMode = "IMAGE";
    await faceLandmarker.setOptions({ runningMode });
  }
  // Remove all landmarks drawed before
  const allCanvas = event.target.parentNode.getElementsByClassName("canvas");
  for (var i = allCanvas.length - 1; i >= 0; i--) {
    const n = allCanvas[i];
    n.parentNode.removeChild(n);
  }

  // We can call faceLandmarker.detect as many times as we like with
  // different image data each time. This returns a promise
  // which we wait to complete and then call a function to
  // print out the results of the prediction.
  const faceLandmarkerResult = faceLandmarker.detect(event.target);
  const canvas = document.createElement("canvas");
  canvas.setAttribute("class", "canvas");
  canvas.setAttribute("width", event.target.naturalWidth + "px");
  canvas.setAttribute("height", event.target.naturalHeight + "px");
  canvas.style.left = "0px";
  canvas.style.top = "0px";
  canvas.style.width = `${event.target.width}px`;
  canvas.style.height = `${event.target.height}px`;

  event.target.parentNode.appendChild(canvas);
  const ctx = canvas.getContext("2d");
  const drawingUtils = new DrawingUtils(ctx);
  for (const landmarks of faceLandmarkerResult.faceLandmarks) {
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
    drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, {
      color: "#E0E0E0"
    });
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
  drawBlendShapes(imageBlendShapes, faceLandmarkerResult.faceBlendshapes);
}

/********************************************************************
// Demo 2: Continuously grab image from webcam stream and detect it.
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

    // --- NEW EMOJI LOGIC ---
    if (results?.faceBlendshapes && results.faceBlendshapes.length > 0) {
      const emotion = classifyEmotion(results.faceBlendshapes);

      // Update the big emoji display
      bigEmojiOutput.innerHTML = `<span class="emoji">${emotion.emoji}</span>`;
      
      // Update the emotion text label
      emotionLabel.textContent = emotion.name;
      
      // Debug logging
      console.log(`🎭 Emotion detected: ${emotion.name} (${emotion.emoji}) - Score: ${emotion.score.toFixed(3)}`);
    } else if (webcamRunning === true) {
      // If no face is detected, display a prompt
      bigEmojiOutput.innerHTML = `<span class="emoji">👤</span>`;
      emotionLabel.textContent = "No Face Detected";
    }
    // --- END NEW EMOJI LOGIC ---

    // Call this function again to keep predicting when the browser is ready.
    if (webcamRunning === true) {
      window.requestAnimationFrame(predictWebcam);
    }
  } catch (error) {
    console.error("❌ Error in predictWebcam:", error);
  }
}

/**
 * Classifies the 52 blend shape scores into a single major emotion.
 * @param {Array} blendShapes - The array of blend shape categories from MediaPipe.
 * @returns {object} An object containing the primary emotion name and its emoji.
 */
function classifyEmotion(blendShapes) {
  if (!blendShapes || blendShapes.length === 0) {
    return { name: "Detecting...", emoji: "🤔" };
  }

  // Convert blendShapes array to a more easily accessible key-value map
  const scores = blendShapes[0].categories.reduce((acc, category) => {
    acc[category.categoryName] = category.score;
    return acc;
  }, {});

  let primaryEmotion = { name: "Neutral", emoji: "😐", score: 0.2 };

  // --- Primary Emotion Classification Logic ---
  // A threshold is used to ensure the emotion is strong enough to be classified.
  const threshold = 0.55; 

  // 1. HAPPY (Smile)
  // A smile is often a combination of Mouth_Smile and Cheek_Squint.
  const happyScore = scores.mouthSmileLeft * 0.5 + scores.mouthSmileRight * 0.5 + scores.cheekSquintLeft * 0.2 + scores.cheekSquintRight * 0.2;
  if (happyScore > threshold) {
    primaryEmotion = { name: "Happy", emoji: "😊", score: happyScore };
  }

  // 2. SURPRISE (Wide eyes and open mouth)
  const surpriseScore = (scores.mouthOpen || 0) * 0.4 + (scores.eyeWideLeft || 0) * 0.3 + (scores.eyeWideRight || 0) * 0.3;
  if (surpriseScore > threshold && surpriseScore > primaryEmotion.score) {
    primaryEmotion = { name: "Surprised", emoji: "😮", score: surpriseScore };
  }

  // 3. ANGER/SADNESS (Frown/Eyebrow furrow)
  // This can be tricky, using Brow_Down and Mouth_Frown as key indicators.
  const angryScore = (scores.browDownLeft || 0) * 0.5 + (scores.browDownRight || 0) * 0.5;
  const sadScore = (scores.mouthFrownLeft || 0) * 0.3 + (scores.mouthFrownRight || 0) * 0.3 + (scores.browInnerUp || 0) * 0.4;
  
  if (angryScore > threshold && angryScore > primaryEmotion.score) {
    primaryEmotion = { name: "Angry", emoji: "😠", score: angryScore };
  } else if (sadScore > threshold && sadScore > primaryEmotion.score) {
    primaryEmotion = { name: "Sad", emoji: "😞", score: sadScore };
  }

  // 4. MOUTH OPEN (Non-emotional, like talking/yawn)
  if ((scores.mouthOpen || 0) > 0.6 && primaryEmotion.name === "Neutral") {
    primaryEmotion = { name: "Talking/Yawn", emoji: "🗣️", score: scores.mouthOpen };
  }
  
  // 5. WINK (A specific action)
  // Check for a left eye blink score significantly higher than the right eye blink score.
  if ((scores.eyeBlinkLeft || 0) > 0.8 && (scores.eyeBlinkRight || 0) < 0.2) {
    // If the person is winking with their left eye (your left, camera's right)
    primaryEmotion = { name: "Wink", emoji: "😉", score: scores.eyeBlinkLeft };
  } else if ((scores.eyeBlinkRight || 0) > 0.8 && (scores.eyeBlinkLeft || 0) < 0.2) {
    // If the person is winking with their right eye (your right, camera's left)
    primaryEmotion = { name: "Wink", emoji: "😉", score: scores.eyeBlinkRight };
  }

  return primaryEmotion;
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