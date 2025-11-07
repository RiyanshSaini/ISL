let model;
let isModelLoaded = false;
const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const predEl = document.getElementById('pred');

console.log(">>> ISL Debug Mode - Loaded");

// Labels for Indian Sign Language
const LABELS = [
  'A','B','C','D','E','F','G','H','I','J',
  'K','L','M','N','O','P','Q','R','S','T',
  'U','V','W','X','Y','Z'
];

// Utility: normalize 21 landmarks to 42 features (x,y only)
function landmarksToFeatures(landmarks, w, h) {
  const pts = landmarks.map(p => ({x: p.x * w, y: p.y * h}));
  const ox = pts[0].x, oy = pts[0].y;
  const rel = pts.map(p => ({x: p.x - ox, y: p.y - oy}));

  let maxd = 0.001;
  for (const p of rel) {
    const d = Math.hypot(p.x, p.y);
    if (d > maxd) maxd = d;
  }
  const norm = rel.map(p => ({x: p.x / maxd, y: p.y / maxd}));

  const feat = [];
  norm.forEach(p => { feat.push(p.x, p.y); });
  
  console.log("✅ Features generated:", feat.length, "dimensions");
  console.log("Sample features:", feat.slice(0, 10)); // First 10 values
  
  return feat;
}

async function init() {
  try {
    console.log("🔄 Starting ISL Debug Mode...");
    predEl.textContent = 'Loading model...';

    // Load TensorFlow.js model
    try {
      console.log("📦 Loading model...");
      model = await tf.loadLayersModel('model/model.json');
      isModelLoaded = true;
      
      // Test model structure
      console.log("✅ Model loaded successfully!");
      console.log("Model layers:", model.layers.map(layer => layer.name));
      
      // Test model with dummy input to understand expected format
      await testModelWithDummyInput();
      
      predEl.textContent = 'Model loaded! Show your hand ✋';
    } catch (modelError) {
      console.error("❌ Model loading failed:", modelError);
      predEl.textContent = 'Model loading failed';
      return;
    }

    // Initialize webcam
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: 640, 
          height: 480,
          facingMode: 'user' 
        } 
      });
      video.srcObject = stream;
      
      video.onloadedmetadata = () => {
        overlay.width = video.videoWidth;
        overlay.height = video.videoHeight;
        startMediaPipe();
      };

    } catch (cameraError) {
      console.error("❌ Camera access failed:", cameraError);
      predEl.textContent = 'Camera access denied';
    }

  } catch (error) {
    console.error("💥 Initialization error:", error);
  }
}

// Test the model with dummy data to understand expected input/output
async function testModelWithDummyInput() {
  console.log("🧪 Testing model with dummy inputs...");
  
  // Test different input shapes
  const testCases = [
    { shape: [1, 42], desc: "Flat 42 features" },
    { shape: [1, 21, 2], desc: "21 points with 2 coordinates" },
    { shape: [1, 21, 2, 1], desc: "21 points with 2 coordinates and channel" }
  ];
  
  for (const testCase of testCases) {
    try {
      console.log(`Testing shape: ${testCase.desc}`);
      
      let dummyInput;
      if (testCase.shape[1] === 42) {
        // Flat array of 42
        dummyInput = tf.tensor2d([new Array(42).fill(0.1)], testCase.shape);
      } else if (testCase.shape[1] === 21) {
        // 21x2 array
        const dummyArray = [];
        for (let i = 0; i < 21; i++) {
          dummyArray.push([0.1, 0.1]);
        }
        dummyInput = tf.tensor3d([dummyArray], testCase.shape);
      }
      
      const output = model.predict(dummyInput);
      const outputData = output.dataSync();
      
      console.log(`✅ ${testCase.desc} - Output shape:`, output.shape);
      console.log(`Output values:`, Array.from(outputData));
      console.log(`Output range: min=${Math.min(...outputData)}, max=${Math.max(...outputData)}`);
      
      tf.dispose([dummyInput, output]);
      
    } catch (error) {
      console.log(`❌ ${testCase.desc} failed:`, error.message);
    }
  }
}

function startMediaPipe() {
  console.log("🖐️ Starting hand detection...");
  
  const hands = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
  });

  hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.7,
    minTrackingConfidence: 0.7
  });

  hands.onResults((results) => {
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    
    if (!isModelLoaded) {
      predEl.textContent = 'Model not ready';
      return;
    }

    if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
      predEl.textContent = 'Show your hand in frame';
      return;
    }

    const landmarks = results.multiHandLandmarks[0];
    console.log("🖐️ Hand detected with", landmarks.length, "landmarks");
    
    // Draw landmarks
    drawLandmarks(landmarks);
    
    // Process and predict
    processHandLandmarks(landmarks);
  });

  const camera = new Camera(video, {
    onFrame: async () => {
      await hands.send({image: video});
    },
    width: 640,
    height: 480
  });
  
  camera.start();
}

function drawLandmarks(landmarks) {
  ctx.strokeStyle = '#00ff00';
  ctx.fillStyle = '#ff0000';
  ctx.lineWidth = 2;
  
  landmarks.forEach(landmark => {
    ctx.beginPath();
    ctx.arc(
      landmark.x * overlay.width, 
      landmark.y * overlay.height, 
      4, 0, Math.PI * 2
    );
    ctx.fill();
    ctx.stroke();
  });
}

function processHandLandmarks(landmarks) {
  try {
    console.log("🎯 Starting prediction...");
    
    const features = landmarksToFeatures(landmarks, overlay.width, overlay.height);
    
    if (features.length !== 42) {
      console.warn(`Expected 42 features, got ${features.length}`);
      return;
    }

    // Try multiple input formats
    let input, output;
    let success = false;
    
    // Format 1: [1, 42] - flat array
    try {
      console.log("Trying format [1, 42]...");
      input = tf.tensor2d([features], [1, 42]);
      output = model.predict(input);
      const outputShape = output.shape;
      const outputData = output.dataSync();
      
      console.log("✅ Format [1, 42] success!");
      console.log("Output shape:", outputShape);
      console.log("Output values:", Array.from(outputData));
      
      processPredictions(outputData);
      success = true;
      
    } catch (error1) {
      console.log("❌ Format [1, 42] failed:", error1.message);
      tf.dispose([input, output]);
      
      // Format 2: [1, 21, 2] - 2D array
      try {
        console.log("Trying format [1, 21, 2]...");
        const features2D = [];
        for (let i = 0; i < 21; i++) {
          features2D.push([features[i*2], features[i*2 + 1]]);
        }
        input = tf.tensor3d([features2D], [1, 21, 2]);
        output = model.predict(input);
        const outputData = output.dataSync();
        
        console.log("✅ Format [1, 21, 2] success!");
        console.log("Output values:", Array.from(outputData));
        
        processPredictions(outputData);
        success = true;
        
      } catch (error2) {
        console.log("❌ Format [1, 21, 2] failed:", error2.message);
        tf.dispose([input, output]);
        
        // Format 3: [1, 1, 42] - with extra dimension
        try {
          console.log("Trying format [1, 1, 42]...");
          input = tf.tensor3d([features], [1, 1, 42]);
          output = model.predict(input);
          const outputData = output.dataSync();
          
          console.log("✅ Format [1, 1, 42] success!");
          console.log("Output values:", Array.from(outputData));
          
          processPredictions(outputData);
          success = true;
          
        } catch (error3) {
          console.log("❌ All formats failed");
          predEl.textContent = 'Model format issue';
        }
      }
    }
    
    if (!success) {
      predEl.textContent = 'Prediction failed - check console';
    }
    
    // Clean up
    if (input) tf.dispose(input);
    if (output) tf.dispose(output);
    
  } catch (error) {
    console.error("💥 Prediction error:", error);
    predEl.textContent = 'Prediction error';
  }
}

function processPredictions(predictions) {
  console.log("📊 Processing predictions...");
  
  // Check if we need softmax
  const needsSoftmax = predictions.some(p => p < 0 || p > 1);
  console.log("Needs softmax?", needsSoftmax);
  
  let finalPredictions;
  if (needsSoftmax) {
    const outputTensor = tf.tensor1d(predictions);
    const probabilities = tf.softmax(outputTensor);
    finalPredictions = probabilities.dataSync();
    tf.dispose([outputTensor, probabilities]);
    console.log("After softmax:", Array.from(finalPredictions));
  } else {
    finalPredictions = predictions;
    console.log("Using raw predictions (already probabilities)");
  }
  
  // Find best prediction
  let bestClass = 0;
  let bestConfidence = finalPredictions[0];
  
  for (let i = 1; i < finalPredictions.length; i++) {
    if (finalPredictions[i] > bestConfidence) {
      bestConfidence = finalPredictions[i];
      bestClass = i;
    }
  }
  
  console.log(`🏆 Best class: ${bestClass} (${LABELS[bestClass]}), Confidence: ${bestConfidence}`);
  
  if (bestConfidence > 0.1) { // Lower threshold for debugging
    const label = LABELS[bestClass] || `Class ${bestClass}`;
    predEl.textContent = `Prediction: ${label} (${bestConfidence.toFixed(3)})`;
  } else {
    predEl.textContent = `Low confidence: ${bestConfidence.toFixed(3)}`;
  }
}

// Start the application
document.addEventListener('DOMContentLoaded', init);