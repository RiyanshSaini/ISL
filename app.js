let model;
let isModelLoaded = false;
const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const predEl = document.getElementById('pred');

console.log(">>> ISL Learning App for Children - Loaded");

// Labels for Indian Sign Language - Child friendly
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
  return feat;     // ✅ 42 values
}


// Draw colorful landmarks for children
function drawLandmarks(landmarks) {
  ctx.strokeStyle = '#00ff00';
  ctx.fillStyle = '#ff0000';
  ctx.lineWidth = 2;
  
  landmarks.forEach((landmark, index) => {
    const x = landmark.x * overlay.width;
    const y = landmark.y * overlay.height;
    
    // Different colors for different parts of hand
    if (index === 0) { // wrist - blue
      ctx.fillStyle = '#0000ff';
    } else if (index <= 4) { // thumb - yellow
      ctx.fillStyle = '#68680dff';
    } else if (index <= 8) { // index finger - green
      ctx.fillStyle = '#00ff00';
    } else if (index <= 12) { // middle finger - orange
      ctx.fillStyle = '#d60707ff';
    } else if (index <= 16) { // ring finger - purple
      ctx.fillStyle = '#800080';
    } else { // pinky - red
      ctx.fillStyle = '#ff0000';
    }
    
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  });
}

async function init() {
  try {
    console.log("🔄 Starting ISL Learning App...");
    predEl.textContent = 'Loading our sign language helper...';
    predEl.style.color = 'red';

    // Load TensorFlow.js model
    try {
      console.log("📦 Loading AI model...");
      model = await tf.loadLayersModel('model.h5');
      // ✅ Cast model weights to float32
    //   model.weights.forEach(w => {
    //     const newVals = w.read().cast('float32');
    //     w.write(newVals);
    //    });
      isModelLoaded = true;
      console.log("✅ AI Model loaded successfully!");
      predEl.textContent = 'Ready! Show me your hand sign ✋';
      predEl.style.color = '#00ff00';
    } catch (modelError) {
      console.error("❌ Model loading failed:", modelError);
      predEl.textContent = 'Oops! Having trouble loading. Please refresh the page.';
      predEl.style.color = 'red';
      return;
    }

    // Initialize webcam
    try {
      console.log("📷 Starting camera...");
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: 640, 
          height: 480,
          facingMode: 'user' 
        } 
      });
      video.srcObject = stream;
      
      video.onloadedmetadata = () => {
        console.log("✅ Camera ready!");
        overlay.width = video.videoWidth;
        overlay.height = video.videoHeight;
        startMediaPipe();
      };

    } catch (cameraError) {
      console.error("❌ Camera access failed:", cameraError);
      predEl.textContent = 'Please allow camera access to learn sign language! 📸';
      predEl.style.color = 'red';
    }

  } catch (error) {
    console.error("💥 Initialization error:", error);
    predEl.textContent = 'Something went wrong. Please try again!';
    predEl.style.color = 'red';
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
      predEl.textContent = 'Getting ready... Please wait!';
      predEl.style.color = 'red';
      return;
    }

    if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0) {
      predEl.textContent = 'Show me your hand! ✋ Make sure it\'s in the camera view.';
      predEl.style.color = 'red';
      return;
    }

    const landmarks = results.multiHandLandmarks[0];
    
    // Draw colorful landmarks
    drawLandmarks(landmarks);
    
    // Process and predict
    processHandLandmarks(landmarks);
  });

  // Start camera with MediaPipe
  const camera = new Camera(video, {
    onFrame: async () => {
      await hands.send({image: video});
    },
    width: 640,
    height: 480
  });
  
  camera.start().then(() => {
    console.log("🎥 Camera started successfully!");
    predEl.textContent = 'Great! Now show me a sign language letter!';
    predEl.style.color = '#00ff00';
  });
}

function processHandLandmarks(landmarks) {
  try {
    const features = landmarksToFeatures(landmarks, overlay.width, overlay.height);

    if (features.length !== 42) {
      predEl.textContent = 'Move your hand a bit ✋';
      predEl.style.color = 'red';
      return;
    }

    // ✅ Use float32 input (browser-compatible)
    const input = tf.tensor2d([features], [1, 42], 'float32');
    const prediction = model.predict(input);
    const data = prediction.dataSync();

    tf.dispose([input, prediction]);

    let best = 0, bestConfidence = data[0];
    for (let i = 1; i < data.length; i++) {
      if (data[i] > bestConfidence) {
        bestConfidence = data[i];
        best = i;
      }
    }

    if (bestConfidence > 0.25) {
      predEl.innerText = `You're showing: ${LABELS[best]} (${Math.round(bestConfidence * 100)}%)`;
      predEl.style.color = '#00ff00';
    } else {
      predEl.innerText = 'Try a clearer sign ✋';
      predEl.style.color = 'red';
    }

  } catch (err) {
    console.error("Prediction error:", err);
    predEl.textContent = 'Let me learn your hand sign better! Try again.';
    predEl.style.color = 'red';
  }
}



// Add some fun instructions for children
function addInstructions() {
  const instructions = document.createElement('div');
  instructions.innerHTML = `
    <div style="margin-top: 20px; text-align: center; color: #ccc; font-size: 16px;">
      <p>🎯 <strong>How to use:</strong></p>
      <p>1. Show one hand in the camera</p>
      <p>2. Make a sign language letter (A, B, C...)</p>
      <p>3. Keep your hand steady for best results!</p>
      <p>4. Have fun learning! 🌟</p>
    </div>
  `;
  document.body.appendChild(instructions);
}

// Start when page loads
document.addEventListener('DOMContentLoaded', () => {
  init();
  addInstructions();
});