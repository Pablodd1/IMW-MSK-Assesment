// PhysioMotion - Enhanced Assessment Workflow with Live Joint Tracking
// Supports: Phone Camera, Laptop Camera, and Femto Mega

// ============================================================================
// GLOBAL STATE
// ============================================================================

const ASSESSMENT_STATE = {
  selectedCamera: null,
  cameraStream: null,
  currentFacingMode: 'user', // 'user' (front) or 'environment' (back)
  pose: null,
  isRecording: false,
  recordingStartTime: null,
  skeletonFrames: [],
  femtoMegaClient: null,
  testId: null,
  assessmentId: null
};

// ============================================================================
// CAMERA SELECTION
// ============================================================================

function selectCameraType(type) {
  // Remove selection from all options
  document.querySelectorAll('.camera-option').forEach(opt => {
    opt.classList.remove('selected');
  });
  
  // Add selection to clicked option
  event.target.closest('.camera-option').classList.add('selected');
  
  ASSESSMENT_STATE.selectedCamera = type;
  document.getElementById('startBtn').disabled = false;
  
  // Show flip button for phone camera
  if (type === 'phone') {
    document.getElementById('flipBtn').style.display = 'flex';
  }
}

async function startAssessment() {
  // Hide modal
  document.getElementById('cameraSelectionModal').style.display = 'none';
  
  // Show camera container
  document.getElementById('cameraContainer').style.display = 'block';
  
  // Update progress
  updateProgress(2);
  
  // Initialize selected camera
  switch (ASSESSMENT_STATE.selectedCamera) {
    case 'phone':
    case 'webcam':
      await initializeWebCamera();
      break;
    case 'femto':
      await initializeFemtoMega();
      break;
    case 'upload':
      handleVideoUpload();
      break;
  }
}

// ============================================================================
// WEB CAMERA INITIALIZATION (Phone & Laptop)
// ============================================================================

async function initializeWebCamera() {
  try {
    showStatus('Initializing camera...', 'warning');
    
    const video = document.getElementById('videoElement');
    const canvas = document.getElementById('canvasElement');
    
    // Request camera access
    const constraints = {
      video: {
        facingMode: ASSESSMENT_STATE.currentFacingMode,
        width: { ideal: 1280 },
        height: { ideal: 720 }
      },
      audio: false
    };
    
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = stream;
    ASSESSMENT_STATE.cameraStream = stream;
    
    // Wait for video to load
    await new Promise((resolve) => {
      video.onloadedmetadata = resolve;
    });
    
    await video.play();
    
    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Initialize MediaPipe Pose
    await initializeMediaPipePose();
    
    showStatus('Camera connected', 'success');
    
  } catch (error) {
    console.error('Camera initialization error:', error);
    showStatus('Camera access denied', 'error');
    alert('Please allow camera access to continue with the assessment.');
  }
}

// ============================================================================
// MEDIAPIPE POSE DETECTION
// ============================================================================

async function initializeMediaPipePose() {
  const video = document.getElementById('videoElement');
  const canvas = document.getElementById('canvasElement');
  const ctx = canvas.getContext('2d');
  
  // Initialize MediaPipe Pose
  const pose = new Pose({
    locateFile: (file) => {
      return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`;
    }
  });
  
  pose.setOptions({
    modelComplexity: 1,
    smoothLandmarks: true,
    enableSegmentation: false,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
  });
  
  pose.onResults((results) => {
    onPoseResults(results, ctx, canvas);
  });
  
  ASSESSMENT_STATE.pose = pose;
  
  // Create camera for continuous detection
  const camera = new Camera(video, {
    onFrame: async () => {
      await pose.send({ image: video });
    },
    width: 1280,
    height: 720
  });
  
  camera.start();
  
  console.log('✅ MediaPipe Pose initialized');
}

// ============================================================================
// POSE RESULTS HANDLER WITH RED JOINT OVERLAY
// ============================================================================

function onPoseResults(results, ctx, canvas) {
  // Clear canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  if (!results.poseLandmarks) {
    return;
  }
  
  const landmarks = results.poseLandmarks;
  
  // Draw connections in YELLOW
  const connections = window.POSE_CONNECTIONS;
  ctx.strokeStyle = '#ffff00'; // YELLOW lines
  ctx.lineWidth = 4;
  ctx.shadowBlur = 10;
  ctx.shadowColor = '#ffff00';
  
  for (const connection of connections) {
    const start = landmarks[connection[0]];
    const end = landmarks[connection[1]];
    
    if (start.visibility > 0.5 && end.visibility > 0.5) {
      ctx.beginPath();
      ctx.moveTo(start.x * canvas.width, start.y * canvas.height);
      ctx.lineTo(end.x * canvas.width, end.y * canvas.height);
      ctx.stroke();
    }
  }
  
  // Draw joints in RED
  ctx.fillStyle = '#ff0000'; // RED circles
  ctx.strokeStyle = '#ffffff'; // WHITE border
  ctx.lineWidth = 2;
  ctx.shadowBlur = 15;
  ctx.shadowColor = '#ff0000';
  
  // Major joints (larger circles)
  const majorJoints = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
  
  landmarks.forEach((landmark, index) => {
    if (landmark.visibility > 0.5) {
      const x = landmark.x * canvas.width;
      const y = landmark.y * canvas.height;
      const radius = majorJoints.includes(index) ? 8 : 5;
      
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
    }
  });
  
  // Reset shadow
  ctx.shadowBlur = 0;
  
  // If recording, store skeleton data
  if (ASSESSMENT_STATE.isRecording) {
    const skeletonData = convertLandmarksToSkeletonData(landmarks);
    ASSESSMENT_STATE.skeletonFrames.push(skeletonData);
    
    // Calculate and display joint angles in real-time
    updateJointAnglesPanel(skeletonData);
  }
}

// ============================================================================
// CONVERT LANDMARKS TO SKELETON DATA
// ============================================================================

function convertLandmarksToSkeletonData(landmarks) {
  const landmarkNames = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
  ];
  
  const skeletonLandmarks = {};
  landmarks.forEach((landmark, index) => {
    skeletonLandmarks[landmarkNames[index]] = {
      x: landmark.x,
      y: landmark.y,
      z: landmark.z,
      visibility: landmark.visibility
    };
  });
  
  return {
    timestamp: Date.now(),
    landmarks: skeletonLandmarks
  };
}

// ============================================================================
// LIVE JOINT ANGLES PANEL UPDATE
// ============================================================================

function updateJointAnglesPanel(skeletonData) {
  const panel = document.getElementById('jointInfoPanel');
  const list = document.getElementById('jointAnglesList');
  
  panel.style.display = 'block';
  
  // Calculate joint angles
  const angles = calculateQuickJointAngles(skeletonData.landmarks);
  
  // Update display
  list.innerHTML = '';
  for (const [name, data] of Object.entries(angles)) {
    const item = document.createElement('div');
    item.className = 'joint-angle-item';
    
    const nameSpan = document.createElement('span');
    nameSpan.className = 'joint-name';
    nameSpan.textContent = name;
    
    const valueSpan = document.createElement('span');
    valueSpan.className = `joint-value ${data.status}`;
    valueSpan.textContent = `${data.angle}°`;
    
    item.appendChild(nameSpan);
    item.appendChild(valueSpan);
    list.appendChild(item);
  }
}

// ============================================================================
// QUICK JOINT ANGLE CALCULATIONS
// ============================================================================

function calculateQuickJointAngles(landmarks) {
  const angles = {};
  
  try {
    // Left Elbow
    const leftElbowAngle = calculateAngle3D(
      landmarks.left_shoulder,
      landmarks.left_elbow,
      landmarks.left_wrist
    );
    angles['Left Elbow'] = {
      angle: Math.round(leftElbowAngle),
      status: leftElbowAngle >= 130 ? 'normal' : 'limited'
    };
    
    // Right Elbow
    const rightElbowAngle = calculateAngle3D(
      landmarks.right_shoulder,
      landmarks.right_elbow,
      landmarks.right_wrist
    );
    angles['Right Elbow'] = {
      angle: Math.round(rightElbowAngle),
      status: rightElbowAngle >= 130 ? 'normal' : 'limited'
    };
    
    // Left Knee
    const leftKneeAngle = calculateAngle3D(
      landmarks.left_hip,
      landmarks.left_knee,
      landmarks.left_ankle
    );
    angles['Left Knee'] = {
      angle: Math.round(leftKneeAngle),
      status: leftKneeAngle >= 120 ? 'normal' : 'limited'
    };
    
    // Right Knee
    const rightKneeAngle = calculateAngle3D(
      landmarks.right_hip,
      landmarks.right_knee,
      landmarks.right_ankle
    );
    angles['Right Knee'] = {
      angle: Math.round(rightKneeAngle),
      status: rightKneeAngle >= 120 ? 'normal' : 'limited'
    };
    
    // Left Hip
    const leftHipAngle = calculateAngle3D(
      landmarks.left_shoulder,
      landmarks.left_hip,
      landmarks.left_knee
    );
    angles['Left Hip'] = {
      angle: Math.round(leftHipAngle),
      status: leftHipAngle >= 90 ? 'normal' : 'limited'
    };
    
    // Right Hip
    const rightHipAngle = calculateAngle3D(
      landmarks.right_shoulder,
      landmarks.right_hip,
      landmarks.right_knee
    );
    angles['Right Hip'] = {
      angle: Math.round(rightHipAngle),
      status: rightHipAngle >= 90 ? 'normal' : 'limited'
    };
    
  } catch (error) {
    console.error('Error calculating angles:', error);
  }
  
  return angles;
}

function calculateAngle3D(a, b, c) {
  const ba = { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
  const bc = { x: c.x - b.x, y: c.y - b.y, z: c.z - b.z };
  
  const dotProduct = ba.x * bc.x + ba.y * bc.y + ba.z * bc.z;
  const magBA = Math.sqrt(ba.x * ba.x + ba.y * ba.y + ba.z * ba.z);
  const magBC = Math.sqrt(bc.x * bc.x + bc.y * bc.y + bc.z * bc.z);
  
  const angleRad = Math.acos(dotProduct / (magBA * magBC));
  return angleRad * (180 / Math.PI);
}

// ============================================================================
// RECORDING CONTROLS
// ============================================================================

function startRecording() {
  ASSESSMENT_STATE.isRecording = true;
  ASSESSMENT_STATE.recordingStartTime = Date.now();
  ASSESSMENT_STATE.skeletonFrames = [];
  
  // Update UI
  document.getElementById('recordingIndicator').style.display = 'flex';
  document.getElementById('recordBtn').style.display = 'none';
  document.getElementById('stopBtn').style.display = 'flex';
  
  // Update progress
  updateProgress(3);
  
  // Start recording timer
  const timerInterval = setInterval(() => {
    if (!ASSESSMENT_STATE.isRecording) {
      clearInterval(timerInterval);
      return;
    }
    
    const elapsed = Date.now() - ASSESSMENT_STATE.recordingStartTime;
    const seconds = Math.floor(elapsed / 1000);
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    
    document.getElementById('recordingTime').textContent = 
      `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
  }, 100);
  
  showNotification('Recording started', 'success');
}

function stopRecording() {
  ASSESSMENT_STATE.isRecording = false;
  
  // Update UI
  document.getElementById('recordingIndicator').style.display = 'none';
  document.getElementById('stopBtn').style.display = 'none';
  document.getElementById('analyzeBtn').style.display = 'flex';
  
  showNotification(`Captured ${ASSESSMENT_STATE.skeletonFrames.length} frames`, 'success');
}

// ============================================================================
// CAMERA FLIP (PHONE)
// ============================================================================

async function flipCamera() {
  // Toggle facing mode
  ASSESSMENT_STATE.currentFacingMode = 
    ASSESSMENT_STATE.currentFacingMode === 'user' ? 'environment' : 'user';
  
  // Stop current stream
  if (ASSESSMENT_STATE.cameraStream) {
    ASSESSMENT_STATE.cameraStream.getTracks().forEach(track => track.stop());
  }
  
  // Restart with new facing mode
  await initializeWebCamera();
  
  showNotification(`Switched to ${ASSESSMENT_STATE.currentFacingMode === 'user' ? 'front' : 'back'} camera`, 'info');
}

// ============================================================================
// FEMTO MEGA INTEGRATION
// ============================================================================

async function initializeFemtoMega() {
  try {
    showStatus('Connecting to Femto Mega...', 'warning');
    
    const femtoClient = new FemtoMegaClient('ws://localhost:8765');
    await femtoClient.connect();
    
    ASSESSMENT_STATE.femtoMegaClient = femtoClient;
    
    // Handle skeleton data from Femto Mega
    femtoClient.onSkeletonData = (skeletonData) => {
      // Draw skeleton on canvas
      drawFemtoMegaSkeleton(skeletonData);
      
      // Store if recording
      if (ASSESSMENT_STATE.isRecording) {
        ASSESSMENT_STATE.skeletonFrames.push(skeletonData);
        updateJointAnglesPanel(skeletonData);
      }
    };
    
    showStatus('Femto Mega connected', 'success');
    showNotification('Professional camera ready', 'success');
    
  } catch (error) {
    console.error('Femto Mega connection error:', error);
    showStatus('Connection failed', 'error');
    alert('Failed to connect to Femto Mega. Please ensure:\n1. Femto Mega camera is connected\n2. Bridge server is running\n3. Bridge server address is correct');
  }
}

// ============================================================================
// MOVEMENT ANALYSIS
// ============================================================================

async function analyzeMovement() {
  if (ASSESSMENT_STATE.skeletonFrames.length === 0) {
    showNotification('No data to analyze', 'error');
    return;
  }
  
  showNotification('Analyzing movement...', 'info');
  
  // Take middle frame as representative
  const middleIndex = Math.floor(ASSESSMENT_STATE.skeletonFrames.length / 2);
  const representativeSkeleton = ASSESSMENT_STATE.skeletonFrames[middleIndex];
  
  try {
    // Call API to analyze
    const response = await fetch(`/api/tests/${ASSESSMENT_STATE.testId}/analyze`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ skeleton_data: representativeSkeleton })
    });
    
    const result = await response.json();
    
    if (result.success) {
      // Update progress
      updateProgress(4);
      
      // Hide camera, show results
      document.getElementById('cameraContainer').style.display = 'none';
      document.getElementById('resultsContainer').style.display = 'block';
      
      // Display results
      displayAnalysisResults(result.data.analysis);
      
      showNotification('Analysis complete!', 'success');
    } else {
      showNotification('Analysis failed: ' + result.error, 'error');
    }
    
  } catch (error) {
    console.error('Analysis error:', error);
    showNotification('Analysis failed', 'error');
  }
}

// ============================================================================
// DISPLAY RESULTS
// ============================================================================

function displayAnalysisResults(analysis) {
  // Movement Quality Score
  document.getElementById('qualityScore').textContent = 
    Math.round(analysis.movement_quality_score);
  
  // Deficiencies
  const deficienciesList = document.getElementById('deficienciesList');
  deficienciesList.innerHTML = '';
  
  analysis.deficiencies.forEach(def => {
    const card = document.createElement('div');
    card.className = 'p-4 border-l-4 border-yellow-500 bg-yellow-50 rounded';
    card.innerHTML = `
      <div class="flex items-start">
        <i class="fas fa-exclamation-circle text-yellow-600 text-xl mr-3 mt-1"></i>
        <div>
          <h4 class="font-bold text-lg">${def.area}</h4>
          <span class="inline-block px-2 py-1 text-xs font-semibold rounded ${
            def.severity === 'severe' ? 'bg-red-200 text-red-800' :
            def.severity === 'moderate' ? 'bg-orange-200 text-orange-800' :
            'bg-yellow-200 text-yellow-800'
          }">${def.severity.toUpperCase()}</span>
          <p class="text-gray-700 mt-2">${def.description}</p>
        </div>
      </div>
    `;
    deficienciesList.appendChild(card);
  });
  
  // Recommendations
  const exercisesList = document.getElementById('exercisesList');
  exercisesList.innerHTML = '';
  
  analysis.recommendations.forEach(rec => {
    const card = document.createElement('div');
    card.className = 'p-4 border-l-4 border-green-500 bg-green-50 rounded';
    card.innerHTML = `
      <div class="flex items-start">
        <i class="fas fa-check-circle text-green-600 text-xl mr-3 mt-1"></i>
        <p class="text-gray-700">${rec}</p>
      </div>
    `;
    exercisesList.appendChild(card);
  });
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

function updateProgress(step) {
  for (let i = 1; i <= 4; i++) {
    const stepEl = document.getElementById(`step${i}`);
    if (i < step) {
      stepEl.classList.add('completed');
      stepEl.classList.remove('active');
    } else if (i === step) {
      stepEl.classList.add('active');
      stepEl.classList.remove('completed');
    } else {
      stepEl.classList.remove('active', 'completed');
    }
  }
}

function showStatus(text, type) {
  const statusEl = document.getElementById('cameraStatus');
  const textEl = document.getElementById('statusText');
  
  textEl.textContent = text;
  statusEl.className = `camera-status ${type === 'success' ? 'connected' : 'disconnected'}`;
}

function showNotification(message, type = 'info') {
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `
    <i class="fas ${
      type === 'success' ? 'fa-check-circle' :
      type === 'error' ? 'fa-exclamation-circle' :
      type === 'warning' ? 'fa-exclamation-triangle' :
      'fa-info-circle'
    }"></i>
    <span>${message}</span>
  `;
  
  document.body.appendChild(toast);
  
  setTimeout(() => {
    toast.remove();
  }, 3000);
}

function startNewAssessment() {
  location.reload();
}

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
  console.log('✅ Assessment workflow initialized');
  console.log('📷 Camera options: Phone, Laptop, Femto Mega');
  console.log('🔴 Live joint tracking enabled');
});
