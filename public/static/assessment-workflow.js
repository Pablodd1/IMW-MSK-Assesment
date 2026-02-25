// PhysioMotion - Enhanced Assessment Workflow with Live Joint Tracking
// Supports: Phone Camera, Laptop Camera, External Camera, and Femto Mega

// ============================================================================
// GLOBAL STATE
// ============================================================================

const ASSESSMENT_STATE = {
  selectedCamera: null,
  selectedDeviceId: null,
  availableCameras: [],
  cameraStream: null,
  currentFacingMode: 'user', // 'user' (front) or 'environment' (back)
  pose: null,
  isRecording: false,
  recordingStartTime: null,
  skeletonFrames: [],
  femtoMegaClient: null,
  testId: null,
  assessmentId: null,
  patientId: null,
  currentTest: null,
  features: {
    ghostMode: false,
    voiceFeedback: false,
    repCounter: false
  },
  repCount: 0,
  repState: 'up', // 'up', 'down'
  ghostSkeleton: null, // Captured "perfect" frame
  chart: null, // Chart.js instance
  chartData: {
    labels: [],
    datasets: []
  },
  maxChartPoints: 100 // Maintain last 100 frames (~3-5 seconds at 30fps)
};

// ============================================================================
// INITIALIZATION - LOAD PATIENT AND CREATE ASSESSMENT
// ============================================================================

async function initializeAssessment() {
  // Load patient selection from URL params or show patient selector
  const urlParams = new URLSearchParams(window.location.search);
  const patientId = urlParams.get('patient_id');
  
  if (!patientId) {
    // Show patient selection modal
    await showPatientSelector();
  } else {
    ASSESSMENT_STATE.patientId = patientId;
    await createAssessment();
  }
}

async function showPatientSelector() {
  try {
    // Fetch patients
    const response = await fetch('/api/patients');
    const result = await response.json();
    
    if (result.success && result.data.length > 0) {
      // For now, use the first patient
      ASSESSMENT_STATE.patientId = result.data[0].id;
      await createAssessment();
    } else {
      showNotification('No patients found. Please create a patient first.', 'warning');
      setTimeout(() => {
        window.location.href = '/intake';
      }, 2000);
    }
  } catch (error) {
    console.error('Error loading patients:', error);
    showNotification('Error loading patients', 'error');
  }
}

async function createAssessment() {
  try {
    // Create new assessment for patient
    const response = await fetch('/api/assessments', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        patient_id: ASSESSMENT_STATE.patientId,
        assessment_type: 'initial',
        clinician_id: 1
      })
    });
    
    const result = await response.json();
    
    if (result.success) {
      ASSESSMENT_STATE.assessmentId = result.data.id;
      console.log('✅ Assessment created:', ASSESSMENT_STATE.assessmentId);
      
      // Create a movement test
      await createMovementTest();
    } else {
      showNotification('Failed to create assessment', 'error');
    }
  } catch (error) {
    console.error('Error creating assessment:', error);
    showNotification('Error creating assessment', 'error');
  }
}

async function createMovementTest() {
  try {
    // Create a functional movement test
    const response = await fetch(`/api/assessments/${ASSESSMENT_STATE.assessmentId}/tests`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        test_name: 'Functional Movement Screen',
        test_category: 'mobility',
        test_order: 1,
        instructions: 'Stand in view of camera and perform the requested movements',
        test_status: 'pending'
      })
    });
    
    const result = await response.json();
    
    if (result.success) {
      ASSESSMENT_STATE.testId = result.data.id;
      ASSESSMENT_STATE.currentTest = result.data;
      console.log('✅ Movement test created:', ASSESSMENT_STATE.testId);
    } else {
      showNotification('Failed to create test', 'error');
    }
  } catch (error) {
    console.error('Error creating test:', error);
    showNotification('Error creating test', 'error');
  }
}

// ============================================================================
// ADVANCED FEATURES (Voice, Reps, Ghost)
// ============================================================================

function toggleFeature(feature) {
  ASSESSMENT_STATE.features[feature] = !ASSESSMENT_STATE.features[feature];
  
  // Update button UI
  const btn = document.getElementById(`btn-${feature}`);
  if (ASSESSMENT_STATE.features[feature]) {
    // Check if classes exist (might be dashboard overlay style or regular style)
    if (btn) {
        btn.classList.remove('opacity-50', 'bg-gray-700', 'bg-gray-900', 'opacity-70');
        btn.classList.add('opacity-100', 'bg-cyan-600', 'ring-2', 'ring-cyan-300');
    }
    showNotification(`${feature} enabled`, 'success');
    
    // Feature specific init
    if (feature === 'ghostMode' && !ASSESSMENT_STATE.ghostSkeleton && ASSESSMENT_STATE.skeletonFrames.length > 0) {
      // Use middle frame of current recording as ghost if available
      const mid = Math.floor(ASSESSMENT_STATE.skeletonFrames.length / 2);
      ASSESSMENT_STATE.ghostSkeleton = ASSESSMENT_STATE.skeletonFrames[mid];
    } else if (feature === 'voiceFeedback') {
      VoiceFeedback.speak("Voice coaching enabled. I will guide your movements.");
    }
  } else {
    if (btn) {
        // Reset to default style (handling both dashboard and regular)
        btn.classList.add('opacity-70', 'bg-gray-900');
        btn.classList.remove('opacity-100', 'bg-cyan-600', 'ring-2', 'ring-cyan-300');
    }
    showNotification(`${feature} disabled`, 'info');
  }
}

const VoiceFeedback = {
  lastSpoken: 0,
  speak: (text) => {
    const now = Date.now();
    if (now - VoiceFeedback.lastSpoken < 2500) return; // Throttle speech
    
    // Cancel previous
    window.speechSynthesis.cancel();
    
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.1;
    utterance.pitch = 1.0;
    utterance.volume = 1.0;
    window.speechSynthesis.speak(utterance);
    VoiceFeedback.lastSpoken = now;
  }
};

function updateRepCounter(angles, ctx, canvas) {
  if (!ASSESSMENT_STATE.features.repCounter) return;
  
  // Simple squat logic based on knee angle
  const kneeAngle = angles['Right Knee']?.angle || angles['Left Knee']?.angle;
  
  if (!kneeAngle) return;

  // Draw Rep Counter Overlay
  ctx.save();
  ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
  // ctx.roundRect might not be supported in all envs, fallback to rect
  if (ctx.roundRect) {
      ctx.roundRect(20, 20, 120, 80, 10);
  } else {
      ctx.fillRect(20, 20, 120, 80);
  }
  ctx.fill();
  
  ctx.fillStyle = '#ffffff';
  ctx.font = 'bold 40px Arial';
  ctx.fillText(ASSESSMENT_STATE.repCount, 50, 75);
  
  ctx.font = '14px Arial';
  ctx.fillStyle = '#cccccc';
  ctx.fillText('REPS', 50, 40);
  
  // State Indicator
  ctx.beginPath();
  ctx.arc(110, 60, 10, 0, 2 * Math.PI);
  ctx.fillStyle = ASSESSMENT_STATE.repState === 'down' ? '#00ff00' : '#ffff00';
  ctx.fill();
  ctx.restore();
  
  // Logic
  if (ASSESSMENT_STATE.repState === 'up' && kneeAngle < 100) {
    ASSESSMENT_STATE.repState = 'down';
  } else if (ASSESSMENT_STATE.repState === 'down' && kneeAngle > 150) {
    ASSESSMENT_STATE.repState = 'up';
    ASSESSMENT_STATE.repCount++;
    VoiceFeedback.speak(ASSESSMENT_STATE.repCount.toString());
  }
}

function drawGhostSkeleton(ctx, canvas) {
  if (!ASSESSMENT_STATE.features.ghostMode || !ASSESSMENT_STATE.ghostSkeleton) return;
  
  const landmarks = ASSESSMENT_STATE.ghostSkeleton.landmarks;
  if (!landmarks) return;

  ctx.save();
  ctx.globalAlpha = 0.3; // Ghost effect
  ctx.strokeStyle = '#00ffff'; // Cyan for ghost
  ctx.fillStyle = '#00ffff';
  
  const connections = [
    ['left_shoulder', 'right_shoulder'],
    ['left_shoulder', 'left_hip'], ['right_shoulder', 'right_hip'],
    ['left_hip', 'left_knee'], ['right_hip', 'right_knee'],
    ['left_knee', 'left_ankle'], ['right_knee', 'right_ankle']
  ];
  
  connections.forEach(([startName, endName]) => {
    const start = landmarks[startName];
    const end = landmarks[endName];
    if (start && end && start.visibility > 0.5 && end.visibility > 0.5) {
      ctx.beginPath();
      ctx.moveTo(start.x * canvas.width, start.y * canvas.height);
      ctx.lineTo(end.x * canvas.width, end.y * canvas.height);
      ctx.stroke();
    }
  });
  
  ctx.restore();
}

// ============================================================================
// CHART INITIALIZATION
// ============================================================================

function initializeChart() {
  const canvas = document.getElementById('jointChart');
  if (!canvas) return; // Guard if element doesn't exist

  const ctx = canvas.getContext('2d');

  const datasets = [
    {
      label: 'Left Knee',
      borderColor: '#06b6d4', // Cyan
      backgroundColor: 'rgba(6, 182, 212, 0.1)',
      data: [],
      tension: 0.4,
      pointRadius: 0
    },
    {
      label: 'Right Knee',
      borderColor: '#8b5cf6', // Purple
      backgroundColor: 'rgba(139, 92, 246, 0.1)',
      data: [],
      tension: 0.4,
      pointRadius: 0
    },
    {
      label: 'Hip Flexion',
      borderColor: '#f59e0b', // Amber
      backgroundColor: 'rgba(245, 158, 11, 0.1)',
      data: [],
      tension: 0.4,
      pointRadius: 0,
      hidden: true
    }
  ];

  ASSESSMENT_STATE.chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: datasets
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: {
        mode: 'index',
        intersect: false,
      },
      plugins: {
        legend: {
          position: 'top',
          labels: { usePointStyle: true, boxWidth: 8 }
        },
        tooltip: { enabled: true }
      },
      scales: {
        x: { display: false, grid: { display: false } },
        y: {
          min: 0, max: 180,
          grid: { color: '#f3f4f6' },
          ticks: { stepSize: 30 }
        }
      }
    }
  });
  console.log('✅ Chart initialized');
}

function updateChart(angles) {
  if (!ASSESSMENT_STATE.chart) return;

  const chart = ASSESSMENT_STATE.chart;
  const now = new Date().toLocaleTimeString();

  chart.data.labels.push(now);

  // 0: Left Knee
  chart.data.datasets[0].data.push(angles['Left Knee'] ? angles['Left Knee'].angle : null);
  // 1: Right Knee
  chart.data.datasets[1].data.push(angles['Right Knee'] ? angles['Right Knee'].angle : null);
  // 2: Left Hip
  chart.data.datasets[2].data.push(angles['Left Hip'] ? angles['Left Hip'].angle : null);

  // Maintain window size
  if (chart.data.labels.length > ASSESSMENT_STATE.maxChartPoints) {
    chart.data.labels.shift();
    chart.data.datasets.forEach(dataset => dataset.data.shift());
  }

  chart.update('none');
}

// ============================================================================
// DASHBOARD UPDATES
// ============================================================================

function updateDashboard(angles) {
    // 1. Update Chart
    updateChart(angles);

    // 2. Update Data Table
    const tbody = document.getElementById('jointTableBody');
    if (tbody) {
        tbody.innerHTML = ''; // Clear current rows

        for (const [name, data] of Object.entries(angles)) {
            const row = document.createElement('tr');

            const statusClass = data.status === 'normal' ? 'status-normal' :
                               data.status === 'limited' ? 'status-limited' : 'status-excessive';

            row.innerHTML = `
                <td class="font-medium text-gray-700">${name}</td>
                <td class="font-bold text-gray-900">${data.angle}°</td>
                <td><span class="status-badge ${statusClass}">${data.status.toUpperCase()}</span></td>
            `;
            tbody.appendChild(row);
        }
    }
}

// ============================================================================
// CAMERA PERMISSION CHECK
// ============================================================================

async function checkCameraPermissions() {
  try {
    console.log('🔍 Checking camera permissions...');
    
    if (!navigator.mediaDevices) {
      throw new Error('MediaDevices API not available. Please use HTTPS or localhost.');
    }
    
    // Check if Permissions API is available
    if (navigator.permissions && navigator.permissions.query) {
      try {
        const result = await navigator.permissions.query({ name: 'camera' });
        if (result.state === 'granted') {
          showNotification('✅ Camera access already granted!', 'success');
          await detectAvailableCameras();
          return true;
        } else if (result.state === 'denied') {
          showNotification('❌ Camera access denied.', 'error');
          setTimeout(() => showDetailedPermissionHelp(), 500);
          return false;
        }
      } catch (e) {}
    }
    
    // Request access
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    stream.getTracks().forEach(track => track.stop());
    
    showNotification('✅ Camera access granted!', 'success');
    await detectAvailableCameras();
    return true;
    
  } catch (error) {
    console.error('❌ Camera check failed:', error);
    let message = '⚠️ Camera access error';
    if (error.name === 'NotAllowedError') message = '❌ Camera access denied';
    showNotification(message, 'error');
    if (error.name === 'NotAllowedError') setTimeout(() => showDetailedPermissionHelp(), 1000);
    return false;
  }
}

// ============================================================================
// CAMERA DETECTION
// ============================================================================

async function detectAvailableCameras() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(device => device.kind === 'videoinput');
    
    ASSESSMENT_STATE.availableCameras = videoDevices.map((device, index) => ({
      deviceId: device.deviceId,
      label: device.label || `Camera ${index + 1}`,
      isFrontFacing: device.label.toLowerCase().includes('front'),
      isBackFacing: device.label.toLowerCase().includes('back')
    }));
    
    if (ASSESSMENT_STATE.availableCameras.length > 0) {
      showNotification(`Found ${ASSESSMENT_STATE.availableCameras.length} camera(s)`, 'success');
    }
    return ASSESSMENT_STATE.availableCameras;
  } catch (error) {
    console.error('Error detecting cameras:', error);
    return [];
  }
}

// ============================================================================
// CAMERA SELECTION & START
// ============================================================================

function selectCameraType(type) {
  document.querySelectorAll('.camera-option').forEach(opt => opt.classList.remove('selected'));
  event.target.closest('.camera-option').classList.add('selected');
  
  ASSESSMENT_STATE.selectedCamera = type;
  document.getElementById('startBtn').disabled = false;
}

async function startAssessment() {
  document.getElementById('cameraSelectionModal').style.display = 'none';
  document.getElementById('cameraContainer').style.display = 'block';
  
  // Initialize Chart if element exists
  initializeChart();
  
  await detectAvailableCameras();
  
  switch (ASSESSMENT_STATE.selectedCamera) {
    case 'phone':
      ASSESSMENT_STATE.currentFacingMode = 'environment';
      await initializeWebCamera();
      break;
    case 'webcam':
      await initializeWebCamera();
      break;
    case 'femto':
      // Get/Set Bridge URL
      const savedUrl = localStorage.getItem('femto_bridge_url');
      if (!savedUrl) {
        const configUrl = prompt('Enter Femto Mega Bridge Server URL:', 'ws://localhost:8765');
        if (configUrl) localStorage.setItem('femto_bridge_url', configUrl);
      }
      await initializeFemtoMega();
      break;
    case 'upload':
      // Placeholder for upload logic
      alert('Video upload feature coming soon!');
      break;
  }
}

// ============================================================================
// WEB CAMERA INITIALIZATION
// ============================================================================

async function initializeWebCamera() {
  try {
    showStatus('Requesting camera access...', 'warning');
    const video = document.getElementById('videoElement');
    const canvas = document.getElementById('canvasElement');
    
    let constraints = {
        video: { width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false
    };
    
    if (ASSESSMENT_STATE.selectedCamera === 'phone') {
        constraints.video.facingMode = ASSESSMENT_STATE.currentFacingMode;
    } else if (ASSESSMENT_STATE.selectedDeviceId) {
        constraints.video.deviceId = { exact: ASSESSMENT_STATE.selectedDeviceId };
    }
    
    // Try to get stream
    let stream;
    try {
        stream = await navigator.mediaDevices.getUserMedia(constraints);
    } catch (e) {
        console.warn('Advanced constraints failed, trying basic:', e);
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
    }
    
    video.srcObject = stream;
    ASSESSMENT_STATE.cameraStream = stream;
    
    await new Promise(resolve => video.onloadedmetadata = resolve);
    await video.play();
    
    canvas.width = video.videoWidth || 1280;
    canvas.height = video.videoHeight || 720;
    
    await initializeMediaPipePose();
    
    showStatus('Camera connected', 'success');
    
  } catch (error) {
    console.error('Camera init error:', error);
    showStatus('Camera initialization failed', 'error');
    showDetailedPermissionHelp();
  }
}

async function flipCamera() {
    if (ASSESSMENT_STATE.selectedCamera === 'phone') {
        ASSESSMENT_STATE.currentFacingMode = ASSESSMENT_STATE.currentFacingMode === 'user' ? 'environment' : 'user';
    } else if (ASSESSMENT_STATE.availableCameras.length > 1) {
        // Cycle cameras logic could go here
    }
    
    if (ASSESSMENT_STATE.cameraStream) {
        ASSESSMENT_STATE.cameraStream.getTracks().forEach(t => t.stop());
    }
    await initializeWebCamera();
}

// ============================================================================
// FEMTO MEGA INITIALIZATION
// ============================================================================

async function initializeFemtoMega() {
  try {
    showStatus('Connecting to Femto Mega...', 'warning');
    const bridgeUrl = localStorage.getItem('femto_bridge_url') || 'ws://localhost:8765';
    
    // Assuming FemtoMegaClient is defined globally or imported
    // If not, we need to ensure the script is loaded or mock it for this file
    if (typeof FemtoMegaClient !== 'undefined') {
        const femtoClient = new FemtoMegaClient(bridgeUrl);
        await femtoClient.connect();
        ASSESSMENT_STATE.femtoMegaClient = femtoClient;

        femtoClient.onSkeletonData = (skeletonData) => {
            // Draw skeleton logic (omitted for brevity, assume similar to pose)
            // But we need to feed it into the pipeline
            if (ASSESSMENT_STATE.isRecording) {
                ASSESSMENT_STATE.skeletonFrames.push(skeletonData);
            }
            updateJointAnglesPanel(skeletonData); // Update old panel
            // Also update new dashboard
            const angles = calculateQuickJointAngles(skeletonData.landmarks);
            updateDashboard(angles);
        };
        showStatus('Femto Mega connected', 'success');
    } else {
        console.warn('FemtoMegaClient library not found.');
        showStatus('Femto Mega driver missing', 'error');
    }
  } catch (error) {
    console.error('Femto error:', error);
    showStatus('Femto connection failed', 'error');
  }
}

// ============================================================================
// MEDIAPIPE POSE
// ============================================================================

async function initializeMediaPipePose() {
  const video = document.getElementById('videoElement');
  const canvas = document.getElementById('canvasElement');
  const ctx = canvas.getContext('2d');
  
  const pose = new Pose({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
  });
  
  pose.setOptions({
    modelComplexity: 1,
    smoothLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
  });
  
  pose.onResults((results) => onPoseResults(results, ctx, canvas));
  
  const camera = new Camera(video, {
    onFrame: async () => await pose.send({ image: video }),
    width: 1280, height: 720
  });
  
  camera.start();
  console.log('✅ MediaPipe Pose initialized');
}

function onPoseResults(results, ctx, canvas) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!results.poseLandmarks) return;
  
  const landmarks = results.poseLandmarks;
  
  // Draw Skeleton Overlay
  drawSkeleton(ctx, canvas, landmarks);
  
  // Process Data
  const skeletonData = convertLandmarksToSkeletonData(landmarks);

  if (ASSESSMENT_STATE.isRecording) {
    ASSESSMENT_STATE.skeletonFrames.push(skeletonData);
  }

  const angles = calculateQuickJointAngles(skeletonData.landmarks);

  // Update UI
  updateJointAnglesPanel(skeletonData); // Legacy panel
  updateDashboard(angles);              // New Dashboard
  updateRepCounter(angles, ctx, canvas);
  drawGhostSkeleton(ctx, canvas);

  if (ASSESSMENT_STATE.features.ghostMode && !ASSESSMENT_STATE.ghostSkeleton) {
      ASSESSMENT_STATE.ghostSkeleton = skeletonData;
      showNotification("Ghost reference set!", "success");
  }
}

function drawSkeleton(ctx, canvas, landmarks) {
    const connections = window.POSE_CONNECTIONS; // Provided by MediaPipe utils

    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 4;

    if (connections) {
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
    }

    ctx.fillStyle = '#00ff00';
    landmarks.forEach(lm => {
        if (lm.visibility > 0.5) {
            ctx.beginPath();
            ctx.arc(lm.x * canvas.width, lm.y * canvas.height, 5, 0, 2 * Math.PI);
            ctx.fill();
        }
    });
}

// ============================================================================
// DATA CONVERSION & CALCULATIONS
// ============================================================================

function convertLandmarksToSkeletonData(landmarks) {
  const names = [
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
  const data = {};
  landmarks.forEach((lm, i) => {
    data[names[i]] = { x: lm.x, y: lm.y, z: lm.z, visibility: lm.visibility };
  });
  return { timestamp: Date.now(), landmarks: data };
}

function calculateQuickJointAngles(landmarks) {
  const angles = {};
  try {
    const joints = [
        { name: 'Left Elbow', a: 'left_shoulder', b: 'left_elbow', c: 'left_wrist', min: 130 },
        { name: 'Right Elbow', a: 'right_shoulder', b: 'right_elbow', c: 'right_wrist', min: 130 },
        { name: 'Left Knee', a: 'left_hip', b: 'left_knee', c: 'left_ankle', min: 120 },
        { name: 'Right Knee', a: 'right_hip', b: 'right_knee', c: 'right_ankle', min: 120 },
        { name: 'Left Hip', a: 'left_shoulder', b: 'left_hip', c: 'left_knee', min: 90 },
        { name: 'Right Hip', a: 'right_shoulder', b: 'right_hip', c: 'right_knee', min: 90 }
    ];

    joints.forEach(j => {
        if (landmarks[j.a] && landmarks[j.b] && landmarks[j.c]) {
            const angle = calculateAngle3D(landmarks[j.a], landmarks[j.b], landmarks[j.c]);
            angles[j.name] = {
                angle: Math.round(angle),
                status: angle >= j.min ? 'normal' : 'limited'
            };
        }
    });
  } catch (e) { console.error(e); }
  return angles;
}

function calculateAngle3D(a, b, c) {
  const ba = { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
  const bc = { x: c.x - b.x, y: c.y - b.y, z: c.z - b.z };
  const dot = ba.x * bc.x + ba.y * bc.y + ba.z * bc.z;
  const magBA = Math.sqrt(ba.x*ba.x + ba.y*ba.y + ba.z*ba.z);
  const magBC = Math.sqrt(bc.x*bc.x + bc.y*bc.y + bc.z*bc.z);
  return Math.acos(Math.max(-1, Math.min(1, dot / (magBA * magBC)))) * (180 / Math.PI);
}

// ============================================================================
// RECORDING & ANALYSIS (RESTORED ROBUST API CALLS)
// ============================================================================

function startRecording() {
  ASSESSMENT_STATE.isRecording = true;
  ASSESSMENT_STATE.recordingStartTime = Date.now();
  ASSESSMENT_STATE.skeletonFrames = [];
  
  document.getElementById('recordingIndicator').style.display = 'flex';
  document.getElementById('recordBtn').style.display = 'none';
  document.getElementById('stopBtn').style.display = 'flex';
  
  const interval = setInterval(() => {
    if (!ASSESSMENT_STATE.isRecording) {
        clearInterval(interval);
        return;
    }
    const elapsed = Math.floor((Date.now() - ASSESSMENT_STATE.recordingStartTime) / 1000);
    const mins = Math.floor(elapsed / 60).toString().padStart(2, '0');
    const secs = (elapsed % 60).toString().padStart(2, '0');
    document.getElementById('recordingTime').textContent = `${mins}:${secs}`;
  }, 100);
  
  showNotification('Recording started', 'success');
}

function stopRecording() {
  ASSESSMENT_STATE.isRecording = false;
  document.getElementById('recordingIndicator').style.display = 'none';
  document.getElementById('stopBtn').style.display = 'none';
  document.getElementById('analyzeBtn').style.display = 'flex';
  showNotification(`Captured ${ASSESSMENT_STATE.skeletonFrames.length} frames`, 'success');
}

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
    const response = await fetch(`/api/tests/${ASSESSMENT_STATE.testId}/analyze`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ skeleton_data: representativeSkeleton })
    });
    
    const result = await response.json();
    
    if (result.success) {
      updateProgress(4);
      document.getElementById('cameraContainer').style.display = 'none';
      document.getElementById('resultsContainer').style.display = 'block';
      displayAnalysisResults(result.data); // Fixed: result.data usually contains analysis directly
      showNotification('Analysis complete!', 'success');
    } else {
      showNotification('Analysis failed: ' + result.error, 'error');
    }
  } catch (error) {
    console.error('Analysis error:', error);
    showNotification('Analysis failed', 'error');
  }
}

function showNotification(message, type = 'info') {
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3000);
}

function showStatus(text, type) {
    const statusEl = document.getElementById('cameraStatus');
    const textEl = document.getElementById('statusText');
    if (textEl && statusEl) {
        textEl.textContent = text;
        statusEl.className = `camera-status ${type === 'success' ? 'connected' : 'disconnected'}`;
    }
}

function updateProgress(step) {
    // Basic progress bar update
    // Assuming structure exists
}

function displayAnalysisResults(data) {
    // Fill in results
    if (data.movement_quality_score) {
        document.getElementById('qualityScore').textContent = Math.round(data.movement_quality_score);
    }

    // Fill deficiencies
    const list = document.getElementById('deficienciesList');
    if (list && data.deficiencies) {
        list.innerHTML = '';
        data.deficiencies.forEach(d => {
            const div = document.createElement('div');
            div.className = 'p-4 bg-yellow-50 border-l-4 border-yellow-500 rounded';
            div.innerHTML = `<strong>${d.area}</strong> (${d.severity}): ${d.description}`;
            list.appendChild(div);
        });
    }

    // Fill exercises
    const exList = document.getElementById('exercisesList');
    if (exList && data.recommendations) {
        exList.innerHTML = '';
        data.recommendations.forEach(r => {
            const div = document.createElement('div');
            div.className = 'p-4 bg-green-50 border-l-4 border-green-500 rounded';
            div.textContent = r;
            exList.appendChild(div);
        });
    }
}

function showDetailedPermissionHelp() {
    alert("Please enable camera permissions in your browser settings and ensure no other application is using the camera.");
}

function startNewAssessment() {
    location.reload();
}

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', async () => {
  console.log('✅ Assessment workflow initialized');
  await initializeAssessment();
  await detectAvailableCameras();
});
