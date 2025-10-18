// PhysioMotion - Medical Movement Assessment Platform
// Complete Frontend Application Logic

// ============================================================================
// GLOBAL STATE
// ============================================================================

const APP_STATE = {
  currentPatient: null,
  currentAssessment: null,
  currentTest: null,
  mediaStream: null,
  poseDetector: null,
  skeletonData: [],
  isRecording: false
};

// ============================================================================
// API CLIENT
// ============================================================================

const API = {
  baseURL: '/api',
  
  async request(endpoint, options = {}) {
    try {
      const response = await fetch(`${this.baseURL}${endpoint}`, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers
        },
        ...options
      });
      
      const data = await response.json();
      
      if (!data.success) {
        throw new Error(data.error || 'API request failed');
      }
      
      return data;
    } catch (error) {
      console.error('API Error:', error);
      throw error;
    }
  },
  
  // Patient endpoints
  patients: {
    create: (data) => API.request('/patients', { method: 'POST', body: JSON.stringify(data) }),
    list: () => API.request('/patients'),
    get: (id) => API.request(`/patients/${id}`),
    addMedicalHistory: (id, data) => API.request(`/patients/${id}/medical-history`, { 
      method: 'POST', 
      body: JSON.stringify(data) 
    }),
    getAssessments: (id) => API.request(`/patients/${id}/assessments`),
    getPrescriptions: (id) => API.request(`/patients/${id}/prescriptions`),
    getSessions: (id) => API.request(`/patients/${id}/sessions`)
  },
  
  // Assessment endpoints
  assessments: {
    create: (data) => API.request('/assessments', { method: 'POST', body: JSON.stringify(data) }),
    addTest: (id, data) => API.request(`/assessments/${id}/tests`, { 
      method: 'POST', 
      body: JSON.stringify(data) 
    }),
    generateNote: (id) => API.request(`/assessments/${id}/generate-note`, { method: 'POST' })
  },
  
  // Test endpoints
  tests: {
    analyze: (id, skeletonData) => API.request(`/tests/${id}/analyze`, {
      method: 'PUT',
      body: JSON.stringify({ skeleton_data: skeletonData })
    }),
    getResults: (id) => API.request(`/tests/${id}/results`)
  },
  
  // Exercise endpoints
  exercises: {
    list: (category) => API.request(`/exercises${category ? `?category=${category}` : ''}`),
    prescribe: (data) => API.request('/prescriptions', { method: 'POST', body: JSON.stringify(data) })
  },
  
  // Session endpoints
  sessions: {
    record: (data) => API.request('/exercise-sessions', { method: 'POST', body: JSON.stringify(data) })
  }
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

function showNotification(message, type = 'info') {
  const notification = document.createElement('div');
  notification.className = `fixed top-4 right-4 px-6 py-3 rounded-lg shadow-lg text-white z-50 ${
    type === 'success' ? 'bg-green-500' :
    type === 'error' ? 'bg-red-500' :
    type === 'warning' ? 'bg-yellow-500' :
    'bg-blue-500'
  }`;
  notification.textContent = message;
  document.body.appendChild(notification);
  
  setTimeout(() => {
    notification.remove();
  }, 3000);
}

function formatDate(dateString) {
  return new Date(dateString).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  });
}

// ============================================================================
// MEDIAPIPE POSE DETECTION
// ============================================================================

async function initMediaPipePose() {
  try {
    // Load MediaPipe Pose via CDN
    await loadMediaPipeScript();
    
    const vision = await window.MediaPipeSolutions.Vision;
    const poseDetector = await vision.PoseLandmarker.createFromOptions({
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
        delegate: 'GPU'
      },
      runningMode: 'VIDEO',
      numPoses: 1
    });
    
    APP_STATE.poseDetector = poseDetector;
    return poseDetector;
  } catch (error) {
    console.error('MediaPipe initialization error:', error);
    // Fallback to simplified pose tracking
    return null;
  }
}

async function loadMediaPipeScript() {
  return new Promise((resolve, reject) => {
    if (window.MediaPipeSolutions) {
      resolve();
      return;
    }
    
    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest';
    script.onload = resolve;
    script.onerror = reject;
    document.head.appendChild(script);
  });
}

// ============================================================================
// SKELETON DATA PROCESSING
// ============================================================================

function convertMediaPipeToSkeletonData(results) {
  if (!results || !results.landmarks || results.landmarks.length === 0) {
    return null;
  }
  
  const landmarks = results.landmarks[0];
  
  // Map MediaPipe indices to our skeleton structure
  return {
    timestamp: Date.now(),
    landmarks: {
      nose: landmarks[0],
      left_eye_inner: landmarks[1],
      left_eye: landmarks[2],
      left_eye_outer: landmarks[3],
      right_eye_inner: landmarks[4],
      right_eye: landmarks[5],
      right_eye_outer: landmarks[6],
      left_ear: landmarks[7],
      right_ear: landmarks[8],
      mouth_left: landmarks[9],
      mouth_right: landmarks[10],
      left_shoulder: landmarks[11],
      right_shoulder: landmarks[12],
      left_elbow: landmarks[13],
      right_elbow: landmarks[14],
      left_wrist: landmarks[15],
      right_wrist: landmarks[16],
      left_pinky: landmarks[17],
      right_pinky: landmarks[18],
      left_index: landmarks[19],
      right_index: landmarks[20],
      left_thumb: landmarks[21],
      right_thumb: landmarks[22],
      left_hip: landmarks[23],
      right_hip: landmarks[24],
      left_knee: landmarks[25],
      right_knee: landmarks[26],
      left_ankle: landmarks[27],
      right_ankle: landmarks[28],
      left_heel: landmarks[29],
      right_heel: landmarks[30],
      left_foot_index: landmarks[31],
      right_foot_index: landmarks[32]
    }
  };
}

// ============================================================================
// CAMERA & VIDEO CAPTURE
// ============================================================================

async function startCamera(videoElement, facingMode = 'user') {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: facingMode,
        width: { ideal: 1280 },
        height: { ideal: 720 }
      },
      audio: false
    });
    
    videoElement.srcObject = stream;
    APP_STATE.mediaStream = stream;
    
    return stream;
  } catch (error) {
    console.error('Camera access error:', error);
    showNotification('Unable to access camera. Please check permissions.', 'error');
    throw error;
  }
}

function stopCamera() {
  if (APP_STATE.mediaStream) {
    APP_STATE.mediaStream.getTracks().forEach(track => track.stop());
    APP_STATE.mediaStream = null;
  }
}

// ============================================================================
// POSE TRACKING & ANALYSIS
// ============================================================================

async function startPoseTracking(videoElement, canvasElement) {
  const ctx = canvasElement.getContext('2d');
  APP_STATE.skeletonData = [];
  APP_STATE.isRecording = true;
  
  async function detectPose() {
    if (!APP_STATE.isRecording) return;
    
    try {
      if (APP_STATE.poseDetector) {
        const results = await APP_STATE.poseDetector.detectForVideo(videoElement, Date.now());
        
        if (results.landmarks && results.landmarks.length > 0) {
          // Draw skeleton
          drawSkeleton(ctx, results.landmarks[0], canvasElement.width, canvasElement.height);
          
          // Store skeleton data
          const skeletonData = convertMediaPipeToSkeletonData(results);
          if (skeletonData) {
            APP_STATE.skeletonData.push(skeletonData);
          }
        }
      }
    } catch (error) {
      console.error('Pose detection error:', error);
    }
    
    requestAnimationFrame(detectPose);
  }
  
  detectPose();
}

function stopPoseTracking() {
  APP_STATE.isRecording = false;
}

function drawSkeleton(ctx, landmarks, width, height) {
  ctx.clearRect(0, 0, width, height);
  
  // Draw connections
  const connections = [
    [11, 12], // shoulders
    [11, 13], [13, 15], // left arm
    [12, 14], [14, 16], // right arm
    [11, 23], [12, 24], // torso
    [23, 24], // hips
    [23, 25], [25, 27], // left leg
    [24, 26], [26, 28], // right leg
  ];
  
  ctx.strokeStyle = '#00ff00';
  ctx.lineWidth = 2;
  
  connections.forEach(([start, end]) => {
    const startPoint = landmarks[start];
    const endPoint = landmarks[end];
    
    if (startPoint.visibility > 0.5 && endPoint.visibility > 0.5) {
      ctx.beginPath();
      ctx.moveTo(startPoint.x * width, startPoint.y * height);
      ctx.lineTo(endPoint.x * width, endPoint.y * height);
      ctx.stroke();
    }
  });
  
  // Draw joints
  ctx.fillStyle = '#ff0000';
  landmarks.forEach(landmark => {
    if (landmark.visibility > 0.5) {
      ctx.beginPath();
      ctx.arc(landmark.x * width, landmark.y * height, 5, 0, 2 * Math.PI);
      ctx.fill();
    }
  });
}

// ============================================================================
// MOVEMENT ANALYSIS
// ============================================================================

async function analyzeMovement(testId) {
  if (APP_STATE.skeletonData.length === 0) {
    showNotification('No skeleton data captured', 'error');
    return null;
  }
  
  // Take average of captured frames for analysis
  const avgSkeleton = averageSkeletonData(APP_STATE.skeletonData);
  
  try {
    const result = await API.tests.analyze(testId, avgSkeleton);
    showNotification('Movement analysis complete!', 'success');
    return result.data;
  } catch (error) {
    showNotification('Analysis failed: ' + error.message, 'error');
    return null;
  }
}

function averageSkeletonData(skeletonArray) {
  if (skeletonArray.length === 0) return null;
  
  // Use the middle frame as representative
  const middleIndex = Math.floor(skeletonArray.length / 2);
  return skeletonArray[middleIndex];
}

// ============================================================================
// UI BUILDERS
// ============================================================================

function buildPatientIntakeForm() {
  return `
    <div class="max-w-4xl mx-auto p-6">
      <h2 class="text-3xl font-bold mb-6">New Patient Intake Form</h2>
      
      <form id="patientIntakeForm" class="space-y-6">
        <!-- Personal Information -->
        <div class="bg-white p-6 rounded-lg shadow">
          <h3 class="text-xl font-semibold mb-4">Personal Information</h3>
          <div class="grid grid-cols-2 gap-4">
            <input type="text" name="first_name" placeholder="First Name" required class="border p-2 rounded">
            <input type="text" name="last_name" placeholder="Last Name" required class="border p-2 rounded">
            <input type="date" name="date_of_birth" required class="border p-2 rounded">
            <select name="gender" required class="border p-2 rounded">
              <option value="">Select Gender</option>
              <option value="male">Male</option>
              <option value="female">Female</option>
              <option value="other">Other</option>
              <option value="prefer_not_to_say">Prefer not to say</option>
            </select>
            <input type="email" name="email" placeholder="Email" class="border p-2 rounded">
            <input type="tel" name="phone" placeholder="Phone" class="border p-2 rounded">
          </div>
        </div>
        
        <!-- Medical History -->
        <div class="bg-white p-6 rounded-lg shadow">
          <h3 class="text-xl font-semibold mb-4">Medical History</h3>
          <div class="space-y-4">
            <select name="surgery_type" class="border p-2 rounded w-full">
              <option value="none">No Surgery</option>
              <option value="pre_surgery">Pre-Surgery</option>
              <option value="post_surgery">Post-Surgery</option>
              <option value="athletic_performance">Athletic Performance</option>
            </select>
            
            <div>
              <label class="block mb-2">Current Pain Level (0-10)</label>
              <input type="range" name="current_pain_level" min="0" max="10" value="0" class="w-full">
              <span id="painLevelValue">0</span>
            </div>
            
            <select name="activity_level" class="border p-2 rounded w-full">
              <option value="sedentary">Sedentary</option>
              <option value="light">Light Activity</option>
              <option value="moderate">Moderate Activity</option>
              <option value="active">Active</option>
              <option value="very_active">Very Active</option>
            </select>
            
            <textarea name="treatment_goals" placeholder="Treatment Goals" rows="3" class="border p-2 rounded w-full"></textarea>
          </div>
        </div>
        
        <button type="submit" class="bg-blue-600 text-white px-6 py-3 rounded-lg w-full hover:bg-blue-700">
          <i class="fas fa-save mr-2"></i>Create Patient
        </button>
      </form>
    </div>
  `;
}

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', async () => {
  console.log('PhysioMotion App Initialized');
  
  // Initialize MediaPipe
  try {
    await initMediaPipePose();
    console.log('MediaPipe Pose initialized successfully');
  } catch (error) {
    console.error('Failed to initialize MediaPipe:', error);
  }
  
  // Update pain level display
  const painLevelInput = document.querySelector('input[name="current_pain_level"]');
  if (painLevelInput) {
    painLevelInput.addEventListener('input', (e) => {
      document.getElementById('painLevelValue').textContent = e.target.value;
    });
  }
});

// Export for use in other scripts
window.PhysioMotion = {
  API,
  APP_STATE,
  startCamera,
  stopCamera,
  startPoseTracking,
  stopPoseTracking,
  analyzeMovement,
  showNotification
};
