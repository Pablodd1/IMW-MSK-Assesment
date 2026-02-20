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
  ghostSkeleton: null // Captured "perfect" frame
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
    btn.classList.remove('opacity-50', 'bg-gray-700');
    btn.classList.add('opacity-100', 'bg-blue-600', 'ring-2', 'ring-blue-300');
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
    btn.classList.add('opacity-50', 'bg-gray-700');
    btn.classList.remove('opacity-100', 'bg-blue-600', 'ring-2', 'ring-blue-300');
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
  // Assuming 'Right Knee' is available. If side view, might need detection.
  const kneeAngle = angles['Right Knee']?.angle || angles['Left Knee']?.angle;
  
  if (!kneeAngle) return;

  // Draw Rep Counter Overlay
  ctx.save();
  ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
  ctx.roundRect(20, 20, 120, 80, 10);
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
    // Optional: Voice feedback for "Down"
  } else if (ASSESSMENT_STATE.repState === 'down' && kneeAngle > 150) {
    ASSESSMENT_STATE.repState = 'up';
    ASSESSMENT_STATE.repCount++;
    VoiceFeedback.speak(ASSESSMENT_STATE.repCount.toString());
    
    // Visual Pulse
    const counter = document.getElementById('repCounterDisplay'); // if we had DOM element
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
  
  // We need to convert our stored skeleton structure back to array-like or iterate keys
  // Our stored structure is object with named keys.
  // We need to map connections manually or reuse logic.
  
  // Simplified drawing for ghost (just major connections)
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
// CAMERA PERMISSION CHECK
// ============================================================================

async function checkCameraPermissions() {
  try {
    console.log('🔍 Checking camera permissions...');
    console.log('🌐 Current URL:', window.location.href);
    console.log('🔒 Protocol:', window.location.protocol);
    console.log('🌐 Host:', window.location.host);
    
    // Check if mediaDevices API is available
    if (!navigator.mediaDevices) {
      throw new Error('MediaDevices API not available. Please use HTTPS or localhost.');
    }
    
    console.log('✅ MediaDevices API is available');
    
    // Check if Permissions API is available
    if (navigator.permissions && navigator.permissions.query) {
      try {
        const result = await navigator.permissions.query({ name: 'camera' });
        console.log('📋 Permission status:', result.state);
        
        if (result.state === 'granted') {
          showNotification('✅ Camera access already granted!', 'success');
          await detectAvailableCameras();
          return true;
        } else if (result.state === 'prompt') {
          showNotification('ℹ️ Camera permission will be requested when you start', 'info');
        } else if (result.state === 'denied') {
          showNotification('❌ Camera access denied. Please enable in browser settings.', 'error');
          setTimeout(() => showDetailedPermissionHelp(), 500);
          return false;
        }
      } catch (permError) {
        console.log('⚠️ Permissions API not fully supported, will check via getUserMedia');
      }
    }
    
    // Try to get camera access to verify
    console.log('📹 Attempting to request camera access...');
    showNotification('🔍 Testing camera access...', 'info');
    
    let stream = null;
    
    // Try multiple approaches
    try {
      console.log('📷 Trying standard getUserMedia...');
      stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      console.log('✅ Standard getUserMedia succeeded!');
    } catch (error1) {
      console.warn('⚠️ Standard approach failed:', error1.name);
      
      try {
        console.log('📷 Trying minimal constraints...');
        stream = await navigator.mediaDevices.getUserMedia({ video: {} });
        console.log('✅ Minimal constraints succeeded!');
      } catch (error2) {
        console.error('❌ All camera access attempts failed');
        throw error2;
      }
    }
    
    if (!stream) {
      throw new Error('Could not obtain camera stream');
    }
    
    console.log('✅ Camera stream obtained!');
    const videoTrack = stream.getVideoTracks()[0];
    if (videoTrack) {
      console.log('📷 Camera label:', videoTrack.label);
      console.log('🎥 Settings:', videoTrack.getSettings());
    }
    
    // Success! Now stop the stream
    stream.getTracks().forEach(track => {
      console.log('🛑 Stopping track:', track.label);
      track.stop();
    });
    
    showNotification('✅ Camera access granted! You can now start the assessment.', 'success');
    
    // Detect available cameras (now with labels)
    await detectAvailableCameras();
    
    console.log('✅ Camera permission check passed');
    return true;
    
  } catch (error) {
    console.error('❌ Camera permission check failed:', error);
    console.error('Error details:', {
      name: error.name,
      message: error.message,
      stack: error.stack
    });
    
    let message = '';
    if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
      message = '❌ Camera access denied. Please allow camera access.';
      showNotification(message, 'error');
      setTimeout(() => showDetailedPermissionHelp(), 1000);
    } else if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
      message = '📷 No camera found on this device.';
      showNotification(message, 'error');
      setTimeout(() => {
        alert(`No Camera Found\n\nPlease check:\n• Camera is connected (for external cameras)\n• Camera is not covered\n• No other app is using the camera\n• Camera works in other applications\n\nTry:\n• Reconnecting USB camera\n• Restarting your browser\n• Checking device manager (Windows)\n• Checking system preferences (Mac)`);
      }, 1000);
    } else if (error.message && error.message.includes('not available')) {
      message = '⚠️ Camera API not available. Please use HTTPS.';
      showNotification(message, 'error');
      setTimeout(() => {
        alert(`Camera API Not Available\n\nThe camera API requires:\n• HTTPS connection (secure)\n• OR localhost for development\n\nCurrent: ${window.location.protocol}//${window.location.host}\n\nPlease access this page via HTTPS.`);
      }, 1000);
    } else {
      message = '⚠️ Camera check failed: ' + error.message;
      showNotification(message, 'error');
    }
    
    return false;
  }
}

// ============================================================================
// CAMERA DETECTION AND ENUMERATION
// ============================================================================

async function detectAvailableCameras() {
  try {
    console.log('🔍 Detecting available cameras...');
    
    // Note: enumerateDevices() might return devices without labels if permission not granted yet
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(device => device.kind === 'videoinput');
    
    console.log(`📷 Raw video devices found: ${videoDevices.length}`);
    videoDevices.forEach((device, idx) => {
      console.log(`  Device ${idx + 1}:`, {
        deviceId: device.deviceId,
        label: device.label || '(No label - permission needed)',
        groupId: device.groupId
      });
    });
    
    ASSESSMENT_STATE.availableCameras = videoDevices.map((device, index) => ({
      deviceId: device.deviceId,
      label: device.label || `Camera ${index + 1}`,
      isFrontFacing: device.label.toLowerCase().includes('front'),
      isBackFacing: device.label.toLowerCase().includes('back')
    }));
    
    if (ASSESSMENT_STATE.availableCameras.length > 0) {
      console.log('✅ Available cameras:', ASSESSMENT_STATE.availableCameras);
      showNotification(`Found ${ASSESSMENT_STATE.availableCameras.length} camera(s)`, 'success');
    } else {
      console.warn('⚠️ No cameras detected. This might be before permission is granted.');
      showNotification('Please grant camera permission to detect available cameras', 'info');
    }
    
    return ASSESSMENT_STATE.availableCameras;
  } catch (error) {
    console.error('❌ Error detecting cameras:', error);
    showNotification('Could not detect cameras. Will use default camera.', 'warning');
    return [];
  }
}

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
  } else {
    document.getElementById('flipBtn').style.display = 'none';
  }
}

async function startAssessment() {
  // Hide modal
  document.getElementById('cameraSelectionModal').style.display = 'none';
  
  // Show camera container
  document.getElementById('cameraContainer').style.display = 'block';
  
  // Update progress
  updateProgress(2);
  
  // Detect available cameras first
  await detectAvailableCameras();
  
  // Initialize selected camera
  switch (ASSESSMENT_STATE.selectedCamera) {
    case 'phone':
      // For phone, try to use back camera by default
      ASSESSMENT_STATE.currentFacingMode = 'environment';
      document.getElementById('flipBtn').style.display = 'flex';
      document.getElementById('flipBtnText').textContent = 'Flip Camera';
      await initializeWebCamera();
      break;
    case 'webcam':
      // For laptop/desktop, show switch button if multiple cameras
      if (ASSESSMENT_STATE.availableCameras.length > 1) {
        document.getElementById('flipBtn').style.display = 'flex';
        document.getElementById('flipBtnText').textContent = 'Switch Camera';
      }
      await initializeWebCamera();
      break;
    case 'femto':
      // Show configuration dialog if bridge URL not set
      const savedUrl = localStorage.getItem('femto_bridge_url');
      if (!savedUrl) {
        const configUrl = prompt(
          'Enter Femto Mega Bridge Server URL:\n\n' +
          'Examples:\n' +
          '  - ws://localhost:8765 (same machine)\n' +
          '  - ws://192.168.1.100:8765 (local network)\n\n' +
          'You can change this later in browser console:\n' +
          'localStorage.setItem(\'femto_bridge_url\', \'ws://YOUR_IP:8765\')',
          'ws://localhost:8765'
        );
        if (configUrl) {
          localStorage.setItem('femto_bridge_url', configUrl);
          console.log(`✅ Femto bridge URL saved: ${configUrl}`);
        }
      }
      await initializeFemtoMega();
      break;
    case 'upload':
      handleVideoUpload();
      break;
  }
}

// ============================================================================
// WEB CAMERA INITIALIZATION (Phone & Laptop & External)
// ============================================================================

async function initializeWebCamera() {
  try {
    showStatus('Requesting camera access...', 'warning');
    
    const video = document.getElementById('videoElement');
    const canvas = document.getElementById('canvasElement');
    
    // Check if getUserMedia is supported
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('Camera API not supported in this browser. Please use Chrome, Firefox, Safari, or Edge.');
    }
    
    // Stop existing stream if any
    if (ASSESSMENT_STATE.cameraStream) {
      ASSESSMENT_STATE.cameraStream.getTracks().forEach(track => track.stop());
    }
    
    console.log('📷 Starting camera initialization...');
    console.log('🌐 Page URL:', window.location.href);
    console.log('🔒 Protocol:', window.location.protocol);
    console.log('📱 Selected camera type:', ASSESSMENT_STATE.selectedCamera);
    console.log('🎥 Available cameras:', ASSESSMENT_STATE.availableCameras.length);
    
    showStatus('Please allow camera access in your browser...', 'warning');
    
    // Try multiple constraint strategies
    let stream = null;
    let lastError = null;
    
    // Strategy 1: Try with specified constraints
    try {
      let constraints;
      
      if (ASSESSMENT_STATE.selectedCamera === 'phone') {
        // Mobile phone - use facingMode
        constraints = {
          video: {
            facingMode: ASSESSMENT_STATE.currentFacingMode,
            width: { ideal: 1280 },
            height: { ideal: 720 }
          },
          audio: false
        };
      } else if (ASSESSMENT_STATE.selectedDeviceId) {
        // Specific device selected
        constraints = {
          video: {
            deviceId: { exact: ASSESSMENT_STATE.selectedDeviceId },
            width: { ideal: 1280 },
            height: { ideal: 720 }
          },
          audio: false
        };
      } else {
        // Default - use any available camera
        constraints = {
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 }
          },
          audio: false
        };
      }
      
      console.log('📷 Strategy 1 - Trying with constraints:', constraints);
      stream = await navigator.mediaDevices.getUserMedia(constraints);
      console.log('✅ Strategy 1 succeeded!');
      
    } catch (error1) {
      console.warn('⚠️ Strategy 1 failed:', error1.name);
      lastError = error1;
      
      // Strategy 2: Try with basic video constraint only
      try {
        console.log('📷 Strategy 2 - Trying with basic constraints...');
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        console.log('✅ Strategy 2 succeeded!');
        
      } catch (error2) {
        console.warn('⚠️ Strategy 2 failed:', error2.name);
        lastError = error2;
        
        // Strategy 3: Try with minimal constraints (just video: {})
        try {
          console.log('📷 Strategy 3 - Trying with minimal constraints...');
          stream = await navigator.mediaDevices.getUserMedia({ video: {} });
          console.log('✅ Strategy 3 succeeded!');
          
        } catch (error3) {
          console.error('❌ All strategies failed');
          lastError = error3;
          throw lastError;
        }
      }
    }
    
    if (!stream) {
      throw lastError || new Error('Failed to get camera stream');
    }
    
    console.log('✅ Camera access granted!');
    console.log('📹 Stream tracks:', stream.getTracks().map(t => `${t.kind}: ${t.label}`));
    
    video.srcObject = stream;
    ASSESSMENT_STATE.cameraStream = stream;
    
    // Get the actual camera being used
    const videoTrack = stream.getVideoTracks()[0];
    const settings = videoTrack.getSettings();
    console.log('✅ Camera settings:', settings);
    
    // Display camera info
    const cameraLabel = videoTrack.label || 'Unknown Camera';
    console.log('📹 Using camera:', cameraLabel);
    
    // Wait for video to load
    await new Promise((resolve) => {
      video.onloadedmetadata = () => {
        console.log('✅ Video metadata loaded');
        resolve();
      };
    });
    
    await video.play();
    console.log('✅ Video playing');
    
    // Check if video has actual dimensions (not black screen)
    if (video.videoWidth === 0 || video.videoHeight === 0) {
      console.error('⚠️ WARNING: Video has no dimensions! Camera may be blocked or in use.');
      showNotification('Camera is connected but showing black screen. Check if another app is using the camera.', 'warning');
    }
    
    // Set canvas size to match video
    canvas.width = video.videoWidth || 1280;
    canvas.height = video.videoHeight || 720;
    
    console.log(`✅ Canvas size: ${canvas.width}x${canvas.height}`);
    
    // Initialize MediaPipe Pose
    await initializeMediaPipePose();
    
    // Add diagnostic check after 2 seconds
    setTimeout(() => {
      const video = document.getElementById('videoElement');
      const canvas = document.getElementById('canvasElement');
      
      if (video.videoWidth === 0 || video.videoHeight === 0) {
        console.error('🚨 DIAGNOSTIC: Camera stream exists but no video dimensions!');
        console.error('   This usually means:');
        console.error('   1. Another app is using the camera (Zoom, Teams, Skype, etc.)');
        console.error('   2. Camera privacy settings are blocking video');
        console.error('   3. Camera driver issue');
        console.error('   Video dimensions:', video.videoWidth, 'x', video.videoHeight);
        console.error('   Video ready state:', video.readyState);
        console.error('   Stream active:', ASSESSMENT_STATE.cameraStream?.active);
        
        // Draw diagnostic message on canvas
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#00ff00';
        ctx.font = 'bold 32px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('⚠️ CAMERA CONNECTED BUT NO VIDEO', canvas.width/2, canvas.height/2 - 100);
        ctx.font = '24px Arial';
        ctx.fillStyle = '#ffffff';
        ctx.fillText('Camera detected: ' + (ASSESSMENT_STATE.cameraStream?.getVideoTracks()[0]?.label || 'Unknown'), canvas.width/2, canvas.height/2 - 40);
        ctx.fillText('Possible causes:', canvas.width/2, canvas.height/2 + 20);
        ctx.font = '20px Arial';
        ctx.fillText('1. Another app is using your camera (close Zoom, Teams, etc.)', canvas.width/2, canvas.height/2 + 60);
        ctx.fillText('2. Camera privacy settings are blocking video', canvas.width/2, canvas.height/2 + 90);
        ctx.fillText('3. Try closing all apps and refresh this page', canvas.width/2, canvas.height/2 + 120);
        ctx.fillText('Press F12 to see console logs for more details', canvas.width/2, canvas.height/2 + 160);
        
        showNotification('Camera connected but showing black screen. Check console (F12) for details.', 'warning');
      } else {
        console.log('✅ Video feed working! Dimensions:', video.videoWidth, 'x', video.videoHeight);
      }
    }, 2000);
    
    showStatus(`Camera connected: ${cameraLabel}`, 'success');
    showNotification('Camera connected successfully!', 'success');
    
  } catch (error) {
    console.error('❌ Camera initialization error:', error);
    console.error('Error name:', error.name);
    console.error('Error message:', error.message);
    
    showStatus('Camera access denied', 'error');
    
    // More helpful error messages with specific instructions
    let errorMessage = '';
    let helpText = '';
    
    if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
      errorMessage = '🚫 Camera Access Denied';
      helpText = `
Please grant camera permissions:

📱 Mobile:
• iPhone: Settings → Safari → Camera → Allow
• Android Chrome: Settings → Site Settings → Camera → Allow for this site

💻 Desktop:
• Chrome: Click the 🔒 lock icon → Camera → Allow
• Firefox: Click the 🔒 lock icon → Permissions → Camera → Allow
• Safari: Settings → Websites → Camera → Allow for this site

After granting permission, please refresh this page.
      `.trim();
    } else if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
      errorMessage = '📷 No Camera Found';
      helpText = `
No camera detected. Please check:

1. ✓ Camera is physically connected (USB cameras)
2. ✓ Camera is not covered or blocked
3. ✓ No other app is using the camera
4. ✓ Camera drivers are installed (Windows/Linux)
5. ✓ Try a different browser

For built-in cameras, try restarting your device.
      `.trim();
    } else if (error.name === 'NotReadableError' || error.name === 'TrackStartError') {
      errorMessage = '⚠️ Camera In Use';
      helpText = `
Camera is already being used by another application.

Please:
1. Close other apps that might be using the camera
2. Close other browser tabs with camera access
3. Restart your browser
4. Try again

Common apps that use camera: Zoom, Skype, Teams, OBS
      `.trim();
    } else if (error.name === 'OverconstrainedError' || error.name === 'ConstraintNotSatisfiedError') {
      errorMessage = '⚙️ Camera Constraints Not Supported';
      helpText = `
The requested camera settings are not supported.

Trying with default settings...
      `.trim();
      
      // Try again with minimal constraints
      setTimeout(() => {
        retryWithMinimalConstraints();
      }, 2000);
      
    } else if (error.message && error.message.includes('not supported')) {
      errorMessage = '🌐 Browser Not Supported';
      helpText = `
This browser doesn't support camera access.

Please use:
• Google Chrome
• Mozilla Firefox  
• Safari (iOS/macOS)
• Microsoft Edge

Make sure you're accessing via HTTPS (secure connection).
      `.trim();
    } else {
      errorMessage = '❌ Camera Error';
      helpText = `
An unexpected error occurred: ${error.message}

Please try:
1. Refresh the page
2. Use a different browser
3. Check browser console for details
4. Contact support if issue persists

Debug Info:
• Error: ${error.name}
• Protocol: ${window.location.protocol}
• Browser: ${navigator.userAgent.split(' ').pop()}
      `.trim();
    }
    
    // Show detailed error message
    alert(`${errorMessage}\n\n${helpText}`);
    
    // Log detailed error for debugging
    console.log('🔍 Full error details:', {
      name: error.name,
      message: error.message,
      stack: error.stack,
      protocol: window.location.protocol,
      userAgent: navigator.userAgent
    });
  }
}

// Retry with minimal constraints if advanced constraints fail
async function retryWithMinimalConstraints() {
  try {
    console.log('🔄 Retrying with minimal constraints...');
    showStatus('Retrying with basic settings...', 'warning');
    
    const video = document.getElementById('videoElement');
    const canvas = document.getElementById('canvasElement');
    
    // Ultra-simple constraints
    const stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false
    });
    
    video.srcObject = stream;
    ASSESSMENT_STATE.cameraStream = stream;
    
    await new Promise((resolve) => {
      video.onloadedmetadata = resolve;
    });
    
    await video.play();
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    await initializeMediaPipePose();
    
    showStatus('Camera connected (basic mode)', 'success');
    showNotification('Camera connected with basic settings', 'success');
    
  } catch (retryError) {
    console.error('❌ Retry also failed:', retryError);
    showStatus('Camera initialization failed', 'error');
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
  
  // Draw connections in BRIGHT GREEN
  const connections = window.POSE_CONNECTIONS;
  ctx.strokeStyle = '#00ff00'; // BRIGHT GREEN lines
  ctx.lineWidth = 4;
  ctx.shadowBlur = 15;
  ctx.shadowColor = '#00ff00';
  
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
  
  // Draw joints in BRIGHT GREEN
  ctx.fillStyle = '#00ff00'; // BRIGHT GREEN circles
  ctx.strokeStyle = '#ffffff'; // WHITE border
  ctx.lineWidth = 3;
  ctx.shadowBlur = 20;
  ctx.shadowColor = '#00ff00';
  
  // Major joints (larger circles)
  const majorJoints = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
  
  landmarks.forEach((landmark, index) => {
    if (landmark.visibility > 0.5) {
      const x = landmark.x * canvas.width;
      const y = landmark.y * canvas.height;
      const radius = majorJoints.includes(index) ? 10 : 6;
      
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
    }
  });
  
  // Reset shadow
  ctx.shadowBlur = 0;
  
  // Convert landmarks to skeleton data for analysis
  const skeletonData = convertLandmarksToSkeletonData(landmarks);

  // If recording, save frame
  if (ASSESSMENT_STATE.isRecording) {
    ASSESSMENT_STATE.skeletonFrames.push(skeletonData);
  }

  // Always update UI components
  updateJointAnglesPanel(skeletonData);
  drawGhostSkeleton(ctx, canvas);
  updateRepCounter(calculateQuickJointAngles(skeletonData.landmarks), ctx, canvas);

  // Ghost mode capture logic
  if (ASSESSMENT_STATE.features.ghostMode && !ASSESSMENT_STATE.ghostSkeleton) {
      ASSESSMENT_STATE.ghostSkeleton = skeletonData;
      showNotification("Ghost reference set!", "success");
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
    
    // Voice Feedback Logic
    if (ASSESSMENT_STATE.features.voiceFeedback) {
        if (angles['Right Knee'].status === 'limited' && angles['Right Knee'].angle < 70) {
            VoiceFeedback.speak("Go lower");
        } else if (angles['Right Knee'].angle > 160) {
            // VoiceFeedback.speak("Good extension");
        }
        
        // Valgus check (simplified 2D check)
        // If knee X is inside ankle X significantly
    }

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
// CAMERA FLIP (PHONE AND EXTERNAL CAMERAS)
// ============================================================================

async function flipCamera() {
  if (ASSESSMENT_STATE.selectedCamera === 'phone') {
    // Mobile phone - toggle facing mode
    ASSESSMENT_STATE.currentFacingMode = 
      ASSESSMENT_STATE.currentFacingMode === 'user' ? 'environment' : 'user';
    
    // Stop current stream
    if (ASSESSMENT_STATE.cameraStream) {
      ASSESSMENT_STATE.cameraStream.getTracks().forEach(track => track.stop());
    }
    
    // Restart with new facing mode
    await initializeWebCamera();
    
    showNotification(`Switched to ${ASSESSMENT_STATE.currentFacingMode === 'user' ? 'front' : 'back'} camera`, 'info');
  } else {
    // Desktop/Laptop - cycle through available cameras
    if (ASSESSMENT_STATE.availableCameras.length > 1) {
      const currentIndex = ASSESSMENT_STATE.availableCameras.findIndex(
        cam => cam.deviceId === ASSESSMENT_STATE.selectedDeviceId
      );
      
      const nextIndex = (currentIndex + 1) % ASSESSMENT_STATE.availableCameras.length;
      ASSESSMENT_STATE.selectedDeviceId = ASSESSMENT_STATE.availableCameras[nextIndex].deviceId;
      
      // Stop current stream
      if (ASSESSMENT_STATE.cameraStream) {
        ASSESSMENT_STATE.cameraStream.getTracks().forEach(track => track.stop());
      }
      
      // Restart with new camera
      await initializeWebCamera();
      
      showNotification(`Switched to ${ASSESSMENT_STATE.availableCameras[nextIndex].label}`, 'info');
    } else {
      showNotification('Only one camera detected', 'info');
    }
  }
}

// ============================================================================
// FEMTO MEGA INTEGRATION
// ============================================================================

async function initializeFemtoMega() {
  try {
    showStatus('Connecting to Femto Mega...', 'warning');
    
    // Get bridge server URL from localStorage or use default
    const bridgeUrl = localStorage.getItem('femto_bridge_url') || 'ws://localhost:8765';
    console.log(`📡 Connecting to Femto Mega bridge at: ${bridgeUrl}`);
    
    const video = document.getElementById('videoElement');
    const canvas = document.getElementById('canvasElement');
    
    // Try the new client with video streaming first
    try {
      const femtoClient = await initializeFemtoMegaWithVideo(video, canvas);
      if (femtoClient) {
        ASSESSMENT_STATE.femtoMegaClient = femtoClient;
        showStatus('Femto Mega connected - Streaming', 'success');
        showNotification('Professional camera ready with video', 'success');
        return;
      }
    } catch (videoError) {
      console.warn('Video streaming failed, trying basic mode:', videoError);
    }
    
    // Fallback to basic client without video
    const femtoClient = new FemtoMegaClient(bridgeUrl);
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
    
    // Start streaming without video
    femtoClient.startStreaming(null, canvas);
    
    showStatus('Femto Mega connected', 'success');
    showNotification('Professional camera ready', 'success');
    
  } catch (error) {
    console.error('Femto Mega connection error:', error);
    showStatus('Connection failed', 'error');
    alert('Failed to connect to Femto Mega. Please ensure:\n1. Femto Mega camera is connected\n2. Bridge server is running\n3. Bridge server address is correct\n\nTo run bridge: python femto_bridge/server_production.py');
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

function showDetailedPermissionHelp() {
  const userAgent = navigator.userAgent.toLowerCase();
  const isIOS = /iphone|ipad|ipod/.test(userAgent);
  const isAndroid = /android/.test(userAgent);
  const isChrome = /chrome/.test(userAgent) && !/edge/.test(userAgent);
  const isFirefox = /firefox/.test(userAgent);
  const isSafari = /safari/.test(userAgent) && !/chrome/.test(userAgent);
  
  let instructions = '';
  
  if (isIOS) {
    instructions = `
📱 iPhone/iPad Camera Permission:

1. Open Settings app
2. Scroll down to Safari (or your browser)
3. Tap "Camera"
4. Select "Allow"
5. Return to this page and refresh

Also make sure:
• Camera is not covered
• No other app is using camera
• iOS is up to date
    `.trim();
  } else if (isAndroid) {
    instructions = `
📱 Android Camera Permission:

1. Go to Settings
2. Apps → Chrome (or your browser)
3. Permissions → Camera
4. Select "Allow"
5. Return to this page and refresh

Or:
1. Long-press the Chrome app icon
2. App info → Permissions → Camera → Allow

Also make sure:
• Camera is not covered
• No other app is using camera
• Android is up to date
    `.trim();
  } else if (isChrome) {
    instructions = `
💻 Chrome Desktop Camera Permission:

1. Click the 🔒 lock icon (left of URL)
2. Find "Camera" in the list
3. Change from "Block" to "Allow"
4. Refresh this page

Alternative:
1. Chrome Settings → Privacy and Security
2. Site Settings → Camera
3. Remove this site from "Blocked"
4. Refresh this page
    `.trim();
  } else if (isFirefox) {
    instructions = `
💻 Firefox Camera Permission:

1. Click the 🔒 lock icon (left of URL)
2. Click ">" next to "Connection secure"
3. Find "Use the Camera"
4. Change to "Allow"
5. Refresh this page

Or clear permission and try again:
1. Click lock icon
2. Clear permissions and cookies
3. Refresh and allow when prompted
    `.trim();
  } else if (isSafari) {
    instructions = `
💻 Safari Camera Permission:

1. Safari menu → Settings for This Website
2. Camera → Allow
3. Refresh this page

Or:
1. Safari menu → Settings
2. Websites → Camera
3. Find this site and set to "Allow"
4. Refresh this page
    `.trim();
  } else {
    instructions = `
💻 Browser Camera Permission:

General steps:
1. Look for camera icon in address bar
2. Click it and select "Allow"
3. Refresh this page

If that doesn't work:
1. Check browser settings
2. Look for camera/permissions section
3. Allow camera for this website
4. Refresh this page
    `.trim();
  }
  
  instructions += `

🌐 Additional Requirements:
• Must use HTTPS (secure connection) ✓
• Camera must not be in use by other apps
• Check if camera is working in other apps
• Try restarting browser if issues persist

Current Info:
• Protocol: ${window.location.protocol}
• Browser: ${navigator.userAgent.split(' ').slice(-2).join(' ')}
  `;
  
  alert(instructions);
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

document.addEventListener('DOMContentLoaded', async () => {
  console.log('✅ Assessment workflow initialized');
  console.log('📷 Camera options: Phone, Laptop, External, Femto Mega');
  console.log('🔴 Live joint tracking enabled');
  
  // Initialize assessment (patient selection and test creation)
  await initializeAssessment();
  
  // Detect available cameras early for better UX
  await detectAvailableCameras();
  
  // Show flip button if laptop with multiple cameras
  if (ASSESSMENT_STATE.selectedCamera === 'webcam' && ASSESSMENT_STATE.availableCameras.length > 1) {
    document.getElementById('flipBtn').style.display = 'flex';
    document.getElementById('flipBtn').innerHTML = '<i class="fas fa-sync-alt"></i> Switch Camera';
  }
});
