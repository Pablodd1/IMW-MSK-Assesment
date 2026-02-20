// Femto Mega Client - Full Control with Video Streaming
// Proper color handling and real-time skeleton tracking

class FemtoMegaClient {
  constructor(bridgeUrl = 'ws://localhost:8765') {
    this.bridgeUrl = bridgeUrl;
    this.ws = null;
    this.isConnected = false;
    this.isStreaming = false;
    this.onSkeletonData = null;
    this.onVideoFrame = null;
    this.onStatusChange = null;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectDelay = 2000;
    this.frameCount = 0;
    this.lastFrameTime = 0;
    this.latency = 0;
    
    // Video element for display
    this.videoElement = null;
    this.canvasElement = null;
    this.ctx = null;
  }

  async connect() {
    return new Promise((resolve, reject) => {
      try {
        console.log(`🔌 Connecting to Femto Mega bridge: ${this.bridgeUrl}`);
        
        this.ws = new WebSocket(this.bridgeUrl);
        
        this.ws.onopen = () => {
          console.log('✅ Femto Mega connected');
          this.isConnected = true;
          this.reconnectAttempts = 0;
          this.updateStatus('connected');
          resolve();
        };
        
        this.ws.onmessage = (event) => {
          this.handleMessage(event);
        };
        
        this.ws.onerror = (error) => {
          console.error('❌ WebSocket error:', error);
          this.updateStatus('error');
        };
        
        this.ws.onclose = () => {
          console.log('🔌 Femto Mega disconnected');
          this.isConnected = false;
          this.isStreaming = false;
          this.updateStatus('disconnected');
          this.attemptReconnect();
        };
        
      } catch (error) {
        console.error('❌ Connection failed:', error);
        reject(error);
      }
    });
  }

  attemptReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts && !this.isConnected) {
      this.reconnectAttempts++;
      console.log(`🔄 Reconnecting... attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts}`);
      setTimeout(() => {
        this.connect().catch(console.error);
      }, this.reconnectDelay);
    }
  }

  disconnect() {
    if (this.ws) {
      this.isStreaming = false;
      this.sendCommand('stop_streaming');
      this.ws.close();
      this.ws = null;
    }
    this.isConnected = false;
  }

  sendCommand(command, data = {}) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        command,
        ...data,
        timestamp: Date.now()
      }));
    }
  }

  handleMessage(event) {
    try {
      const data = JSON.parse(event.data);
      const now = Date.now();
      
      // Calculate latency if timestamp included
      if (data.timestamp) {
        this.latency = now - data.timestamp;
      }

      switch (data.type) {
        case 'skeleton':
          this.frameCount++;
          this.lastFrameTime = now;
          
          if (this.onSkeletonData) {
            this.onSkeletonData(data.skeleton);
          }
          break;
          
        case 'video_frame':
          // Handle video frame from bridge
          if (this.onVideoFrame && data.image) {
            this.onVideoFrame(data.image);
          }
          break;
          
        case 'status':
          this.updateStatus(data.status, data.message);
          break;
          
        case 'error':
          console.error('Femto Mega error:', data.message);
          break;
          
        case 'pong':
          // Response to ping
          break;
          
        default:
          console.log('Unknown message type:', data.type);
      }
    } catch (error) {
      console.error('Error handling message:', error);
    }
  }

  startStreaming(videoElement = null, canvasElement = null) {
    if (!this.isConnected) {
      console.error('Not connected to Femto Mega');
      return false;
    }
    
    // Set up video display
    this.videoElement = videoElement;
    this.canvasElement = canvasElement;
    
    if (canvasElement) {
      this.ctx = canvasElement.getContext('2d', { willReadFrequently: true });
    }
    
    // Request video stream from bridge
    this.sendCommand('start_streaming', {
      include_video: true,
      video_quality: 'high',
      target_fps: 30
    });
    
    this.isStreaming = true;
    this.updateStatus('streaming');
    
    // Start ping/pong for connection health
    this.startPingInterval();
    
    return true;
  }

  stopStreaming() {
    this.sendCommand('stop_streaming');
    this.isStreaming = false;
    this.updateStatus('connected');
    this.stopPingInterval();
  }

  startPingInterval() {
    this.pingInterval = setInterval(() => {
      if (this.isConnected) {
        this.sendCommand('ping');
      }
    }, 5000);
  }

  stopPingInterval() {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }

  // Camera control commands
  setAutoExposure(enabled) {
    this.sendCommand('set_auto_exposure', { enabled });
  }

  setExposure(value) {
    this.sendCommand('set_exposure', { value });
  }

  setWhiteBalance(mode) {
    this.sendCommand('set_white_balance', { mode }); // 'auto', 'manual', 'daylight', etc.
  }

  setGain(value) {
    this.sendCommand('set_gain', { value });
  }

  setLaser(enabled) {
    this.sendCommand('set_laser', { enabled });
  }

  // Get camera info
  getCameraInfo() {
    this.sendCommand('get_camera_info');
  }

  // Get streaming stats
  getStats() {
    return {
      isConnected: this.isConnected,
      isStreaming: this.isStreaming,
      frameCount: this.frameCount,
      fps: this.calculateFPS(),
      latency: this.latency,
      reconnectAttempts: this.reconnectAttempts
    };
  }

  calculateFPS() {
    if (this.lastFrameTime === 0) return 0;
    const timeDiff = Date.now() - this.lastFrameTime;
    if (timeDiff > 1000) return 0;
    return Math.round(this.frameCount / (timeDiff / 1000));
  }

  updateStatus(status, message = '') {
    if (this.onStatusChange) {
      this.onStatusChange(status, message);
    }
  }
}

// Global instance
window.FemtoMegaClient = FemtoMegaClient;

// ============================================================================
// DRAWING FUNCTIONS FOR FEMTO MEGA SKELETON
// ============================================================================

function drawFemtoMegaSkeleton(skeletonData) {
  const canvas = document.getElementById('canvasElement');
  if (!canvas) return;
  
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  
  // Clear canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  if (!skeletonData || !skeletonData.landmarks) {
    return;
  }
  
  const landmarks = skeletonData.landmarks;
  const connections = getSkeletonConnections();
  
  // Draw connections (bones)
  ctx.lineWidth = 3;
  ctx.lineCap = 'round';
  
  for (const [start, end] of connections) {
    const startJoint = landmarks[start];
    const endJoint = landmarks[end];
    
    if (startJoint && endJoint && 
        startJoint.visibility > 0.5 && 
        endJoint.visibility > 0.5) {
      
      // Color based on confidence
      const avgConfidence = (startJoint.visibility + endJoint.visibility) / 2;
      ctx.strokeStyle = avgConfidence > 0.7 ? 
        'rgba(255, 255, 0, 0.9)' : // Yellow for high confidence
        'rgba(255, 165, 0, 0.7)';  // Orange for lower
      
      ctx.beginPath();
      ctx.moveTo(startJoint.x * canvas.width, startJoint.y * canvas.height);
      ctx.lineTo(endJoint.x * canvas.width, endJoint.y * canvas.height);
      ctx.stroke();
    }
  }
  
  // Draw joints
  const majorJoints = [
    'nose', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    'left_hip', 'right_hip', 'left_knee', 'right_knee',
    'left_ankle', 'right_ankle'
  ];
  
  for (const jointName of majorJoints) {
    const joint = landmarks[jointName];
    if (joint && joint.visibility > 0.5) {
      const x = joint.x * canvas.width;
      const y = joint.y * canvas.height;
      
      // Draw circle
      ctx.beginPath();
      ctx.arc(x, y, 8, 0, 2 * Math.PI);
      ctx.fillStyle = joint.visibility > 0.7 ? 
        'rgba(255, 0, 0, 0.9)' :  // Red for high confidence
        'rgba(255, 100, 0, 0.7)'; // Orange for lower
      ctx.fill();
      ctx.strokeStyle = 'white';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }
  
  // Draw minor joints
  for (const [name, joint] of Object.entries(landmarks)) {
    if (majorJoints.includes(name)) continue;
    
    if (joint && joint.visibility > 0.5) {
      const x = joint.x * canvas.width;
      const y = joint.y * canvas.height;
      
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, 2 * Math.PI);
      ctx.fillStyle = joint.visibility > 0.7 ? 
        'rgba(255, 0, 0, 0.7)' : 
        'rgba(255, 165, 0, 0.5)';
      ctx.fill();
    }
  }
}

function getSkeletonConnections() {
  // MediaPipe Pose connections
  return [
    // Face
    ['nose', 'left_eye_inner'],
    ['left_eye_inner', 'left_eye'],
    ['left_eye', 'left_eye_outer'],
    ['left_eye_outer', 'left_ear'],
    ['nose', 'right_eye_inner'],
    ['right_eye_inner', 'right_eye'],
    ['right_eye', 'right_eye_outer'],
    ['right_eye_outer', 'right_ear'],
    ['mouth_left', 'mouth_right'],
    
    // Torso
    ['left_shoulder', 'right_shoulder'],
    ['left_shoulder', 'left_hip'],
    ['right_shoulder', 'right_hip'],
    ['left_hip', 'right_hip'],
    
    // Left arm
    ['left_shoulder', 'left_elbow'],
    ['left_elbow', 'left_wrist'],
    ['left_wrist', 'left_pinky'],
    ['left_wrist', 'left_index'],
    ['left_wrist', 'left_thumb'],
    
    // Right arm
    ['right_shoulder', 'right_elbow'],
    ['right_elbow', 'right_wrist'],
    ['right_wrist', 'right_pinky'],
    ['right_wrist', 'right_index'],
    ['right_wrist', 'right_thumb'],
    
    // Left leg
    ['left_hip', 'left_knee'],
    ['left_knee', 'left_ankle'],
    ['left_ankle', 'left_heel'],
    ['left_ankle', 'left_foot_index'],
    
    // Right leg
    ['right_hip', 'right_knee'],
    ['right_knee', 'right_ankle'],
    ['right_ankle', 'right_heel'],
    ['right_ankle', 'right_foot_index']
  ];
}

// Initialize Femto Mega with proper video
async function initializeFemtoMegaWithVideo(videoElement, canvasElement) {
  try {
    showStatus('Connecting to Femto Mega...', 'warning');
    
    const bridgeUrl = localStorage.getItem('femto_bridge_url') || 'ws://localhost:8765';
    console.log(`📡 Connecting to Femto Mega bridge at: ${bridgeUrl}`);
    
    const femtoClient = new FemtoMegaClient(bridgeUrl);
    
    femtoClient.onStatusChange = (status, message) => {
      console.log(`Femto Mega status: ${status} - ${message}`);
      showStatus(`Femto Mega: ${status}`, status === 'streaming' ? 'success' : 'warning');
    };
    
    // Set up video frame handler
    femtoClient.onVideoFrame = (imageData) => {
      if (videoElement && imageData) {
        // Create image from base64 or blob
        const img = new Image();
        img.onload = () => {
          if (videoElement) {
            videoElement.src = img.src;
          }
        };
        img.src = `data:image/jpeg;base64,${imageData}`;
      }
    };
    
    await femtoClient.connect();
    
    // Start streaming with video
    femtoClient.startStreaming(videoElement, canvasElement);
    
    ASSESSMENT_STATE.femtoMegaClient = femtoClient;
    
    showStatus('Femto Mega connected - Streaming', 'success');
    showNotification('Professional camera ready with video', 'success');
    
    return femtoClient;
    
  } catch (error) {
    console.error('Femto Mega connection error:', error);
    showStatus('Connection failed', 'error');
    alert('Failed to connect to Femto Mega. Please ensure:\n1. Femto Mega camera is connected\n2. Bridge server is running\n3. Bridge server address is correct');
    return null;
  }
}

// Export for use
window.initializeFemtoMegaWithVideo = initializeFemtoMegaWithVideo;
window.drawFemtoMegaSkeleton = drawFemtoMegaSkeleton;
