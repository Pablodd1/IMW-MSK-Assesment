// ============================================================================
// PhysioMotion - High-Quality Multi-Camera Manager
// Supports: Laptop Camera, Mobile Phone, Orbbec Femto Mega
// Optimized for: 1080p+, 30fps+, highest accuracy pose detection
// ============================================================================

const IMW_CAMERA_DEFAULT_CONFIG = {
    poseEngineUrl: 'wss://pablodd1--pose-engine-ws-serve.modal.run/ws',
    supabaseUrl: '',
    demoMode: false,
    voiceEnabled: true
};

function getIMWCameraConfig() {
    return { ...IMW_CAMERA_DEFAULT_CONFIG, ...(window.IMW_CONFIG || {}) };
}

const CameraManager = {
    // State
    state: {
        activeStream: null,
        activeCamera: null,
        cameraType: null, // 'webcam', 'phone', 'femto'
        facingMode: 'environment', // 'user' (front), 'environment' (back)
        availableCameras: [],
        preferredResolution: { width: 1920, height: 1080 },
        fallbackResolution: { width: 1280, height: 720 },
        frameRate: 30,
        isMirror: false,
        qualityScore: 0,
        lightingLevel: 'good', // 'good', 'low', 'poor'
        depthFrame: null,
        skeletonFrame: null,
        femtoConnected: false,
        onDepthFrame: null,
        onSkeletonFrame: null,
    },

    // Camera type definitions with optimal settings
    cameraProfiles: {
        webcam: {
            name: 'Laptop Camera',
            icon: 'fa-laptop',
            constraints: {
                video: {
                    width: { ideal: 1920, min: 1280 },
                    height: { ideal: 1080, min: 720 },
                    frameRate: { ideal: 30, min: 24 },
                    facingMode: 'user'
                }
            },
            mirror: true,
            tips: [
                'Ensure good lighting from the front',
                'Position camera at hip height for full body',
                'Clear background for better tracking'
            ]
        },
        phone: {
            name: 'Mobile Phone',
            icon: 'fa-mobile-alt',
            constraints: {
                video: {
                    width: { ideal: 1920, min: 1280 },
                    height: { ideal: 1080, min: 720 },
                    frameRate: { ideal: 30, min: 24 },
                    facingMode: 'environment' // Default to back camera
                }
            },
            mirror: false,
            tips: [
                'Use rear camera for best quality',
                'Mount phone at hip height',
                'Use landscape orientation',
                'Ensure subject is 6-10 feet away'
            ]
        },
        femto: {
            name: 'Orbbec Femto Mega',
            icon: 'fa-video',
            constraints: null, // Uses WebSocket, not getUserMedia
            mirror: false,
            websocketUrl: getIMWCameraConfig().poseEngineUrl || 'ws://localhost:8765',
            tips: [
                'Ensure Femto bridge server is running',
                'Connect via USB-C or Ethernet',
                'Position 2-4 meters from subject',
                'Avoid direct sunlight on depth sensor'
            ]
        }
    },

    // Initialize camera manager
    async initialize() {
        console.log('📷 Camera Manager initializing...');
        await this.detectCameras();
        this.loadPreferences();
        return this.state.availableCameras.length > 0;
    },

    // Detect all available cameras
    async detectCameras() {
        try {
            // Request permission first to get labeled devices
            const tempStream = await navigator.mediaDevices.getUserMedia({ video: true });
            tempStream.getTracks().forEach(t => t.stop());

            const devices = await navigator.mediaDevices.enumerateDevices();
            const videoDevices = devices.filter(d => d.kind === 'videoinput');

            this.state.availableCameras = videoDevices.map((device, index) => {
                const label = device.label.toLowerCase();
                return {
                    deviceId: device.deviceId,
                    label: device.label || `Camera ${index + 1}`,
                    type: this.classifyCamera(label),
                    isFrontFacing: label.includes('front') || label.includes('user'),
                    isBackFacing: label.includes('back') || label.includes('rear') || label.includes('environment'),
                    isExternal: label.includes('usb') || label.includes('hdmi') || label.includes('capture'),
                    capabilities: null // Will be populated on selection
                };
            });

            console.log(`✅ Found ${this.state.availableCameras.length} camera(s):`, 
                this.state.availableCameras.map(c => c.label));
            
            return this.state.availableCameras;
        } catch (error) {
            console.error('❌ Camera detection failed:', error);
            return [];
        }
    },

    // Classify camera based on label
    classifyCamera(label) {
        if (label.includes('femto') || label.includes('orbbec')) return 'femto';
        if (label.includes('iphone') || label.includes('android') || label.includes('mobile')) return 'phone';
        if (label.includes('integrated') || label.includes('built-in') || label.includes('webcam')) return 'webcam';
        return 'unknown';
    },

    // Load saved preferences
    loadPreferences() {
        const prefs = localStorage.getItem('physio_camera_prefs');
        if (prefs) {
            const parsed = JSON.parse(prefs);
            this.state.preferredResolution = parsed.resolution || this.state.preferredResolution;
            this.state.facingMode = parsed.facingMode || this.state.facingMode;
        }
    },

    // Save preferences
    savePreferences() {
        localStorage.setItem('physio_camera_prefs', JSON.stringify({
            resolution: this.state.preferredResolution,
            facingMode: this.state.facingMode
        }));
    },

    // Start camera with optimal settings
    async startCamera(type, deviceId = null) {
        this.state.cameraType = type;
        const profile = this.cameraProfiles[type];
        
        if (!profile) {
            throw new Error(`Unknown camera type: ${type}`);
        }

        console.log(`📷 Starting ${profile.name}...`);

        // Stop any existing stream
        await this.stopCamera();

        try {
            if (type === 'femto') {
                return await this.startFemtoCamera();
            } else {
                return await this.startWebCamera(type, deviceId);
            }
        } catch (error) {
            console.error(`❌ Failed to start ${type}:`, error);
            // Try fallback
            if (type !== 'webcam' && this.state.availableCameras.length > 0) {
                console.log('🔄 Trying fallback to webcam...');
                return await this.startWebCamera('webcam');
            }
            throw error;
        }
    },

    // Start web camera (webcam or phone)
    async startWebCamera(type, deviceId = null) {
        const profile = this.cameraProfiles[type];
        let constraints = JSON.parse(JSON.stringify(profile.constraints));

        // Apply device ID if specified
        if (deviceId) {
            constraints.video.deviceId = { exact: deviceId };
            delete constraints.video.facingMode;
        }

        // Apply facing mode for phone
        if (type === 'phone' && !deviceId) {
            constraints.video.facingMode = this.state.facingMode;
        }

        // Try high quality first
        let stream;
        try {
            console.log('🎯 Trying 1080p...');
            stream = await navigator.mediaDevices.getUserMedia(constraints);
        } catch (e) {
            console.warn('⚠️ 1080p failed, trying 720p:', e.message);
            // Fallback to lower resolution
            constraints.video.width = { ideal: 1280, min: 640 };
            constraints.video.height = { ideal: 720, min: 480 };
            stream = await navigator.mediaDevices.getUserMedia(constraints);
        }

        this.state.activeStream = stream;
        this.state.isMirror = profile.mirror;

        // Get actual settings
        const track = stream.getVideoTracks()[0];
        const settings = track.getSettings();
        console.log('✅ Camera active:', {
            resolution: `${settings.width}x${settings.height}`,
            frameRate: settings.frameRate,
            facingMode: settings.facingMode,
            deviceId: settings.deviceId?.slice(0, 8) + '...'
        });

        // Monitor quality
        this.startQualityMonitoring(track);

        return {
            stream,
            settings,
            mirrored: this.state.isMirror,
            type
        };
    },

    // Start Femto Mega camera
    async startFemtoCamera() {
        const url = localStorage.getItem('femto_bridge_url') ||
                    getIMWCameraConfig().poseEngineUrl ||
                    this.cameraProfiles.femto.websocketUrl;

        if (typeof FemtoMegaClient === 'undefined') {
            throw new Error('FemtoMegaClient not loaded. Include femto-client.js');
        }

        const client = new FemtoMegaClient(url);
        await client.connect();

        client.onDepthFrame = (frame) => {
            this.state.depthFrame = frame;
            this.state.femtoConnected = true;
            if (typeof this.state.onDepthFrame === 'function') {
                this.state.onDepthFrame(frame);
            }
        };

        client.onSkeletonData = (frame) => {
            this.state.skeletonFrame = frame;
            this.state.femtoConnected = true;
            if (typeof this.state.onSkeletonFrame === 'function') {
                this.state.onSkeletonFrame(frame);
            }
        };

        this.state.activeCamera = {
            type: 'femto',
            client,
            isWebSocket: true
        };
        this.state.femtoConnected = true;

        console.log('✅ Femto Mega connected via', url);

        return {
            type: 'femto',
            client,
            settings: {
                resolution: '1280x720 (Depth)',
                frameRate: 30,
                type: 'Orbbec Femto Mega'
            }
        };
    },

    // Stop active camera
    async stopCamera() {
        if (this.state.activeStream) {
            this.state.activeStream.getTracks().forEach(t => t.stop());
            this.state.activeStream = null;
        }

        if (this.state.activeCamera?.type === 'femto') {
            this.state.activeCamera.client.disconnect();
        }

        this.state.activeCamera = null;
        this.state.depthFrame = null;
        this.state.skeletonFrame = null;
        this.state.femtoConnected = false;
        this.stopQualityMonitoring();
        console.log('📷 Camera stopped');
    },

    // Flip camera (for mobile)
    async flipCamera() {
        if (this.state.cameraType !== 'phone') {
            console.log('⚠️ Flip only supported on mobile');
            return null;
        }

        this.state.facingMode = this.state.facingMode === 'user' ? 'environment' : 'user';
        this.savePreferences();

        // Restart with new facing mode
        return await this.startCamera('phone');
    },

    // Cycle through available cameras
    async cycleCamera() {
        if (this.state.availableCameras.length <= 1) {
            console.log('⚠️ Only one camera available');
            return null;
        }

        const currentIndex = this.state.availableCameras.findIndex(
            c => c.deviceId === this.state.activeCamera?.deviceId
        );
        const nextIndex = (currentIndex + 1) % this.state.availableCameras.length;
        const nextCamera = this.state.availableCameras[nextIndex];

        console.log(`🔄 Switching to: ${nextCamera.label}`);
        return await this.startWebCamera(this.state.cameraType, nextCamera.deviceId);
    },

    // Monitor camera quality
    startQualityMonitoring(track) {
        this.stopQualityMonitoring();
        
        this.qualityInterval = setInterval(async () => {
            try {
                const capabilities = track.getCapabilities();
                const settings = track.getSettings();
                
                // Calculate quality score
                const resolutionScore = (settings.width * settings.height) / (1920 * 1080);
                const frameRateScore = settings.frameRate / 30;
                this.state.qualityScore = Math.min(100, Math.round((resolutionScore + frameRateScore) * 50));

                // Check lighting (if brightness is available)
                if ('brightness' in settings) {
                    const brightness = settings.brightness;
                    this.state.lightingLevel = brightness < 50 ? 'poor' : brightness < 100 ? 'low' : 'good';
                }

            } catch (e) {}
        }, 2000);
    },

    stopQualityMonitoring() {
        if (this.qualityInterval) {
            clearInterval(this.qualityInterval);
            this.qualityInterval = null;
        }
    },

    // Get current quality info
    getQualityInfo() {
        return {
            score: this.state.qualityScore,
            lighting: this.state.lightingLevel,
            depth: this.state.depthFrame ? {
                timestamp: this.state.depthFrame.timestamp || Date.now(),
                width: this.state.depthFrame.width,
                height: this.state.depthFrame.height,
                format: this.state.depthFrame.format || 'unknown'
            } : null,
            resolution: this.state.activeStream ? 
                this.state.activeStream.getVideoTracks()[0]?.getSettings() : null
        };
    },

    // Latest Femto Mega depth payload, forwarded without reshaping protocol fields
    getDepthFrame() {
        return this.state.depthFrame;
    },

    getSkeletonFrame() {
        return this.state.skeletonFrame;
    },

    onDepthFrame(callback) {
        this.state.onDepthFrame = callback;
    },

    onSkeletonFrame(callback) {
        this.state.onSkeletonFrame = callback;
    },

    // Check if mobile device
    isMobile() {
        return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    },

    // Get recommended camera for device
    getRecommendedCamera() {
        if (this.isMobile()) return 'phone';
        if (this.state.availableCameras.some(c => c.type === 'femto')) return 'femto';
        return 'webcam';
    },

    // Get camera tips
    getTips(type) {
        return this.cameraProfiles[type]?.tips || [];
    }
};

// ============================================================================
// High-Quality Pose Detection Engine
// Uses MediaPipe with maximum accuracy settings
// ============================================================================

const PoseEngine = {
    pose: null,
    camera: null,
    isInitialized: false,
    config: {
        modelComplexity: 2, // 0=lite, 1=full, 2=heavy (best accuracy)
        smoothLandmarks: true,
        enableSegmentation: false,
        smoothSegmentation: false,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
    },

    // Initialize with highest quality
    async initialize(videoElement, canvasElement, onResults) {
        console.log('🦴 Initializing High-Quality Pose Engine...');

        // Load MediaPipe scripts if needed
        await this.loadScripts();

        // Create pose detector
        this.pose = new Pose({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
        });

        this.pose.setOptions(this.config);

        // Set up results callback with performance tracking
        let lastFrameTime = performance.now();
        let frameCount = 0;

        this.pose.onResults((results) => {
            const now = performance.now();
            const fps = 1000 / (now - lastFrameTime);
            lastFrameTime = now;
            frameCount++;

            // Log FPS every 100 frames
            if (frameCount % 100 === 0) {
                console.log(`🎬 Tracking FPS: ${fps.toFixed(1)}`);
            }

            onResults(results, fps);
        });

        // Create camera with optimal settings
        if (videoElement && videoElement.srcObject) {
            this.camera = new Camera(videoElement, {
                onFrame: async () => {
                    await this.pose.send({ image: videoElement });
                },
                width: 1920,
                height: 1080
            });
        }

        this.isInitialized = true;
        console.log('✅ High-Quality Pose Engine ready');
        return true;
    },

    // Load required scripts
    async loadScripts() {
        const scripts = [
            'https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js',
            'https://cdn.jsdelivr.net/npm/@mediapipe/control_utils/control_utils.js',
            'https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils/drawing_utils.js',
            'https://cdn.jsdelivr.net/npm/@mediapipe/pose/pose.js'
        ];

        for (const src of scripts) {
            if (!document.querySelector(`script[src="${src}"]`)) {
                await new Promise((resolve, reject) => {
                    const script = document.createElement('script');
                    script.src = src;
                    script.crossOrigin = 'anonymous';
                    script.onload = resolve;
                    script.onerror = reject;
                    document.head.appendChild(script);
                });
            }
        }
    },

    // Start tracking
    start() {
        if (this.camera) {
            this.camera.start();
            console.log('🎬 Pose tracking started');
        }
    },

    // Stop tracking
    stop() {
        if (this.camera) {
            // Camera doesn't have a direct stop method in MediaPipe utils
            // We control this via the video element
        }
    },

    // Get landmark quality score
    getLandmarkQuality(landmarks) {
        if (!landmarks || landmarks.length === 0) return 0;

        const avgVisibility = landmarks.reduce((sum, lm) => sum + (lm.visibility || 0), 0) / landmarks.length;
        const keyJoints = [11, 12, 23, 24, 25, 26, 27, 28]; // Shoulders, hips, knees, ankles
        const keyJointVisibility = keyJoints.reduce((sum, idx) => sum + (landmarks[idx]?.visibility || 0), 0) / keyJoints.length;

        return {
            overall: Math.round(avgVisibility * 100),
            keyJoints: Math.round(keyJointVisibility * 100),
            isGood: keyJointVisibility > 0.7
        };
    }
};

// ============================================================================
// Skeleton Renderer - High-Quality Visualization
// ============================================================================

const SkeletonRenderer = {
    // Draw skeleton with quality indicators
    draw(ctx, canvas, landmarks, options = {}) {
        const { 
            showConnections = true, 
            showJoints = true,
            highlightErrors = false,
            ghostLandmarks = null,
            quality = null
        } = options;

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw ghost skeleton first (if enabled)
        if (ghostLandmarks) {
            this.drawGhost(ctx, canvas, ghostLandmarks);
        }

        if (!landmarks) return;

        const width = canvas.width;
        const height = canvas.height;

        // Define connections
        const connections = window.POSE_CONNECTIONS || [
            [11, 12], // shoulders
            [11, 13], [13, 15], // left arm
            [12, 14], [14, 16], // right arm
            [11, 23], [12, 24], // spine
            [23, 24], // hips
            [23, 25], [25, 27], // left leg
            [24, 26], [26, 28], // right leg
        ];

        // Draw connections
        if (showConnections) {
            connections.forEach(([start, end]) => {
                const startPoint = landmarks[start];
                const endPoint = landmarks[end];

                if (startPoint?.visibility > 0.5 && endPoint?.visibility > 0.5) {
                    const confidence = Math.min(startPoint.visibility, endPoint.visibility);
                    
                    ctx.beginPath();
                    ctx.moveTo(startPoint.x * width, startPoint.y * height);
                    ctx.lineTo(endPoint.x * width, endPoint.y * height);
                    
                    // Color based on confidence
                    ctx.strokeStyle = confidence > 0.8 ? '#00ff00' : 
                                      confidence > 0.6 ? '#ffff00' : '#ff6600';
                    ctx.lineWidth = confidence > 0.8 ? 4 : 3;
                    ctx.lineCap = 'round';
                    ctx.stroke();
                }
            });
        }

        // Draw joints
        if (showJoints) {
            landmarks.forEach((landmark, index) => {
                if (landmark.visibility > 0.5) {
                    const x = landmark.x * width;
                    const y = landmark.y * height;

                    // Joint circle
                    ctx.beginPath();
                    ctx.arc(x, y, landmark.visibility > 0.8 ? 6 : 4, 0, 2 * Math.PI);
                    ctx.fillStyle = landmark.visibility > 0.8 ? '#00ff00' : '#ffff00';
                    ctx.fill();

                    // Highlight key joints
                    if ([11, 12, 23, 24, 25, 26, 27, 28].includes(index)) {
                        ctx.beginPath();
                        ctx.arc(x, y, 8, 0, 2 * Math.PI);
                        ctx.strokeStyle = 'rgba(0, 255, 0, 0.3)';
                        ctx.lineWidth = 2;
                        ctx.stroke();
                    }
                }
            });
        }

        // Draw quality indicator
        if (quality) {
            this.drawQualityIndicator(ctx, quality);
        }
    },

    // Draw ghost skeleton (reference frame)
    drawGhost(ctx, canvas, landmarks) {
        const width = canvas.width;
        const height = canvas.height;

        ctx.save();
        ctx.globalAlpha = 0.25;
        ctx.strokeStyle = '#00ffff';
        ctx.fillStyle = '#00ffff';
        ctx.lineWidth = 3;

        const connections = window.POSE_CONNECTIONS || [
            [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
            [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [24, 26], [26, 28]
        ];

        connections.forEach(([start, end]) => {
            const startPoint = landmarks[start];
            const endPoint = landmarks[end];

            if (startPoint?.visibility > 0.5 && endPoint?.visibility > 0.5) {
                ctx.beginPath();
                ctx.moveTo(startPoint.x * width, startPoint.y * height);
                ctx.lineTo(endPoint.x * width, endPoint.y * height);
                ctx.stroke();
            }
        });

        ctx.restore();
    },

    // Draw quality indicator overlay
    drawQualityIndicator(ctx, quality) {
        ctx.save();
        ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
        ctx.fillRect(10, 10, 120, 60);

        ctx.fillStyle = quality.overall > 80 ? '#00ff00' : 
                        quality.overall > 60 ? '#ffff00' : '#ff0000';
        ctx.font = 'bold 16px Arial';
        ctx.fillText(`Quality: ${quality.overall}%`, 20, 35);

        ctx.fillStyle = '#ffffff';
        ctx.font = '12px Arial';
        ctx.fillText(quality.isGood ? 'Good tracking' : 'Poor visibility', 20, 55);

        ctx.restore();
    }
};

// ============================================================================
// Joint Angle Calculator - Biomechanical Analysis
// ============================================================================

const Biomechanics = {
    // Calculate angle between three points (in 3D)
    calculateAngle(a, b, c) {
        const ba = { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
        const bc = { x: c.x - b.x, y: c.y - b.y, z: c.z - b.z };

        const dot = ba.x * bc.x + ba.y * bc.y + ba.z * bc.z;
        const magBA = Math.sqrt(ba.x**2 + ba.y**2 + ba.z**2);
        const magBC = Math.sqrt(bc.x**2 + bc.y**2 + bc.z**2);

        const cosAngle = dot / (magBA * magBC);
        const angle = Math.acos(Math.max(-1, Math.min(1, cosAngle))) * (180 / Math.PI);

        return Math.round(angle);
    },

    // Define joint configurations for analysis
    jointDefinitions: [
        { name: 'Left Elbow', points: [11, 13, 15], min: 130, max: 180 },
        { name: 'Right Elbow', points: [12, 14, 16], min: 130, max: 180 },
        { name: 'Left Shoulder', points: [23, 11, 13], min: 90, max: 180 },
        { name: 'Right Shoulder', points: [24, 12, 14], min: 90, max: 180 },
        { name: 'Left Hip', points: [11, 23, 25], min: 80, max: 120 },
        { name: 'Right Hip', points: [12, 24, 26], min: 80, max: 120 },
        { name: 'Left Knee', points: [23, 25, 27], min: 120, max: 180 },
        { name: 'Right Knee', points: [24, 26, 28], min: 120, max: 180 },
        { name: 'Left Ankle', points: [25, 27, 31], min: 80, max: 110 },
        { name: 'Right Ankle', points: [26, 28, 32], min: 80, max: 110 },
    ],

    // Calculate all joint angles
    calculateAllAngles(landmarks) {
        const angles = {};

        this.jointDefinitions.forEach(joint => {
            const [a, b, c] = joint.points.map(idx => landmarks[idx]);
            
            if (a?.visibility > 0.5 && b?.visibility > 0.5 && c?.visibility > 0.5) {
                const angle = this.calculateAngle(a, b, c);
                
                let status = 'normal';
                if (angle < joint.min) status = 'limited';
                if (angle > joint.max) status = 'excessive';

                angles[joint.name] = {
                    angle,
                    status,
                    min: joint.min,
                    max: joint.max,
                    confidence: Math.min(a.visibility, b.visibility, c.visibility)
                };
            }
        });

        return angles;
    },

    // Detect movement phase (for rep counting)
    detectPhase(angles, previousPhase) {
        // Simple squat phase detection
        const leftKnee = angles['Left Knee']?.angle;
        const rightKnee = angles['Right Knee']?.angle;
        
        if (!leftKnee && !rightKnee) return previousPhase;

        const avgKnee = (leftKnee || rightKnee) || 0;

        if (previousPhase === 'up' && avgKnee < 100) {
            return 'down';
        }
        if (previousPhase === 'down' && avgKnee > 150) {
            return 'up';
        }
        return previousPhase;
    }
};

// Export for global access
window.CameraManager = CameraManager;
window.PoseEngine = PoseEngine;
window.SkeletonRenderer = SkeletonRenderer;
window.Biomechanics = Biomechanics;

console.log('✅ PhysioMotion High-Quality Camera System loaded');
