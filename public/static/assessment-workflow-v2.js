// PhysioMotion - Enhanced Assessment Workflow v2
// High-Quality Multi-Camera Pose Assessment
// Integrates: CameraManager, PoseEngine, SkeletonRenderer, Biomechanics

// ============================================================================
// GLOBAL STATE
// ============================================================================

const ASSESSMENT_STATE = {
    patientId: null,
    assessmentId: null,
    testId: null,
    currentTest: null,
    
    // Camera & Tracking
    cameraActive: false,
    cameraType: null,
    isRecording: false,
    recordingStartTime: null,
    
    // Data
    skeletonFrames: [],
    videoBlob: null,
    repCount: 0,
    repPhase: 'up',
    ghostSkeleton: null,
    
    // Features
    features: {
        ghostMode: false,
        voiceFeedback: false,
        repCounter: false,
        qualityOverlay: true
    },
    
    // Chart
    chart: null,
    maxChartPoints: 150,

    // Calibration
    calibrationWizard: null,
    isCalibrating: false,
    calibrationProfile: null
};

// ============================================================================
// INITIALIZATION
// ============================================================================

document.addEventListener('DOMContentLoaded', async () => {
    console.log('🚀 PhysioMotion Assessment v2 initializing...');
    
    // Check auth
    if (typeof checkAuth === 'function') checkAuth();
    
    // Initialize camera manager
    await CameraManager.initialize();
    
    // Load patient & create assessment
    await initializeAssessmentFlow();
    
    console.log('✅ Assessment workflow ready');
});

async function initializeAssessmentFlow() {
    const urlParams = new URLSearchParams(window.location.search);
    const patientId = urlParams.get('patient_id');

    if (patientId) {
        ASSESSMENT_STATE.patientId = patientId;
        await createAssessment();
    } else {
        await showPatientSelector();
    }
}

async function showPatientSelector() {
    try {
        const response = await fetch('/api/patients');
        const result = await response.json();

        if (result.success && result.data.length > 0) {
            ASSESSMENT_STATE.patientId = result.data[0].id;
            await createAssessment();
        } else {
            showNotification('No patients found. Redirecting to intake...', 'warning');
            setTimeout(() => window.location.href = '/intake', 2000);
        }
    } catch (error) {
        console.error('Error:', error);
        showNotification('Error loading patients', 'error');
    }
}

async function createAssessment() {
    try {
        const response = await fetch('/api/assessments', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                patient_id: ASSESSMENT_STATE.patientId,
                assessment_type: 'movement_analysis',
                clinician_id: 1
            })
        });

        const result = await response.json();
        if (result.success) {
            ASSESSMENT_STATE.assessmentId = result.data.id;
            await createMovementTest();
        }
    } catch (error) {
        showNotification('Failed to create assessment', 'error');
    }
}

async function createMovementTest() {
    try {
        const response = await fetch(`/api/assessments/${ASSESSMENT_STATE.assessmentId}/tests`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                test_name: 'Comprehensive Movement Screen',
                test_category: 'functional',
                test_order: 1,
                instructions: 'Stand 6-10 feet from camera. Perform movements as instructed.',
                test_status: 'pending'
            })
        });

        const result = await response.json();
        if (result.success) {
            ASSESSMENT_STATE.testId = result.data.id;
            console.log('✅ Assessment ready:', ASSESSMENT_STATE.assessmentId);
        }
    } catch (error) {
        showNotification('Failed to create test', 'error');
    }
}

// ============================================================================
// CAMERA SELECTION & START
// ============================================================================

function selectCameraType(type) {
    // Update UI
    document.querySelectorAll('.camera-option').forEach(opt => {
        opt.classList.remove('selected', 'ring-2', 'ring-cyan-500');
    });
    event.target.closest('.camera-option').classList.add('selected', 'ring-2', 'ring-cyan-500');

    ASSESSMENT_STATE.cameraType = type;
    document.getElementById('startBtn').disabled = false;

    // Show calibration option
    const calOpt = document.getElementById('calibrationOption');
    if (calOpt) calOpt.style.display = 'block';
    
    // Show tips for this camera type
    showCameraTips(type);
}

function showCameraTips(type) {
    const tips = CameraManager.getTips(type);
    const tipsContainer = document.getElementById('cameraTips');
    if (tipsContainer && tips.length > 0) {
        tipsContainer.innerHTML = `
            <div class="mt-4 p-4 bg-blue-50 rounded-lg">
                <h4 class="font-semibold text-blue-800 mb-2"><i class="fas fa-lightbulb mr-2"></i>Tips for best results:</h4>
                <ul class="text-sm text-blue-700 space-y-1">
                    ${tips.map(t => `<li>• ${t}</li>`).join('')}
                </ul>
            </div>
        `;
    }
}

async function startAssessment() {
    const type = ASSESSMENT_STATE.cameraType;
    if (!type) return;

    // Hide modal, show camera
    document.getElementById('cameraSelectionModal').style.display = 'none';
    document.getElementById('cameraContainer').style.display = 'block';
    document.getElementById('progressSteps').style.display = 'none';

    // Initialize chart
    initializeChart();

    try {
        // Start camera with quality settings
        showStatus('Initializing camera...', 'warning');
        const cameraResult = await CameraManager.startCamera(type);
        
        // Set up video element
        const video = document.getElementById('videoElement');
        const canvas = document.getElementById('canvasElement');

        if (type === 'femto') {
            // Femto uses WebSocket, different setup
            await setupFemtoCamera(cameraResult, canvas);
        } else {
            // Web camera setup
            video.srcObject = cameraResult.stream;
            video.style.transform = cameraResult.mirrored ? 'scaleX(-1)' : 'none';
            
            await video.play();
            
            // Match canvas to video
            canvas.width = video.videoWidth || 1280;
            canvas.height = video.videoHeight || 720;
            
            // Initialize high-quality pose tracking
            await initializePoseTracking(video, canvas);
        }

        // Show flip button for mobile
        if (type === 'phone') {
            document.getElementById('flipBtn').style.display = 'flex';
        }

        ASSESSMENT_STATE.cameraActive = true;
        showStatus('Camera active - Ready to record', 'success');

        // Start calibration wizard if requested
        const calibrateChecked = document.getElementById('calibrateCheckbox');
        if (calibrateChecked && calibrateChecked.checked && type !== 'femto') {
            startCalibrationFlowV2();
        }

    } catch (error) {
        console.error('Camera error:', error);
        showStatus('Camera failed: ' + error.message, 'error');
        showNotification('Camera initialization failed. Please check permissions.', 'error');
    }
}

// ============================================================================
// POSE TRACKING SETUP
// ============================================================================

async function initializePoseTracking(video, canvas) {
    const ctx = canvas.getContext('2d');

    await PoseEngine.initialize(video, canvas, (results, fps) => {
        onPoseResults(results, ctx, canvas, fps);
    });

    PoseEngine.start();
}

function onPoseResults(results, ctx, canvas, fps) {
    if (!results.poseLandmarks) {
        // No person detected
        SkeletonRenderer.draw(ctx, canvas, null);
        updateStatusText('No person detected - please step into frame');
        return;
    }

    const landmarks = results.poseLandmarks;
    
    // Feed calibration wizard if active
    if (ASSESSMENT_STATE.isCalibrating && ASSESSMENT_STATE.calibrationWizard) {
        ASSESSMENT_STATE.calibrationWizard.update(landmarks);
        return; // Wizard handles its own rendering
    }
    
    // Check tracking quality
    const quality = PoseEngine.getLandmarkQuality(landmarks);
    
    // Calculate biomechanics
    const angles = Biomechanics.calculateAllAngles(landmarks);
    
    // Draw skeleton with quality indicators
    SkeletonRenderer.draw(ctx, canvas, landmarks, {
        ghostLandmarks: ASSESSMENT_STATE.features.ghostMode ? ASSESSMENT_STATE.ghostSkeleton : null,
        quality: ASSESSMENT_STATE.features.qualityOverlay ? quality : null
    });

    // Update UI
    updateDashboard(angles, quality);
    
    // Rep counter
    if (ASSESSMENT_STATE.features.repCounter) {
        updateRepCounter(angles);
    }
    
    // Voice feedback
    if (ASSESSMENT_STATE.features.voiceFeedback && quality.isGood) {
        provideVoiceCoaching(angles);
    }
    
    // Record data if active
    if (ASSESSMENT_STATE.isRecording) {
        recordFrame(landmarks, angles);
    }
    
    // Update status
    const statusText = quality.isGood ? 'Good tracking' : 'Adjust position for better visibility';
    updateStatusText(statusText);
}

async function setupFemtoCamera(cameraResult, canvas) {
    const ctx = canvas.getContext('2d');
    canvas.width = 1280;
    canvas.height = 720;

    cameraResult.client.onSkeletonData = (skeletonData) => {
        // Convert Femto format to MediaPipe format
        const landmarks = convertFemtoToMediaPipe(skeletonData);
        
        const quality = { overall: 95, keyJoints: 95, isGood: true };
        const angles = Biomechanics.calculateAllAngles(landmarks);
        
        SkeletonRenderer.draw(ctx, canvas, landmarks, {
            ghostLandmarks: ASSESSMENT_STATE.features.ghostMode ? ASSESSMENT_STATE.ghostSkeleton : null,
            quality: ASSESSMENT_STATE.features.qualityOverlay ? quality : null
        });

        updateDashboard(angles, quality);
        
        if (ASSESSMENT_STATE.isRecording) {
            recordFrame(landmarks, angles);
        }
    };

    showStatus('Femto Mega active', 'success');
}

function convertFemtoToMediaPipe(femtoData) {
    // Map Femto joints to MediaPipe format
    const mapping = {
        'PELVIS': 0, 'SPINE_NAVEL': 1, 'SPINE_CHEST': 2, 'NECK': 3,
        'CLAVICLE_LEFT': 11, 'SHOULDER_LEFT': 13, 'ELBOW_LEFT': 15, 'WRIST_LEFT': 17,
        'CLAVICLE_RIGHT': 12, 'SHOULDER_RIGHT': 14, 'ELBOW_RIGHT': 16, 'WRIST_RIGHT': 18,
        'HIP_LEFT': 23, 'KNEE_LEFT': 25, 'ANKLE_LEFT': 27, 'FOOT_LEFT': 31,
        'HIP_RIGHT': 24, 'KNEE_RIGHT': 26, 'ANKLE_RIGHT': 28, 'FOOT_RIGHT': 32
    };

    const landmarks = new Array(33).fill(null).map(() => ({ x: 0, y: 0, z: 0, visibility: 0 }));
    
    if (femtoData.joints) {
        for (const [name, joint] of Object.entries(femtoData.joints)) {
            const idx = mapping[name];
            if (idx !== undefined) {
                landmarks[idx] = {
                    x: joint.x / 1280, // Normalize to 0-1
                    y: joint.y / 720,
                    z: joint.z || 0,
                    visibility: joint.confidence || 0.9
                };
            }
        }
    }

    return landmarks;
}

// ============================================================================
// RECORDING & DATA CAPTURE
// ============================================================================

function recordFrame(landmarks, angles) {
    const frame = {
        timestamp: Date.now(),
        landmarks: landmarks.map(l => ({
            x: l.x, y: l.y, z: l.z, visibility: l.visibility
        })),
        angles: angles
    };
    
    ASSESSMENT_STATE.skeletonFrames.push(frame);
}

async function startRecording() {
    ASSESSMENT_STATE.isRecording = true;
    ASSESSMENT_STATE.recordingStartTime = Date.now();
    ASSESSMENT_STATE.skeletonFrames = [];
    
    // Initialize video recorder
    const video = document.getElementById('videoElement');
    if (video && video.srcObject && typeof VideoRecorder !== 'undefined') {
        await VideoRecorder.initialize(video);
        VideoRecorder.startRecording();
    }
    
    // Update UI
    document.getElementById('recordingIndicator').style.display = 'flex';
    document.getElementById('recordBtn').style.display = 'none';
    document.getElementById('stopBtn').style.display = 'flex';
    
    // Start timer
    startRecordingTimer();
    
    // Set ghost reference if enabled
    if (ASSESSMENT_STATE.features.ghostMode && !ASSESSMENT_STATE.ghostSkeleton) {
        // Will be set on first good frame
    }
    
    showNotification('Recording started', 'success');
}

function stopRecording() {
    ASSESSMENT_STATE.isRecording = false;
    
    // Stop video recorder
    if (typeof VideoRecorder !== 'undefined') {
        VideoRecorder.stopRecording();
    }
    
    // Stop timer
    stopRecordingTimer();
    
    // Update UI
    document.getElementById('recordingIndicator').style.display = 'none';
    document.getElementById('stopBtn').style.display = 'none';
    document.getElementById('analyzeBtn').style.display = 'flex';
    
    const frameCount = ASSESSMENT_STATE.skeletonFrames.length;
    const duration = Math.floor((Date.now() - ASSESSMENT_STATE.recordingStartTime) / 1000);
    
    showNotification(`Captured ${frameCount} frames in ${duration}s`, 'success');
}

function startRecordingTimer() {
    const timerEl = document.getElementById('recordingTime');
    
    ASSESSMENT_STATE.recordingTimer = setInterval(() => {
        const elapsed = Math.floor((Date.now() - ASSESSMENT_STATE.recordingStartTime) / 1000);
        const mins = Math.floor(elapsed / 60).toString().padStart(2, '0');
        const secs = (elapsed % 60).toString().padStart(2, '0');
        timerEl.textContent = `${mins}:${secs}`;
    }, 1000);
}

function stopRecordingTimer() {
    if (ASSESSMENT_STATE.recordingTimer) {
        clearInterval(ASSESSMENT_STATE.recordingTimer);
        ASSESSMENT_STATE.recordingTimer = null;
    }
}

// ============================================================================
// FEATURES
// ============================================================================

function toggleFeature(feature) {
    ASSESSMENT_STATE.features[feature] = !ASSESSMENT_STATE.features[feature];
    
    const btn = document.getElementById(`btn-${feature}`);
    const isActive = ASSESSMENT_STATE.features[feature];
    
    if (isActive) {
        btn.classList.add('bg-cyan-600', 'ring-2', 'ring-cyan-300');
        btn.classList.remove('bg-gray-900', 'opacity-70');
        
        if (feature === 'ghostMode') {
            // Capture current frame as ghost
            const currentFrame = ASSESSMENT_STATE.skeletonFrames[ASSESSMENT_STATE.skeletonFrames.length - 1];
            if (currentFrame) {
                ASSESSMENT_STATE.ghostSkeleton = currentFrame.landmarks;
                showNotification('Ghost reference captured', 'success');
            }
        }
        
        if (feature === 'voiceFeedback') {
            VoiceFeedback.speak('Voice coaching enabled');
        }
    } else {
        btn.classList.remove('bg-cyan-600', 'ring-2', 'ring-cyan-300');
        btn.classList.add('bg-gray-900', 'opacity-70');
    }
}

async function flipCamera() {
    showStatus('Switching camera...', 'warning');
    await CameraManager.flipCamera();
    showStatus('Camera switched', 'success');
}

// ============================================================================
// DASHBOARD & UI UPDATES
// ============================================================================

function initializeChart() {
    const canvas = document.getElementById('jointChart');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');

    ASSESSMENT_STATE.chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Left Knee',
                    borderColor: '#06b6d4',
                    backgroundColor: 'rgba(6, 182, 212, 0.1)',
                    data: [],
                    tension: 0.4,
                    pointRadius: 0
                },
                {
                    label: 'Right Knee',
                    borderColor: '#8b5cf6',
                    backgroundColor: 'rgba(139, 92, 246, 0.1)',
                    data: [],
                    tension: 0.4,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: {
                legend: { position: 'top', labels: { usePointStyle: true, boxWidth: 8 } }
            },
            scales: {
                x: { display: false },
                y: { min: 0, max: 180, ticks: { stepSize: 30 } }
            }
        }
    });
}

function updateDashboard(angles, quality) {
    // Update chart
    if (ASSESSMENT_STATE.chart) {
        const chart = ASSESSMENT_STATE.chart;
        const now = new Date().toLocaleTimeString();
        
        chart.data.labels.push(now);
        chart.data.datasets[0].data.push(angles['Left Knee']?.angle || null);
        chart.data.datasets[1].data.push(angles['Right Knee']?.angle || null);
        
        if (chart.data.labels.length > ASSESSMENT_STATE.maxChartPoints) {
            chart.data.labels.shift();
            chart.data.datasets.forEach(d => d.data.shift());
        }
        
        chart.update('none');
    }
    
    // Update joint table
    const tbody = document.getElementById('jointTableBody');
    if (tbody) {
        tbody.innerHTML = Object.entries(angles)
            .filter(([name, data]) => data.confidence > 0.5)
            .map(([name, data]) => {
                const statusClass = data.status === 'normal' ? 'status-normal' :
                                   data.status === 'limited' ? 'status-limited' : 'status-excessive';
                return `
                    <tr>
                        <td class="font-medium text-gray-700">${name}</td>
                        <td class="font-bold text-gray-900">${data.angle}°</td>
                        <td><span class="status-badge ${statusClass}">${data.status.toUpperCase()}</span></td>
                    </tr>
                `;
            }).join('');
    }
}

function updateRepCounter(angles) {
    const newPhase = Biomechanics.detectPhase(angles, ASSESSMENT_STATE.repPhase);
    
    if (ASSESSMENT_STATE.repPhase === 'down' && newPhase === 'up') {
        ASSESSMENT_STATE.repCount++;
        VoiceFeedback.speak(ASSESSMENT_STATE.repCount.toString());
    }
    
    ASSESSMENT_STATE.repPhase = newPhase;
    
    // Draw rep counter overlay
    const canvas = document.getElementById('canvasElement');
    const ctx = canvas.getContext('2d');
    
    ctx.save();
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.roundRect(20, 20, 100, 80, 10);
    ctx.fill();
    
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 36px Arial';
    ctx.fillText(ASSESSMENT_STATE.repCount, 50, 70);
    
    ctx.font = '12px Arial';
    ctx.fillStyle = '#cccccc';
    ctx.fillText('REPS', 50, 40);
    ctx.restore();
}

function provideVoiceCoaching(angles) {
    // Simple coaching based on knee angles during squat
    const leftKnee = angles['Left Knee']?.angle;
    
    if (leftKnee && leftKnee < 90 && ASSESSMENT_STATE.repPhase === 'down') {
        VoiceFeedback.speak('Good depth');
    }
}

// ============================================================================
// ANALYSIS & RESULTS
// ============================================================================

async function analyzeMovement() {
    if (ASSESSMENT_STATE.skeletonFrames.length === 0) {
        showNotification('No data captured', 'error');
        return;
    }

    showNotification('Analyzing movement...', 'info');
    
    // Upload video if available
    let videoData = null;
    if (typeof VideoRecorder !== 'undefined' && VideoRecorder.getStats().size > 0) {
        videoData = await VideoRecorder.uploadVideo(
            ASSESSMENT_STATE.patientId,
            ASSESSMENT_STATE.assessmentId,
            {
                title: `Assessment - ${new Date().toLocaleString()}`,
                description: `Multi-camera movement assessment`,
                videoType: 'assessment'
            }
        );
    }
    
    // Prepare analysis data
    const midIndex = Math.floor(ASSESSMENT_STATE.skeletonFrames.length / 2);
    const representativeFrame = ASSESSMENT_STATE.skeletonFrames[midIndex];
    
    try {
        const response = await fetch(`/api/tests/${ASSESSMENT_STATE.testId}/analyze`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                skeleton_data: representativeFrame,
                video_id: videoData?.id || null,
                frame_count: ASSESSMENT_STATE.skeletonFrames.length
            })
        });

        const result = await response.json();
        
        if (result.success) {
            showResults(result.data);
        } else {
            showNotification('Analysis failed', 'error');
        }
    } catch (error) {
        console.error('Analysis error:', error);
        showNotification('Analysis failed', 'error');
    }
}

function showResults(data) {
    document.getElementById('cameraContainer').style.display = 'none';
    document.getElementById('resultsContainer').style.display = 'block';
    
    // Update score
    document.getElementById('qualityScore').textContent = Math.round(data.movement_quality_score || 75);
    
    // Show deficiencies
    const defList = document.getElementById('deficienciesList');
    if (defList && data.deficiencies) {
        defList.innerHTML = data.deficiencies.map(d => `
            <div class="p-4 bg-yellow-50 border-l-4 border-yellow-500 rounded">
                <strong>${d.area}</strong> (${d.severity}): ${d.description}
            </div>
        `).join('');
    }
    
    // Show exercises
    const exList = document.getElementById('exercisesList');
    if (exList && data.recommendations) {
        exList.innerHTML = data.recommendations.map(r => `
            <div class="p-4 bg-green-50 border-l-4 border-green-500 rounded">
                ${r}
            </div>
        `).join('');
    }
}

// ============================================================================
// CALIBRATION WIZARD (v2 flow)
// ============================================================================

function startCalibrationFlowV2() {
    console.log('[AssessmentV2] Starting calibration wizard...');
    ASSESSMENT_STATE.isCalibrating = true;

    ASSESSMENT_STATE.calibrationWizard = new CalibrationWizard({
        canvasId: 'canvasElement',
        statusEl: 'calibrationStatusV2',
        wsUrl: window.IMW_CONFIG?.poseEngineUrl || 'wss://pablodd1--pose-engine-ws-serve.modal.run/ws',
        framesRequired: 30,
        onStepChange: (stepIndex, step) => {
            const statusEl = document.getElementById('calibrationStatusV2');
            if (statusEl) {
                statusEl.style.display = 'block';
                statusEl.textContent = '⚙️ ' + step.title + ': ' + step.message;
            }
        },
        onComplete: (profile) => {
            onCalibrationCompleteV2(profile);
        }
    });

    ASSESSMENT_STATE.calibrationWizard.start();

    const statusEl = document.getElementById('calibrationStatusV2');
    if (statusEl) {
        statusEl.style.display = 'block';
        statusEl.textContent = '⚙️ Calibration starting...';
    }
    showStatus('Calibrating...', 'warning');
}

function onCalibrationCompleteV2(profile) {
    console.log('[AssessmentV2] Calibration complete:', profile);
    ASSESSMENT_STATE.isCalibrating = false;
    ASSESSMENT_STATE.calibrationProfile = profile;

    renderCalibrationResultsV2(profile);

    showStatus('Calibrated - Ready', 'success');

    if (ASSESSMENT_STATE.calibrationWizard) {
        ASSESSMENT_STATE.calibrationWizard.stop();
    }

    // Clear canvas for normal rendering
    const canvas = document.getElementById('canvasElement');
    if (canvas) {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
}

function renderCalibrationResultsV2(profile) {
    const el = document.getElementById('calibrationResultsV2');
    if (!el) return;
    el.style.display = 'block';

    const score = Math.round(profile.overallScore * 100);
    const color = score >= 80 ? '#059669' : score >= 60 ? '#d97706' : '#dc2626';

    el.innerHTML = '<div style="padding:8px;background:#ecfdf5;border-radius:6px;border:1px solid #a7f3d0">' +
        '<div style="font-size:0.8rem;font-weight:700;color:' + color + ';margin-bottom:4px">✅ Calibrated — ' + score + '%</div>' +
        '<div style="font-size:0.7rem;color:#6b7280">' +
        profile.steps.map(function(s) {
            return '<span style="display:inline-block;margin-right:10px">' + s.id + ': ' + Math.round(s.score * 100) + '%</span>';
        }).join('') +
        '</div>' +
        '<div style="font-size:0.7rem;color:#6b7280;margin-top:4px">' +
        'Pose Engine: ' + (profile.wsConnected ? '🟢 Connected' : '🟡 Local fallback') +
        '</div>' +
        '</div>';
}

// Listen for calibration complete event from the wizard
window.addEventListener('calibrationComplete', function(e) {
    if (e.detail && e.detail.profile) {
        onCalibrationCompleteV2(e.detail.profile);
    }
});

// ============================================================================
// UTILITIES
// ============================================================================

function showStatus(text, type) {
    const statusEl = document.getElementById('cameraStatus');
    const textEl = document.getElementById('statusText');
    
    if (textEl) textEl.textContent = text;
    if (statusEl) {
        statusEl.className = `camera-status ${type === 'success' ? 'connected' : type === 'error' ? 'error' : 'disconnected'}`;
    }
}

function updateStatusText(text) {
    const el = document.getElementById('statusText');
    if (el) el.textContent = text;
}

function showNotification(message, type = 'info') {
    if (typeof window.showNotification === 'function' && window.showNotification !== showNotification) {
        window.showNotification(message, type);
    } else {
        console.log(`[${type}] ${message}`);
    }
}

const VoiceFeedback = {
    lastSpoken: 0,
    speak(text) {
        const now = Date.now();
        if (now - this.lastSpoken < 2000) return;
        
        window.speechSynthesis.cancel();
        const u = new SpeechSynthesisUtterance(text);
        u.rate = 1.1;
        window.speechSynthesis.speak(u);
        this.lastSpoken = now;
    }
};

function startNewAssessment() {
    location.reload();
}

// Legacy compatibility
function generateMedicalNote() {
    showNotification('Generating medical note...', 'info');
}

function prescribeExercises() {
    showNotification('Saving prescription...', 'info');
}

console.log('✅ Enhanced Assessment Workflow v2 loaded');
