// ============================================================================
// IMW-MSK Movement Photo Capture Engine
// Captures camera frames + skeleton overlay at key movement phases
// Integrates with GaitAnalyzer, MuscleAssessment, and ProgressTracker
// ============================================================================

const MovementCapture = {
    state: {
        videoEl: null,
        canvasEl: null,
        overlayCtx: null,
        keypoints: [],
        connections: [],
        snapshots: [],
        isCapturing: false,
        captureInterval: null,
        onSnapshot: null,
        lastPhase: null,
        phaseTriggers: ['heel_strike', 'midstance', 'toe_off', 'swing'],
        currentPhase: null,
        frameCount: 0,
    },

    // COCO skeleton connections for overlay drawing
    SKELETON_CONNECTIONS: [
        [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
        [5, 11], [6, 12], [11, 12],
        [11, 13], [13, 15], [12, 14], [14, 16],
        [0, 1], [0, 2], [1, 3], [2, 4],
    ],

    KEYPOINT_COLORS: {
        stance: '#3b82f6', swing: '#22c55e', neutral: '#60a5fa',
    },

    init(videoElement, canvasElement, options = {}) {
        this.state.videoEl = videoElement;
        this.state.canvasEl = canvasElement;
        this.state.overlayCtx = canvasElement.getContext('2d');
        this.state.onSnapshot = options.onSnapshot || null;
        if (options.phaseTriggers) this.state.phaseTriggers = options.phaseTriggers;
        
        // Match canvas size to video
        if (videoElement) {
            canvasElement.width = videoElement.videoWidth || 640;
            canvasElement.height = videoElement.videoHeight || 480;
        }
    },

    // Update keypoints from pose detection (called from Boxer3D or PoseEngine)
    updateKeypoints(keypoints, connections = null) {
        this.state.keypoints = keypoints;
        if (connections) this.state.connections = connections;
        this.state.frameCount++;
        this.drawOverlay();
    },

    // Set current gait/movement phase (triggers auto-capture)
    setPhase(phase) {
        if (phase !== this.state.currentPhase && this.state.phaseTriggers.includes(phase)) {
            this.state.currentPhase = phase;
            if (this.state.isCapturing) {
                this.captureSnapshot(phase);
            }
        }
    },

    // Draw skeleton overlay on the camera canvas
    drawOverlay() {
        const ctx = this.state.overlayCtx;
        const canvas = this.state.canvasEl;
        if (!ctx || !canvas) return;

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (!this.state.keypoints || this.state.keypoints.length === 0) return;

        const kp = this.state.keypoints;
        const w = canvas.width;
        const h = canvas.height;

        // Draw bones
        const conns = this.state.connections.length > 0 
            ? this.state.connections 
            : this.SKELETON_CONNECTIONS;

        conns.forEach(([a, b]) => {
            if (!kp[a] || !kp[b]) return;
            const pa = kp[a], pb = kp[b];
            if (pa[2] < 0.3 || pb[2] < 0.3) return;

            const isLower = a >= 11 || b >= 11;
            const isStance = this.state.currentPhase !== 'swing';

            ctx.strokeStyle = isLower 
                ? (isStance ? this.KEYPOINT_COLORS.stance : this.KEYPOINT_COLORS.swing)
                : this.KEYPOINT_COLORS.neutral;
            ctx.lineWidth = isLower ? 4 : 3;
            ctx.lineCap = 'round';
            ctx.shadowColor = 'rgba(0,0,0,0.5)';
            ctx.shadowBlur = 3;

            ctx.beginPath();
            ctx.moveTo(pa[0] * w, pa[1] * h);
            ctx.lineTo(pb[0] * w, pb[1] * h);
            ctx.stroke();
        });

        // Draw keypoints
        ctx.shadowBlur = 0;
        kp.forEach((pt, i) => {
            if (!pt || pt[2] < 0.3) return;
            const isLower = i >= 11;
            const isStance = this.state.currentPhase !== 'swing';
            ctx.fillStyle = isLower 
                ? (isStance ? this.KEYPOINT_COLORS.stance : this.KEYPOINT_COLORS.swing)
                : this.KEYPOINT_COLORS.neutral;
            ctx.beginPath();
            ctx.arc(pt[0] * w, pt[1] * h, isLower ? 7 : 5, 0, Math.PI * 2);
            ctx.fill();
        });
    },

    // Capture a snapshot: draw video frame + skeleton to a data URL
    captureSnapshot(phase) {
        const video = this.state.videoEl;
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        const ctx = canvas.getContext('2d');

        // Draw video frame
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Draw skeleton overlay
        const kp = this.state.keypoints;
        if (kp && kp.length > 0) {
            const w = canvas.width, h = canvas.height;
            const conns = this.state.connections.length > 0 
                ? this.state.connections 
                : this.SKELETON_CONNECTIONS;

            conns.forEach(([a, b]) => {
                if (!kp[a] || !kp[b]) return;
                if (kp[a][2] < 0.3 || kp[b][2] < 0.3) return;
                const isLower = a >= 11 || b >= 11;
                ctx.strokeStyle = isLower ? '#3b82f6' : '#60a5fa';
                ctx.lineWidth = isLower ? 4 : 3;
                ctx.lineCap = 'round';
                ctx.shadowColor = 'rgba(0,0,0,0.5)';
                ctx.shadowBlur = 3;
                ctx.beginPath();
                ctx.moveTo(kp[a][0] * w, kp[a][1] * h);
                ctx.lineTo(kp[b][0] * w, kp[b][1] * h);
                ctx.stroke();
            });
            ctx.shadowBlur = 0;
            kp.forEach((pt, i) => {
                if (!pt || pt[2] < 0.3) return;
                ctx.fillStyle = i >= 11 ? '#3b82f6' : '#60a5fa';
                ctx.beginPath();
                ctx.arc(pt[0] * w, pt[1] * h, i >= 11 ? 7 : 5, 0, Math.PI * 2);
                ctx.fill();
            });
        }

        // Add phase label
        ctx.fillStyle = 'rgba(0,0,0,0.6)';
        ctx.fillRect(8, canvas.height - 36, 160, 28);
        ctx.fillStyle = '#fff';
        ctx.font = '14px sans-serif';
        ctx.fillText(`${phase.replace('_', ' ')} · ${new Date().toLocaleTimeString()}`, 16, canvas.height - 16);

        const dataUrl = canvas.toDataURL('image/jpeg', 0.85);

        const snapshot = {
            dataUrl,
            phase,
            timestamp: Date.now(),
            frameNumber: this.state.frameCount,
        };

        this.state.snapshots.push(snapshot);

        // Keep max 20 snapshots
        if (this.state.snapshots.length > 20) {
            this.state.snapshots.shift();
        }

        if (typeof this.state.onSnapshot === 'function') {
            this.state.onSnapshot(snapshot);
        }

        return snapshot;
    },

    // Start continuous capture at interval
    startCapture(intervalMs = 2000) {
        this.state.isCapturing = true;
        this.state.captureInterval = setInterval(() => {
            if (this.state.videoEl && this.state.videoEl.readyState >= 2) {
                this.captureSnapshot(this.state.currentPhase || 'capture');
            }
        }, intervalMs);
    },

    // Stop capturing
    stopCapture() {
        this.state.isCapturing = false;
        if (this.state.captureInterval) {
            clearInterval(this.state.captureInterval);
            this.state.captureInterval = null;
        }
    },

    // Get all snapshots
    getSnapshots() {
        return [...this.state.snapshots];
    },

    // Clear snapshots
    clearSnapshots() {
        this.state.snapshots = [];
    },

    // Export snapshots for ProgressTracker
    exportForProgressTracker() {
        return this.state.snapshots.map(s => ({
            dataUrl: s.dataUrl,
            phase: s.phase,
            timestamp: s.timestamp,
            frameNumber: s.frameNumber,
        }));
    },

    // Manual snapshot trigger
    snapshot() {
        return this.captureSnapshot(this.state.currentPhase || 'manual');
    },
};

// Export for global access
window.MovementCapture = MovementCapture;
