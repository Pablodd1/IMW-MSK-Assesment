/**
 * Femto Mega Calibration Wizard v2
 * Step-by-step patient positioning guide integrated into camera flow.
 * Wires calibration data to deployed pose engine WebSocket.
 */

const IMW_CALIBRATION_DEFAULT_CONFIG = {
    poseEngineUrl: 'wss://pablodd1--pose-engine-ws-serve.modal.run/ws',
    supabaseUrl: '',
    demoMode: false,
    voiceEnabled: true
};

function getIMWCalibrationConfig() {
    return { ...IMW_CALIBRATION_DEFAULT_CONFIG, ...(window.IMW_CONFIG || {}) };
}

class CalibrationWizard {
    constructor(options = {}) {
        this.canvasId = options.canvasId || 'canvasEl';
        this.canvas = document.getElementById(this.canvasId);
        this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
        this.currentStep = 0;
        this.completed = false;
        this.active = false;
        this.stepScores = {};         // per-step confidence scores
        this.framesInStep = 0;
        this.framesRequired = options.framesRequired || 30; // frames needed for step completion
        this.calibrationProfile = null;
        this.onStepChange = options.onStepChange || null;
        this.onComplete = options.onComplete || null;
        this.statusEl = options.statusEl || null;

        // Pose engine WebSocket (wired to deployed instance)
        this.wsUrl = options.wsUrl || getIMWCalibrationConfig().poseEngineUrl;
        this.ws = null;
        this.wsConnected = false;

        this.steps = [
            {
                id: 't-pose',
                title: 'Step 1: T-Pose',
                message: 'Stand in T-pose — arms straight out to sides, feet together',
                icon: '🧍',
                check: (kpts) => this.checkTPose(kpts),
            },
            {
                id: 'distance',
                title: 'Step 2: Distance',
                message: 'Step back until your full body is visible in the frame',
                icon: '📏',
                check: (kpts) => this.checkDistance(kpts),
            },
            {
                id: 'turn-profile',
                title: 'Step 3: Side Profile',
                message: 'Turn 90° to your right for side-view calibration',
                icon: '🔄',
                check: (kpts) => this.checkProfile(kpts),
            },
        ];
    }

    // ─── Pose Engine WebSocket ────────────────────────────────────────────
    connectPoseEngine() {
        return new Promise((resolve, reject) => {
            try {
                console.log(`[CalibrationWizard] Connecting to pose engine: ${this.wsUrl}`);
                this.ws = new WebSocket(this.wsUrl);

                this.ws.onopen = () => {
                    console.log('[CalibrationWizard] ✅ Connected to pose engine');
                    this.wsConnected = true;
                    resolve();
                };

                this.ws.onerror = (err) => {
                    console.warn('[CalibrationWizard] ⚠️ Pose engine WS error (will use local fallback):', err.message || err);
                    this.wsConnected = false;
                    // Resolve anyway — fall back to local pose detection
                    resolve();
                };

                this.ws.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        if (data.type === 'calibration_result') {
                            this.handleRemoteCalibration(data);
                        }
                    } catch (e) { /* ignore parse errors */ }
                };

                this.ws.onclose = () => {
                    console.log('[CalibrationWizard] Pose engine disconnected');
                    this.wsConnected = false;
                };

                // Timeout: resolve after 5s even if connection fails
                setTimeout(() => {
                    if (this.ws.readyState !== WebSocket.OPEN) {
                        console.warn('[CalibrationWizard] Pose engine connection timed out, using local');
                        resolve();
                    }
                }, 5000);

            } catch (e) {
                console.warn('[CalibrationWizard] Cannot connect to pose engine:', e.message);
                this.wsConnected = false;
                resolve();
            }
        });
    }

    sendCalibrationData(stepId, keypoints, result) {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
        try {
            this.ws.send(JSON.stringify({
                type: 'calibration_data',
                step: stepId,
                step_index: this.currentStep,
                keypoints: keypoints.map(k => ({
                    id: k.id, x: k.x, y: k.y, z: k.z || 0,
                    confidence: k.confidence || k.visibility || 0
                })),
                result: {
                    ok: result.ok,
                    hint: result.hint,
                    confidence: result.confidence || (result.ok ? 1.0 : 0.5)
                },
                timestamp: Date.now(),
                profile: this.stepScores
            }));
        } catch (e) {
            console.warn('[CalibrationWizard] Failed to send to pose engine:', e.message);
        }
    }

    handleRemoteCalibration(data) {
        console.log('[CalibrationWizard] Remote calibration result:', data);
        if (data.step_id) {
            this.stepScores[data.step_id] = (this.stepScores[data.step_id] || 0) + 0.1;
        }
    }

    // ─── Step Checks ──────────────────────────────────────────────────────
    checkTPose(kpts) {
        const ls = kpts.find(k => k.id === 5);
        const rs = kpts.find(k => k.id === 6);
        const lw = kpts.find(k => k.id === 9);
        const rw = kpts.find(k => k.id === 10);
        if (!ls || !rs || !lw || !rw) {
            return { ok: false, hint: 'Cannot see your arms — stand facing the camera', confidence: 0 };
        }
        const armSpan = Math.abs(lw.x - rw.x);
        const shoulderWidth = Math.abs(ls.x - rs.x);
        const armsOut = armSpan > shoulderWidth * 1.8;
        const level = Math.abs(lw.y - ls.y) < 0.15 && Math.abs(rw.y - rs.y) < 0.15;
        const conf = ((armsOut ? 1 : 0) + (level ? 1 : 0)) / 2;
        return {
            ok: armsOut && level,
            hint: armsOut && level ? 'Perfect T-pose!' : 'Raise arms to shoulder height, fully extended',
            confidence: conf
        };
    }

    checkDistance(kpts) {
        const nose = kpts.find(k => k.id === 0);
        const lAnkle = kpts.find(k => k.id === 15);
        const rAnkle = kpts.find(k => k.id === 16);
        if (!nose || (!lAnkle && !rAnkle)) {
            return { ok: false, hint: 'Cannot see full body — move back from camera', confidence: 0 };
        }
        const ankle = lAnkle || rAnkle;
        const fullBody = nose.y > 0.03 && ankle.y < 0.94;
        const headRoom = nose.y > 0.05;
        const footRoom = ankle.y < 0.90;
        const conf = ((fullBody ? 1 : 0) + (headRoom ? 1 : 0) + (footRoom ? 1 : 0)) / 3;
        return {
            ok: fullBody,
            hint: fullBody ? 'Good distance — full body visible' : 'Move back until feet are visible, and head has room',
            confidence: conf
        };
    }

    checkProfile(kpts) {
        const ls = kpts.find(k => k.id === 5);
        const rs = kpts.find(k => k.id === 6);
        if (!ls || !rs) {
            return { ok: false, hint: 'Cannot see shoulders — face the camera', confidence: 0 };
        }
        const overlap = Math.abs(ls.x - rs.x) < 0.12;
        const conf = overlap ? 0.9 : Math.max(0, 1 - Math.abs(ls.x - rs.x) / 0.3);
        return {
            ok: overlap,
            hint: overlap ? 'Good side view — shoulders aligned' : 'Turn more to your right until shoulders overlap',
            confidence: Math.min(1, conf)
        };
    }

    // ─── Main Update Loop ─────────────────────────────────────────────────
    update(keypoints) {
        if (!this.active || this.completed || !this.ctx || !this.canvas) return;

        const step = this.steps[this.currentStep];
        if (!step) return;

        const result = step.check(keypoints);
        this.drawOverlay(step, result);

        // Send to pose engine
        if (this.wsConnected) {
            this.sendCalibrationData(step.id, keypoints, result);
        }

        // Accumulate confidence
        if (!this.stepScores[step.id]) this.stepScores[step.id] = 0;
        this.stepScores[step.id] = Math.max(this.stepScores[step.id], result.confidence);
        this.framesInStep++;

        // Advance step if condition met consistently
        const required = result.ok ? this.framesRequired : this.framesRequired * 2;
        if (this.framesInStep >= required) {
            this.advanceStep();
        }
    }

    advanceStep() {
        const step = this.steps[this.currentStep];
        if (step) {
            console.log(`[CalibrationWizard] ✅ Step complete: ${step.id} (score: ${this.stepScores[step.id]?.toFixed(2)})`);
        }

        this.framesInStep = 0;
        this.currentStep++;

        if (this.currentStep >= this.steps.length) {
            this.completed = true;
            this.active = false;
            this.calibrationProfile = this.buildProfile();
            this.dispatchComplete();
            return;
        }

        if (this.onStepChange) {
            this.onStepChange(this.currentStep, this.steps[this.currentStep]);
        }
    }

    buildProfile() {
        const now = Date.now();
        return {
            id: `calib_${now}`,
            timestamp: new Date(now).toISOString(),
            steps: this.steps.map(s => ({
                id: s.id,
                score: this.stepScores[s.id] || 0
            })),
            overallScore: this.steps.reduce((sum, s) => sum + (this.stepScores[s.id] || 0), 0) / this.steps.length,
            wsUrl: this.wsUrl,
            wsConnected: this.wsConnected
        };
    }

    dispatchComplete() {
        console.log('[CalibrationWizard] 🎉 Calibration complete!', this.calibrationProfile);
        saveCalibrationProfile(this.calibrationProfile);

        window.dispatchEvent(new CustomEvent('calibrationComplete', {
            detail: {
                profile: this.calibrationProfile,
                wizard: this
            }
        }));

        if (this.onComplete) {
            this.onComplete(this.calibrationProfile);
        }
    }

    // ─── Drawing ──────────────────────────────────────────────────────────
    drawOverlay(step, result) {
        const cw = this.canvas.width || 640;
        const ch = this.canvas.height || 480;

        // Clear
        this.ctx.clearRect(0, 0, cw, ch);

        // Target bracket
        const pad = 40;
        this.ctx.strokeStyle = result.ok ? '#00ff88' : '#ffaa00';
        this.ctx.lineWidth = 3;
        this.ctx.setLineDash([8, 4]);
        this.ctx.strokeRect(pad, pad, cw - pad * 2, ch - pad * 2);
        this.ctx.setLineDash([]);

        // Corner brackets
        const cb = 20;
        this.ctx.strokeStyle = '#00e5ff';
        this.ctx.lineWidth = 2;
        // Top-left
        this.ctx.beginPath(); this.ctx.moveTo(pad, pad + cb); this.ctx.lineTo(pad, pad); this.ctx.lineTo(pad + cb, pad); this.ctx.stroke();
        // Top-right
        this.ctx.beginPath(); this.ctx.moveTo(cw - pad - cb, pad); this.ctx.lineTo(cw - pad, pad); this.ctx.lineTo(cw - pad, pad + cb); this.ctx.stroke();
        // Bottom-left
        this.ctx.beginPath(); this.ctx.moveTo(pad, ch - pad - cb); this.ctx.lineTo(pad, ch - pad); this.ctx.lineTo(pad + cb, ch - pad); this.ctx.stroke();
        // Bottom-right
        this.ctx.beginPath(); this.ctx.moveTo(cw - pad - cb, ch - pad); this.ctx.lineTo(cw - pad, ch - pad); this.ctx.lineTo(cw - pad, ch - pad - cb); this.ctx.stroke();

        // Step indicator (top)
        this.ctx.fillStyle = 'rgba(0,0,0,0.6)';
        this.ctx.fillRect(pad, pad - 30, cw - pad * 2, 28);
        this.ctx.fillStyle = '#00e5ff';
        this.ctx.font = 'bold 14px sans-serif';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(`⚙️ Calibration: ${step.title}`, cw / 2, pad - 8);

        // Progress dots
        const dotY = pad + 18;
        const dotSpacing = 30;
        const startX = cw / 2 - ((this.steps.length - 1) * dotSpacing) / 2;
        for (let i = 0; i < this.steps.length; i++) {
            const x = startX + i * dotSpacing;
            this.ctx.beginPath();
            this.ctx.arc(x, dotY, 6, 0, Math.PI * 2);
            if (i < this.currentStep) {
                this.ctx.fillStyle = '#00ff88';
            } else if (i === this.currentStep) {
                this.ctx.fillStyle = result.ok ? '#00ff88' : '#ffaa00';
            } else {
                this.ctx.fillStyle = '#333';
            }
            this.ctx.fill();
            this.ctx.strokeStyle = '#555';
            this.ctx.lineWidth = 1;
            this.ctx.stroke();
        }

        // Instruction bar (bottom)
        this.ctx.fillStyle = 'rgba(0,0,0,0.75)';
        this.ctx.fillRect(0, ch - 70, cw, 70);
        this.ctx.fillStyle = result.ok ? '#00ff88' : '#ffffff';
        this.ctx.font = 'bold 16px sans-serif';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(step.icon + ' ' + step.message, cw / 2, ch - 40);
        this.ctx.fillStyle = result.ok ? '#00ff88' : '#ffaa00';
        this.ctx.font = '13px sans-serif';
        this.ctx.fillText(result.hint, cw / 2, ch - 18);

        // Update status element
        if (this.statusEl) {
            const el = document.getElementById(this.statusEl);
            if (el) {
                el.textContent = `Calibrating: ${step.title} — ${result.hint}`;
                el.style.color = result.ok ? '#00ff88' : '#ffaa00';
            }
        }
    }

    // ─── Controls ─────────────────────────────────────────────────────────
    start() {
        this.reset();
        this.active = true;
        console.log('[CalibrationWizard] ▶ Calibration started');
        // Connect to pose engine in background
        this.connectPoseEngine().then(() => {
            console.log('[CalibrationWizard] Pose engine connection resolved');
        });
    }

    stop() {
        this.active = false;
        console.log('[CalibrationWizard] ■ Calibration stopped');
    }

    reset() {
        this.currentStep = 0;
        this.completed = false;
        this.active = false;
        this.stepScores = {};
        this.framesInStep = 0;
        this.calibrationProfile = null;
    }

    disconnect() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
            this.wsConnected = false;
        }
    }
}

// ─── Calibration Profile Storage ────────────────────────────────────────
function saveCalibrationProfile(profile) {
    try {
        const profiles = JSON.parse(localStorage.getItem('physio_calibration_profiles') || '[]');
        profiles.unshift(profile);
        // Keep last 10 profiles
        if (profiles.length > 10) profiles.length = 10;
        localStorage.setItem('physio_calibration_profiles', JSON.stringify(profiles));
        localStorage.setItem('physio_last_calibration', JSON.stringify(profile));
        console.log('[CalibrationWizard] 💾 Profile saved:', profile.id);
    } catch (e) {
        console.warn('[CalibrationWizard] Failed to save profile:', e.message);
    }
}

function getCalibrationProfiles() {
    try {
        return JSON.parse(localStorage.getItem('physio_calibration_profiles') || '[]');
    } catch (e) {
        return [];
    }
}

function getLastCalibration() {
    try {
        return JSON.parse(localStorage.getItem('physio_last_calibration'));
    } catch (e) {
        return null;
    }
}

// Export globally
if (typeof window !== 'undefined') {
    window.CalibrationWizard = CalibrationWizard;
    window.saveCalibrationProfile = saveCalibrationProfile;
    window.getCalibrationProfiles = getCalibrationProfiles;
    window.getLastCalibration = getLastCalibration;
}
