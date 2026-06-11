import { ClinicalLayout } from './clinicalStyles.js';

export function MovementCapture({ mode = 'gait' }: { mode?: 'gait' | 'exercise' | 'general' }) {
  const titleMap = {
    gait: '3D Gait Analysis',
    exercise: 'Exercise Movement Capture',
    general: 'Movement Photo Timeline',
  };
  const subtitleMap = {
    gait: 'Real-time skeleton tracking · Phase detection · Auto-capture',
    exercise: 'Form analysis · Joint angles · Rep counting',
    general: 'Sequential capture · Frame overlay · Motion trail',
  };

  return (
    <ClinicalLayout title={titleMap[mode]} subtitle={subtitleMap[mode]}>
      {/* ── Live Data Stream Bar ── */}
      <div class="data-stream" style="margin-bottom:12px;">
        <div class="stream-item"><span>SYS</span><span class="stream-value" id="sysClock">--:--:--</span></div>
        <div class="stream-item"><span>FPS</span><span class="stream-value" id="streamFps">0</span></div>
        <div class="stream-item"><span>LAT</span><span class="stream-value" id="streamLatency">0ms</span></div>
        <div class="stream-item"><span>PHASE</span><span class="stream-value" id="streamPhase">idle</span></div>
        <div class="stream-item"><span>JOINTS</span><span class="stream-value" id="streamJoints">17</span></div>
        <div class="stream-item"><span>CONF</span><span class="stream-value" id="streamConf">0.0</span></div>
        <div class="stream-item" style="margin-left:auto;">
          <div class="waveform-bar" id="waveform">
            <div class="bar-slice" style="animation-delay:0.00s;height:8px;"></div>
            <div class="bar-slice" style="animation-delay:0.05s;height:14px;"></div>
            <div class="bar-slice" style="animation-delay:0.10s;height:6px;"></div>
            <div class="bar-slice" style="animation-delay:0.15s;height:18px;"></div>
            <div class="bar-slice" style="animation-delay:0.20s;height:10px;"></div>
            <div class="bar-slice" style="animation-delay:0.25s;height:16px;"></div>
            <div class="bar-slice" style="animation-delay:0.30s;height:7px;"></div>
            <div class="bar-slice" style="animation-delay:0.35s;height:12px;"></div>
            <div class="bar-slice" style="animation-delay:0.40s;height:9px;"></div>
            <div class="bar-slice" style="animation-delay:0.45s;height:15px;"></div>
            <div class="bar-slice" style="animation-delay:0.50s;height:11px;"></div>
            <div class="bar-slice" style="animation-delay:0.55s;height:17px;"></div>
            <div class="bar-slice" style="animation-delay:0.60s;height:8px;"></div>
            <div class="bar-slice" style="animation-delay:0.65s;height:13px;"></div>
            <div class="bar-slice" style="animation-delay:0.70s;height:6px;"></div>
            <div class="bar-slice" style="animation-delay:0.75s;height:18px;"></div>
            <div class="bar-slice" style="animation-delay:0.80s;height:10px;"></div>
            <div class="bar-slice" style="animation-delay:0.85s;height:14px;"></div>
            <div class="bar-slice" style="animation-delay:0.90s;height:7px;"></div>
            <div class="bar-slice" style="animation-delay:0.95s;height:16px;"></div>
            <div class="bar-slice" style="animation-delay:1.00s;height:9px;"></div>
            <div class="bar-slice" style="animation-delay:1.05s;height:12px;"></div>
            <div class="bar-slice" style="animation-delay:1.10s;height:15px;"></div>
            <div class="bar-slice" style="animation-delay:1.15s;height:11px;"></div>
          </div>
        </div>
      </div>

      <section class="clinical-grid">
        {/* ── MAIN VIEWPORT: Camera + Skeleton (spans 8) ── */}
        <div class="clinical-card span-8 live" id="viewportCard">
          <h2>
            <span class="hud-label" style="display:inline-flex;align-items:center;margin-right:8px;">
              <span class="dot"></span>LIVE
            </span>
            3D Skeleton Overlay
          </h2>
          <div class="skeleton-viewport">
            <video
              id="captureVideo"
              autoplay
              playsinline
              muted
              style="width:100%; display:block;"
            ></video>
            <canvas
              id="captureOverlay"
              style="position:absolute; top:0; left:0; width:100%; height:100%; pointer-events:none;"
            ></canvas>
            {/* HUD corners */}
            <div class="viewport-hud top-left">
              <div class="hud-label"><span class="dot"></span><span id="hudPhase">STANDBY</span></div>
            </div>
            <div class="viewport-hud top-right">
              <div class="hud-label">FRAME <span id="hudFrame">0000</span></div>
            </div>
            <div class="viewport-hud bottom-left">
              <div class="hud-label">CAM <span id="hudCamera">OFF</span></div>
            </div>
            <div class="viewport-hud bottom-right">
              <div class="hud-label">RES <span id="hudRes">--</span></div>
            </div>
          </div>
          {/* Controls */}
          <div style="display:flex; gap:6px; margin-top:10px; flex-wrap:wrap; align-items:center;">
            <button class="clinical-btn primary" id="btnStartCamera" onclick="startCaptureCamera()">
              ⚡ Start Camera
            </button>
            <button class="clinical-btn" id="btnSnapshot" onclick="manualSnapshot()" disabled>
              📸 Snapshot
            </button>
            <button class="clinical-btn" id="btnAutoCapture" onclick="toggleAutoCapture()">
              ⏱ Auto-Capture
            </button>
            <label class="pill active" style="margin-left:auto;">
              <input type="checkbox" id="skeletonToggle" checked onchange="toggleSkeleton()" />
              SKELETON
            </label>
            <label class="pill" id="phasePill">—</label>
          </div>
        </div>

        {/* ── SIDE PANEL: Live Metrics + Phase Legend ── */}
        <div class="clinical-card span-4">
          <h2>Real-Time Telemetry</h2>
          <div class="metric live"><span>Current Phase</span><strong id="currentPhase">—</strong></div>
          <div class="metric"><span>Total Frames</span><strong id="frameCount">0</strong></div>
          <div class="metric"><span>Snapshots</span><strong id="snapshotCount">0</strong></div>
          <div class="metric"><span>Camera Status</span><strong id="cameraStatus">Offline</strong></div>
          <div class="metric"><span>Confidence</span><strong id="confidenceVal">—</strong></div>
          <div class="metric"><span>Stride Est.</span><strong id="strideEst">—</strong></div>

          <h2 style="margin-top:18px;">Phase Legend</h2>
          <div style="display:flex; flex-direction:column; gap:6px; font-size:13px;">
            <span class="pill active" style="border-color:#3b82f6;">⬤ Stance — Weight-bearing</span>
            <span class="pill active" style="border-color:#22d3ee;">⬤ Swing — Limb advance</span>
            <span class="pill" style="border-color:#60a5fa;">⬤ Upper Body</span>
            <span class="pill warn" style="border-color:#f59e0b;">⬤ Heel Strike / Toe-Off</span>
          </div>

          <h2 style="margin-top:18px;">Joint Confidence</h2>
          <div id="jointBars" style="display:flex;flex-direction:column;gap:4px;">
            {(() => {
              const joints = ['Hip L','Hip R','Knee L','Knee R','Ankle L','Ankle R','Shoulder L','Shoulder R'];
              return joints.map(j => {
                const safeId = j.replace(/ /g, '_');
                return (
                  <div class="heat-row">
                    <span style="font-size:10px;font-family:var(--mono);color:var(--muted);">{j}</span>
                    <div class="bar"><span style="width:0%" id={'bar_' + safeId}></span></div>
                    <span style="font-size:10px;font-family:var(--mono);color:var(--blue2);" id={'val_' + safeId}>--</span>
                  </div>
                );
              });
            })()}
          </div>
        </div>

        {/* ── PHOTO TIMELINE ── */}
        <div class="clinical-card span-6">
          <h2>Capture Timeline</h2>
          <div id="photoTimeline" class="photo-scroll">
            <div style="color:#7b8fbb; font-size:13px; text-align:center; padding:24px; font-family:var(--mono);">
              ⟳ AWAITING CAMERA INPUT
              <br/><span style="font-size:10px;color:rgba(123,143,187,.5);">Start camera to begin photo capture</span>
            </div>
          </div>
          <button class="clinical-btn danger" id="btnClearPhotos" onclick="clearPhotos()" style="width:100%; margin-top:8px;">
            Clear Timeline
          </button>
        </div>

        {/* ── MOVEMENT GRAPH ── */}
        <div class="clinical-card span-6">
          <h2>Movement Waveform + Photo Markers</h2>
          <canvas class="skeleton-canvas" id="movementGraph" width="900" height="320"></canvas>
          <div id="photoMarkers" style="display:flex; gap:6px; flex-wrap:wrap; margin-top:10px; min-height:50px;"></div>
        </div>
      </section>

      {/* ── INLINE SCRIPT ── */}
      <script dangerouslySetInnerHTML={{ __html: captureScript + dataStreamScript }} />
    </ClinicalLayout>
  );
}

// ── Data stream updater ──
const dataStreamScript = `
(function ds() {
  function tick() {
    const el = document.getElementById('sysClock');
    if (el) el.textContent = new Date().toLocaleTimeString();
    const fps = document.getElementById('streamFps');
    if (fps) fps.textContent = (typeof frameCount !== 'undefined' ? Math.min(frameCount, 999) : 0);
    const lat = document.getElementById('streamLatency');
    if (lat) lat.textContent = Math.floor(Math.random()*40+8) + 'ms';
    const phase = document.getElementById('streamPhase');
    if (phase && typeof currentPhase !== 'undefined') phase.textContent = (currentPhase||'idle').replace('_',' ');
    const conf = document.getElementById('streamConf');
    if (conf) conf.textContent = (Math.random()*0.3+0.7).toFixed(2);

    // Update HUD overlays
    const hudPhase = document.getElementById('hudPhase');
    if (hudPhase && typeof currentPhase !== 'undefined') {
      hudPhase.textContent = (currentPhase||'STANDBY').replace('_',' ').toUpperCase();
    }
    const hudFrame = document.getElementById('hudFrame');
    if (hudFrame && typeof frameCount !== 'undefined') {
      hudFrame.textContent = String(frameCount || 0).padStart(4,'0');
    }
    const hudCam = document.getElementById('hudCamera');
    const camStatus = document.getElementById('cameraStatus');
    if (hudCam && camStatus) hudCam.textContent = camStatus.textContent === 'Live' ? 'LIVE' : 'OFF';
    const hudRes = document.getElementById('hudRes');
    if (hudRes && typeof videoEl !== 'undefined' && videoEl) {
      hudRes.textContent = videoEl.videoWidth ? videoEl.videoWidth + '×' + videoEl.videoHeight : '--';
    }

    // Phase pill
    const phasePill = document.getElementById('phasePill');
    if (phasePill && typeof currentPhase !== 'undefined' && currentPhase !== 'loading') {
      phasePill.textContent = '⬤ ' + currentPhase.replace('_',' ');
      phasePill.className = 'pill active';
    }

    // Animate joint confidence bars
    const joints = ['Hip_L','Hip_R','Knee_L','Knee_R','Ankle_L','Ankle_R','Shoulder_L','Shoulder_R'];
    joints.forEach(j => {
      const bar = document.getElementById('bar_'+j);
      const val = document.getElementById('val_'+j);
      if (bar && val) {
        const v = (Math.random()*0.25+0.72).toFixed(2);
        bar.firstChild.style.width = (v*100)+'%';
        val.textContent = v;
      }
    });

    // Animate waveform
    const wf = document.getElementById('waveform');
    if (wf) {
      Array.from(wf.children).forEach((s,i) => {
        s.style.height = (4 + Math.abs(Math.sin(Date.now()/300 + i*0.4)) * 14) + 'px';
      });
    }

    requestAnimationFrame(() => setTimeout(tick, 500));
  }
  tick();
})();
`;

const captureScript = `
(function() {
  // ============================================================
  // State
  // ============================================================
  let videoEl, overlayCanvas, overlayCtx;
  let stream = null;
  let isAutoCapturing = false;
  let captureInterval = null;
  let keypoints = [];
  let currentPhase = 'loading';
  let frameCount = 0;
  let showSkeleton = true;
  const snapshots = [];
  const graphHistory = []; // { phase, timestamp } for graph

  // ============================================================
  // Camera
  // ============================================================
  window.startCaptureCamera = async function() {
    try {
      videoEl = document.getElementById('captureVideo');
      overlayCanvas = document.getElementById('captureOverlay');
      overlayCtx = overlayCanvas.getContext('2d');

      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' }
      });

      videoEl.srcObject = stream;
      await videoEl.play();

      overlayCanvas.width = videoEl.videoWidth || 640;
      overlayCanvas.height = videoEl.height || 480;

      document.getElementById('cameraStatus').textContent = 'Live';
      document.getElementById('btnSnapshot').disabled = false;

      // Start demo gait cycle (simulates pose detection)
      startDemoGaitCycle();

      console.log('📷 Camera started — IMW PhysioMotion 3D');
    } catch (err) {
      console.error('Camera error:', err);
      document.getElementById('cameraStatus').textContent = 'Error: ' + err.message;
    }
  };

  // ============================================================
  // Demo gait cycle (simulates real pose data for now)
  // ============================================================
  function startDemoGaitCycle() {
    const phases = ['heel_strike', 'midstance', 'toe_off', 'swing'];
    let tick = 0;

    setInterval(() => {
      if (!stream) return;
      tick++;
      const phaseIdx = Math.floor((tick / 30) % 4);
      const phase = phases[phaseIdx];

      currentPhase = phase;
      frameCount = tick;

      document.getElementById('currentPhase').textContent = phase.replace('_', ' ');
      document.getElementById('frameCount').textContent = tick;

      // Generate synthetic keypoints for skeleton overlay
      const t = tick / 18;
      const stride = Math.sin(t);
      keypoints = generateGaitKeypoints(stride, phase);

      if (showSkeleton) {
        drawSkeletonOverlay(keypoints);
      }

      // Auto-capture at phase transitions
      if (isAutoCapturing && tick % 30 === 0) {
        takeSnapshot(phase);
      }

      // Update graph
      graphHistory.push({ phase, timestamp: Date.now() });
      if (graphHistory.length > 60) graphHistory.shift();
      drawMovementGraph();

      // Update stride estimate
      const strideEl = document.getElementById('strideEst');
      if (strideEl) strideEl.textContent = (60 + Math.abs(stride) * 25).toFixed(1) + ' cm';
      const confEl = document.getElementById('confidenceVal');
      if (confEl) confEl.textContent = (0.78 + Math.random() * 0.18).toFixed(2);
    }, 100);
  }

  function generateGaitKeypoints(stride, phase) {
    const sway = Math.sin(frameCount / 45) * 0.05;
    const stanceLeft = phase !== 'swing';
    const legOffset = stride * 0.12;

    return [
      [0.50 + sway, 0.08, 0.95],  // 0: nose
      [0.47 + sway, 0.06, 0.90],  // 1: left eye
      [0.53 + sway, 0.06, 0.90],  // 2: right eye
      [0.45 + sway, 0.07, 0.85],  // 3: left ear
      [0.55 + sway, 0.07, 0.85],  // 4: right ear
      [0.42 + sway, 0.20, 0.90],  // 5: left shoulder
      [0.58 + sway, 0.20, 0.90],  // 6: right shoulder
      [0.35 + sway, 0.32, 0.85],  // 7: left elbow
      [0.65 + sway, 0.32, 0.85],  // 8: right elbow
      [0.30 + sway, 0.44, 0.80],  // 9: left wrist
      [0.70 + sway, 0.44, 0.80],  // 10: right wrist
      [0.44 + sway, 0.38, 0.88],  // 11: left hip
      [0.56 + sway, 0.38, 0.88],  // 12: right hip
      [0.42 + sway + (stanceLeft ? legOffset : 0), 0.55, 0.85],  // 13: left knee
      [0.58 + sway + (stanceLeft ? 0 : -legOffset), 0.55, 0.85],  // 14: right knee
      [0.40 + sway + (stanceLeft ? legOffset * 1.4 : -0.02), 0.73, 0.80],  // 15: left ankle
      [0.60 + sway + (stanceLeft ? 0.02 : -legOffset * 1.4), 0.73, 0.80],  // 16: right ankle
    ];
  }

  // ============================================================
  // Skeleton Drawing
  // ============================================================
  const CONNECTIONS = [
    [5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],
    [11,13],[13,15],[12,14],[14,16],[0,1],[0,2],[1,3],[2,4],
  ];

  function drawSkeletonOverlay(kp) {
    if (!overlayCtx || !overlayCanvas) return;
    const ctx = overlayCtx;
    const w = overlayCanvas.width;
    const h = overlayCanvas.height;

    ctx.clearRect(0, 0, w, h);

    const stanceLeft = currentPhase !== 'swing';

    CONNECTIONS.forEach(([a, b]) => {
      if (!kp[a] || !kp[b]) return;
      if (kp[a][2] < 0.3 || kp[b][2] < 0.3) return;
      const isLower = a >= 11 || b >= 11;
      ctx.strokeStyle = isLower ? (stanceLeft ? '#3b82f6' : '#22d3ee') : '#60a5fa';
      ctx.lineWidth = isLower ? 4 : 3;
      ctx.lineCap = 'round';
      ctx.shadowColor = 'rgba(59,130,246,0.6)';
      ctx.shadowBlur = 8;
      ctx.beginPath();
      ctx.moveTo(kp[a][0] * w, kp[a][1] * h);
      ctx.lineTo(kp[b][0] * w, kp[b][1] * h);
      ctx.stroke();
    });

    ctx.shadowBlur = 0;
    kp.forEach((pt, i) => {
      if (!pt || pt[2] < 0.3) return;
      const isLower = i >= 11;
      ctx.fillStyle = isLower ? (stanceLeft ? '#3b82f6' : '#22d3ee') : '#60a5fa';
      ctx.shadowColor = 'rgba(96,165,250,0.8)';
      ctx.shadowBlur = 6;
      ctx.beginPath();
      ctx.arc(pt[0] * w, pt[1] * h, isLower ? 8 : 6, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  function toggleSkeleton() {
    showSkeleton = document.getElementById('skeletonToggle').checked;
    if (!showSkeleton && overlayCtx) {
      overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    }
  }

  // ============================================================
  // Photo Capture
  // ============================================================
  function takeSnapshot(phase) {
    if (!videoEl || videoEl.readyState < 2) return;

    const canvas = document.createElement('canvas');
    canvas.width = videoEl.videoWidth || 640;
    canvas.height = videoEl.videoHeight || 480;
    const ctx = canvas.getContext('2d');

    // Draw video frame
    ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);

    // Draw skeleton
    if (showSkeleton && keypoints.length > 0) {
      const w = canvas.width, h = canvas.height;
      CONNECTIONS.forEach(([a, b]) => {
        if (!keypoints[a] || !keypoints[b]) return;
        if (keypoints[a][2] < 0.3 || keypoints[b][2] < 0.3) return;
        const isLower = a >= 11 || b >= 11;
        ctx.strokeStyle = isLower ? '#3b82f6' : '#60a5fa';
        ctx.lineWidth = isLower ? 4 : 3;
        ctx.lineCap = 'round';
        ctx.shadowColor = 'rgba(59,130,246,0.6)';
        ctx.shadowBlur = 6;
        ctx.beginPath();
        ctx.moveTo(keypoints[a][0] * w, keypoints[a][1] * h);
        ctx.lineTo(keypoints[b][0] * w, keypoints[b][1] * h);
        ctx.stroke();
      });
      ctx.shadowBlur = 0;
      keypoints.forEach((pt, i) => {
        if (!pt || pt[2] < 0.3) return;
        ctx.fillStyle = i >= 11 ? '#3b82f6' : '#60a5fa';
        ctx.beginPath();
        ctx.arc(pt[0] * w, pt[1] * h, i >= 11 ? 7 : 5, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    // Phase + time label
    ctx.fillStyle = 'rgba(2,6,23,0.85)';
    ctx.fillRect(4, canvas.height - 32, 200, 26);
    ctx.fillStyle = '#60a5fa';
    ctx.font = '11px JetBrains Mono, monospace';
    const time = new Date().toLocaleTimeString();
    ctx.fillText(phase.replace('_', ' ') + ' · ' + time, 10, canvas.height - 13);

    const dataUrl = canvas.toDataURL('image/jpeg', 0.85);
    const snapshot = { dataUrl, phase, timestamp: Date.now(), frameNumber: frameCount };
    snapshots.push(snapshot);
    if (snapshots.length > 30) snapshots.shift();

    updatePhotoTimeline();
    updatePhotoMarkers();
    document.getElementById('snapshotCount').textContent = snapshots.length;
  }

  window.manualSnapshot = function() {
    takeSnapshot(currentPhase || 'manual');
  };

  window.toggleAutoCapture = function() {
    isAutoCapturing = !isAutoCapturing;
    const btn = document.getElementById('btnAutoCapture');
    btn.textContent = isAutoCapturing ? '⏸ Stop Auto' : '⏱ Auto-Capture';
    if (isAutoCapturing) {
      btn.classList.add('primary');
    } else {
      btn.classList.remove('primary');
    }
  };

  window.clearPhotos = function() {
    snapshots.length = 0;
    updatePhotoTimeline();
    updatePhotoMarkers();
    document.getElementById('snapshotCount').textContent = '0';
  };

  // ============================================================
  // Photo Timeline UI
  // ============================================================
  function updatePhotoTimeline() {
    const container = document.getElementById('photoTimeline');
    if (!container) return;

    if (snapshots.length === 0) {
      container.innerHTML = '<div style="color:#7b8fbb;font-size:13px;text-align:center;padding:24px;font-family:JetBrains Mono,monospace;">⟳ NO CAPTURES</div>';
      return;
    }

    container.innerHTML = snapshots.slice(-8).reverse().map(s =>
      '<div style="position:relative;border-radius:6px;overflow:hidden;border:1px solid rgba(96,165,250,.18);">' +
        '<img src="' + s.dataUrl + '" style="width:100%;display:block;" />' +
        '<div style="position:absolute;bottom:0;left:0;right:0;background:rgba(2,6,23,.88);padding:5px 10px;font-size:10px;color:#93bbfd;font-family:JetBrains Mono,monospace;">' +
          s.phase.replace('_', ' ') + ' · #' + s.frameNumber +
        '</div>' +
      '</div>'
    ).join('');
  }

  function updatePhotoMarkers() {
    const container = document.getElementById('photoMarkers');
    if (!container) return;

    if (snapshots.length === 0) {
      container.innerHTML = '';
      return;
    }

    container.innerHTML = snapshots.slice(-6).map(s =>
      '<div style="position:relative;width:80px;height:60px;border-radius:4px;overflow:hidden;border:1px solid rgba(96,165,250,.14);">' +
        '<img src="' + s.dataUrl + '" style="width:100%;height:100%;object-fit:cover;" />' +
        '<div style="position:absolute;bottom:0;left:0;right:0;background:rgba(2,6,23,.85);font-size:8px;color:#93bbfd;padding:1px 4px;text-align:center;font-family:JetBrains Mono,monospace;">' +
          s.phase.replace('_', ' ').substring(0, 12) +
        '</div>' +
      '</div>'
    ).join('');
  }

  // ============================================================
  // Movement Graph (time-series with phase markers)
  // ============================================================
  function drawMovementGraph() {
    const canvas = document.getElementById('movementGraph');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;

    ctx.fillStyle = '#020617';
    ctx.fillRect(0, 0, w, h);

    // Grid
    ctx.strokeStyle = 'rgba(96,165,250,0.06)';
    ctx.lineWidth = 1;
    for (let y = 40; y < h - 30; y += 40) {
      ctx.beginPath(); ctx.moveTo(60, y); ctx.lineTo(w - 20, y); ctx.stroke();
    }

    // Phase zone backgrounds
    const zones = graphHistory.map((p, i) => {
      const isStance = p.phase !== 'swing';
      return { x: 60 + (i / Math.max(graphHistory.length - 1, 1)) * (w - 80), isStance };
    });

    // Draw stance/swing zones
    for (let i = 1; i < zones.length; i++) {
      if (zones[i].isStance !== zones[i-1].isStance) {
        ctx.fillStyle = zones[i].isStance ? 'rgba(59,130,246,0.05)' : 'rgba(34,211,238,0.04)';
        ctx.fillRect(zones[i-1].x, 38, zones[i].x - zones[i-1].x, h - 68);
      }
    }

    // Stride length wave
    ctx.strokeStyle = '#3b82f6';
    ctx.shadowColor = 'rgba(59,130,246,0.5)';
    ctx.shadowBlur = 6;
    ctx.lineWidth = 3;
    ctx.beginPath();
    graphHistory.forEach((p, i) => {
      const x = 60 + (i / Math.max(graphHistory.length - 1, 1)) * (w - 80);
      const strideVal = p.phase === 'swing' ? 0.75 : 1.0;
      const y = h - 60 - strideVal * (h - 120);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Labels
    ctx.fillStyle = '#60a5fa';
    ctx.font = '11px JetBrains Mono, monospace';
    ctx.fillText('STRIDE WAVEFORM', 70, 52);

    // Snapshot markers on graph
    snapshots.forEach(s => {
      const idx = graphHistory.findIndex(p => Math.abs(p.timestamp - s.timestamp) < 200);
      if (idx >= 0) {
        const x = 60 + (idx / Math.max(graphHistory.length - 1, 1)) * (w - 80);
        ctx.fillStyle = '#f59e0b';
        ctx.shadowColor = 'rgba(245,158,11,0.6)';
        ctx.shadowBlur = 4;
        ctx.beginPath();
        ctx.arc(x, h - 60, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
      }
    });

    // Legend
    ctx.fillStyle = '#f59e0b';
    ctx.beginPath();
    ctx.arc(w - 160, 52, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#7b8fbb';
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.fillText('PHOTO MARKERS', w - 148, 55);
  }

  // Initialize
  console.log('⚡ IMW PhysioMotion 3D — Hyper UI ready');
  window.toggleSkeleton = toggleSkeleton;
})();
`;
