import { ClinicalLayout } from './clinicalStyles.js';

export function MovementCapture({ mode = 'gait' }: { mode?: 'gait' | 'exercise' | 'general' }) {
  const titleMap = {
    gait: 'Gait Analysis with Photo Capture',
    exercise: 'Exercise Movement Capture',
    general: 'Movement Photo Timeline',
  };

  return (
    <ClinicalLayout title={titleMap[mode]} subtitle="Real-time camera + skeleton overlay with phase-triggered photo snapshots.">
      <section class="clinical-grid">
        {/* Camera Feed + Skeleton Overlay */}
        <div class="clinical-card span-6">
          <h2>Live Camera + Skeleton</h2>
          <div style="position:relative; background:#000; border-radius: 8px; overflow: hidden;">
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
          </div>
          <div style="display:flex; gap:8px; margin-top:10px; flex-wrap:wrap;">
            <button class="clinical-btn" id="btnStartCamera" onclick="startCaptureCamera()">📷 Start Camera</button>
            <button class="clinical-btn" id="btnSnapshot" onclick="manualSnapshot()" disabled>📸 Snapshot</button>
            <button class="clinical-btn" id="btnAutoCapture" onclick="toggleAutoCapture()">⏱ Auto-Capture</button>
            <label class="pill">
              <input type="checkbox" id="skeletonToggle" checked onchange="toggleSkeleton()" /> Skeleton
            </label>
          </div>
        </div>

        {/* Live Metrics */}
        <div class="clinical-card span-3">
          <h2>Movement Phase</h2>
          <div class="metric"><span>Current phase</span><strong id="currentPhase">—</strong></div>
          <div class="metric"><span>Frames</span><strong id="frameCount">0</strong></div>
          <div class="metric"><span>Snapshots</span><strong id="snapshotCount">0</strong></div>
          <div class="metric"><span>Camera</span><strong id="cameraStatus">Off</strong></div>
          <h2 style="margin-top:16px;">Phase Legend</h2>
          <div style="font-size:13px; line-height:1.8;">
            <div><span class="pill" style="border-color:#3b82f6; display:inline-block; margin-bottom:4px;">Blue = Stance</span></div>
            <div><span class="pill" style="border-color:#22c55e; display:inline-block; margin-bottom:4px;">Green = Swing</span></div>
            <div><span class="pill" style="border-color:#60a5fa; display:inline-block;">Light Blue = Upper body</span></div>
          </div>
        </div>

        {/* Photo Timeline */}
        <div class="clinical-card span-3">
          <h2>Photo Timeline</h2>
          <div id="photoTimeline" style="max-height: 400px; overflow-y: auto; display:flex; flex-direction:column; gap:8px;">
            <div style="color:#64748b; font-size:13px; text-align:center; padding:20px;">
              Start camera to capture movement photos
            </div>
          </div>
          <button class="clinical-btn" id="btnClearPhotos" onclick="clearPhotos()" style="width:100%; margin-top:8px;">
            🗑 Clear Photos
          </button>
        </div>

        {/* Movement Graph with Photo Markers */}
        <div class="clinical-card span-12">
          <h2>Movement Graph + Photo Markers</h2>
          <canvas class="skeleton-canvas" id="movementGraph" width="900" height="320"></canvas>
          <div id="photoMarkers" style="display:flex; gap:8px; flex-wrap:wrap; margin-top:12px; min-height:60px;"></div>
        </div>
      </section>
      <script dangerouslySetInnerHTML={{ __html: captureScript }} />
    </ClinicalLayout>
  );
}

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
      overlayCanvas.height = videoEl.videoHeight || 480;

      document.getElementById('cameraStatus').textContent = 'Live';
      document.getElementById('btnSnapshot').disabled = false;

      // Start demo gait cycle (simulates pose detection)
      startDemoGaitCycle();

      console.log('📷 Camera started');
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
      ctx.strokeStyle = isLower ? (stanceLeft ? '#3b82f6' : '#22c55e') : '#60a5fa';
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
      const isLower = i >= 11;
      ctx.fillStyle = isLower ? (stanceLeft ? '#3b82f6' : '#22c55e') : '#60a5fa';
      ctx.beginPath();
      ctx.arc(pt[0] * w, pt[1] * h, isLower ? 7 : 5, 0, Math.PI * 2);
      ctx.fill();
    });

    // Phase badge
    ctx.fillStyle = 'rgba(0,0,0,0.6)';
    ctx.fillRect(8, 8, 140, 28);
    ctx.fillStyle = '#fff';
    ctx.font = '13px sans-serif';
    ctx.fillText(currentPhase.replace('_', ' ') + ' · frame ' + frameCount, 16, 27);
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
        ctx.shadowColor = 'rgba(0,0,0,0.5)';
        ctx.shadowBlur = 3;
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
    ctx.fillStyle = 'rgba(0,0,0,0.6)';
    ctx.fillRect(4, canvas.height - 32, 170, 26);
    ctx.fillStyle = '#fff';
    ctx.font = '12px sans-serif';
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
    btn.style.background = isAutoCapturing ? '#ef4444' : '';
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
      container.innerHTML = '<div style="color:#64748b; font-size:13px; text-align:center; padding:20px;">No photos yet</div>';
      return;
    }

    container.innerHTML = snapshots.slice(-8).reverse().map(s => 
      '<div style="position:relative; border-radius:6px; overflow:hidden; border:2px solid #1e293b;">' +
        '<img src="' + s.dataUrl + '" style="width:100%; display:block;" />' +
        '<div style="position:absolute; bottom:0; left:0; right:0; background:rgba(0,0,0,0.7); padding:4px 8px; font-size:11px; color:#fff;">' +
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
      '<div style="position:relative; width:80px; height:60px; border-radius:4px; overflow:hidden; border:1px solid #334155;">' +
        '<img src="' + s.dataUrl + '" style="width:100%; height:100%; object-fit:cover;" />' +
        '<div style="position:absolute; bottom:0; left:0; right:0; background:rgba(0,0,0,0.7); font-size:9px; color:#fff; padding:1px 4px; text-align:center;">' +
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

    ctx.fillStyle = '#050a16';
    ctx.fillRect(0, 0, w, h);

    // Grid
    ctx.strokeStyle = 'rgba(148,163,184,0.08)';
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
        ctx.strokeStyle = zones[i].isStance ? 'rgba(59,130,246,0.08)' : 'rgba(34,197,94,0.08)';
        ctx.lineWidth = (w - 80) / graphHistory.length;
        ctx.beginPath();
        ctx.moveTo(zones[i].x, 40);
        ctx.lineTo(zones[i].x, h - 30);
        ctx.stroke();
      }
    }

    // Stride length wave
    ctx.strokeStyle = '#3b82f6';
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

    // Labels
    ctx.fillStyle = '#3b82f6';
    ctx.font = '13px sans-serif';
    ctx.fillText('Stride Length', 70, 52);

    // Snapshot markers on graph
    snapshots.forEach(s => {
      const idx = graphHistory.findIndex(p => Math.abs(p.timestamp - s.timestamp) < 200);
      if (idx >= 0) {
        const x = 60 + (idx / Math.max(graphHistory.length - 1, 1)) * (w - 80);
        ctx.fillStyle = '#f59e0b';
        ctx.beginPath();
        ctx.arc(x, h - 60, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    });

    // Legend
    ctx.fillStyle = '#f59e0b';
    ctx.beginPath();
    ctx.arc(w - 140, 52, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#94a3b8';
    ctx.font = '12px sans-serif';
    ctx.fillText('Photo markers', w - 128, 56);
  }

  // Initialize
  console.log('📸 Movement Photo Capture ready');
  window.toggleSkeleton = toggleSkeleton;
})();
`;
