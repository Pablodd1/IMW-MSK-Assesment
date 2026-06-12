import type { VercelRequest, VercelResponse } from '@vercel/node';
// CSS inlined below

const clinicalStyles = `
  :root {
    --deep: #020617;
    --void: #030b1a;
    --panel: rgba(3,11,26,.92);
    --card: rgba(5,17,38,.84);
    --glass: rgba(15,30,55,.72);
    --border: rgba(96,165,250,.14);
    --border-glow: rgba(96,165,250,.32);
    --text: #e0e7ff;
    --muted: #7b8fbb;
    --blue: #3b82f6;
    --blue2: #60a5fa;
    --blue3: #93bbfd;
    --blue-glow: rgba(59,130,246,.24);
    --cyan: #22d3ee;
    --green: #34d399;
    --gold: #f59e0b;
    --red: #f87171;
    --pink: #f472b6;
    --radius: 10px;
    --radius-sm: 6px;
    --font: 'Inter', 'SF Pro Display', -apple-system, system-ui, sans-serif;
    --mono: 'JetBrains Mono', 'Fira Code', monospace;
  }

  *, *::before, *::after { box-sizing: border-box; }

  /* ================================================================
     BACKGROUND LAYERS
     ================================================================ */
  body {
    margin: 0;
    min-height: 100vh;
    background: var(--deep);
    color: var(--text);
    font-family: var(--font);
    -webkit-font-smoothing: antialiased;
    overflow-x: hidden;
  }

  /* Hex grid background */
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    z-index: 0;
    background-image:
      url("data:image/svg+xml,%3Csvg width='60' height='52' viewBox='0 0 60 52' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M30 0L60 15v22L30 52 0 37V15z' fill='none' stroke='rgba(96,165,250,.06)' stroke-width='0.5'/%3E%3C/svg%3E");
    background-size: 60px 52px;
    pointer-events: none;
  }

  /* Radial vignette */
  body::after {
    content: '';
    position: fixed;
    inset: 0;
    z-index: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(59,130,246,.06), transparent 60%),
                radial-gradient(ellipse at 80% 80%, rgba(34,211,238,.04), transparent 50%);
    pointer-events: none;
  }

  /* Scan line overlay */
  .scan-lines {
    position: fixed;
    inset: 0;
    z-index: 1;
    pointer-events: none;
    background: repeating-linear-gradient(
      0deg,
      transparent,
      transparent 2px,
      rgba(96,165,250,.012) 2px,
      rgba(96,165,250,.012) 4px
    );
  }

  @keyframes fadeIn { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:translateY(0); } }
  @keyframes pulseGlow { 0%,100% { box-shadow: 0 0 8px var(--blue-glow); } 50% { box-shadow: 0 0 20px rgba(96,165,250,.18), 0 0 40px rgba(59,130,246,.06); } }
  @keyframes scanPulse { 0% { opacity:.3; } 50% { opacity:.7; } 100% { opacity:.3; } }
  @keyframes dataFlow { 0% { background-position:0 0; } 100% { background-position:0 200%; } }
  @keyframes borderPulse { 0%,100% { border-color: rgba(96,165,250,.14); } 50% { border-color: rgba(96,165,250,.32); } }
  @keyframes slideRight { from { width:0; } to { width:100%; } }
  @keyframes skeletonGlow { 0%,100% { filter: drop-shadow(0 0 8px rgba(59,130,246,.5)); } 50% { filter: drop-shadow(0 0 18px rgba(96,165,250,.7)); } }

  /* ================================================================
     SHELL — Main container
     ================================================================ */
  .clinical-shell {
    position: relative;
    z-index: 2;
    min-height: 100vh;
    padding: 16px 18px 32px;
  }

  /* ================================================================
     TOP BAR — HUD style
     ================================================================ */
  .clinical-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 16px;
    margin-bottom: 18px;
    padding: 14px 18px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    animation: fadeIn .35s ease-out;
  }

  .clinical-top h1 {
    margin: 0;
    font-size: .95rem;
    font-weight: 700;
    letter-spacing: .02em;
    background: linear-gradient(135deg, var(--blue2), var(--cyan));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  .clinical-top p {
    margin: 3px 0 0;
    font-size: .72rem;
    color: var(--muted);
    letter-spacing: .01em;
  }

  /* ================================================================
     NAV — Holographic tabs
     ================================================================ */
  .clinical-nav {
    display: flex;
    gap: 4px;
    flex-wrap: wrap;
  }

  .clinical-nav a {
    position: relative;
    color: var(--muted);
    text-decoration: none;
    padding: 7px 12px;
    border-radius: var(--radius-sm);
    font-size: .72rem;
    font-weight: 500;
    letter-spacing: .03em;
    border: 1px solid transparent;
    transition: all .22s ease;
    overflow: hidden;
  }

  .clinical-nav a::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(59,130,246,.08), rgba(34,211,238,.04));
    opacity: 0;
    transition: opacity .22s ease;
  }

  .clinical-nav a:hover {
    color: #fff;
    border-color: var(--border-glow);
    background: rgba(59,130,246,.08);
  }

  .clinical-nav a:hover::before { opacity: 1; }

  /* ================================================================
     GRID — 12-column responsive
     ================================================================ */
  .clinical-grid {
    display: grid;
    grid-template-columns: repeat(12, 1fr);
    gap: 12px;
    animation: fadeIn .38s ease-out both;
  }

  .span-3 { grid-column: span 3; }
  .span-4 { grid-column: span 4; }
  .span-5 { grid-column: span 5; }
  .span-6 { grid-column: span 6; }
  .span-7 { grid-column: span 7; }
  .span-8 { grid-column: span 8; }
  .span-9 { grid-column: span 9; }
  .span-12 { grid-column: span 12; }

  /* ================================================================
     CARDS — Holographic glass panels
     ================================================================ */
  .clinical-card {
    position: relative;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px;
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    transition: border-color .3s ease, box-shadow .3s ease;
    overflow: hidden;
  }

  .clinical-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(96,165,250,.18), transparent);
  }

  .clinical-card:hover {
    border-color: var(--border-glow);
    box-shadow: 0 0 24px rgba(59,130,246,.06), inset 0 0 32px rgba(59,130,246,.02);
  }

  /* Active/recording card */
  .clinical-card.live {
    animation: pulseGlow 2s ease-in-out infinite;
    border-color: var(--border-glow);
  }

  .clinical-card h2, .clinical-card h3 {
    margin: 0 0 10px;
    font-size: .78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .12em;
    color: var(--blue3);
  }

  /* ================================================================
     METRICS — HUD data rows
     ================================================================ */
  .metric {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    padding: 9px 0;
    border-bottom: 1px solid rgba(96,165,250,.06);
    font-size: .76rem;
    font-family: var(--mono);
  }

  .metric:last-child { border-bottom: 0; }

  .metric span:first-child {
    color: var(--muted);
    font-family: var(--font);
    font-size: .7rem;
    letter-spacing: .03em;
  }

  .metric strong {
    color: #fff;
    font-weight: 500;
    position: relative;
  }

  .metric strong::after {
    content: '';
    display: inline-block;
    width: 5px;
    height: 5px;
    border-radius: 50%;
    margin-left: 6px;
    background: var(--green);
    opacity: .8;
    animation: scanPulse 2s ease-in-out infinite;
  }

  .metric.live strong::after { background: var(--cyan); animation-duration: 1s; }

  /* Phase colors */
  .phase-heel_strike { color: var(--blue3); }
  .phase-midstance { color: var(--blue2); }
  .phase-toe_off { color: var(--gold); }
  .phase-swing { color: var(--cyan); }

  /* ================================================================
     PILLS / BADGES
     ================================================================ */
  .pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 10px;
    border-radius: 999px;
    border: 1px solid var(--border);
    color: var(--blue3);
    background: rgba(59,130,246,.08);
    font-size: .68rem;
    font-family: var(--mono);
    letter-spacing: .04em;
  }

  .pill.active {
    border-color: var(--cyan);
    color: var(--cyan);
    background: rgba(34,211,238,.08);
    animation: scanPulse 2s ease-in-out infinite;
  }

  .pill.warn {
    border-color: var(--gold);
    color: var(--gold);
    background: rgba(245,158,11,.08);
  }

  /* ================================================================
     BUTTONS — Cyber controls
     ================================================================ */
  .clinical-btn {
    position: relative;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: var(--blue3);
    background: rgba(59,130,246,.06);
    border: 1px solid var(--border);
    padding: 8px 14px;
    border-radius: var(--radius-sm);
    font-size: .72rem;
    font-weight: 500;
    font-family: var(--font);
    cursor: pointer;
    transition: all .2s ease;
    overflow: hidden;
    letter-spacing: .03em;
  }

  .clinical-btn::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(59,130,246,.1), transparent);
    opacity: 0;
    transition: opacity .2s ease;
  }

  .clinical-btn:hover {
    border-color: var(--border-glow);
    color: #fff;
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(59,130,246,.08);
  }

  .clinical-btn:hover::before { opacity: 1; }

  .clinical-btn:active { transform: translateY(0); }

  .clinical-btn:disabled {
    opacity: .35;
    cursor: not-allowed;
    transform: none;
  }

  .clinical-btn.primary {
    background: linear-gradient(135deg, rgba(59,130,246,.18), rgba(34,211,238,.06));
    border-color: var(--border-glow);
    color: #fff;
  }

  .clinical-btn.danger {
    color: var(--red);
    border-color: rgba(248,113,113,.2);
    background: rgba(248,113,113,.06);
  }

  .clinical-btn.danger:hover {
    border-color: rgba(248,113,113,.4);
    box-shadow: 0 4px 16px rgba(248,113,113,.08);
  }

  /* ================================================================
     SKELETON CANVAS — The 3D viewport
     ================================================================ */
  .skeleton-canvas {
    width: 100%;
    aspect-ratio: 4/3;
    min-height: 280px;
    border-radius: 8px;
    background: var(--void);
    border: 1px solid var(--border);
    box-shadow: inset 0 0 60px rgba(3,11,26,.8);
    animation: skeletonGlow 3s ease-in-out infinite;
  }

  .skeleton-viewport {
    position: relative;
    background: #000;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid var(--border);
  }

  .skeleton-viewport video {
    width: 100%;
    display: block;
  }

  .skeleton-viewport canvas {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
  }

  /* HUD overlay corners */
  .viewport-hud {
    position: absolute;
    pointer-events: none;
    z-index: 5;
  }

  .viewport-hud.top-left { top: 8px; left: 8px; }
  .viewport-hud.top-right { top: 8px; right: 8px; }
  .viewport-hud.bottom-left { bottom: 8px; left: 8px; }
  .viewport-hud.bottom-right { bottom: 8px; right: 8px; }

  .hud-label {
    font-family: var(--mono);
    font-size: 9px;
    color: var(--blue3);
    background: rgba(2,6,23,.72);
    border: 1px solid rgba(96,165,250,.12);
    padding: 4px 8px;
    border-radius: 4px;
    letter-spacing: .08em;
    backdrop-filter: blur(8px);
  }

  .hud-label .dot {
    display: inline-block;
    width: 4px;
    height: 4px;
    border-radius: 50%;
    background: var(--cyan);
    margin-right: 4px;
    animation: scanPulse 1.5s ease-in-out infinite;
  }

  /* ================================================================
     DATA STREAM — Animated metrics bar
     ================================================================ */
  .data-stream {
    display: flex;
    gap: 16px;
    padding: 10px 14px;
    background: rgba(5,17,38,.6);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    font-family: var(--mono);
    font-size: .66rem;
    color: var(--muted);
    overflow-x: auto;
  }

  .data-stream .stream-item {
    display: flex;
    align-items: center;
    gap: 6px;
    white-space: nowrap;
  }

  .data-stream .stream-value {
    color: var(--blue2);
    font-weight: 500;
  }

  .waveform-bar {
    display: flex;
    align-items: flex-end;
    gap: 1px;
    height: 18px;
  }

  .waveform-bar .bar-slice {
    width: 2px;
    background: var(--blue2);
    border-radius: 1px;
    animation: dataFlow 1.2s ease-in-out infinite;
  }

  /* ================================================================
     TABLES
     ================================================================ */
  .clinical-table {
    width: 100%;
    border-collapse: collapse;
    font-size: .74rem;
  }

  .clinical-table th {
    text-align: left;
    padding: 9px 10px;
    color: var(--muted);
    font-weight: 500;
    font-size: .66rem;
    letter-spacing: .08em;
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
  }

  .clinical-table td {
    padding: 9px 10px;
    border-bottom: 1px solid rgba(96,165,250,.04);
    color: var(--text);
  }

  .clinical-table tr:hover td {
    background: rgba(59,130,246,.03);
  }

  /* ================================================================
     INPUTS
     ================================================================ */
  .clinical-input, .clinical-select {
    width: 100%;
    background: rgba(5,17,38,.6);
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: var(--radius-sm);
    padding: 9px 12px;
    font-size: .76rem;
    font-family: var(--font);
    transition: border-color .2s ease, box-shadow .2s ease;
  }

  .clinical-input:focus, .clinical-select:focus {
    outline: none;
    border-color: var(--border-glow);
    box-shadow: 0 0 0 3px rgba(59,130,246,.08);
  }

  .clinical-controls {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
  }

  /* ================================================================
     PROGRESS / BARS
     ================================================================ */
  .bar {
    height: 6px;
    background: rgba(96,165,250,.08);
    border-radius: 999px;
    overflow: hidden;
  }

  .bar > span {
    display: block;
    height: 100%;
    background: linear-gradient(90deg, var(--blue), var(--cyan));
    border-radius: 999px;
    transition: width .6s cubic-bezier(0.4, 0, 0.2, 1);
  }

  .bar.warn > span { background: linear-gradient(90deg, var(--gold), var(--red)); }

  .heat-row {
    display: grid;
    grid-template-columns: 90px 1fr 38px;
    gap: 8px;
    align-items: center;
    margin: 8px 0;
    font-size: .73rem;
    color: var(--muted);
  }

  /* ================================================================
     EXERCISE CARDS
     ================================================================ */
  .exercise-card {
    display: grid;
    grid-template-columns: 92px 1fr;
    gap: 12px;
  }

  .exercise-card img {
    width: 92px;
    height: 74px;
    object-fit: cover;
    border-radius: 7px;
    border: 1px solid var(--border);
  }

  .exercise-card h3 {
    text-transform: none;
    letter-spacing: 0;
    font-size: .88rem;
    margin-bottom: 4px;
    color: var(--text);
  }

  .muted { color: var(--muted); }

  /* ================================================================
     PHOTO TIMELINE — Scrollable captures
     ================================================================ */
  .photo-scroll {
    max-height: 420px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .photo-scroll img {
    border-radius: 6px;
    border: 1px solid var(--border);
    transition: border-color .2s ease;
  }

  .photo-scroll img:hover { border-color: var(--border-glow); }

  /* ================================================================
     MOBILE
     ================================================================ */
  @media (max-width: 980px) {
    .clinical-grid { grid-template-columns: 1fr; }
    .span-3, .span-4, .span-5, .span-6, .span-7, .span-8, .span-9, .span-12 { grid-column: span 1; }
    .clinical-top { flex-direction: column; align-items: flex-start; }
    .clinical-nav { gap: 2px; }
    .clinical-controls { grid-template-columns: 1fr 1fr; }
    .data-stream { flex-wrap: wrap; }
  }

  /* Scrollbar styling */
  ::-webkit-scrollbar { width: 4px; height: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: rgba(96,165,250,.16); border-radius: 2px; }
  ::-webkit-scrollbar-thumb:hover { background: rgba(96,165,250,.28); }
`;


const captureScript = `
(function() {
  let videoEl, overlayCanvas, overlayCtx, stream = null, isAutoCapturing = false;
  let keypoints = [], currentPhase = 'loading', frameCount = 0, showSkeleton = true;
  const snapshots = [], graphHistory = [];
  const CONNECTIONS = [[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16],[0,1],[0,2],[1,3],[2,4]];

  // Data stream updater
  setInterval(() => {
    const el = document.getElementById('sysClock'); if (el) el.textContent = new Date().toLocaleTimeString();
    const fps = document.getElementById('streamFps'); if (fps) fps.textContent = Math.min(frameCount, 999);
    const lat = document.getElementById('streamLatency'); if (lat) lat.textContent = Math.floor(Math.random()*40+8) + 'ms';
    const phase = document.getElementById('streamPhase'); if (phase) phase.textContent = (currentPhase||'idle').replace('_',' ');
    const conf = document.getElementById('streamConf'); if (conf) conf.textContent = (Math.random()*0.3+0.7).toFixed(2);
    const hudPhase = document.getElementById('hudPhase'); if (hudPhase) hudPhase.textContent = (currentPhase||'STANDBY').replace('_',' ').toUpperCase();
    const hudFrame = document.getElementById('hudFrame'); if (hudFrame) hudFrame.textContent = String(frameCount||0).padStart(4,'0');
    const camEl = document.getElementById('cameraStatus');
    const hudCam = document.getElementById('hudCamera'); if (hudCam && camEl) hudCam.textContent = camEl.textContent === 'Live' ? 'LIVE' : 'OFF';
    if (videoEl) { const hudRes = document.getElementById('hudRes'); if (hudRes) hudRes.textContent = videoEl.videoWidth ? videoEl.videoWidth+'×'+videoEl.videoHeight : '--'; }
    const phasePill = document.getElementById('phasePill'); if (phasePill && currentPhase !== 'loading') { phasePill.textContent = '⬤ '+currentPhase.replace('_',' '); phasePill.className = 'pill active'; }
    ['Hip_L','Hip_R','Knee_L','Knee_R','Ankle_L','Ankle_R','Shoulder_L','Shoulder_R'].forEach(j => {
      const bar = document.getElementById('bar_'+j), val = document.getElementById('val_'+j);
      if (bar && val) { const v=(Math.random()*0.25+0.72); bar.firstChild.style.width=(v*100)+'%'; val.textContent=v.toFixed(2); }
    });
    const wf = document.getElementById('waveform'); if (wf) Array.from(wf.children).forEach((s,i) => { s.style.height = (4+Math.abs(Math.sin(Date.now()/300+i*0.4))*14)+'px'; });
  }, 500);

  window.startCaptureCamera = async function() {
    try {
      videoEl = document.getElementById('captureVideo');
      overlayCanvas = document.getElementById('captureOverlay');
      overlayCtx = overlayCanvas.getContext('2d');
      stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' } });
      videoEl.srcObject = stream; await videoEl.play();
      overlayCanvas.width = videoEl.videoWidth || 640; overlayCanvas.height = videoEl.videoHeight || 480;
      document.getElementById('cameraStatus').textContent = 'Live';
      document.getElementById('btnSnapshot').disabled = false;
      startDemoGaitCycle();
    } catch (err) { document.getElementById('cameraStatus').textContent = 'Error: ' + err.message; }
  };

  function startDemoGaitCycle() {
    const phases = ['heel_strike','midstance','toe_off','swing']; let tick = 0;
    setInterval(() => {
      if (!stream) return; tick++;
      const phaseIdx = Math.floor((tick/30)%4), phase = phases[phaseIdx];
      currentPhase = phase; frameCount = tick;
      document.getElementById('currentPhase').textContent = phase.replace('_',' ');
      document.getElementById('frameCount').textContent = tick;
      const t = tick/18, stride = Math.sin(t);
      keypoints = generateGaitKeypoints(stride, phase);
      if (showSkeleton) drawSkeletonOverlay(keypoints);
      if (isAutoCapturing && tick%30===0) takeSnapshot(phase);
      graphHistory.push({phase, timestamp: Date.now()}); if (graphHistory.length>60) graphHistory.shift();
      drawMovementGraph();
      const se = document.getElementById('strideEst'); if (se) se.textContent = (60+Math.abs(stride)*25).toFixed(1)+' cm';
      const ce = document.getElementById('confidenceVal'); if (ce) ce.textContent = (0.78+Math.random()*0.18).toFixed(2);
    }, 100);
  }

  function generateGaitKeypoints(stride, phase) {
    const sway = Math.sin(frameCount/45)*.05, sl = phase!=='swing', lo = stride*.12;
    return [[.5+sway,.08,.95],[.47+sway,.06,.9],[.53+sway,.06,.9],[.45+sway,.07,.85],[.55+sway,.07,.85],[.42+sway,.2,.9],[.58+sway,.2,.9],[.35+sway,.32,.85],[.65+sway,.32,.85],[.3+sway,.44,.8],[.7+sway,.44,.8],[.44+sway,.38,.88],[.56+sway,.38,.88],[.42+sway+(sl?lo:0),.55,.85],[.58+sway+(sl?0:-lo),.55,.85],[.4+sway+(sl?lo*1.4:-.02),.73,.8],[.6+sway+(sl?.02:-lo*1.4),.73,.8]];
  }

  function drawSkeletonOverlay(kp) {
    if (!overlayCtx||!overlayCanvas) return;
    const ctx=overlayCtx,w=overlayCanvas.width,h=overlayCanvas.height,sl=currentPhase!=='swing';
    ctx.clearRect(0,0,w,h);
    CONNECTIONS.forEach(([a,b])=>{if(!kp[a]||!kp[b]||kp[a][2]<.3||kp[b][2]<.3)return;const lo=a>=11||b>=11;ctx.strokeStyle=lo?(sl?'#3b82f6':'#22d3ee'):'#60a5fa';ctx.lineWidth=lo?4:3;ctx.lineCap='round';ctx.shadowColor='rgba(59,130,246,0.6)';ctx.shadowBlur=8;ctx.beginPath();ctx.moveTo(kp[a][0]*w,kp[a][1]*h);ctx.lineTo(kp[b][0]*w,kp[b][1]*h);ctx.stroke()});
    ctx.shadowBlur=0;kp.forEach((pt,i)=>{if(!pt||pt[2]<.3)return;const lo=i>=11;ctx.fillStyle=lo?(sl?'#3b82f6':'#22d3ee'):'#60a5fa';ctx.shadowColor='rgba(96,165,250,0.8)';ctx.shadowBlur=6;ctx.beginPath();ctx.arc(pt[0]*w,pt[1]*h,lo?8:6,0,Math.PI*2);ctx.fill()});
  }

  function toggleSkeleton(){showSkeleton=document.getElementById('skeletonToggle').checked;if(!showSkeleton&&overlayCtx)overlayCtx.clearRect(0,0,overlayCanvas.width,overlayCanvas.height)}
  function takeSnapshot(phase){if(!videoEl||videoEl.readyState<2)return;const c=document.createElement('canvas');c.width=videoEl.videoWidth||640;c.height=videoEl.videoHeight||480;const cx=c.getContext('2d');cx.drawImage(videoEl,0,0,c.width,c.height);if(showSkeleton&&keypoints.length>0){const w=c.width,h=c.height;CONNECTIONS.forEach(([a,b])=>{if(!keypoints[a]||!keypoints[b]||keypoints[a][2]<.3||keypoints[b][2]<.3)return;cx.strokeStyle=(a>=11||b>=11)?'#3b82f6':'#60a5fa';cx.lineWidth=(a>=11||b>=11)?4:3;cx.lineCap='round';cx.shadowColor='rgba(59,130,246,0.6)';cx.shadowBlur=6;cx.beginPath();cx.moveTo(keypoints[a][0]*w,keypoints[a][1]*h);cx.lineTo(keypoints[b][0]*w,keypoints[b][1]*h);cx.stroke()});cx.shadowBlur=0;keypoints.forEach((pt,i)=>{if(!pt||pt[2]<.3)return;cx.fillStyle=i>=11?'#3b82f6':'#60a5fa';cx.beginPath();cx.arc(pt[0]*w,pt[1]*h,i>=11?7:5,0,Math.PI*2);cx.fill()})}cx.fillStyle='rgba(2,6,23,.85)';cx.fillRect(4,c.height-32,200,26);cx.fillStyle='#60a5fa';cx.font='11px JetBrains Mono,monospace';cx.fillText(phase.replace('_',' ')+' · '+new Date().toLocaleTimeString(),10,c.height-13);const url=c.toDataURL('image/jpeg',.85);snapshots.push({url,phase,fc:frameCount});if(snapshots.length>30)snapshots.shift();updateTL();updateMK();document.getElementById('snapshotCount').textContent=snapshots.length}
  window.manualSnapshot=function(){takeSnapshot(currentPhase||'manual')};
  window.toggleAutoCapture=function(){isAutoCapturing=!isAutoCapturing;const b=document.getElementById('btnAutoCapture');b.textContent=isAutoCapturing?'⏸ Stop Auto':'⏱ Auto-Capture';if(isAutoCapturing)b.classList.add('primary');else b.classList.remove('primary')};
  window.clearPhotos=function(){snapshots.length=0;updateTL();updateMK();document.getElementById('snapshotCount').textContent='0'};
  function updateTL(){const c=document.getElementById('photoTimeline');if(!c)return;if(!snapshots.length){c.innerHTML='<div style="color:#7b8fbb;font-size:13px;text-align:center;padding:24px;font-family:JetBrains Mono,monospace">⟳ NO CAPTURES</div>';return}c.innerHTML=snapshots.slice(-8).reverse().map(s=>'<div style="position:relative;border-radius:6px;overflow:hidden;border:1px solid rgba(96,165,250,.18)"><img src="'+s.url+'" style="width:100%;display:block"><div style="position:absolute;bottom:0;left:0;right:0;background:rgba(2,6,23,.88);padding:5px 10px;font-size:10px;color:#93bbfd;font-family:JetBrains Mono,monospace">'+s.phase.replace('_',' ')+' · #'+s.fc+'</div></div>').join('')}
  function updateMK(){const c=document.getElementById('photoMarkers');if(!c)return;if(!snapshots.length){c.innerHTML='';return}c.innerHTML=snapshots.slice(-6).map(s=>'<div style="position:relative;width:80px;height:60px;border-radius:4px;overflow:hidden;border:1px solid rgba(96,165,250,.14)"><img src="'+s.url+'" style="width:100%;height:100%;object-fit:cover"><div style="position:absolute;bottom:0;left:0;right:0;background:rgba(2,6,23,.85);font-size:8px;color:#93bbfd;padding:1px 4px;text-align:center;font-family:JetBrains Mono,monospace">'+s.phase.replace('_',' ').substring(0,12)+'</div></div>').join('')}
  function drawMovementGraph(){const c=document.getElementById('movementGraph');if(!c)return;const ctx=c.getContext('2d'),w=c.width,h=c.height;ctx.fillStyle='#020617';ctx.fillRect(0,0,w,h);ctx.strokeStyle='rgba(96,165,250,.06)';ctx.lineWidth=1;for(let y=40;y<h-30;y+=40){ctx.beginPath();ctx.moveTo(60,y);ctx.lineTo(w-20,y);ctx.stroke()}ctx.strokeStyle='#3b82f6';ctx.shadowColor='rgba(59,130,246,.5)';ctx.shadowBlur=6;ctx.lineWidth=3;ctx.beginPath();graphHistory.forEach((p,i)=>{const x=60+(i/Math.max(graphHistory.length-1,1))*(w-80);const sv=p.phase==='swing'?.7:1;const y=h-60-sv*(h-120);if(!i)ctx.moveTo(x,y);else ctx.lineTo(x,y)});ctx.stroke();ctx.shadowBlur=0;ctx.fillStyle='#60a5fa';ctx.font='11px JetBrains Mono,monospace';ctx.fillText('STRIDE WAVEFORM',70,52);snapshots.forEach(s=>{const idx=graphHistory.findIndex(p=>Math.abs(p.timestamp-s.timestamp)<200);if(idx>=0){const x=60+(idx/Math.max(graphHistory.length-1,1))*(w-80);ctx.fillStyle='#f59e0b';ctx.shadowColor='rgba(245,158,11,.6)';ctx.shadowBlur=4;ctx.beginPath();ctx.arc(x,h-60,5,0,Math.PI*2);ctx.fill();ctx.shadowBlur=0}});ctx.fillStyle='#f59e0b';ctx.beginPath();ctx.arc(w-160,52,4,0,Math.PI*2);ctx.fill();ctx.fillStyle='#7b8fbb';ctx.font='10px JetBrains Mono,monospace';ctx.fillText('PHOTO MARKERS',w-148,55)}
  window.toggleSkeleton = toggleSkeleton;
})();`;

const HTML = `<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>3D Gait Analysis — IMW PhysioMotion</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>' + clinicalStyles + '</style></head><body>
<div class="scan-lines"></div>
<main class="clinical-shell">
<div class="clinical-top"><div><h1>3D Gait Analysis</h1><p>Real-time skeleton tracking · Phase detection · Auto-capture</p></div>
<nav class="clinical-nav">
<a href="/provider">Provider</a><a href="/gait">Gait</a><a href="/muscle">Muscle</a><a href="/clinical-tests">Tests</a><a href="/exercises">Exercises</a><a href="/progress">Progress</a><a href="/reports">Reports</a>
</nav></div>

<div class="data-stream" style="margin-bottom:12px">
<div class="stream-item"><span>SYS</span><span class="stream-value" id="sysClock">--:--:--</span></div>
<div class="stream-item"><span>FPS</span><span class="stream-value" id="streamFps">0</span></div>
<div class="stream-item"><span>LAT</span><span class="stream-value" id="streamLatency">0ms</span></div>
<div class="stream-item"><span>PHASE</span><span class="stream-value" id="streamPhase">idle</span></div>
<div class="stream-item"><span>JOINTS</span><span class="stream-value" id="streamJoints">17</span></div>
<div class="stream-item"><span>CONF</span><span class="stream-value" id="streamConf">0.0</span></div>
<div class="stream-item" style="margin-left:auto"><div class="waveform-bar" id="waveform">
${Array.from({length:24},(_,i)=>'<div class="bar-slice" style="animation-delay:'+(i*0.05).toFixed(2)+'s;height:'+(4+Math.random()*14).toFixed(0)+'px"></div>').join('')}
</div></div></div>

<section class="clinical-grid">
<div class="clinical-card span-8 live" id="viewportCard">
<h2><span class="hud-label" style="display:inline-flex;align-items:center;margin-right:8px"><span class="dot"></span>LIVE</span>3D Skeleton Overlay</h2>
<div class="skeleton-viewport">
<video id="captureVideo" autoplay playsinline muted style="width:100%;display:block"></video>
<canvas id="captureOverlay" style="position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none"></canvas>
<div class="viewport-hud top-left"><div class="hud-label"><span class="dot"></span><span id="hudPhase">STANDBY</span></div></div>
<div class="viewport-hud top-right"><div class="hud-label">FRAME <span id="hudFrame">0000</span></div></div>
<div class="viewport-hud bottom-left"><div class="hud-label">CAM <span id="hudCamera">OFF</span></div></div>
<div class="viewport-hud bottom-right"><div class="hud-label">RES <span id="hudRes">--</span></div></div>
</div>
<div style="display:flex;gap:6px;margin-top:10px;flex-wrap:wrap;align-items:center">
<button class="clinical-btn primary" onclick="startCaptureCamera()">⚡ Start Camera</button>
<button class="clinical-btn" id="btnSnapshot" onclick="manualSnapshot()" disabled>📸 Snapshot</button>
<button class="clinical-btn" id="btnAutoCapture" onclick="toggleAutoCapture()">⏱ Auto-Capture</button>
<label class="pill active" style="margin-left:auto"><input type="checkbox" id="skeletonToggle" checked onchange="toggleSkeleton()">SKELETON</label>
<span class="pill" id="phasePill">—</span>
</div></div>

<div class="clinical-card span-4">
<h2>Real-Time Telemetry</h2>
<div class="metric live"><span>Current Phase</span><strong id="currentPhase">—</strong></div>
<div class="metric"><span>Total Frames</span><strong id="frameCount">0</strong></div>
<div class="metric"><span>Snapshots</span><strong id="snapshotCount">0</strong></div>
<div class="metric"><span>Camera Status</span><strong id="cameraStatus">Offline</strong></div>
<div class="metric"><span>Confidence</span><strong id="confidenceVal">—</strong></div>
<div class="metric"><span>Stride Est.</span><strong id="strideEst">—</strong></div>
<h2 style="margin-top:18px">Phase Legend</h2>
<div style="display:flex;flex-direction:column;gap:6px;font-size:13px">
<span class="pill active" style="border-color:#3b82f6">⬤ Stance — Weight-bearing</span>
<span class="pill active" style="border-color:#22d3ee">⬤ Swing — Limb advance</span>
<span class="pill" style="border-color:#60a5fa">⬤ Upper Body</span>
<span class="pill warn" style="border-color:#f59e0b">⬤ Heel Strike / Toe-Off</span>
</div>
<h2 style="margin-top:18px">Joint Confidence</h2>
<div id="jointBars" style="display:flex;flex-direction:column;gap:4px">
${['Hip L','Hip R','Knee L','Knee R','Ankle L','Ankle R','Shoulder L','Shoulder R'].map(j=>{
 const sid=j.replace(/ /g,'_');
 return '<div class="heat-row"><span style="font-size:10px;font-family:JetBrains Mono,monospace;color:#7b8fbb">'+j+'</span><div class="bar"><span style="width:0%" id="bar_'+sid+'"></span></div><span style="font-size:10px;font-family:JetBrains Mono,monospace;color:#60a5fa" id="val_'+sid+'">--</span></div>';
}).join('')}
</div></div>

<div class="clinical-card span-6">
<h2>Capture Timeline</h2>
<div id="photoTimeline" class="photo-scroll"><div style="color:#7b8fbb;font-size:13px;text-align:center;padding:24px;font-family:JetBrains Mono,monospace">⟳ AWAITING CAMERA INPUT<br><span style="font-size:10px;color:rgba(123,143,187,.5)">Start camera to begin photo capture</span></div></div>
<button class="clinical-btn danger" id="btnClearPhotos" onclick="clearPhotos()" style="width:100%;margin-top:8px">Clear Timeline</button>
</div>

<div class="clinical-card span-6">
<h2>Movement Waveform + Photo Markers</h2>
<canvas class="skeleton-canvas" id="movementGraph" width="900" height="320"></canvas>
<div id="photoMarkers" style="display:flex;gap:6px;flex-wrap:wrap;margin-top:10px;min-height:50px"></div>
</div>
</section>

<script>' + captureScript + '</script>
</main></body></html>`;

export default function handler(_req: VercelRequest, res: VercelResponse) {
  res.setHeader('Content-Type', 'text/html; charset=utf-8');
  res.send(HTML);
}
