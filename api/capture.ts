import type { VercelRequest, VercelResponse } from '@vercel/node';
import { clinicalStyles } from '../src/components/clinicalStyles.js';

const HTML = `<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>3D Gait Analysis — IMW PhysioMotion</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>${clinicalStyles}</style></head><body>
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

<script>${captureScript}</script>
</main></body></html>`;

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

export default function handler(_req: VercelRequest, res: VercelResponse) {
  res.setHeader('Content-Type', 'text/html; charset=utf-8');
  res.send(HTML);
}
