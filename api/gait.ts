import type { VercelRequest, VercelResponse } from '@vercel/node';
import { clinicalStyles } from '../src/components/clinicalStyles.js';

const gaitScript = `
(function(){
  const canvas = document.getElementById('gaitCanvas');
  const ctx = canvas.getContext('2d');
  let tick = 0;
  const phases = ['heel_strike','midstance','toe_off','swing'];
  const pairs = [[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]];
  function points(t){
    const tm = document.getElementById('treadmillMode').checked;
    const stride = Math.sin(t/18);
    const sway = tm ? 0 : Math.sin(t/45)*.07;
    return {5:[.42+sway,.30],6:[.58+sway,.30],7:[.35+sway,.47+stride*.04],8:[.65+sway,.47-stride*.04],9:[.31+sway,.64+stride*.06],10:[.69+sway,.64-stride*.06],11:[.44+sway,.55],12:[.56+sway,.55],13:[.41+sway+stride*.04,.73],14:[.59+sway-stride*.04,.73],15:[.36+sway+stride*.10,.91],16:[.64+sway-stride*.10,.91]};
  }
  function setText(id, text){ const el=document.getElementById(id); if(el) el.textContent=text; }
  function draw(){
    tick++;
    const p = points(tick);
    const phase = phases[Math.floor((tick/18)%4)];
    const stanceLeft = phase !== 'swing';
    const stride = Math.round(63 + Math.abs(Math.sin(tick/18))*22);
    const cadence = Math.round((document.getElementById('treadmillMode').checked ? 112 : 98) + Math.abs(Math.sin(tick/30))*18);
    const width = Math.round(8 + Math.abs(p[15][0]-p[16][0]-.28)*22);
    const pronation = (p[15][0]-p[11][0]) > .02 ? 'pronation' : (p[16][0]-p[12][0]) < -.02 ? 'supination' : 'neutral';
    setText('phase', phase.replace('_',' ')); setText('stride', stride+' cm'); setText('cadence', cadence+' spm');
    setText('stepWidth', width+' cm'); setText('singleSupport', (phase==='swing'?38:32)+'%');
    setText('doubleSupport', (phase==='heel_strike'||phase==='toe_off'?22:14)+'%'); setText('pronation', pronation);
    setText('pelvicTilt', (Math.sin(tick/28)*4).toFixed(1)+' deg'); setText('armSwing', Math.round(86+Math.abs(Math.sin(tick/20))*12)+'%');
    setText('dsPhase', phase.replace('_',' ')); setText('dsStride', stride+'cm'); setText('dsCadence', cadence+'spm');
    setText('dsWidth', width+'cm'); setText('dsPronation', pronation); setText('dsTilt', (Math.sin(tick/28)*4).toFixed(1)+'\u00b0');
    setText('dsSymmetry', Math.round(86+Math.abs(Math.sin(tick/20))*12)+'%');
    const hudPhase = document.getElementById('hudGaitPhase'); if (hudPhase) hudPhase.textContent = phase.replace('_',' ').toUpperCase();
    setText('hudGaitCycle', String(tick).padStart(3,'0'));
    ctx.clearRect(0,0,canvas.width,canvas.height); ctx.fillStyle='#020617'; ctx.fillRect(0,0,canvas.width,canvas.height);
    ctx.strokeStyle='rgba(96,165,250,.08)'; ctx.lineWidth=1;
    for(let x=0;x<canvas.width;x+=42){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,canvas.height);ctx.stroke()}
    pairs.forEach(([a,b])=>{if(!p[a]||!p[b])return;const lower=a>=11||b>=11;ctx.strokeStyle=lower?(stanceLeft?'#3b82f6':'#22d3ee'):'#60a5fa';ctx.lineWidth=lower?10:7;ctx.lineCap='round';ctx.shadowColor=lower?'rgba(59,130,246,0.7)':'rgba(96,165,250,0.5)';ctx.shadowBlur=10;ctx.beginPath();ctx.moveTo(p[a][0]*canvas.width,p[a][1]*canvas.height);ctx.lineTo(p[b][0]*canvas.width,p[b][1]*canvas.height);ctx.stroke()});
    ctx.shadowBlur=0;
    Object.entries(p).forEach(([id,pt])=>{const isLower=Number(id)>=11;ctx.shadowColor=isLower?'rgba(59,130,246,0.8)':'rgba(96,165,250,0.6)';ctx.shadowBlur=8;ctx.beginPath();ctx.arc(pt[0]*canvas.width,pt[1]*canvas.height,10,0,Math.PI*2);ctx.fillStyle=isLower?(stanceLeft?'#3b82f6':'#22d3ee'):'#60a5fa';ctx.fill()});
    ctx.shadowBlur=0; requestAnimationFrame(draw);
  }
  draw();
})();`;

const HTML = '<!DOCTYPE html>\n<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">\n<title>3D Gait Analyzer \u2014 IMW PhysioMotion</title>\n<link rel="preconnect" href="https://fonts.googleapis.com">\n<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&amp;family=JetBrains+Mono:wght@400;500&amp;display=swap" rel="stylesheet">\n<style>' + clinicalStyles + '</style></head><body>\n<div class="scan-lines"></div>\n<main class="clinical-shell">\n<div class="clinical-top"><div><h1>3D Gait Analyzer</h1><p>Real-time skeleton tracking \u00b7 Phase detection \u00b7 Treadmill mode \u00b7 Clinical metrics</p></div>\n<nav class="clinical-nav">\n<a href="/provider">Provider</a><a href="/gait">Gait</a><a href="/muscle">Muscle</a><a href="/clinical-tests">Tests</a><a href="/exercises">Exercises</a><a href="/progress">Progress</a><a href="/reports">Reports</a>\n</nav></div>\n\n<div class="data-stream" style="margin-bottom:12px">\n<div class="stream-item"><span>CYCLE</span><span class="stream-value" id="dsPhase">\u2014</span></div>\n<div class="stream-item"><span>STRIDE</span><span class="stream-value" id="dsStride">\u2014</span></div>\n<div class="stream-item"><span>CADENCE</span><span class="stream-value" id="dsCadence">\u2014</span></div>\n<div class="stream-item"><span>WIDTH</span><span class="stream-value" id="dsWidth">\u2014</span></div>\n<div class="stream-item"><span>POSTURE</span><span class="stream-value" id="dsPronation">\u2014</span></div>\n<div class="stream-item"><span>TILT</span><span class="stream-value" id="dsTilt">\u2014</span></div>\n<div class="stream-item"><span>SYMM</span><span class="stream-value" id="dsSymmetry">\u2014</span></div>\n</div>\n\n<section class="clinical-grid">\n<div class="clinical-card span-8 live">\n<h2><span class="hud-label" style="display:inline-flex;align-items:center;margin-right:8px"><span class="dot"></span>3D VIEWPORT</span>Gait Phase Skeleton Overlay</h2>\n<div class="skeleton-viewport">\n<canvas class="skeleton-canvas" id="gaitCanvas" width="900" height="620"></canvas>\n<div class="viewport-hud top-left"><div class="hud-label"><span class="dot"></span><span id="hudGaitPhase">MIDSTANCE</span></div></div>\n<div class="viewport-hud top-right"><div class="hud-label">CYCLE <span id="hudGaitCycle">000</span></div></div>\n</div>\n<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:10px;align-items:center">\n<span class="pill active" style="border-color:#3b82f6">\u2b24 Stance</span>\n<span class="pill active" style="border-color:#22d3ee">\u2b24 Swing</span>\n<span class="pill" style="border-color:#60a5fa">\u2b24 Upper</span>\n<label class="pill" style="margin-left:auto"><input type="checkbox" id="treadmillMode">TREADMILL</label>\n</div></div>\n\n<div class="clinical-card span-4">\n<h2>Live Telemetry</h2>\n<div class="metric live"><span>Current Phase</span><strong id="phase">midstance</strong></div>\n<div class="metric"><span>Stride Length</span><strong id="stride">68 cm</strong></div>\n<div class="metric"><span>Cadence</span><strong id="cadence">104 spm</strong></div>\n<div class="metric"><span>Step Width</span><strong id="stepWidth">9 cm</strong></div>\n<div class="metric"><span>Single Support</span><strong id="singleSupport">32%</strong></div>\n<div class="metric"><span>Double Support</span><strong id="doubleSupport">18%</strong></div>\n<div class="metric"><span>Foot Posture</span><strong id="pronation">neutral</strong></div>\n<div class="metric"><span>Pelvic Tilt</span><strong id="pelvicTilt">3.2 deg</strong></div>\n<div class="metric"><span>Arm Swing Symmetry</span><strong id="armSwing">91%</strong></div>\n</div>\n\n<div class="clinical-card span-12">\n<h2>Clinical Interpretation</h2>\n<table class="clinical-table">\n<thead><tr><th>Measure</th><th>Clinical Use</th><th>Flag</th></tr></thead>\n<tbody id="gaitFindings">\n<tr><td>Step Width</td><td>Frontal-plane balance and base of support.</td><td>Within screen</td></tr>\n<tr><td>Pronation/Supination</td><td>Foot collapse or rigid lateral loading from ankle keypoints.</td><td>Neutral</td></tr>\n<tr><td>Arm Swing</td><td>Reciprocal trunk rotation and neurologic symmetry.</td><td>Symmetric</td></tr>\n</tbody></table>\n</div>\n</section>\n\n<script>' + gaitScript + '<\/script>\n</main></body></html>';

export default function handler(_req: VercelRequest, res: VercelResponse) {
  res.setHeader('Content-Type', 'text/html; charset=utf-8');
  res.send(HTML);
}
