import type { VercelRequest, VercelResponse } from '@vercel/node';
import { Hono } from 'hono';

// Lightweight standalone Hono app for page routes only
// Full API routes handled by /api/* rewrite

const app = new Hono();

// ============================================================================
// CLINICAL PAGE ROUTES (self-contained HTML rendering)
// ============================================================================

function renderLayout(title: string, subtitle: string, body: string): string {
  return `<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>${title}</title>
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  :root{--navy:#0a1628;--panel:#0d1b31;--border:#1d3355;--gold:#f59e0b;--text:#e2e8f0;--muted:#94a3b8;--blue:#60a5fa;--green:#22c55e}
  body{background:var(--navy);color:var(--text);font-family:Inter,sans-serif}
  .clinical-layout{padding:24px;max-width:1440px;margin:0 auto}
  .clinical-header{margin-bottom:24px}
  .clinical-header h1{font-size:1.4rem;color:var(--gold)}
  .clinical-header p{color:var(--muted);font-size:.85rem;margin-top:4px}
  .clinical-grid{display:grid;grid-template-columns:repeat(12,1fr);gap:16px}
  .clinical-card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:16px}
  .clinical-card h2{font-size:.9rem;color:var(--blue);margin-bottom:12px;letter-spacing:.02em}
  .span-3{grid-column:span 3}.span-4{grid-column:span 4}.span-6{grid-column:span 6}.span-8{grid-column:span 8}.span-12{grid-column:span 12}
  .metric{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid rgba(29,51,85,.4);font-size:.78rem}
  .metric span{color:var(--muted)}.metric strong{color:var(--text);font-weight:600}
  .clinical-btn{background:var(--blue);color:#fff;border:none;border-radius:6px;padding:8px 16px;font-size:.75rem;font-weight:600;cursor:pointer}
  .clinical-btn:hover{opacity:.9}
  .pill{display:inline-flex;align-items:center;gap:4px;padding:4px 10px;border:1px solid var(--border);border-radius:999px;font-size:.7rem;color:var(--muted)}
  .skeleton-canvas{border-radius:8px;width:100%;background:#050a16}
  .clinical-table{width:100%;border-collapse:collapse;font-size:.75rem}
  .clinical-table th{text-align:left;color:var(--muted);padding:6px 8px;border-bottom:1px solid var(--border);font-weight:600}
  .clinical-table td{padding:8px;border-bottom:1px solid rgba(29,51,85,.4)}
  .heat-row{display:flex;align-items:center;gap:8px;padding:4px 0;font-size:.72rem}
  .heat-row span{width:50px;color:var(--muted)}
  .heat-row .bar{flex:1;height:6px;background:rgba(29,51,85,.5);border-radius:3px;overflow:hidden}
  .heat-row .bar span{display:block;height:100%;background:var(--blue);border-radius:3px}
  @media(max-width:900px){.span-3,.span-4,.span-6,.span-8{grid-column:span 12}}
</style></head><body><div class="clinical-layout">
<div class="clinical-header"><h1>${title}</h1><p>${subtitle}</p></div>
${body}
</div></body></html>`;
}

// Movement Photo Capture page
app.get('/', (c) => {
  const body = `
<section class="clinical-grid">
  <div class="clinical-card span-6">
    <h2>Live Camera + Skeleton Overlay</h2>
    <div style="position:relative;background:#000;border-radius:8px;overflow:hidden">
      <video id="captureVideo" autoplay playsinline muted style="width:100%;display:block"></video>
      <canvas id="captureOverlay" style="position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none"></canvas>
    </div>
    <div style="display:flex;gap:8px;margin-top:10px;flex-wrap:wrap">
      <button class="clinical-btn" onclick="startCamera()">📷 Start Camera</button>
      <button class="clinical-btn" onclick="snapshot()">📸 Snapshot</button>
      <button class="clinical-btn" id="autoBtn" onclick="toggleAuto()">⏱ Auto-Capture</button>
      <label class="pill"><input type="checkbox" id="skelToggle" checked onchange="toggleSkel()">Skeleton</label>
    </div>
  </div>
  <div class="clinical-card span-3">
    <h2>Status</h2>
    <div class="metric"><span>Phase</span><strong id="phase">—</strong></div>
    <div class="metric"><span>Frames</span><strong id="frames">0</strong></div>
    <div class="metric"><span>Snapshots</span><strong id="snapCount">0</strong></div>
    <div class="metric"><span>Camera</span><strong id="camStatus">Off</strong></div>
    <h2 style="margin-top:16px">Legend</h2>
    <div style="font-size:12px;line-height:1.8">
      <div><span class="pill" style="border-color:#3b82f6">Blue = Stance</span></div>
      <div><span class="pill" style="border-color:#22c55e">Green = Swing</span></div>
      <div><span class="pill" style="border-color:#60a5fa">Light Blue = Upper</span></div>
    </div>
  </div>
  <div class="clinical-card span-3">
    <h2>Photo Timeline</h2>
    <div id="timeline" style="max-height:400px;overflow-y:auto;display:flex;flex-direction:column;gap:8px">
      <div style="color:#64748b;font-size:12px;text-align:center;padding:20px">Start camera to capture</div>
    </div>
    <button class="clinical-btn" onclick="clearPhotos()" style="width:100%;margin-top:8px;background:#ef4444">🗑 Clear</button>
  </div>
  <div class="clinical-card span-12">
    <h2>Movement Graph + Photo Markers</h2>
    <canvas class="skeleton-canvas" id="graph" width="900" height="280"></canvas>
    <div id="photoMarkers" style="display:flex;gap:6px;flex-wrap:wrap;margin-top:10px;min-height:50px"></div>
  </div>
</section>
<script>
let stream=null,auto=false,showSkel=true,snaps=[],phase='ready',fc=0,graph=[],kp=[];
const CONN=[[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16],[0,1],[0,2],[1,3],[2,4]];
const PHASES=['heel_strike','midstance','toe_off','swing'];
function set(k,v){const e=document.getElementById(k);if(e)e.textContent=v}
async function startCamera(){try{stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720},facingMode:'user'}});document.getElementById('captureVideo').srcObject=stream;await document.getElementById('captureVideo').play();set('camStatus','Live');setInterval(()=>{if(!stream)return;fc++;set('frames',fc);const pi=Math.floor((fc/30)%4);phase=PHASES[pi];set('phase',phase.replace('_',' '));const st=Math.sin(fc/18),sw=Math.sin(fc/45)*.05,lo=st*.12,sl=phase!=='swing';kp=[[.5+sw,.08,.95],[.47+sw,.06,.9],[.53+sw,.06,.9],[.45+sw,.07,.85],[.55+sw,.07,.85],[.42+sw,.2,.9],[.58+sw,.2,.9],[.35+sw,.32,.85],[.65+sw,.32,.85],[.3+sw,.44,.8],[.7+sw,.44,.8],[.44+sw,.38,.88],[.56+sw,.38,.88],[.42+sw+(sl?lo:0),.55,.85],[.58+sw+(sl?0:-lo),.55,.85],[.4+sw+(sl?lo*1.4:-.02),.73,.8],[.6+sw+(sl?.02:-lo*1.4),.73,.8]];if(showSkel)drawSkel();if(auto&&fc%30===0)captureSnap();graph.push({phase,t:Date.now()});if(graph.length>60)graph.shift();drawGraph()},100)}catch(e){set('camStatus','Error: '+e.message)}}
function drawSkel(){const c=document.getElementById('captureOverlay'),ctx=c.getContext('2d');c.width=c.clientWidth||640;c.height=c.clientHeight||480;ctx.clearRect(0,0,c.width,c.height);const w=c.width,h=c.height,sl=phase!=='swing';CONN.forEach(([a,b])=>{if(!kp[a]||!kp[b]||kp[a][2]<.3||kp[b][2]<.3)return;const lo=a>=11||b>=11;ctx.strokeStyle=lo?(sl?'#3b82f6':'#22c55e'):'#60a5fa';ctx.lineWidth=lo?4:3;ctx.lineCap='round';ctx.beginPath();ctx.moveTo(kp[a][0]*w,kp[a][1]*h);ctx.lineTo(kp[b][0]*w,kp[b][1]*h);ctx.stroke()});kp.forEach((p,i)=>{if(!p||p[2]<.3)return;const lo=i>=11;ctx.fillStyle=lo?(sl?'#3b82f6':'#22c55e'):'#60a5fa';ctx.beginPath();ctx.arc(p[0]*w,p[1]*h,lo?7:5,0,Math.PI*2);ctx.fill()});ctx.fillStyle='rgba(0,0,0,.6)';ctx.fillRect(8,8,140,28);ctx.fillStyle='#fff';ctx.font='13px sans-serif';ctx.fillText(phase.replace('_',' ')+' · '+fc,16,27)}
function captureSnap(){const v=document.getElementById('captureVideo');const c2=document.createElement('canvas');c2.width=v.videoWidth||640;c2.height=v.videoHeight||480;const cx=c2.getContext('2d');cx.drawImage(v,0,0,c2.width,c2.height);const w=c2.width,h=c2.height,sl=phase!=='swing';CONN.forEach(([a,b])=>{if(!kp[a]||!kp[b]||kp[a][2]<.3||kp[b][2]<.3)return;const lo=a>=11||b>=11;cx.strokeStyle=lo?(sl?'#3b82f6':'#22c55e'):'#60a5fa';cx.lineWidth=lo?4:3;cx.lineCap='round';cx.beginPath();cx.moveTo(kp[a][0]*w,kp[a][1]*h);cx.lineTo(kp[b][0]*w,kp[b][1]*h);cx.stroke()});kp.forEach((p,i)=>{if(!p||p[2]<.3)return;cx.fillStyle=i>=11?(sl?'#3b82f6':'#22c55e'):'#60a5fa';cx.beginPath();cx.arc(p[0]*w,p[1]*h,i>=11?7:5,0,Math.PI*2);cx.fill()});cx.fillStyle='rgba(0,0,0,.6)';cx.fillRect(4,c2.height-32,170,26);cx.fillStyle='#fff';cx.font='12px sans-serif';cx.fillText(phase.replace('_',' ')+' · '+new Date().toLocaleTimeString(),10,c2.height-13);const url=c2.toDataURL('image/jpeg',.85);snaps.push({url,phase,fc});if(snaps.length>30)snaps.shift();set('snapCount',snaps.length);updateTimeline();updateMarkers()}
function snapshot(){captureSnap()}
function toggleAuto(){auto=!auto;const b=document.getElementById('autoBtn');b.textContent=auto?'⏸ Stop Auto':'⏱ Auto-Capture';b.style.background=auto?'#ef4444':''}
function toggleSkel(){showSkel=document.getElementById('skelToggle').checked;if(!showSkel){const c=document.getElementById('captureOverlay');c.getContext('2d').clearRect(0,0,c.width,c.height)}}
function clearPhotos(){snaps=[];set('snapCount','0');updateTimeline();updateMarkers()}
function updateTimeline(){const c=document.getElementById('timeline');if(snaps.length===0){c.innerHTML='<div style="color:#64748b;font-size:12px;text-align:center;padding:20px">No photos yet</div>';return}c.innerHTML=snaps.slice(-8).reverse().map(s=>'<div style="position:relative;border-radius:6px;overflow:hidden;border:2px solid #1e293b"><img src="'+s.url+'" style="width:100%;display:block"><div style="position:absolute;bottom:0;left:0;right:0;background:rgba(0,0,0,.7);padding:4px 8px;font-size:11px;color:#fff">'+s.phase.replace('_',' ')+' · #'+s.fc+'</div></div>').join('')}
function updateMarkers(){const c=document.getElementById('photoMarkers');if(snaps.length===0){c.innerHTML='';return}c.innerHTML=snaps.slice(-6).map(s=>'<div style="position:relative;width:70px;height:52px;border-radius:3px;overflow:hidden;border:1px solid #334155"><img src="'+s.url+'" style="width:100%;height:100%;object-fit:cover"><div style="position:absolute;bottom:0;left:0;right:0;background:rgba(0,0,0,.7);font-size:8px;color:#fbbf24;padding:1px 3px;text-align:center">'+s.phase.replace('_',' ')+'</div></div>').join('')}
function drawGraph(){const c=document.getElementById('graph'),ctx=c.getContext('2d'),w=c.width,h=c.height;ctx.fillStyle='#050a16';ctx.fillRect(0,0,w,h);ctx.strokeStyle='rgba(148,163,184,.08)';ctx.lineWidth=1;for(let y=40;y<h-30;y+=40){ctx.beginPath();ctx.moveTo(60,y);ctx.lineTo(w-20,y);ctx.stroke()}ctx.strokeStyle='#3b82f6';ctx.lineWidth=3;ctx.beginPath();graph.forEach((p,i)=>{const x=60+(i/Math.max(graph.length-1,1))*(w-80);const sv=p.phase==='swing'?.7:1;const y=h-60-sv*(h-120);if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y)});ctx.stroke();ctx.fillStyle='#3b82f6';ctx.font='13px sans-serif';ctx.fillText('Stride Length',70,52);snaps.forEach(s=>{const idx=graph.findIndex(p=>Math.abs(p.t-s.timestamp)<200);if(idx>=0){const x=60+(idx/Math.max(graph.length-1,1))*(w-80);ctx.fillStyle='#f59e0b';ctx.beginPath();ctx.arc(x,h-60,6,0,Math.PI*2);ctx.fill();ctx.strokeStyle='#fff';ctx.lineWidth=1;ctx.stroke()}});ctx.fillStyle='#f59e0b';ctx.beginPath();ctx.arc(w-80,52,5,0,Math.PI*2);ctx.fill();ctx.fillStyle='#94a3b8';ctx.font='12px sans-serif';ctx.fillText('Photo markers',w-68,56)}
</script>`;
  return c.html(renderLayout('Movement Photo Capture', 'Real-time camera + skeleton overlay with phase-triggered photo snapshots', body));
});

export default app;
