import type { VercelRequest, VercelResponse } from '@vercel/node';

const HTML = `<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Movement Photo Capture — IMW MSK</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--navy:#0a1628;--panel:#0d1b31;--border:#1d3355;--gold:#f59e0b;--text:#e2e8f0;--muted:#94a3b8;--blue:#60a5fa;--green:#22c55e;--red:#ef4444}
body{background:var(--navy);color:var(--text);font-family:Inter,sans-serif}
.layout{padding:20px;max-width:1440px;margin:0 auto}
h1{font-size:1.3rem;color:var(--gold);margin-bottom:4px}
h2{font-size:.85rem;color:var(--blue);margin-bottom:10px}
.subtitle{color:var(--muted);font-size:.8rem;margin-bottom:20px}
.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:14px}
.card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:14px}
.s3{grid-column:span 3}.s4{grid-column:span 4}.s6{grid-column:span 6}.s9{grid-column:span 9}.s12{grid-column:span 12}
.metric{display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid rgba(29,51,85,.4);font-size:.75rem}
.metric span{color:var(--muted)}.metric strong{color:var(--text)}
.btn{background:var(--blue);color:#fff;border:none;border-radius:6px;padding:7px 14px;font-size:.73rem;font-weight:600;cursor:pointer;transition:opacity .15s}
.btn:hover{opacity:.9}.btn.red{background:var(--red)}.btn.gold{background:var(--gold);color:#111}
.pill{display:inline-flex;align-items:center;gap:4px;padding:3px 8px;border:1px solid var(--border);border-radius:999px;font-size:.68rem;color:var(--muted)}
.canvas{border-radius:8px;width:100%;background:#050a16}
.feed{position:relative;background:#000;border-radius:8px;overflow:hidden}
.feed video{width:100%;display:block}
.feed canvas{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none}
.timeline{max-height:360px;overflow-y:auto;display:flex;flex-direction:column;gap:6px}
.thumb{position:relative;border-radius:6px;overflow:hidden;border:2px solid #1e293b}
.thumb img{width:100%;display:block}
.thumb .label{position:absolute;bottom:0;left:0;right:0;background:rgba(0,0,0,.7);padding:3px 6px;font-size:10px;color:#fff}
.marker-thumb{position:relative;width:65px;height:48px;border-radius:3px;overflow:hidden;border:1px solid #334155;flex-shrink:0}
.marker-thumb img{width:100%;height:100%;object-fit:cover}
.marker-thumb .label{position:absolute;bottom:0;left:0;right:0;background:rgba(0,0,0,.7);font-size:7px;color:#fbbf24;padding:1px 2px;text-align:center}
.ctrl-row{display:flex;gap:8px;margin-top:10px;flex-wrap:wrap;align-items:center}
@media(max-width:900px){.s3,.s4,.s6,.s9{grid-column:span 12}}
</style></head><body>
<div class="layout">
<h1>📸 Movement Photo Capture</h1>
<p class="subtitle">Real-time camera + skeleton overlay · Phase-triggered snapshots · Photo markers on movement graph</p>
<div class="grid">
<div class="card s6">
<h2>Live Camera + Skeleton Overlay</h2>
<div class="feed"><video id="vid" autoplay playsinline muted></video><canvas id="overlay"></canvas></div>
<div class="ctrl-row">
<button class="btn" onclick="startCam()">📷 Start Camera</button>
<button class="btn gold" onclick="snap()">📸 Snapshot</button>
<button class="btn" id="autoBtn" onclick="toggleAuto()">⏱ Auto-Capture</button>
<label class="pill"><input type="checkbox" id="skelTog" checked onchange="showSkel=this.checked;if(!showSkel)document.getElementById('overlay').getContext('2d').clearRect(0,0,999,999)">Skeleton</label>
</div>
</div>
<div class="card s3">
<h2>Status</h2>
<div class="metric"><span>Phase</span><strong id="phaseTxt">—</strong></div>
<div class="metric"><span>Frames</span><strong id="fcTxt">0</strong></div>
<div class="metric"><span>Snapshots</span><strong id="snapTxt">0</strong></div>
<div class="metric"><span>Camera</span><strong id="camTxt">Off</strong></div>
<h2 style="margin-top:14px">Legend</h2>
<div style="font-size:11px;line-height:1.9">
<div><span class="pill" style="border-color:#3b82f6">Blue = Stance</span></div>
<div><span class="pill" style="border-color:#22c55e">Green = Swing</span></div>
<div><span class="pill" style="border-color:#60a5fa">Light Blue = Upper</span></div>
<div><span class="pill" style="border-color:#f59e0b">Gold = Photo marker</span></div>
</div>
</div>
<div class="card s3">
<h2>Photo Timeline</h2>
<div class="timeline" id="timeline"><div style="color:#64748b;font-size:11px;text-align:center;padding:30px">Start camera to capture<br>movement photos</div></div>
<button class="btn red" onclick="clearAll()" style="width:100%;margin-top:8px">🗑 Clear All Photos</button>
</div>
<div class="card s12">
<h2>Movement Graph with Photo Markers</h2>
<canvas class="canvas" id="graph" width="900" height="280"></canvas>
<div id="markers" style="display:flex;gap:5px;flex-wrap:wrap;margin-top:10px;min-height:50px"></div>
</div>
</div>
</div>
<script>
let stream=null,auto=false,showSkel=true,snaps=[],phase='ready',fc=0,graph=[],kp=[];
const CONN=[[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16],[0,1],[0,2],[1,3],[2,4]];
const PHASES=['heel_strike','midstance','toe_off','swing'];
function $(id){return document.getElementById(id)}
function set(id,v){const e=$(id);if(e)e.textContent=v}
async function startCam(){try{stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720},facingMode:'user'}});$('vid').srcObject=stream;await $('vid').play();set('camTxt','Live');setInterval(()=>{if(!stream)return;fc++;set('fcTxt',fc);const pi=Math.floor((fc/30)%4);phase=PHASES[pi];set('phaseTxt',phase.replace('_',' '));const st=Math.sin(fc/18),sw=Math.sin(fc/45)*.05,lo=st*.12,sl=phase!=='swing';kp=[[.5+sw,.08,.95],[.47+sw,.06,.9],[.53+sw,.06,.9],[.45+sw,.07,.85],[.55+sw,.07,.85],[.42+sw,.2,.9],[.58+sw,.2,.9],[.35+sw,.32,.85],[.65+sw,.32,.85],[.3+sw,.44,.8],[.7+sw,.44,.8],[.44+sw,.38,.88],[.56+sw,.38,.88],[.42+sw+(sl?lo:0),.55,.85],[.58+sw+(sl?0:-lo),.55,.85],[.4+sw+(sl?lo*1.4:-.02),.73,.8],[.6+sw+(sl?.02:-lo*1.4),.73,.8]];if(showSkel)drawSkel();if(auto&&fc%30===0)capture();graph.push({phase,t:Date.now()});if(graph.length>60)graph.shift();drawGraph()},100)}catch(e){set('camTxt','Error: '+e.message)}}
function drawSkel(){const c=$('overlay'),ctx=c.getContext('2d');c.width=c.clientWidth||640;c.height=c.clientHeight||480;ctx.clearRect(0,0,c.width,c.height);const w=c.width,h=c.height,sl=phase!=='swing';CONN.forEach(([a,b])=>{if(!kp[a]||!kp[b]||kp[a][2]<.3||kp[b][2]<.3)return;const lo=a>=11||b>=11;ctx.strokeStyle=lo?(sl?'#3b82f6':'#22c55e'):'#60a5fa';ctx.lineWidth=lo?4:3;ctx.lineCap='round';ctx.beginPath();ctx.moveTo(kp[a][0]*w,kp[a][1]*h);ctx.lineTo(kp[b][0]*w,kp[b][1]*h);ctx.stroke()});kp.forEach((p,i)=>{if(!p||p[2]<.3)return;const lo=i>=11;ctx.fillStyle=lo?(sl?'#3b82f6':'#22c55e'):'#60a5fa';ctx.beginPath();ctx.arc(p[0]*w,p[1]*h,lo?7:5,0,Math.PI*2);ctx.fill()});ctx.fillStyle='rgba(0,0,0,.6)';ctx.fillRect(8,8,140,28);ctx.fillStyle='#fff';ctx.font='13px sans-serif';ctx.fillText(phase.replace('_',' ')+' · '+fc,16,27)}
function capture(){const v=$('vid'),c2=document.createElement('canvas');c2.width=v.videoWidth||640;c2.height=v.videoHeight||480;const cx=c2.getContext('2d');cx.drawImage(v,0,0,c2.width,c2.height);const w=c2.width,h=c2.height,sl=phase!=='swing';CONN.forEach(([a,b])=>{if(!kp[a]||!kp[b]||kp[a][2]<.3||kp[b][2]<.3)return;const lo=a>=11||b>=11;cx.strokeStyle=lo?(sl?'#3b82f6':'#22c55e'):'#60a5fa';cx.lineWidth=lo?4:3;cx.lineCap='round';cx.beginPath();cx.moveTo(kp[a][0]*w,kp[a][1]*h);cx.lineTo(kp[b][0]*w,kp[b][1]*h);cx.stroke()});kp.forEach((p,i)=>{if(!p||p[2]<.3)return;cx.fillStyle=i>=11?(sl?'#3b82f6':'#22c55e'):'#60a5fa';cx.beginPath();cx.arc(p[0]*w,p[1]*h,i>=11?7:5,0,Math.PI*2);cx.fill()});cx.fillStyle='rgba(0,0,0,.6)';cx.fillRect(4,c2.height-32,170,26);cx.fillStyle='#fff';cx.font='12px sans-serif';cx.fillText(phase.replace('_',' ')+' · '+new Date().toLocaleTimeString(),10,c2.height-13);const url=c2.toDataURL('image/jpeg',.85);snaps.push({url,phase,fc});if(snaps.length>30)snaps.shift();set('snapTxt',snaps.length);updateTL();updateMK()}
function snap(){capture()}
function toggleAuto(){auto=!auto;const b=$('autoBtn');b.textContent=auto?'⏸ Stop Auto':'⏱ Auto-Capture';b.className=auto?'btn red':'btn'}
function clearAll(){snaps=[];set('snapTxt','0');updateTL();updateMK()}
function updateTL(){const c=$('timeline');if(!snaps.length){c.innerHTML='<div style="color:#64748b;font-size:11px;text-align:center;padding:30px">No photos yet</div>';return}c.innerHTML=snaps.slice(-8).reverse().map(s=>'<div class="thumb"><img src="'+s.url+'"><div class="label">'+s.phase.replace('_',' ')+' · #'+s.fc+'</div></div>').join('')}
function updateMK(){const c=$('markers');if(!snaps.length){c.innerHTML='';return}c.innerHTML=snaps.slice(-6).map(s=>'<div class="marker-thumb"><img src="'+s.url+'"><div class="label">'+s.phase.replace('_',' ').substring(0,12)+'</div></div>').join('')}
function drawGraph(){const c=$('graph'),ctx=c.getContext('2d'),w=c.width,h=c.height;ctx.fillStyle='#050a16';ctx.fillRect(0,0,w,h);ctx.strokeStyle='rgba(148,163,184,.08)';ctx.lineWidth=1;for(let y=40;y<h-30;y+=40){ctx.beginPath();ctx.moveTo(60,y);ctx.lineTo(w-20,y);ctx.stroke()}ctx.strokeStyle='#3b82f6';ctx.lineWidth=3;ctx.beginPath();graph.forEach((p,i)=>{const x=60+(i/Math.max(graph.length-1,1))*(w-80);const sv=p.phase==='swing'?.7:1;const y=h-60-sv*(h-120);if(!i)ctx.moveTo(x,y);else ctx.lineTo(x,y)});ctx.stroke();ctx.fillStyle='#3b82f6';ctx.font='13px sans-serif';ctx.fillText('Stride Length',70,52);snaps.forEach(s=>{const idx=graph.findIndex(p=>Math.abs(p.t-s.timestamp)<200);if(idx>=0){const x=60+(idx/Math.max(graph.length-1,1))*(w-80);ctx.fillStyle='#f59e0b';ctx.beginPath();ctx.arc(x,h-60,6,0,Math.PI*2);ctx.fill();ctx.strokeStyle='#fff';ctx.lineWidth=1;ctx.stroke()}});ctx.fillStyle='#f59e0b';ctx.beginPath();ctx.arc(w-80,52,5,0,Math.PI*2);ctx.fill();ctx.fillStyle='#94a3b8';ctx.font='12px sans-serif';ctx.fillText('Photo markers',w-68,56)}
</script></body></html>`;

export default function handler(req: VercelRequest, res: VercelResponse) {
  res.setHeader('Content-Type', 'text/html; charset=utf-8');
  res.send(HTML);
}
