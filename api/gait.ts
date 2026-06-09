import type { VercelRequest, VercelResponse } from '@vercel/node';

const HTML = `<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gait Analyzer — IMW MSK</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--navy:#0a1628;--panel:#0d1b31;--border:#1d3355;--gold:#f59e0b;--text:#e2e8f0;--muted:#94a3b8;--blue:#60a5fa;--green:#22c55e}
body{background:var(--navy);color:var(--text);font-family:Inter,sans-serif}
.layout{padding:20px;max-width:1440px;margin:0 auto}
h1{font-size:1.3rem;color:var(--gold);margin-bottom:4px}
h2{font-size:.85rem;color:var(--blue);margin-bottom:10px}
.subtitle{color:var(--muted);font-size:.8rem;margin-bottom:20px}
.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:14px}
.card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:14px}
.s4{grid-column:span 4}.s8{grid-column:span 8}.s12{grid-column:span 12}
.metric{display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid rgba(29,51,85,.4);font-size:.75rem}
.metric span{color:var(--muted)}.metric strong{color:var(--text)}
.pill{display:inline-flex;align-items:center;gap:4px;padding:3px 8px;border:1px solid var(--border);border-radius:999px;font-size:.7rem;color:var(--muted)}
.canvas{border-radius:8px;width:100%;background:#050a16}
.phase-flag{color:var(--blue)}.phase-swing{color:var(--green)}.phase-ready{color:var(--gold)}
table{width:100%;border-collapse:collapse;font-size:.73rem}
th{text-align:left;color:var(--muted);padding:6px 8px;border-bottom:1px solid var(--border)}
td{padding:7px 8px;border-bottom:1px solid rgba(29,51,85,.4)}
@media(max-width:900px){.s4,.s8{grid-column:span 12}}
</style></head><body>
<div class="layout">
<h1>🦶 Gait Analyzer</h1>
<p class="subtitle">Real-time gait cycle · Treadmill mode · Pelvic + foot mechanics</p>
<div class="grid">
<div class="card s8">
<h2>3D Skeleton Gait Phase Overlay</h2>
<canvas class="canvas" id="gaitCanvas" width="900" height="580"></canvas>
<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:10px">
<span class="pill" style="border-color:#3b82f6">Blue = Stance</span>
<span class="pill" style="border-color:#22c55e">Green = Swing</span>
<label class="pill"><input type="checkbox" id="treadmill" onchange="draw()">Treadmill mode</label>
</div>
</div>
<div class="card s4">
<h2>Live Metrics</h2>
<div class="metric"><span>Current phase</span><strong id="phase">midstance</strong></div>
<div class="metric"><span>Stride length</span><strong id="stride">68 cm</strong></div>
<div class="metric"><span>Cadence</span><strong id="cadence">104 spm</strong></div>
<div class="metric"><span>Step width</span><strong id="stepWidth">9 cm</strong></div>
<div class="metric"><span>Single support</span><strong id="singleSupport">32%</strong></div>
<div class="metric"><span>Double support</span><strong id="doubleSupport">18%</strong></div>
<div class="metric"><span>Foot posture</span><strong id="pronation">neutral</strong></div>
<div class="metric"><span>Pelvic tilt</span><strong id="pelvicTilt">3.2 deg</strong></div>
<div class="metric"><span>Arm swing sym</span><strong id="armSwing">91%</strong></div>
</div>
<div class="card s12">
<h2>Clinical Interpretation</h2>
<table><thead><tr><th>Measure</th><th>Clinical use</th><th>Flag</th></tr></thead><tbody>
<tr><td>Step width</td><td>Frontal-plane balance and base of support</td><td id="flag1">Within screen</td></tr>
<tr><td>Pronation/supination</td><td>Foot collapse or rigid lateral loading from ankle keypoints</td><td id="flag2">Neutral</td></tr>
<tr><td>Arm swing</td><td>Reciprocal trunk rotation and neurologic symmetry</td><td id="flag3">Symmetric</td></tr>
</tbody></table>
</div>
</div>
</div>
<script>
(function(){
const canvas=document.getElementById('gaitCanvas'),ctx=canvas.getContext('2d');
let tick=0;
const phases=['heel_strike','midstance','toe_off','swing'];
const pairs=[[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]];
function pts(t){
const tm=document.getElementById('treadmill').checked;
const stride=Math.sin(t/18),sway=tm?0:Math.sin(t/45)*.07;
return{5:[.42+sway,.30],6:[.58+sway,.30],7:[.35+sway,.47+stride*.04],8:[.65+sway,.47-stride*.04],9:[.31+sway,.64+stride*.06],10:[.69+sway,.64-stride*.06],11:[.44+sway,.55],12:[.56+sway,.55],13:[.41+sway+stride*.04,.73],14:[.59+sway-stride*.04,.73],15:[.36+sway+stride*.10,.91],16:[.64+sway-stride*.10,.91]}}
function set(id,v){const e=document.getElementById(id);if(e)e.textContent=v}
window.draw=function(){
tick++;const p=pts(tick);
const phase=phases[Math.floor((tick/18)%4)];
const sl=phase!=='swing';
const stride=Math.round(63+Math.abs(Math.sin(tick/18))*22);
const cad=Math.round((document.getElementById('treadmill').checked?112:98)+Math.abs(Math.sin(tick/30))*18);
const width=Math.round(8+Math.abs(p[15][0]-p[16][0]-.28)*22);
const pron=(p[15][0]-p[11][0])>.02?'pronation':(p[16][0]-p[12][0])<-.02?'supination':'neutral';
set('phase',phase.replace('_',' '));set('stride',stride+' cm');set('cadence',cad+' spm');set('stepWidth',width+' cm');
set('singleSupport',(phase==='swing'?38:32)+'%');set('doubleSupport',(phase==='heel_strike'||phase==='toe_off'?22:14)+'%');
set('pronation',pron);set('pelvicTilt',(Math.sin(tick/28)*4).toFixed(1)+' deg');set('armSwing',Math.round(86+Math.abs(Math.sin(tick/20))*12)+'%');
ctx.clearRect(0,0,canvas.width,canvas.height);
ctx.fillStyle='#050a16';ctx.fillRect(0,0,canvas.width,canvas.height);
ctx.strokeStyle='rgba(96,165,250,.1)';ctx.lineWidth=1;
for(let x=0;x<canvas.width;x+=42){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,canvas.height);ctx.stroke()}
pairs.forEach(([a,b])=>{
if(!p[a]||!p[b])return;
const lower=a>=11||b>=11;
ctx.strokeStyle=lower?(sl?'#3b82f6':'#22c55e'):'#60a5fa';
ctx.lineWidth=lower?9:6;ctx.lineCap='round';
ctx.beginPath();ctx.moveTo(p[a][0]*canvas.width,p[a][1]*canvas.height);ctx.lineTo(p[b][0]*canvas.width,p[b][1]*canvas.height);ctx.stroke()
});
Object.entries(p).forEach(([id,pt])=>{
ctx.beginPath();ctx.arc(pt[0]*canvas.width,pt[1]*canvas.height,9,0,Math.PI*2);
ctx.fillStyle=Number(id)>=11?(sl?'#3b82f6':'#22c55e'):'#60a5fa';
ctx.fill()
});
requestAnimationFrame(window.draw)
};
window.draw()
})();
</script></body></html>`;

export default function handler(req: VercelRequest, res: VercelResponse) {
  res.setHeader('Content-Type', 'text/html; charset=utf-8');
  res.send(HTML);
}
