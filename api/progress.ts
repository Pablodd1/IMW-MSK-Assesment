import type { VercelRequest, VercelResponse } from '@vercel/node';

const HTML = `<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Progress Tracking — IMW MSK</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--navy:#0a1628;--panel:#0d1b31;--border:#1d3355;--gold:#f59e0b;--text:#e2e8f0;--muted:#94a3b8;--blue:#60a5fa;--green:#22c55e;--red:#ef4444;--amber:#f59e0b}
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
.btn{background:var(--blue);color:#fff;border:none;border-radius:6px;padding:7px 14px;font-size:.73rem;font-weight:600;cursor:pointer}
.canvas{border-radius:8px;width:100%;background:#050a16}
table{width:100%;border-collapse:collapse;font-size:.73rem}
th{text-align:left;color:var(--muted);padding:6px 8px;border-bottom:1px solid var(--border)}
td{padding:7px 8px;border-bottom:1px solid rgba(29,51,85,.4)}
.heat-row{display:flex;align-items:center;gap:8px;padding:4px 0;font-size:.7rem}
.heat-row span:first-child{width:50px;color:var(--muted)}
.heat-row .bar{flex:1;height:6px;background:rgba(29,51,85,.5);border-radius:3px;overflow:hidden}
.heat-row .bar i{display:block;height:100%;background:var(--blue);border-radius:3px}
.before-after{display:flex;gap:10px}
.before-after>div{flex:1;border-radius:6px;overflow:hidden;background:#1e293b}
.before-after .ba-header{padding:4px 8px;font-size:11px;font-weight:600}
.before-after .ba-body{aspect-ratio:4/3;display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:11px}
.before{border:2px solid var(--red)}.before .ba-header{background:var(--red);color:#fff}
.after{border:2px solid var(--green)}.after .ba-header{background:var(--green);color:#000}
.marker-thumb{position:relative;width:65px;height:48px;border-radius:3px;overflow:hidden;border:1px solid #334155;flex-shrink:0}
.marker-thumb img{width:100%;height:100%;object-fit:cover}
.marker-thumb .mlabel{position:absolute;bottom:0;left:0;right:0;background:rgba(0,0,0,.7);font-size:7px;color:#fbbf24;padding:1px 2px;text-align:center}
@media(max-width:900px){.s4,.s8{grid-column:span 12}}
</style></head><body>
<div class="layout">
<h1>📊 Progress Tracking</h1>
<p class="subtitle">Trend charts, before/after photos, session history, and goals</p>
<div class="grid">
<div class="card s8">
<h2>Trend Chart with Photo Markers</h2>
<canvas class="canvas" id="progressChart" width="900" height="400"></canvas>
<div id="photoMarkers" style="display:flex;gap:5px;flex-wrap:wrap;margin-top:10px;min-height:50px"></div>
</div>
<div class="card s4">
<h2>Before / After</h2>
<div class="metric"><span>FMS total</span><strong>11 → 16 (+45%)</strong></div>
<div class="metric"><span>Lumbar flexion</span><strong>45 → 58 deg (+29%)</strong></div>
<div class="metric"><span>Gait cadence</span><strong>92 → 108 spm (+17%)</strong></div>
<div class="metric"><span>Pain</span><strong>7 → 3 (-57%)</strong></div>
<h2 style="margin-top:16px">Movement Photos</h2>
<div class="before-after">
<div class="before"><div class="ba-header">BEFORE · Session 1</div><div class="ba-body" id="beforePhoto">Start gait capture</div></div>
<div class="after"><div class="ba-header">AFTER · Session 3</div><div class="ba-body" id="afterPhoto">Start gait capture</div></div>
</div>
<h2 style="margin-top:16px">Goals</h2>
<div class="heat-row"><span>Pain</span><div class="bar"><i style="width:76%"></i></div><strong>76%</strong></div>
<div class="heat-row"><span>ROM</span><div class="bar"><i style="width:88%"></i></div><strong>88%</strong></div>
<div class="heat-row"><span>HEP</span><div class="bar"><i style="width:64%"></i></div><strong>64%</strong></div>
<button class="btn" onclick="window.print()" style="width:100%;margin-top:12px">📄 Export Progress Report</button>
</div>
<div class="card s12">
<h2>Session History</h2>
<table><thead><tr><th>Date</th><th>FMS</th><th>ROM</th><th>Gait</th><th>Pain</th><th>Notes</th></tr></thead><tbody>
<tr><td>2026-05-06</td><td>11/21</td><td>45°</td><td>92 spm</td><td>7/10</td><td>Initial evaluation</td></tr>
<tr><td>2026-05-20</td><td>14/21</td><td>52°</td><td>101 spm</td><td>5/10</td><td>Improved tolerance</td></tr>
<tr><td>2026-06-03</td><td>16/21</td><td>58°</td><td>108 spm</td><td>3/10</td><td>Progressing to loaded control</td></tr>
</tbody></table>
</div>
</div>
</div>
<script>
(function(){
const c=document.getElementById('progressChart'),ctx=c.getContext('2d');
const w=c.width,h=c.height;
const series={FMS:[11,14,16],ROM:[45,52,58],Gait:[92,101,108],Pain:[7,5,3]};
const colors={FMS:'#f59e0b',ROM:'#60a5fa',Gait:'#22c55e',Pain:'#ef4444'};
ctx.fillStyle='#050a16';ctx.fillRect(0,0,w,h);
ctx.strokeStyle='rgba(148,163,184,.1)';ctx.lineWidth=1;
for(let y=40;y<h-30;y+=48){ctx.beginPath();ctx.moveTo(48,y);ctx.lineTo(w-24,y);ctx.stroke()}
Object.entries(series).forEach(([name,vals],idx)=>{
const max=Math.max(...vals),min=Math.min(...vals);
ctx.strokeStyle=colors[name];ctx.lineWidth=3;ctx.beginPath();
vals.forEach((v,i)=>{
const x=70+i*(w-140)/(vals.length-1);
const y=h-60-((v-min)/(max-min||1))*(h-130)-idx*5;
if(!i)ctx.moveTo(x,y);else ctx.lineTo(x,y);
ctx.fillStyle=colors[name];ctx.fillRect(x-4,y-4,8,8)
});
ctx.stroke();ctx.fillStyle=colors[name];ctx.font='14px sans-serif';ctx.fillText(name,70+idx*110,28)
});
// Photo markers (gold dots)
const snaps=window.__imwPhotos||[];
snaps.forEach((s,i)=>{
const t=(s.session||i)/Math.max(snaps.length-1||1,1);
const x=70+t*(w-140);
const y=h-50-Math.sin(t*Math.PI)*30;
ctx.fillStyle='#f59e0b';ctx.beginPath();ctx.arc(x,y,7,0,Math.PI*2);ctx.fill();
ctx.strokeStyle='#fff';ctx.lineWidth=1.5;ctx.stroke();
ctx.fillStyle='#fbbf24';ctx.font='9px sans-serif';ctx.fillText((s.phase||'').replace('_',' ').substring(0,8),x-16,y-12)
});
const mk=document.getElementById('photoMarkers');
if(mk&&snaps.length){mk.innerHTML=snaps.slice(-5).map(s=>'<div class="marker-thumb"><img src="'+s.url+'" onerror="this.style.display=\\'none\\'"><div class="mlabel">'+(s.phase||'S').replace('_',' ')+'</div></div>').join('')}
if(snaps.length>=2){
const b=document.getElementById('beforePhoto'),a=document.getElementById('afterPhoto');
if(b)b.innerHTML='<img src="'+snaps[0].url+'" style="width:100%;height:100%;object-fit:cover">';
if(a)a.innerHTML='<img src="'+snaps[snaps.length-1].url+'" style="width:100%;height:100%;object-fit:cover">'
}
})();
</script></body></html>`;

export default function handler(req: VercelRequest, res: VercelResponse) {
  res.setHeader('Content-Type', 'text/html; charset=utf-8');
  res.send(HTML);
}
