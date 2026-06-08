import { ClinicalLayout } from './clinicalStyles.js';

export function ProgressTracker() {
  return (
    <ClinicalLayout title="Progress Tracking" subtitle="Session history, trend charts, before/after comparison, goals, and exportable progress data.">
      <section class="clinical-grid">
        <div class="clinical-card span-8">
          <h2>Trend Chart with Photo Markers</h2>
          <canvas class="skeleton-canvas" id="progressChart" width="900" height="420"></canvas>
          <div id="progressPhotoMarkers" style="display:flex; gap:6px; flex-wrap:wrap; margin-top:10px; min-height:50px;"></div>
        </div>
        <div class="clinical-card span-4">
          <h2>Before / After</h2>
          <div class="metric"><span>FMS total</span><strong>11 {'->'} 16 (+45%)</strong></div>
          <div class="metric"><span>Lumbar flexion</span><strong>45 {'->'} 58 deg (+29%)</strong></div>
          <div class="metric"><span>Gait cadence</span><strong>92 {'->'} 108 spm (+17%)</strong></div>
          <div class="metric"><span>Pain</span><strong>7 {'->'} 3 (-57%)</strong></div>
          <h2 style="margin-top:16px">Movement Photos</h2>
          <div id="beforeAfterPhotos" style="display:flex; gap:8px;">
            <div style="flex:1; border-radius:6px; overflow:hidden; border:2px solid #ef4444; background:#1e293b;">
              <div style="padding:4px 8px; background:#ef4444; color:#fff; font-size:11px; font-weight:600;">BEFORE · Session 1</div>
              <div id="beforePhoto" style="aspect-ratio:4/3; display:flex; align-items:center; justify-content:center; color:#64748b; font-size:12px;">
                Capture during gait
              </div>
            </div>
            <div style="flex:1; border-radius:6px; overflow:hidden; border:2px solid #22c55e; background:#1e293b;">
              <div style="padding:4px 8px; background:#22c55e; color:#000; font-size:11px; font-weight:600;">AFTER · Session 3</div>
              <div id="afterPhoto" style="aspect-ratio:4/3; display:flex; align-items:center; justify-content:center; color:#64748b; font-size:12px;">
                Capture during gait
              </div>
            </div>
          </div>
          <h2 style="margin-top:16px">Goals</h2>
          <div class="heat-row"><span>Pain</span><div class="bar"><span style="width:76%"></span></div><strong>76%</strong></div>
          <div class="heat-row"><span>ROM</span><div class="bar"><span style="width:88%"></span></div><strong>88%</strong></div>
          <div class="heat-row"><span>HEP</span><div class="bar"><span style="width:64%"></span></div><strong>64%</strong></div>
          <button class="clinical-btn" onclick="window.print()" style="width:100%;margin-top:12px">Export Progress Report</button>
        </div>
        <div class="clinical-card span-12">
          <h2>Session History</h2>
          <table class="clinical-table">
            <thead><tr><th>Date</th><th>FMS</th><th>ROM</th><th>Gait</th><th>Pain</th><th>Notes</th></tr></thead>
            <tbody>
              <tr><td>2026-05-06</td><td>11/21</td><td>45 deg</td><td>92 spm</td><td>7/10</td><td>Initial evaluation</td></tr>
              <tr><td>2026-05-20</td><td>14/21</td><td>52 deg</td><td>101 spm</td><td>5/10</td><td>Improved tolerance</td></tr>
              <tr><td>2026-06-03</td><td>16/21</td><td>58 deg</td><td>108 spm</td><td>3/10</td><td>Progressing to loaded control</td></tr>
            </tbody>
          </table>
        </div>
      </section>
      <script dangerouslySetInnerHTML={{ __html: progressScript }} />
    </ClinicalLayout>
  );
}

const progressScript = `
(function(){
  const canvas=document.getElementById('progressChart'),ctx=canvas.getContext('2d');
  const series={FMS:[11,14,16],ROM:[45,52,58],Gait:[92,101,108],Pain:[7,5,3]};
  const colors={FMS:'#f59e0b',ROM:'#60a5fa',Gait:'#22c55e',Pain:'#ef4444'};
  ctx.fillStyle='#050a16'; ctx.fillRect(0,0,canvas.width,canvas.height);
  ctx.strokeStyle='rgba(148,163,184,.14)'; ctx.lineWidth=1;
  for(let y=40;y<canvas.height-30;y+=48){ctx.beginPath();ctx.moveTo(48,y);ctx.lineTo(canvas.width-24,y);ctx.stroke();}
  Object.entries(series).forEach(([name,vals],idx)=>{
    const max=Math.max(...vals),min=Math.min(...vals);
    ctx.strokeStyle=colors[name]; ctx.lineWidth=4; ctx.beginPath();
    vals.forEach((v,i)=>{ const x=70+i*(canvas.width-140)/(vals.length-1); const y=canvas.height-60-((v-min)/(max-min||1))*(canvas.height-130)-idx*5; if(i===0)ctx.moveTo(x,y); else ctx.lineTo(x,y); ctx.fillStyle=colors[name]; ctx.fillRect(x-4,y-4,8,8); });
    ctx.stroke(); ctx.fillStyle=colors[name]; ctx.font='16px sans-serif'; ctx.fillText(name,70+idx*110,28);
 });

 // Draw photo snapshot markers (gold dots on timeline)
 const photoSnaps = window.__imwPhotoSnapshots || [];
 photoSnaps.forEach((snap, i) => {
   const t = (snap.sessionIndex || i) / Math.max(photoSnaps.length - 1 || 1, 1);
   const x = 70 + t * (canvas.width - 140);
   const y = canvas.height - 50 - Math.sin(t * Math.PI) * 30;
    
   // Gold marker dot
   ctx.fillStyle = '#f59e0b';
   ctx.beginPath();
   ctx.arc(x, y, 7, 0, Math.PI * 2);
   ctx.fill();
   ctx.strokeStyle = '#fff';
   ctx.lineWidth = 1.5;
   ctx.stroke();
    
   // Phase label
   ctx.fillStyle = '#fbbf24';
   ctx.font = '9px sans-serif';
   ctx.fillText(snap.phase?.replace('_',' ').substring(0, 8) || '', x - 16, y - 12);
 });

 // Draw photo thumbnails below chart
 const markersDiv = document.getElementById('progressPhotoMarkers');
 if (markersDiv && photoSnaps.length > 0) {
   markersDiv.innerHTML = photoSnaps.slice(-5).map((s, i) =>
     '<div style="position:relative; width:70px; height:52px; border-radius:3px; overflow:hidden; border:1px solid #334155;">' +
       '<img src="' + s.dataUrl + '" style="width:100%; height:100%; object-fit:cover;" onerror="this.style.display=\\'none\\'" />' +
       '<div style="position:absolute; bottom:0; left:0; right:0; background:rgba(0,0,0,0.7); font-size:8px; color:#fbbf24; padding:1px 3px; text-align:center;">' +
         (s.phase || 'S' + (i+1)).replace('_',' ') +
       '</div>' +
     '</div>'
   ).join('');
 }

 // ============================================================
 // Before/After photo loading (from MovementCapture)
 // ============================================================
 function loadBeforeAfterPhotos() {
   const snaps = window.__imwPhotoSnapshots || [];
   if (snaps.length >= 2) {
     const beforeEl = document.getElementById('beforePhoto');
     const afterEl = document.getElementById('afterPhoto');
     if (beforeEl) {
       beforeEl.innerHTML = '<img src="' + snaps[0].dataUrl + '" style="width:100%; height:100%; object-fit:cover;" />';
     }
     if (afterEl) {
       afterEl.innerHTML = '<img src="' + snaps[snaps.length - 1].dataUrl + '" style="width:100%; height:100%; object-fit:cover;" />';
     }
   }
 }
 setTimeout(loadBeforeAfterPhotos, 500);
})();
`;
