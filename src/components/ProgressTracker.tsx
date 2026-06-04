import { ClinicalLayout } from './clinicalStyles.js';

export function ProgressTracker() {
  return (
    <ClinicalLayout title="Progress Tracking" subtitle="Session history, trend charts, before/after comparison, goals, and exportable progress data.">
      <section class="clinical-grid">
        <div class="clinical-card span-8">
          <h2>Trend Chart</h2>
          <canvas class="skeleton-canvas" id="progressChart" width="900" height="420"></canvas>
        </div>
        <div class="clinical-card span-4">
          <h2>Before / After</h2>
          <div class="metric"><span>FMS total</span><strong>11 {'->'} 16 (+45%)</strong></div>
          <div class="metric"><span>Lumbar flexion</span><strong>45 {'->'} 58 deg (+29%)</strong></div>
          <div class="metric"><span>Gait cadence</span><strong>92 {'->'} 108 spm (+17%)</strong></div>
          <div class="metric"><span>Pain</span><strong>7 {'->'} 3 (-57%)</strong></div>
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
})();
`;
