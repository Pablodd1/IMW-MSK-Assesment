import type { GaitMetrics } from '../utils/clinical.js';
import { ClinicalLayout } from './clinicalStyles.js';

const initialMetrics: GaitMetrics = {
  phase: 'midstance',
  strideLengthCm: 68,
  cadenceSpm: 104,
  stepWidthCm: 9,
  singleSupportPct: 32,
  doubleSupportPct: 18,
  pronation: 'neutral',
  pelvicTiltDeg: 3.2,
  armSwingSymmetryPct: 91,
  stanceSide: 'left',
};

export function GaitAnalyzer({ metrics = initialMetrics }: { metrics?: GaitMetrics }) {
  return (
    <ClinicalLayout title="Gait Analyzer" subtitle="Real-time gait cycle, treadmill mode, pelvic and foot mechanics.">
      <section class="clinical-grid">
        <div class="clinical-card span-8">
          <h2>3D Skeleton Gait Phase Overlay</h2>
          <canvas class="skeleton-canvas" id="gaitCanvas" width="900" height="620"></canvas>
          <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:10px">
            <span class="pill" style="border-color:#3b82f6">Blue stance</span>
            <span class="pill" style="border-color:#22c55e">Green swing</span>
            <label class="pill"><input type="checkbox" id="treadmillMode" /> Treadmill mode</label>
          </div>
        </div>
        <div class="clinical-card span-4">
          <h2>Live Metrics</h2>
          <div class={`metric phase-${metrics.phase}`}><span>Current phase</span><strong id="phase">{metrics.phase.replace('_', ' ')}</strong></div>
          <div class="metric"><span>Stride length</span><strong id="stride">{metrics.strideLengthCm} cm</strong></div>
          <div class="metric"><span>Cadence</span><strong id="cadence">{metrics.cadenceSpm} spm</strong></div>
          <div class="metric"><span>Step width</span><strong id="stepWidth">{metrics.stepWidthCm} cm</strong></div>
          <div class="metric"><span>Single support</span><strong id="singleSupport">{metrics.singleSupportPct}%</strong></div>
          <div class="metric"><span>Double support</span><strong id="doubleSupport">{metrics.doubleSupportPct}%</strong></div>
          <div class="metric"><span>Foot posture</span><strong id="pronation">{metrics.pronation}</strong></div>
          <div class="metric"><span>Pelvic tilt</span><strong id="pelvicTilt">{metrics.pelvicTiltDeg} deg</strong></div>
          <div class="metric"><span>Arm swing symmetry</span><strong id="armSwing">{metrics.armSwingSymmetryPct}%</strong></div>
        </div>
        <div class="clinical-card span-12">
          <h2>Clinical Interpretation</h2>
          <table class="clinical-table">
            <thead><tr><th>Measure</th><th>Clinical use</th><th>Flag</th></tr></thead>
            <tbody id="gaitFindings">
              <tr><td>Step width</td><td>Frontal-plane balance and base of support.</td><td>Within screen</td></tr>
              <tr><td>Pronation/supination</td><td>Foot collapse or rigid lateral loading from ankle keypoints.</td><td>Neutral</td></tr>
              <tr><td>Arm swing</td><td>Reciprocal trunk rotation and neurologic symmetry.</td><td>Symmetric</td></tr>
            </tbody>
          </table>
        </div>
      </section>
      <script dangerouslySetInnerHTML={{ __html: gaitScript }} />
    </ClinicalLayout>
  );
}

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
    return {
      5:[.42+sway,.30],6:[.58+sway,.30],7:[.35+sway,.47+stride*.04],8:[.65+sway,.47-stride*.04],9:[.31+sway,.64+stride*.06],10:[.69+sway,.64-stride*.06],
      11:[.44+sway,.55],12:[.56+sway,.55],13:[.41+sway+stride*.04,.73],14:[.59+sway-stride*.04,.73],15:[.36+sway+stride*.10,.91],16:[.64+sway-stride*.10,.91]
    };
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
    setText('phase', phase.replace('_',' ')); setText('stride', stride+' cm'); setText('cadence', cadence+' spm'); setText('stepWidth', width+' cm');
    setText('singleSupport', (phase==='swing'?38:32)+'%'); setText('doubleSupport', (phase==='heel_strike'||phase==='toe_off'?22:14)+'%');
    setText('pronation', pronation); setText('pelvicTilt', (Math.sin(tick/28)*4).toFixed(1)+' deg'); setText('armSwing', Math.round(86+Math.abs(Math.sin(tick/20))*12)+'%');
    ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx.fillStyle='#050a16'; ctx.fillRect(0,0,canvas.width,canvas.height);
    ctx.strokeStyle='rgba(96,165,250,.13)'; ctx.lineWidth=1;
    for(let x=0;x<canvas.width;x+=42){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,canvas.height);ctx.stroke();}
    pairs.forEach(([a,b])=>{
      if(!p[a]||!p[b]) return;
      const lower = a>=11 || b>=11;
      ctx.strokeStyle = lower ? (stanceLeft ? '#3b82f6' : '#22c55e') : '#60a5fa';
      ctx.lineWidth = lower ? 9 : 6; ctx.lineCap='round';
      ctx.beginPath(); ctx.moveTo(p[a][0]*canvas.width,p[a][1]*canvas.height); ctx.lineTo(p[b][0]*canvas.width,p[b][1]*canvas.height); ctx.stroke();
    });
    Object.entries(p).forEach(([id,pt])=>{ ctx.beginPath(); ctx.arc(pt[0]*canvas.width,pt[1]*canvas.height,9,0,Math.PI*2); ctx.fillStyle = Number(id)>=11 ? (stanceLeft ? '#3b82f6' : '#22c55e') : '#60a5fa'; ctx.fill(); });
    requestAnimationFrame(draw);
  }
  draw();
})();
`;

