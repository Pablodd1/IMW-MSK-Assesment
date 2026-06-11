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
    <ClinicalLayout title="3D Gait Analyzer" subtitle="Real-time skeleton tracking · Phase detection · Treadmill mode · Clinical metrics">
      {/* ── Data Stream ── */}
      <div class="data-stream" style="margin-bottom:12px;">
        <div class="stream-item"><span>CYCLE</span><span class="stream-value" id="dsPhase">—</span></div>
        <div class="stream-item"><span>STRIDE</span><span class="stream-value" id="dsStride">—</span></div>
        <div class="stream-item"><span>CADENCE</span><span class="stream-value" id="dsCadence">—</span></div>
        <div class="stream-item"><span>WIDTH</span><span class="stream-value" id="dsWidth">—</span></div>
        <div class="stream-item"><span>POSTURE</span><span class="stream-value" id="dsPronation">—</span></div>
        <div class="stream-item"><span>TILT</span><span class="stream-value" id="dsTilt">—</span></div>
        <div class="stream-item"><span>SYMM</span><span class="stream-value" id="dsSymmetry">—</span></div>
      </div>

      <section class="clinical-grid">
        {/* ── 3D SKELETON VIEWPORT ── */}
        <div class="clinical-card span-8 live">
          <h2>
            <span class="hud-label" style="display:inline-flex;align-items:center;margin-right:8px;">
              <span class="dot"></span>3D VIEWPORT
            </span>
            Gait Phase Skeleton Overlay
          </h2>
          <div class="skeleton-viewport">
            <canvas class="skeleton-canvas" id="gaitCanvas" width="900" height="620"></canvas>
            <div class="viewport-hud top-left">
              <div class="hud-label"><span class="dot"></span><span id="hudGaitPhase">MIDSTANCE</span></div>
            </div>
            <div class="viewport-hud top-right">
              <div class="hud-label">CYCLE <span id="hudGaitCycle">000</span></div>
            </div>
          </div>
          <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:10px;align-items:center;">
            <span class="pill active" style="border-color:#3b82f6;">⬤ Stance</span>
            <span class="pill active" style="border-color:#22d3ee;">⬤ Swing</span>
            <span class="pill" style="border-color:#60a5fa;">⬤ Upper</span>
            <label class="pill" style="margin-left:auto;">
              <input type="checkbox" id="treadmillMode" /> TREADMILL
            </label>
          </div>
        </div>

        {/* ── LIVE METRICS PANEL ── */}
        <div class="clinical-card span-4">
          <h2>Live Telemetry</h2>
          <div class={`metric live phase-${metrics.phase}`}>
            <span>Current Phase</span><strong id="phase">{metrics.phase.replace('_', ' ')}</strong>
          </div>
          <div class="metric"><span>Stride Length</span><strong id="stride">{metrics.strideLengthCm} cm</strong></div>
          <div class="metric"><span>Cadence</span><strong id="cadence">{metrics.cadenceSpm} spm</strong></div>
          <div class="metric"><span>Step Width</span><strong id="stepWidth">{metrics.stepWidthCm} cm</strong></div>
          <div class="metric"><span>Single Support</span><strong id="singleSupport">{metrics.singleSupportPct}%</strong></div>
          <div class="metric"><span>Double Support</span><strong id="doubleSupport">{metrics.doubleSupportPct}%</strong></div>
          <div class="metric"><span>Foot Posture</span><strong id="pronation">{metrics.pronation}</strong></div>
          <div class="metric"><span>Pelvic Tilt</span><strong id="pelvicTilt">{metrics.pelvicTiltDeg} deg</strong></div>
          <div class="metric"><span>Arm Swing Symmetry</span><strong id="armSwing">{metrics.armSwingSymmetryPct}%</strong></div>
        </div>

        {/* ── CLINICAL FINDINGS TABLE ── */}
        <div class="clinical-card span-12">
          <h2>Clinical Interpretation</h2>
          <table class="clinical-table">
            <thead><tr><th>Measure</th><th>Clinical Use</th><th>Flag</th></tr></thead>
            <tbody id="gaitFindings">
              <tr><td>Step Width</td><td>Frontal-plane balance and base of support.</td><td>Within screen</td></tr>
              <tr><td>Pronation/Supination</td><td>Foot collapse or rigid lateral loading from ankle keypoints.</td><td>Neutral</td></tr>
              <tr><td>Arm Swing</td><td>Reciprocal trunk rotation and neurologic symmetry.</td><td>Symmetric</td></tr>
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

    setText('phase', phase.replace('_',' '));
    setText('stride', stride+' cm');
    setText('cadence', cadence+' spm');
    setText('stepWidth', width+' cm');
    setText('singleSupport', (phase==='swing'?38:32)+'%');
    setText('doubleSupport', (phase==='heel_strike'||phase==='toe_off'?22:14)+'%');
    setText('pronation', pronation);
    setText('pelvicTilt', (Math.sin(tick/28)*4).toFixed(1)+' deg');
    setText('armSwing', Math.round(86+Math.abs(Math.sin(tick/20))*12)+'%');

    // Data stream
    setText('dsPhase', phase.replace('_',' '));
    setText('dsStride', stride+'cm');
    setText('dsCadence', cadence+'spm');
    setText('dsWidth', width+'cm');
    setText('dsPronation', pronation);
    setText('dsTilt', (Math.sin(tick/28)*4).toFixed(1)+'°');
    setText('dsSymmetry', Math.round(86+Math.abs(Math.sin(tick/20))*12)+'%');

    // HUD
    const hudPhase = document.getElementById('hudGaitPhase');
    if (hudPhase) hudPhase.textContent = phase.replace('_',' ').toUpperCase();
    setText('hudGaitCycle', String(tick).padStart(3,'0'));

    ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx.fillStyle='#020617';
    ctx.fillRect(0,0,canvas.width,canvas.height);

    // Grid
    ctx.strokeStyle='rgba(96,165,250,.08)';
    ctx.lineWidth=1;
    for(let x=0;x<canvas.width;x+=42){
      ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,canvas.height);ctx.stroke();
    }

    // Bones with glow
    pairs.forEach(([a,b])=>{
      if(!p[a]||!p[b]) return;
      const lower = a>=11 || b>=11;
      ctx.strokeStyle = lower ? (stanceLeft ? '#3b82f6' : '#22d3ee') : '#60a5fa';
      ctx.lineWidth = lower ? 10 : 7;
      ctx.lineCap='round';
      ctx.shadowColor = lower ? 'rgba(59,130,246,0.7)' : 'rgba(96,165,250,0.5)';
      ctx.shadowBlur = 10;
      ctx.beginPath();
      ctx.moveTo(p[a][0]*canvas.width,p[a][1]*canvas.height);
      ctx.lineTo(p[b][0]*canvas.width,p[b][1]*canvas.height);
      ctx.stroke();
    });
    ctx.shadowBlur = 0;

    // Joints
    Object.entries(p).forEach(([id,pt])=>{
      const isLower = Number(id)>=11;
      ctx.shadowColor = isLower ? 'rgba(59,130,246,0.8)' : 'rgba(96,165,250,0.6)';
      ctx.shadowBlur = 8;
      ctx.beginPath();
      ctx.arc(pt[0]*canvas.width,pt[1]*canvas.height,10,0,Math.PI*2);
      ctx.fillStyle = isLower ? (stanceLeft ? '#3b82f6' : '#22d3ee') : '#60a5fa';
      ctx.fill();
    });
    ctx.shadowBlur = 0;
    requestAnimationFrame(draw);
  }
  draw();
})();
`;
