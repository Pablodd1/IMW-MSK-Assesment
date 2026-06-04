import { ClinicalLayout } from './clinicalStyles.js';
import { MUSCLE_GROUPS, type MuscleGrade } from '../utils/clinical.js';

const grades: MuscleGrade[] = [
  { joint: 'Shoulder', movement: 'abduction', muscleGroup: 'deltoid/supraspinatus', grade: 4, forceEstimateN: 82, side: 'left' },
  { joint: 'Shoulder', movement: 'abduction', muscleGroup: 'deltoid/supraspinatus', grade: 5, forceEstimateN: 104, side: 'right' },
  { joint: 'Hip', movement: 'abduction', muscleGroup: 'gluteus medius', grade: 3, forceEstimateN: 61, side: 'left' },
  { joint: 'Knee', movement: 'extension', muscleGroup: 'quadriceps', grade: 4, forceEstimateN: 95, side: 'right' },
  { joint: 'Ankle', movement: 'dorsiflexion', muscleGroup: 'tibialis anterior', grade: 4, forceEstimateN: 70, side: 'bilateral' },
];

export function MuscleAssessment({ muscleGrades = grades }: { muscleGrades?: MuscleGrade[] }) {
  const fmaUpper = Math.min(66, muscleGrades.filter((g) => ['Shoulder', 'Elbow'].includes(g.joint)).reduce((s, g) => s + g.grade, 0) * 3);
  const fmaLower = Math.min(34, muscleGrades.filter((g) => ['Hip', 'Knee', 'Ankle'].includes(g.joint)).reduce((s, g) => s + g.grade, 0) * 2);
  return (
    <ClinicalLayout title="Muscle Assessment" subtitle="Clinical MMT 0-5 grading, Fugl-Meyer scoring, and keypoint-derived force estimates.">
      <section class="clinical-grid">
        <div class="clinical-card span-7">
          <h2>Muscle Heat Map Overlay</h2>
          <canvas class="skeleton-canvas" id="muscleCanvas" width="900" height="620"></canvas>
          <p class="muted" style="font-size:.76rem">Red indicates weak groups, green indicates strong groups. Grades are screening estimates and should be confirmed with clinician-applied resistance.</p>
        </div>
        <div class="clinical-card span-5">
          <h2>Fugl-Meyer Assessment</h2>
          <div class="metric"><span>Upper extremity</span><strong>{fmaUpper}/66</strong></div>
          <div class="metric"><span>Lower extremity</span><strong>{fmaLower}/34</strong></div>
          <div class="metric"><span>Total motor score</span><strong>{fmaUpper + fmaLower}/100</strong></div>
          <h2 style="margin-top:16px">Dynamometer Simulation</h2>
          {muscleGrades.map((grade) => (
            <div class="heat-row">
              <span>{grade.side} {grade.joint}</span>
              <div class="bar"><span style={`width:${grade.grade * 20}%`}></span></div>
              <strong style={grade.grade < 3 ? 'color:var(--red)' : grade.grade < 5 ? 'color:var(--gold)' : 'color:var(--green)'}>{grade.grade}/5</strong>
            </div>
          ))}
        </div>
        <div class="clinical-card span-12">
          <h2>Manual Muscle Testing Matrix</h2>
          <table class="clinical-table">
            <thead><tr><th>Joint</th><th>Movements</th><th>Muscle groups</th><th>MMT grade meaning</th></tr></thead>
            <tbody>
              {MUSCLE_GROUPS.map((row) => (
                <tr>
                  <td>{row.joint}</td>
                  <td>{row.movements.join(', ')}</td>
                  <td>{row.groups.join(', ')}</td>
                  <td>0 none, 1 trace, 2 gravity eliminated, 3 against gravity, 4 resistance, 5 normal</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
      <script dangerouslySetInnerHTML={{ __html: muscleScript }} />
    </ClinicalLayout>
  );
}

const muscleScript = `
(function(){
  const canvas=document.getElementById('muscleCanvas'),ctx=canvas.getContext('2d');
  const grades={leftShoulder:4,rightShoulder:5,leftHip:3,rightHip:4,leftKnee:4,rightKnee:4,leftAnkle:4,rightAnkle:4};
  const pts={head:[.50,.14],leftShoulder:[.39,.30],rightShoulder:[.61,.30],leftElbow:[.31,.47],rightElbow:[.69,.47],leftWrist:[.27,.63],rightWrist:[.73,.63],leftHip:[.43,.55],rightHip:[.57,.55],leftKnee:[.40,.74],rightKnee:[.60,.74],leftAnkle:[.38,.92],rightAnkle:[.62,.92]};
  const bones=[['leftShoulder','rightShoulder'],['leftShoulder','leftElbow'],['leftElbow','leftWrist'],['rightShoulder','rightElbow'],['rightElbow','rightWrist'],['leftShoulder','leftHip'],['rightShoulder','rightHip'],['leftHip','rightHip'],['leftHip','leftKnee'],['leftKnee','leftAnkle'],['rightHip','rightKnee'],['rightKnee','rightAnkle']];
  function color(g){ const r=Math.round(239-(g/5)*205), gr=Math.round(68+(g/5)*153); return 'rgb('+r+','+gr+',80)'; }
  function draw(){
    ctx.clearRect(0,0,canvas.width,canvas.height); ctx.fillStyle='#050a16'; ctx.fillRect(0,0,canvas.width,canvas.height);
    bones.forEach(([a,b])=>{ const pa=pts[a],pb=pts[b],g=Math.min(grades[a]||4,grades[b]||4); ctx.strokeStyle=color(g); ctx.lineWidth=11; ctx.lineCap='round'; ctx.beginPath(); ctx.moveTo(pa[0]*canvas.width,pa[1]*canvas.height); ctx.lineTo(pb[0]*canvas.width,pb[1]*canvas.height); ctx.stroke(); });
    Object.entries(pts).forEach(([name,p])=>{ const g=grades[name]||4; ctx.beginPath(); ctx.arc(p[0]*canvas.width,p[1]*canvas.height,name==='head'?24:11,0,Math.PI*2); ctx.fillStyle=name==='head'?'rgba(59,130,246,.38)':color(g); ctx.fill(); ctx.strokeStyle='#bfdbfe'; ctx.lineWidth=1; ctx.stroke(); });
  }
  draw();
})();
`;

