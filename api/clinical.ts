import type { VercelRequest, VercelResponse } from '@vercel/node';

const PAGES: Record<string, { title: string; subtitle: string; body: string }> = {
  muscle: {
    title: '💪 Muscle Assessment',
    subtitle: 'Manual muscle testing grades · Fugl-Meyer scoring · ROM normative comparisons',
    body: `<div class="grid">
<div class="card s6"><h2>Muscle Strength Grades</h2><table><thead><tr><th>Muscle Group</th><th>Left</th><th>Right</th><th>Normative</th><th>Flag</th></tr></thead><tbody>
<tr><td>Hip Flexors</td><td>4/5</td><td>5/5</td><td>5/5</td><td style="color:#fbbf24">⚠ Mild L</td></tr>
<tr><td>Knee Extensors</td><td>5/5</td><td>5/5</td><td>5/5</td><td style="color:#4ade80">✓</td></tr>
<tr><td>Ankle Dorsiflexors</td><td>4/5</td><td>4+/5</td><td>5/5</td><td style="color:#fbbf24">⚠ Bilateral</td></tr>
<tr><td>Ankle Plantarflexors</td><td>5/5</td><td>5/5</td><td>5/5</td><td style="color:#4ade80">✓</td></tr>
<tr><td>Shoulder Abductors</td><td>5/5</td><td>5/5</td><td>5/5</td><td style="color:#4ade80">✓</td></tr>
<tr><td>Elbow Flexors</td><td>5/5</td><td>5/5</td><td>5/5</td><td style="color:#4ade80">✓</td></tr>
<tr><td>Trunk Flexors</td><td>4/5</td><td>—</td><td>5/5</td><td style="color:#fbbf24">⚠</td></tr>
</tbody></table></div>
<div class="card s3"><h2>Fugl-Meyer Score</h2>
<div class="metric"><span>Upper Extremity</span><strong style="color:#4ade80">60/66</strong></div>
<div class="metric"><span>Lower Extremity</span><strong style="color:#fbbf24">28/34</strong></div>
<div class="metric"><span>Balance</span><strong>12/14</strong></div>
<div class="metric"><span>Total</span><strong style="font-size:1.2rem;color:var(--gold)">100/114</strong></div>
<h2 style="margin-top:16px">ROM Comparison</h2>
<div class="metric"><span>Lumbar Flexion</span><strong>58° <small style="color:var(--muted)">/ 60-80°</small></strong></div>
<div class="metric"><span>Hip Flexion L</span><strong style="color:#fbbf24">105° <small style="color:var(--muted)">/ 120°</small></strong></div>
<div class="metric"><span>Hip Flexion R</span><strong>118° <small style="color:var(--muted)">/ 120°</small></strong></div>
<div class="metric"><span>Knee Flexion L</span><strong>130° <small style="color:var(--muted)">/ 135°</small></strong></div>
<div class="metric"><span>Knee Flexion R</span><strong>135° <small style="color:var(--muted)">/ 135°</small></strong></div></div>
<div class="card s3"><h2>Symmetry Index</h2>
<div class="metric"><span>Hip Flexion</span><strong style="color:#fbbf24">89%</strong></div>
<div class="metric"><span>Knee Flexion</span><strong style="color:#4ade80">96%</strong></div>
<div class="metric"><span>Ankle DF</span><strong style="color:#fbbf24">85%</strong></div>
<div class="metric"><span>Overall</span><strong>90%</strong></div>
<h2 style="margin-top:16px">Recommendations</h2>
<div style="font-size:.73rem;color:var(--muted);line-height:1.6">
• Target left hip flexor strengthening<br>
• Bilateral ankle mobility work<br>
• Core stabilization for trunk control<br>
• Re-assess in 2 weeks
</div></div>
</div>`
  },
  clinical: {
    title: '🔬 Clinical Tests',
    subtitle: 'Special tests · Movement screening · Outcome measures',
    body: `<div class="grid">
<div class="card s6"><h2>Special Tests</h2><table><thead><tr><th>Test</th><th>Result</th><th>Interpretation</th></tr></thead><tbody>
<tr><td>Thomas Test (L)</td><td style="color:#fbbf24">Positive</td><td>Hip flexor tightness</td></tr>
<tr><td>Thomas Test (R)</td><td style="color:#4ade80">Negative</td><td>Normal hip flexor length</td></tr>
<tr><td>Ober Test (L)</td><td style="color:#fbbf24">Positive</td><td>IT band/TFL tightness</td></tr>
<tr><td>Ober Test (R)</td><td style="color:#4ade80">Negative</td><td>Normal</td></tr>
<tr><td>Straight Leg Raise (L)</td><td style="color:#4ade80">Negative</td><td>No neural tension</td></tr>
<tr><td>Straight Leg Raise (R)</td><td style="color:#4ade80">Negative</td><td>No neural tension</td></tr>
<tr><td>Trendelenburg</td><td style="color:#fbbf24">Mild + L</td><td>Gluteus medius weakness</td></tr>
<tr><td>Slump Test</td><td style="color:#4ade80">Negative</td><td>No adverse neural dynamics</td></tr>
</tbody></table></div>
<div class="card s3"><h2>Movement Screening</h2>
<div class="metric"><span>Deep Squat</span><strong style="color:#fbbf24">2/3</strong></div>
<div class="metric"><span>Hurdle Step L</span><strong>3/3</strong></div>
<div class="metric"><span>Hurdle Step R</span><strong>3/3</strong></div>
<div class="metric"><span>Inline Lunge L</span><strong style="color:#fbbf24">2/3</strong></div>
<div class="metric"><span>Inline Lunge R</span><strong>3/3</strong></div>
<div class="metric"><span>Shoulder Mobility</span><strong>3/3</strong></div>
<div class="metric"><span>Active SLR L</span><strong style="color:#fbbf24">2/3</strong></div>
<div class="metric"><span>Active SLR R</span><strong>3/3</strong></div></div>
<div class="card s3"><h2>Outcome Measures</h2>
<div class="metric"><span>ODI Score</span><strong>22%</strong></div>
<div class="metric"><span>LEFS</span><strong>68/80</strong></div>
<div class="metric"><span>NPRS (Pain)</span><strong style="color:#fbbf24">3/10</strong></div>
<div class="metric"><span>PSFS</span><strong>6.2/10</strong></div>
<div class="metric"><span>TUG Test</span><strong>8.2 sec</strong></div>
<h2 style="margin-top:16px">ICD-10 Codes</h2>
<div style="display:flex;flex-wrap:wrap;gap:4px">
<span style="background:#0f172a;color:var(--blue);padding:2px 6px;border-radius:3px;font-size:.68rem">M62.81</span>
<span style="background:#0f172a;color:var(--blue);padding:2px 6px;border-radius:3px;font-size:.68rem">M25.552</span>
<span style="background:#0f172a;color:var(--blue);padding:2px 6px;border-radius:3px;font-size:.68rem">R26.2</span>
</div></div>
</div>`
  },
  reports: {
    title: '📋 Report Generator',
    subtitle: 'Clinical summaries · SOAP notes · ICD-10/CPT coding · Exportable reports',
    body: `<div class="grid">
<div class="card s8"><h2>Patient Report Preview</h2>
<div style="background:#050a16;border-radius:6px;padding:16px;font-size:.78rem;line-height:1.6">
<div style="margin-bottom:10px;padding-bottom:10px;border-bottom:1px solid var(--border)">
<strong style="color:var(--gold)">PATIENT:</strong> John D. · DOB: 1985-03-15 · ID: PT-2026-0042<br>
<strong style="color:var(--gold)">CLINICIAN:</strong> Dr. Sarah Chen, PT, DPT · License: PT-28491<br>
<strong style="color:var(--gold)">DATE:</strong> June 8, 2026 · Assessment #3
</div>
<div style="margin-bottom:8px"><strong style="color:#f87171">SUBJECTIVE:</strong><br>
Patient reports 3/10 low back pain (down from 7/10 initial). Pain localized to L4-L5 region. Aggravated by prolonged sitting (>45 min) and forward bending. Relieved by walking and supine positioning. Reports improved sleep quality. Functional goal: return to recreational basketball.</div>
<div style="margin-bottom:8px"><strong style="color:#60a5fa">OBJECTIVE:</strong><br>
Lumbar flexion AROM: 58° (improved from 45°). Hip flexor strength: 4/5 L, 5/5 R. Gait cadence: 108 spm (improved from 92). Thomas test: positive L, negative R. FMS total: 16/21.</div>
<div style="margin-bottom:8px"><strong style="color:#fbbf24">ASSESSMENT:</strong><br>
Continued improvement in lumbar mobility and gait mechanics. Persistent left hip flexor tightness and mild gluteus medius weakness contributing to Trendelenburg gait pattern. Progressing toward functional goals.</div>
<div style="margin-bottom:8px"><strong style="color:#22c55e">PLAN:</strong><br>
1. Continue PT 2x/week x 4 weeks<br>
2. Progress core stabilization to dynamic control<br>
3. Initiate sport-specific training protocol<br>
4. HEP: clamshells, bridges, dead bugs — 3x10 daily<br>
5. Re-evaluate in 2 weeks</div>
<div style="display:flex;gap:8px;margin-top:10px">
<span style="background:#0f172a;color:var(--blue);padding:2px 6px;border-radius:3px;font-size:.68rem">M54.5</span>
<span style="background:#0f172a;color:var(--blue);padding:2px 6px;border-radius:3px;font-size:.68rem">M62.81</span>
<span style="background:#0f172a;color:var(--green);padding:2px 6px;border-radius:3px;font-size:.68rem">97110</span>
<span style="background:#0f172a;color:var(--green);padding:2px 6px;border-radius:3px;font-size:.68rem">97112</span>
<span style="background:#0f172a;color:var(--green);padding:2px 6px;border-radius:3px;font-size:.68rem">97140</span>
</div>
</div></div>
<div class="card s4"><h2>Quick Actions</h2>
<button class="btn" style="width:100%;margin-bottom:6px">📄 Generate Full Report</button>
<button class="btn" style="width:100%;margin-bottom:6px;background:var(--green)">📧 Email to Patient</button>
<button class="btn" style="width:100%;margin-bottom:6px;background:#7c3aed">📋 Copy SOAP Note</button>
<button class="btn" style="width:100%;margin-bottom:6px">🖨 Print Report</button>
<h2 style="margin-top:16px">Billing Codes</h2>
<div class="metric"><span>97110</span><strong>Therapeutic Exercise</strong></div>
<div class="metric"><span>97112</span><strong>Neuromuscular Re-ed</strong></div>
<div class="metric"><span>97140</span><strong>Manual Therapy</strong></div>
<div class="metric"><span>97530</span><strong>Therapeutic Activities</strong></div>
<h2 style="margin-top:16px">Diagnosis</h2>
<div class="metric"><span>M54.5</span><strong>Low back pain</strong></div>
<div class="metric"><span>M62.81</span><strong>Muscle weakness</strong></div>
<div class="metric"><span>R26.2</span><strong>Difficulty walking</strong></div></div>
</div>`
  },
  exercises: {
    title: '💊 Exercise Prescriptions',
    subtitle: 'Therapeutic exercises · HEP programs · CPT-coded · Progress tracking',
    body: `<div class="grid">
<div class="card s6"><h2>Current Prescriptions</h2>
<div style="display:flex;flex-direction:column;gap:8px">
<div style="background:#1a2a4a40;border:1px solid #1a2a4a;border-radius:6px;padding:10px"><h3 style="color:#4ade80;font-size:.75rem;margin-bottom:4px">Clamshells</h3><p style="color:var(--muted);font-size:.7rem">Gluteus medius strengthening · Sidelying hip abduction</p><div style="color:#facc15;font-size:.68rem;margin-top:4px">3 sets × 12 reps · Daily · CPT 97110</div></div>
<div style="background:#1a2a4a40;border:1px solid #1a2a4a;border-radius:6px;padding:10px"><h3 style="color:#4ade80;font-size:.75rem;margin-bottom:4px">Bridges</h3><p style="color:var(--muted);font-size:.7rem">Gluteal activation · Supine hip extension</p><div style="color:#facc15;font-size:.68rem;margin-top:4px">3 sets × 15 reps · Daily · CPT 97110</div></div>
<div style="background:#1a2a4a40;border:1px solid #1a2a4a;border-radius:6px;padding:10px"><h3 style="color:#4ade80;font-size:.75rem;margin-bottom:4px">Dead Bugs</h3><p style="color:var(--muted);font-size:.7rem">Core stabilization · Supine alternating limb</p><div style="color:#facc15;font-size:.68rem;margin-top:4px">3 sets × 8 reps each side · Daily · CPT 97110</div></div>
<div style="background:#1a2a4a40;border:1px solid #1a2a4a;border-radius:6px;padding:10px"><h3 style="color:#4ade80;font-size:.75rem;margin-bottom:4px">Cat-Cow</h3><p style="color:var(--muted);font-size:.7rem">Spinal mobility · Quadruped flexion/extension</p><div style="color:#facc15;font-size:.68rem;margin-top:4px">2 sets × 10 reps · Daily · CPT 97112</div></div>
</div></div>
<div class="card s3"><h2>Progress</h2>
<div class="metric"><span>Adherence</span><strong style="color:#4ade80">82%</strong></div>
<div class="metric"><span>Sessions completed</span><strong>8/12</strong></div>
<div class="metric"><span>Pain with HEP</span><strong style="color:#4ade80">1.5/10</strong></div>
<div class="metric"><span>Difficulty</span><strong>Moderate</strong></div>
<h2 style="margin-top:16px">Add Exercise</h2>
<select style="width:100%;background:#1a2a4a;border:1px solid var(--border);border-radius:4px;padding:6px;color:var(--text);font-size:.73rem;margin-bottom:6px"><option>Select exercise...</option><option>Quad Sets</option><option>Straight Leg Raises</option><option>Wall Slides</option><option>Pendulum (Codman)</option><option>Prone Press-ups</option></select>
<button class="btn" style="width:100%">+ Add to Program</button></div>
<div class="card s3"><h2>HEP Instructions</h2>
<div style="font-size:.73rem;color:var(--muted);line-height:1.6">
<strong style="color:var(--text)">Daily Routine:</strong><br>
1. Clamshells — morning<br>
2. Bridges — morning<br>
3. Dead Bugs — afternoon<br>
4. Cat-Cow — evening<br><br>
<strong style="color:var(--text)">Precautions:</strong><br>
• Stop if pain > 3/10<br>
• Avoid loaded flexion<br>
• Use mirror for form check<br><br>
<strong style="color:var(--text)">Progression:</strong><br>
Add resistance band at week 3
</div></div>
</div>`
  }
};

const SHARED_CSS = `*{margin:0;padding:0;box-sizing:border-box}
:root{--navy:#0a1628;--panel:#0d1b31;--border:#1d3355;--gold:#f59e0b;--text:#e2e8f0;--muted:#94a3b8;--blue:#60a5fa;--green:#22c55e;--red:#ef4444}
body{background:var(--navy);color:var(--text);font-family:Inter,sans-serif}
.layout{padding:20px;max-width:1440px;margin:0 auto}
h1{font-size:1.3rem;color:var(--gold);margin-bottom:4px}
h2{font-size:.85rem;color:var(--blue);margin-bottom:10px}
.subtitle{color:var(--muted);font-size:.8rem;margin-bottom:20px}
.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:14px}
.card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:14px}
.s3{grid-column:span 3}.s4{grid-column:span 4}.s6{grid-column:span 6}.s8{grid-column:span 8}.s12{grid-column:span 12}
.metric{display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid rgba(29,51,85,.4);font-size:.75rem}
.metric span{color:var(--muted)}.metric strong{color:var(--text)}
.btn{background:var(--blue);color:#fff;border:none;border-radius:6px;padding:7px 14px;font-size:.73rem;font-weight:600;cursor:pointer}
table{width:100%;border-collapse:collapse;font-size:.73rem}
th{text-align:left;color:var(--muted);padding:6px 8px;border-bottom:1px solid var(--border)}
td{padding:7px 8px;border-bottom:1px solid rgba(29,51,85,.4)}
@media(max-width:900px){.s3,.s4,.s6,.s8{grid-column:span 12}}`;

function getPage(path: string) {
  if (path.includes('/muscle')) return PAGES.muscle;
  if (path.includes('/clinical')) return PAGES.clinical;
  if (path.includes('/reports')) return PAGES.reports;
  if (path.includes('/exercises')) return PAGES.exercises;
  return PAGES.muscle;
}

export default function handler(req: VercelRequest, res: VercelResponse) {
  const page = getPage(req.url || '');
  const html = `<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>${page.title} — IMW MSK</title><style>${SHARED_CSS}</style></head><body><div class="layout"><h1>${page.title}</h1><p class="subtitle">${page.subtitle}</p>${page.body}</div></body></html>`;
  res.setHeader('Content-Type', 'text/html; charset=utf-8');
  res.send(html);
}
