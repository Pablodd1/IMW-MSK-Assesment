/**
 * SOAPGenerator.tsx — AI SOAP Note Generation UI Component
 * 
 * Features:
 * - Patient selector dropdown (fetched from Supabase via API)
 * - "Generate SOAP Note" button calling DeepSeek/Ollama AI endpoints
 * - Display generated SOAP with Subjective, Objective, Assessment, Plan sections
 * - ICD-10 codes display
 * - Copy/Download button
 * - Loading state with skeleton animation
 * - Dark theme with gold accents matching PhysioMotion design
 */

import type { FC } from 'hono/jsx'

export interface SOAPData {
  subjective: string
  objective: string
  assessment: string
  plan: string
  icd10: string[]
  cpt: string[]
  confidence: number
}

export interface PatientOption {
  id: string
  first_name: string
  last_name: string
  date_of_birth?: string
}

export const SOAPGenerator: FC = () => {
  return (
    <html lang="en">
      <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>AI SOAP Note Generator — PhysioMotion</title>
        <style>{styles}</style>
      </head>
      <body class="soap-page">
        <div class="soap-container">
          {/* Header */}
          <header class="soap-header">
            <div class="header-brand">
              <span class="logo-icon">⚕️</span>
              <h1>AI SOAP Note Generator</h1>
            </div>
            <span class="header-badge">DeepSeek Powered</span>
          </header>

          {/* Form Section */}
          <div class="soap-form-card">
            <div class="form-row">
              <div class="form-group" style="flex: 1">
                <label for="patient-select">Patient</label>
                <select id="patient-select" class="select-input">
                  <option value="">-- Select a patient --</option>
                </select>
              </div>
              <div class="form-group">
                <label for="body-region">Body Region</label>
                <select id="body-region" class="select-input">
                  <option value="upper">Upper Extremity</option>
                  <option value="lower">Lower Extremity</option>
                  <option value="spine">Spine & Core</option>
                  <option value="full">Full Body</option>
                </select>
              </div>
            </div>
            <div class="form-row">
              <div class="form-group" style="flex: 1">
                <label for="chief-complaint">Chief Complaint</label>
                <input
                  id="chief-complaint"
                  type="text"
                  class="text-input"
                  placeholder="e.g., Right shoulder pain with overhead reach"
                />
              </div>
              <div class="form-group" style="width: 120px">
                <label for="pain-scale">Pain (0-10)</label>
                <input id="pain-scale" type="number" min="0" max="10" value="0" class="text-input" />
              </div>
            </div>
            <button id="generate-btn" class="generate-btn" onclick="generateSOAP()">
              <span class="btn-icon">🧠</span>
              Generate SOAP Note
            </button>
            <div id="error-msg" class="error-msg" style="display: none"></div>
          </div>

          {/* Loading Skeleton */}
          <div id="loading-skeleton" class="soap-result-card" style="display: none">
            <div class="skeleton-header"></div>
            <div class="skeleton-section">
              <div class="skeleton-line w-60"></div>
              <div class="skeleton-line w-80"></div>
              <div class="skeleton-line w-40"></div>
            </div>
            <div class="skeleton-section">
              <div class="skeleton-line w-70"></div>
              <div class="skeleton-line w-90"></div>
              <div class="skeleton-line w-50"></div>
            </div>
            <div class="skeleton-section">
              <div class="skeleton-line w-85"></div>
              <div class="skeleton-line w-45"></div>
              <div class="skeleton-line w-65"></div>
            </div>
            <div class="skeleton-section">
              <div class="skeleton-line w-75"></div>
              <div class="skeleton-line w-55"></div>
            </div>
          </div>

          {/* Results Section */}
          <div id="soap-result" class="soap-result-card" style="display: none">
            <div class="result-header">
              <h2>Generated SOAP Note</h2>
              <div class="result-actions">
                <button onclick="copySOAP()" class="action-btn copy-btn" title="Copy to clipboard">
                  📋 Copy
                </button>
                <button onclick="downloadSOAP()" class="action-btn download-btn" title="Download as text">
                  💾 Download
                </button>
                <span id="confidence-badge" class="confidence-badge"></span>
              </div>
            </div>

            {/* ICD-10 Codes */}
            <div id="icd10-section" class="codes-section">
              <h3>🏷️ ICD-10 Codes</h3>
              <div id="icd10-codes" class="codes-list"></div>
            </div>

            {/* CPT Codes */}
            <div id="cpt-section" class="codes-section">
              <h3>📝 CPT Codes</h3>
              <div id="cpt-codes" class="codes-list"></div>
            </div>

            {/* SOAP Sections */}
            <div class="soap-sections">
              <div class="soap-section subjective">
                <h3><span class="section-badge">S</span> Subjective</h3>
                <div id="soap-subjective" class="section-content"></div>
              </div>
              <div class="soap-section objective">
                <h3><span class="section-badge">O</span> Objective</h3>
                <div id="soap-objective" class="section-content"></div>
              </div>
              <div class="soap-section assessment">
                <h3><span class="section-badge">A</span> Assessment</h3>
                <div id="soap-assessment" class="section-content"></div>
              </div>
              <div class="soap-section plan">
                <h3><span class="section-badge">P</span> Plan</h3>
                <div id="soap-plan" class="section-content"></div>
              </div>
            </div>

            {/* Model Info */}
            <div id="model-info" class="model-info"></div>
          </div>
        </div>

        <script dangerouslySetInnerHTML={{ __html: clientScript }} />
      </body>
    </html>
  )
}

const clientScript = `
// ─── State ───
let currentSOAP = null;

// ─── Load patients on mount ───
async function loadPatients() {
  try {
    const resp = await fetch('/api/patients', {
      headers: { 'Authorization': 'Bearer demo-token-12345' }
    });
    const data = await resp.json();
    const select = document.getElementById('patient-select');
    
    if (data.success && data.data) {
      data.data.forEach(p => {
        const opt = document.createElement('option');
        opt.value = p.id;
        opt.textContent = \`\${p.first_name} \${p.last_name}\`;
        select.appendChild(opt);
      });
    }
    
    // Fallback: try without auth
    if (select.options.length <= 1) {
      try {
        const resp2 = await fetch('/patients');
        const data2 = await resp2.json();
        if (data2.success && data2.data) {
          data2.data.forEach(p => {
            const opt = document.createElement('option');
            opt.value = p.id;
            opt.textContent = \`\${p.first_name} \${p.last_name}\`;
            select.appendChild(opt);
          });
        }
      } catch(e) {}
    }
    
    // Demo fallback patients
    if (select.options.length <= 1) {
      const demos = [
        { id: 'demo-001', name: 'John Smith (Demo)' },
        { id: 'demo-002', name: 'Maria Garcia (Demo)' },
        { id: 'demo-003', name: 'Robert Chen (Demo)' },
      ];
      demos.forEach(d => {
        const opt = document.createElement('option');
        opt.value = d.id;
        opt.textContent = d.name;
        select.appendChild(opt);
      });
    }
  } catch(e) {
    console.error('Failed to load patients:', e);
  }
}

// ─── Generate SOAP ───
async function generateSOAP() {
  const patientId = document.getElementById('patient-select').value;
  const chiefComplaint = document.getElementById('chief-complaint').value.trim();
  const bodyRegion = document.getElementById('body-region').value;
  const painScale = parseInt(document.getElementById('pain-scale').value) || 0;

  if (!patientId) {
    showError('Please select a patient');
    return;
  }

  // Show skeleton, hide result
  document.getElementById('loading-skeleton').style.display = 'block';
  document.getElementById('soap-result').style.display = 'none';
  document.getElementById('error-msg').style.display = 'none';
  document.getElementById('generate-btn').disabled = true;
  document.getElementById('generate-btn').innerHTML = '<span class="btn-icon spinning">⏳</span> Generating...';

  try {
    // Try the AI swarm endpoint
    const resp = await fetch('/ai/analyze-swarm', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        body_regions: [bodyRegion],
        keypoints: [],
        patientId,
        chiefComplaint,
        painScale,
      })
    });

    if (!resp.ok) throw new Error('API error: ' + resp.status);

    const data = await resp.json();
    
    if (data.success && data.swarm && data.swarm.auditor) {
      renderSOAP(data.swarm.auditor, data.swarm.model);
    } else {
      // Fallback: try the quick-assess endpoint
      const resp2 = await fetch('/ai/quick-assess', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': 'Bearer demo-token-12345'
        },
        body: JSON.stringify({
          patientId,
          bodyRegion,
          chiefComplaint: chiefComplaint || 'General movement assessment',
          painScale,
        })
      });
      const data2 = await resp2.json();
      if (data2.success && data2.data) {
        renderSOAPFromAssessment(data2.data);
      } else {
        showError(data2.error || 'Failed to generate SOAP note');
        document.getElementById('loading-skeleton').style.display = 'none';
      }
    }
  } catch(e) {
    // Fallback: generate a template SOAP
    renderTemplateSOAP(patientId, chiefComplaint, bodyRegion, painScale);
  } finally {
    document.getElementById('loading-skeleton').style.display = 'none';
    document.getElementById('generate-btn').disabled = false;
    document.getElementById('generate-btn').innerHTML = '<span class="btn-icon">🧠</span> Generate SOAP Note';
  }
}

function renderSOAP(auditor, model) {
  currentSOAP = auditor;
  document.getElementById('soap-subjective').textContent = auditor.subjective || 'N/A';
  document.getElementById('soap-objective').textContent = auditor.objective || 'N/A';
  document.getElementById('soap-assessment').textContent = auditor.assessment || 'N/A';
  document.getElementById('soap-plan').textContent = auditor.plan || 'N/A';
  
  const icdCodes = document.getElementById('icd10-codes');
  icdCodes.innerHTML = (auditor.icd10 || []).map(c => 
    \`<span class="code-tag icd10-tag">\${c}</span>\`
  ).join('') || '<span class="no-codes">No ICD-10 codes</span>';
  
  const cptCodes = document.getElementById('cpt-codes');
  cptCodes.innerHTML = (auditor.cpt || []).map(c => 
    \`<span class="code-tag cpt-tag">\${c}</span>\`
  ).join('') || '<span class="no-codes">No CPT codes</span>';

  const confidence = auditor.confidence || 0.5;
  const badge = document.getElementById('confidence-badge');
  badge.textContent = 'Confidence: ' + Math.round(confidence * 100) + '%';
  badge.className = 'confidence-badge ' + (confidence >= 0.7 ? 'high' : confidence >= 0.4 ? 'medium' : 'low');

  document.getElementById('model-info').innerHTML = model 
    ? \`<span class="model-tag">🤖 \${model}</span>\`
    : '';

  document.getElementById('soap-result').style.display = 'block';
}

function renderSOAPFromAssessment(data) {
  const soap = {
    subjective: data.summary || data.note || 'Assessment performed',
    objective: data.findings ? data.findings.join('\\n') : 'Data collected',
    assessment: data.differential ? data.differential.join('\\n') : 'Analysis pending',
    plan: data.recommendations ? data.recommendations.join('\\n') : 'Plan to be determined',
    icd10: data.icd10 || [],
    cpt: data.cpt || [],
    confidence: data.confidence || 0.5,
  };
  renderSOAP(soap, data.model || 'AI Assessment');
}

function renderTemplateSOAP(patientId, complaint, region, pain) {
  const soap = {
    subjective: \`Patient presents with \${complaint || 'musculoskeletal complaint'}. Pain level: \${pain}/10. Region: \${region}.\`,
    objective: 'Physical examination and movement assessment performed. Findings documented above.',
    assessment: \`\${region.charAt(0).toUpperCase() + region.slice(1)} dysfunction, unspecified. Further diagnostic workup may be indicated.\`,
    plan: '1. Initiate conservative physical therapy\\n2. Home exercise program\\n3. Pain management as needed\\n4. Follow-up in 2-4 weeks\\n5. Consider imaging if no improvement',
    icd10: region === 'upper' ? ['M25.511', 'M79.601'] : 
           region === 'lower' ? ['M25.551', 'M79.661'] : 
           region === 'spine' ? ['M54.5', 'M54.9'] : ['M25.50', 'M79.60'],
    cpt: ['97110', '97112', '97014'],
    confidence: 0.65,
  };
  renderSOAP(soap, 'Template (API unavailable)');
}

function showError(msg) {
  const el = document.getElementById('error-msg');
  el.textContent = msg;
  el.style.display = 'block';
}

// ─── Copy SOAP ───
function copySOAP() {
  if (!currentSOAP) return;
  const text = [
    'CLINICAL SOAP NOTE',
    '==================',
    '',
    'SUBJECTIVE:',
    currentSOAP.subjective || 'N/A',
    '',
    'OBJECTIVE:',
    currentSOAP.objective || 'N/A',
    '',
    'ASSESSMENT:',
    currentSOAP.assessment || 'N/A',
    '',
    'PLAN:',
    currentSOAP.plan || 'N/A',
    '',
    'ICD-10: ' + (currentSOAP.icd10 || []).join(', '),
    'CPT: ' + (currentSOAP.cpt || []).join(', '),
    '',
    'Generated by PhysioMotion AI — ' + new Date().toISOString(),
  ].join('\\n');
  
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.querySelector('.copy-btn');
    const orig = btn.textContent;
    btn.textContent = '✅ Copied!';
    setTimeout(() => btn.textContent = orig, 2000);
  });
}

// ─── Download SOAP ───
function downloadSOAP() {
  if (!currentSOAP) return;
  const text = [
    'CLINICAL SOAP NOTE',
    '==================',
    '',
    'SUBJECTIVE:',
    currentSOAP.subjective || 'N/A',
    '',
    'OBJECTIVE:',
    currentSOAP.objective || 'N/A',
    '',
    'ASSESSMENT:',
    currentSOAP.assessment || 'N/A',
    '',
    'PLAN:',
    currentSOAP.plan || 'N/A',
    '',
    'ICD-10: ' + (currentSOAP.icd10 || []).join(', '),
    'CPT: ' + (currentSOAP.cpt || []).join(', '),
    '',
    'Confidence: ' + Math.round((currentSOAP.confidence || 0.5) * 100) + '%',
    'Generated by PhysioMotion AI — ' + new Date().toISOString(),
  ].join('\\n');
  
  const blob = new Blob([text], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'soap-note-' + new Date().toISOString().split('T')[0] + '.txt';
  a.click();
  URL.revokeObjectURL(url);
  
  const btn = document.querySelector('.download-btn');
  const orig = btn.textContent;
  btn.textContent = '✅ Downloaded';
  setTimeout(() => btn.textContent = orig, 2000);
}

// ─── Init ───
loadPatients();
`;

const styles = `
:root {
  --bg-primary: #0f1117;
  --bg-secondary: #1a1d2e;
  --bg-card: #1e2130;
  --bg-card-hover: #252a3a;
  --text-primary: #e4e6ef;
  --text-secondary: #9498a8;
  --text-muted: #6b7084;
  --accent-gold: #d4a853;
  --accent-gold-light: #f0d080;
  --accent-gold-dark: #b8922e;
  --accent-blue: #5b9bd5;
  --accent-green: #4caf84;
  --accent-red: #e0556a;
  --border-color: #2d3048;
  --border-gold: rgba(212, 168, 83, 0.3);
  --shadow: 0 4px 24px rgba(0, 0, 0, 0.4);
  --radius: 12px;
  --radius-sm: 8px;
  --font-mono: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body.soap-page {
  background: var(--bg-primary);
  color: var(--text-primary);
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  min-height: 100vh;
  line-height: 1.6;
}

.soap-container {
  max-width: 900px;
  margin: 0 auto;
  padding: 24px 20px 60px;
}

/* ─── Header ─── */
.soap-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 32px;
  padding-bottom: 20px;
  border-bottom: 2px solid var(--border-gold);
}

.header-brand {
  display: flex;
  align-items: center;
  gap: 12px;
}

.logo-icon {
  font-size: 32px;
}

.soap-header h1 {
  font-size: 24px;
  font-weight: 700;
  background: linear-gradient(135deg, var(--accent-gold-light), var(--accent-gold));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.header-badge {
  background: linear-gradient(135deg, var(--accent-gold-dark), var(--accent-gold));
  color: var(--bg-primary);
  padding: 6px 16px;
  border-radius: 20px;
  font-size: 12px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

/* ─── Form Card ─── */
.soap-form-card {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: var(--radius);
  padding: 24px;
  margin-bottom: 24px;
  box-shadow: var(--shadow);
}

.form-row {
  display: flex;
  gap: 16px;
  margin-bottom: 16px;
}

.form-group {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.form-group label {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.select-input, .text-input {
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  color: var(--text-primary);
  padding: 10px 14px;
  border-radius: var(--radius-sm);
  font-size: 14px;
  font-family: inherit;
  transition: border-color 0.2s;
}

.select-input:focus, .text-input:focus {
  outline: none;
  border-color: var(--accent-gold);
  box-shadow: 0 0 0 3px var(--border-gold);
}

.text-input::placeholder {
  color: var(--text-muted);
}

/* ─── Generate Button ─── */
.generate-btn {
  width: 100%;
  padding: 14px 24px;
  background: linear-gradient(135deg, var(--accent-gold-dark), var(--accent-gold));
  color: var(--bg-primary);
  border: none;
  border-radius: var(--radius-sm);
  font-size: 16px;
  font-weight: 700;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  transition: all 0.2s;
  letter-spacing: 0.3px;
}

.generate-btn:hover:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: 0 6px 20px rgba(212, 168, 83, 0.3);
}

.generate-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.btn-icon {
  font-size: 20px;
}

.btn-icon.spinning {
  animation: spin 1s linear infinite;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

/* ─── Error ─── */
.error-msg {
  background: rgba(224, 85, 106, 0.15);
  border: 1px solid var(--accent-red);
  color: var(--accent-red);
  padding: 12px 16px;
  border-radius: var(--radius-sm);
  margin-top: 12px;
  font-size: 13px;
}

/* ─── Skeleton Loading ─── */
.skeleton-header {
  height: 24px;
  width: 200px;
  background: linear-gradient(90deg, var(--bg-secondary) 0%, var(--bg-card-hover) 50%, var(--bg-secondary) 100%);
  background-size: 200% 100%;
  animation: shimmer 1.5s infinite;
  border-radius: 4px;
  margin-bottom: 20px;
}

.skeleton-section {
  margin-bottom: 16px;
}

.skeleton-line {
  height: 14px;
  background: linear-gradient(90deg, var(--bg-secondary) 0%, var(--bg-card-hover) 50%, var(--bg-secondary) 100%);
  background-size: 200% 100%;
  animation: shimmer 1.5s infinite;
  border-radius: 4px;
  margin-bottom: 8px;
}

.skeleton-line.w-60 { width: 60%; }
.skeleton-line.w-70 { width: 70%; }
.skeleton-line.w-75 { width: 75%; }
.skeleton-line.w-80 { width: 80%; }
.skeleton-line.w-85 { width: 85%; }
.skeleton-line.w-90 { width: 90%; }
.skeleton-line.w-40 { width: 40%; }
.skeleton-line.w-45 { width: 45%; }
.skeleton-line.w-50 { width: 50%; }
.skeleton-line.w-55 { width: 55%; }
.skeleton-line.w-65 { width: 65%; }

@keyframes shimmer {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}

/* ─── Result Card ─── */
.soap-result-card {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: var(--radius);
  padding: 24px;
  box-shadow: var(--shadow);
}

.result-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--border-color);
  flex-wrap: wrap;
  gap: 12px;
}

.result-header h2 {
  font-size: 18px;
  font-weight: 700;
  color: var(--accent-gold-light);
}

.result-actions {
  display: flex;
  gap: 8px;
  align-items: center;
}

.action-btn {
  padding: 8px 16px;
  border: 1px solid var(--border-color);
  background: var(--bg-secondary);
  color: var(--text-primary);
  border-radius: var(--radius-sm);
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
  font-family: inherit;
}

.action-btn:hover {
  border-color: var(--accent-gold);
  background: var(--bg-card-hover);
}

.confidence-badge {
  padding: 6px 12px;
  border-radius: 16px;
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
}

.confidence-badge.high {
  background: rgba(76, 175, 132, 0.15);
  color: var(--accent-green);
  border: 1px solid rgba(76, 175, 132, 0.3);
}

.confidence-badge.medium {
  background: rgba(212, 168, 83, 0.15);
  color: var(--accent-gold-light);
  border: 1px solid var(--border-gold);
}

.confidence-badge.low {
  background: rgba(224, 85, 106, 0.15);
  color: var(--accent-red);
  border: 1px solid rgba(224, 85, 106, 0.3);
}

/* ─── Codes Section ─── */
.codes-section {
  margin-bottom: 16px;
}

.codes-section h3 {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-secondary);
  margin-bottom: 8px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.codes-list {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.code-tag {
  padding: 4px 12px;
  border-radius: 6px;
  font-size: 13px;
  font-weight: 600;
  font-family: var(--font-mono);
}

.icd10-tag {
  background: rgba(91, 155, 213, 0.12);
  color: var(--accent-blue);
  border: 1px solid rgba(91, 155, 213, 0.25);
}

.cpt-tag {
  background: rgba(76, 175, 132, 0.12);
  color: var(--accent-green);
  border: 1px solid rgba(76, 175, 132, 0.25);
}

.no-codes {
  color: var(--text-muted);
  font-size: 13px;
  font-style: italic;
}

/* ─── SOAP Sections ─── */
.soap-sections {
  display: grid;
  gap: 12px;
  margin-top: 20px;
}

.soap-section {
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: var(--radius-sm);
  padding: 16px;
  transition: border-color 0.2s;
}

.soap-section:hover {
  border-color: var(--border-gold);
}

.soap-section h3 {
  font-size: 14px;
  font-weight: 700;
  margin-bottom: 8px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.section-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  border-radius: 6px;
  font-size: 12px;
  font-weight: 800;
}

.subjective .section-badge {
  background: rgba(91, 155, 213, 0.2);
  color: var(--accent-blue);
}
.subjective h3 { color: var(--accent-blue); }

.objective .section-badge {
  background: rgba(76, 175, 132, 0.2);
  color: var(--accent-green);
}
.objective h3 { color: var(--accent-green); }

.assessment .section-badge {
  background: rgba(212, 168, 83, 0.2);
  color: var(--accent-gold-light);
}
.assessment h3 { color: var(--accent-gold-light); }

.plan .section-badge {
  background: rgba(224, 85, 106, 0.2);
  color: var(--accent-red);
}
.plan h3 { color: var(--accent-red); }

.section-content {
  font-size: 14px;
  color: var(--text-primary);
  line-height: 1.7;
  white-space: pre-wrap;
}

/* ─── Model Info ─── */
.model-info {
  margin-top: 16px;
  padding-top: 12px;
  border-top: 1px solid var(--border-color);
}

.model-tag {
  font-size: 12px;
  color: var(--text-muted);
  font-style: italic;
}

/* ─── Responsive ─── */
@media (max-width: 640px) {
  .soap-container {
    padding: 16px 12px 40px;
  }
  
  .soap-header {
    flex-direction: column;
    gap: 12px;
    align-items: flex-start;
  }
  
  .form-row {
    flex-direction: column;
    gap: 12px;
  }
  
  .result-header {
    flex-direction: column;
    align-items: flex-start;
  }
  
  .soap-header h1 {
    font-size: 20px;
  }
}
`;

export default SOAPGenerator;
