/**
 * ProviderPortal — Real-time clinical provider dashboard
 * Features: Patient queue, 3D skeleton viewer, clinical assessment panel
 * WebSocket connection to deployed pose engine at Modal
 */

import type { FC } from 'hono/jsx'

// ── Mock patient data (in production, fetched from API) ──
const MOCK_QUEUE = [
  { id: '1', name: 'John Smith', dob: '1979-05-15', condition: 'Chronic LBP', status: 'in-session', pain: 4, lastFMS: 11 },
  { id: '2', name: 'Maria Garcia', dob: '1963-08-22', condition: 'Post TKA (R)', status: 'waiting', pain: 3, lastFMS: 9 },
  { id: '3', name: 'David Chen', dob: '1987-03-10', condition: 'Shoulder Impingement', status: 'waiting', pain: 5, lastFMS: 13 },
  { id: '4', name: 'Sarah Williams', dob: '1992-11-03', condition: 'ACL Rehab (L)', status: 'completed', pain: 2, lastFMS: 15 },
  { id: '5', name: 'Robert Kim', dob: '1955-07-28', condition: 'Cervical Radiculopathy', status: 'waiting', pain: 6, lastFMS: 8 },
]

// ── Styles ──
const styles = `
  :root {
    --bg-primary: #0a1628;
    --bg-secondary: #0d1b31;
    --bg-card: rgba(26, 32, 53, 0.82);
    --bg-hover: #14243d;
    --border: rgba(148, 163, 184, 0.1);
    --text-primary: #e2e8f0;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --gold: #f59e0b;
    --gold-light: #fbbf24;
    --gold-dim: rgba(245, 158, 11, 0.15);
    --green: #10b981;
    --red: #ef4444;
    --blue: #3b82f6;
    --teal: #14b8a6;
    --purple: #8b5cf6;
    --sidebar-w: 300px;
    --panel-w: 360px;
  }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    height: 100vh;
    overflow: hidden;
  }
  .app { display: flex; height: 100vh; }
  .app { animation: fadeIn 0.35s ease-out both; }

  /* ── LEFT SIDEBAR ── */
  .sidebar {
    width: var(--sidebar-w);
    min-width: var(--sidebar-w);
    background: var(--bg-secondary);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .sidebar-header {
    padding: 16px 20px;
    border-bottom: 1px solid var(--border);
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
  }
  .sidebar-header h2 {
    font-size: 14px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--gold);
    margin-bottom: 2px;
  }
  .sidebar-header .subtitle {
    font-size: 11px;
    color: var(--text-muted);
  }
  .patient-list {
    flex: 1;
    overflow-y: auto;
    padding: 8px;
  }
  .patient-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px;
    border-radius: 10px;
    cursor: pointer;
    transition: all 0.15s;
    border: 1px solid transparent;
    margin-bottom: 4px;
  }
  .patient-item:hover { background: var(--bg-hover); border-color: var(--border); }
  .patient-item.active { 
    background: var(--gold-dim); 
    border-color: rgba(245, 158, 11, 0.3);
    box-shadow: 0 0 20px rgba(245, 158, 11, 0.05);
  }
  .patient-avatar {
    width: 38px; height: 38px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 13px;
    flex-shrink: 0;
  }
  .patient-info { flex: 1; min-width: 0; }
  .patient-name { font-size: 13px; font-weight: 600; color: var(--text-primary); }
  .patient-condition {
    font-size: 11px; color: var(--text-secondary);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .patient-meta {
    display: flex; gap: 8px; margin-top: 3px;
    font-size: 10px; color: var(--text-muted);
  }
  .status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .status-dot.in-session { background: var(--green); box-shadow: 0 0 6px var(--green); }
  .status-dot.waiting { background: var(--gold); }
  .status-dot.completed { background: var(--text-muted); }
  .sidebar-footer {
    padding: 12px 16px;
    border-top: 1px solid var(--border);
    font-size: 11px;
  }
  .sidebar-footer .connection-status {
    display: flex; align-items: center; gap: 6px;
    color: var(--text-muted);
  }
  .connection-dot { width: 6px; height: 6px; border-radius: 50%; }
  .connection-dot.connected { background: var(--green); }
  .connection-dot.disconnected { background: var(--red); }
  .connection-dot.connecting { background: var(--gold); animation: pulse 1s infinite; }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

  /* ── MAIN VIEWER ── */
  .main-viewer {
    flex: 1;
    display: flex;
    flex-direction: column;
    background: var(--bg-primary);
    position: relative;
  }
  .viewer-toolbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 16px;
    border-bottom: 1px solid var(--border);
    background: var(--bg-secondary);
  }
  .toolbar-title {
    font-size: 13px; font-weight: 600;
    color: var(--gold);
    display: flex; align-items: center; gap: 8px;
  }
  .toolbar-controls { display: flex; gap: 8px; }
  .btn {
    padding: 6px 14px;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: var(--bg-card);
    color: var(--text-secondary);
    font-size: 12px;
    cursor: pointer;
    transition: all 0.15s;
    display: flex; align-items: center; gap: 6px;
  }
  .btn:hover { background: var(--bg-hover); color: var(--text-primary); }
  .btn.primary { background: var(--gold); color: #000; border-color: var(--gold); font-weight: 600; }
  .btn.primary:hover { background: var(--gold-light); }
  .btn.danger { background: rgba(239, 68, 68, 0.15); color: var(--red); border-color: rgba(239, 68, 68, 0.3); }
  .btn.danger:hover { background: rgba(239, 68, 68, 0.25); }
  .btn.active-recording { background: rgba(239, 68, 68, 0.2); border-color: var(--red); color: var(--red); animation: pulse 1.5s infinite; }
  .btn.recording-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--red); }
  .btn .gold-circle { width: 8px; height: 8px; border-radius: 50%; background: var(--gold); }

  .three-container {
    flex: 1;
    position: relative;
    min-height: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
  }
  .three-container canvas { display: block; width: 100% !important; height: 100% !important; }
  .viewer-overlay {
    position: absolute;
    bottom: 16px; left: 16px;
    background: rgba(0,0,0,0.7);
    backdrop-filter: blur(8px);
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 11px;
    color: var(--text-secondary);
    border: 1px solid var(--border);
  }
  .viewer-overlay span { color: var(--gold); font-weight: 600; margin: 0 4px; }

  /* ── RIGHT PANEL ── */
  .right-panel {
    width: var(--panel-w);
    min-width: var(--panel-w);
    background: var(--bg-secondary);
    border-left: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .panel-tabs {
    display: flex;
    border-bottom: 1px solid var(--border);
  }
  .panel-tab {
    flex: 1;
    padding: 10px 8px;
    text-align: center;
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
    cursor: pointer;
    border-bottom: 2px solid transparent;
    transition: all 0.15s;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .panel-tab:hover { color: var(--text-secondary); }
  .panel-tab.active { color: var(--gold); border-bottom-color: var(--gold); }
  .panel-content {
    flex: 1;
    overflow-y: auto;
    padding: 12px;
  }
  .panel-section {
    margin-bottom: 16px;
    background: var(--bg-card);
    border-radius: 10px;
    border: 1px solid var(--border);
    padding: 14px;
    backdrop-filter: blur(14px);
  }
  .panel-section h3 {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--gold);
    margin-bottom: 10px;
    display: flex; align-items: center; gap: 6px;
  }
  .fms-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
  }
  .fms-item {
    background: var(--bg-hover);
    border-radius: 6px;
    padding: 8px;
    text-align: center;
  }
  .fms-item .score {
    font-size: 24px; font-weight: 700;
  }
  .fms-item .score.good { color: var(--green); }
  .fms-item .score.warn { color: var(--gold); }
  .fms-item .score.bad { color: var(--red); }
  .fms-item .label { font-size: 10px; color: var(--text-muted); margin-top: 2px; }
  .joint-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 6px 0; border-bottom: 1px solid var(--border);
    font-size: 12px;
  }
  .joint-row:last-child { border-bottom: none; }
  .joint-name { color: var(--text-secondary); font-weight: 500; }
  .joint-values { display: flex; gap: 12px; }
  .joint-val { text-align: right; }
  .joint-val .deg { font-weight: 700; }
  .joint-val .side { font-size: 9px; color: var(--text-muted); }
  .asymmetry-badge {
    display: inline-block;
    padding: 2px 6px; border-radius: 4px;
    font-size: 10px; font-weight: 600;
  }
  .asymmetry-badge.normal { background: rgba(16,185,129,0.15); color: var(--green); }
  .asymmetry-badge.mild { background: rgba(245,158,11,0.15); color: var(--gold); }
  .asymmetry-badge.moderate { background: rgba(239,68,68,0.15); color: var(--red); }
  .chiropractic-indicator {
    display: flex; gap: 8px; margin-top: 6px;
    flex-wrap: wrap;
  }
  .chiro-chip {
    padding: 4px 10px; border-radius: 20px;
    font-size: 10px; font-weight: 600;
    border: 1px solid;
  }
  .chiro-chip.deviation { border-color: rgba(139, 92, 246, 0.4); color: var(--purple); background: rgba(139,92,246,0.1); }
  .chiro-chip.rotation { border-color: rgba(59, 130, 246, 0.4); color: var(--blue); background: rgba(59,130,246,0.1); }
  .chiro-chip.tilt { border-color: rgba(20, 184, 166, 0.4); color: var(--teal); background: rgba(20,184,166,0.1); }
  .swarm-log {
    font-family: 'Fira Code', 'Cascadia Code', monospace;
    font-size: 11px;
    max-height: 180px;
    overflow-y: auto;
    background: rgba(0,0,0,0.3);
    border-radius: 6px;
    padding: 8px;
    border: 1px solid var(--border);
  }
  .swarm-line {
    padding: 2px 0;
    border-bottom: 1px solid rgba(255,255,255,0.03);
    color: var(--text-muted);
  }
  .swarm-line .agent { color: var(--teal); font-weight: 600; }
  .swarm-line .finding { color: var(--purple); }
  .soap-preview {
    font-size: 11px;
    line-height: 1.5;
    max-height: 200px;
    overflow-y: auto;
  }
  .soap-preview .soap-s { color: var(--blue); margin-bottom: 4px; }
  .soap-preview .soap-o { color: var(--teal); margin-bottom: 4px; }
  .soap-preview .soap-a { color: var(--purple); margin-bottom: 4px; }
  .soap-preview .soap-p { color: var(--gold); }
  .soap-preview b { color: var(--text-primary); }
  .metric-row {
    display: flex; justify-content: space-between;
    font-size: 12px; padding: 4px 0;
    border-bottom: 1px solid rgba(255,255,255,0.03);
  }
  .metric-label { color: var(--text-secondary); }
  .metric-value { font-weight: 600; }

  /* ── Session Controls (Bottom Bar) ── */
  .session-bar {
    display: flex; align-items: center; gap: 12px;
    padding: 10px 16px;
    border-top: 1px solid var(--border);
    background: var(--bg-secondary);
  }
  .session-bar .elapsed {
    font-family: monospace; font-size: 18px; font-weight: 700;
    color: var(--text-primary);
    margin-right: auto;
  }
  .session-bar .elapsed.recording { color: var(--red); }

  /* ── EMPTY STATE ── */
  .empty-state {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    height: 100%;
    color: var(--text-muted);
    text-align: center;
    padding: 20px;
  }
  .empty-state .icon { font-size: 40px; margin-bottom: 10px; opacity: 0.3; }
  .empty-state h3 { font-size: 14px; margin-bottom: 4px; }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

  /* Notifications */
  .notification {
    position: fixed;
    top: 16px; right: 16px;
    background: var(--bg-card);
    border: 1px solid var(--gold);
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 12px;
    color: var(--text-primary);
    z-index: 999;
    animation: slideIn 0.3s ease-out;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    display: flex; align-items: center; gap: 8px;
    max-width: 350px;
  }
  @keyframes slideIn { from { transform: translateX(100%); opacity: 0; } to { transform: translateX(0); opacity: 1; } }
  @media (max-width: 1180px) {
    :root { --sidebar-w: 260px; --panel-w: 330px; }
    .toolbar-controls { flex-wrap: wrap; justify-content: flex-end; }
  }
  @media (max-width: 920px) {
    body { overflow: auto; }
    .app { min-height: 100vh; height: auto; flex-direction: column; }
    .sidebar, .right-panel { width: 100%; min-width: 0; }
    .sidebar { max-height: 280px; border-right: 0; border-bottom: 1px solid var(--border); }
    .main-viewer { min-height: 560px; }
    .right-panel { border-left: 0; border-top: 1px solid var(--border); min-height: 520px; }
    .session-bar, .viewer-toolbar { flex-wrap: wrap; }
  }
`

// ── Provider Portal Layout ──
function Layout({ children }: { children: any }) {
  return (
    <html lang="en">
      <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Provider Portal — PhysioMotion</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.4.0/css/all.min.css" rel="stylesheet" />
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        {/* OrbitControls would need to be bundled, we'll use a minimal approach */}
        <style>{styles}</style>
      </head>
      <body>
        {children}
      </body>
    </html>
  )
}

// ── Main Component ──
export const ProviderPortal: FC = () => {
  return (
    <Layout>
      <div class="app" id="app">
        {/* ── LEFT SIDEBAR: Patient Queue ── */}
        <aside class="sidebar" id="sidebar">
          <div class="sidebar-header">
            <h2><i class="fas fa-user-md" style="margin-right:6px"></i>Patient Queue</h2>
            <div class="subtitle" id="queue-count">5 patients • 1 in session</div>
          </div>
          <div class="patient-list" id="patient-list">
            {MOCK_QUEUE.map((p, i) => (
              <div
                class={`patient-item ${i === 0 ? 'active' : ''}`}
                data-patient-id={p.id}
                id={`patient-${p.id}`}
              >
                <div
                  class="patient-avatar"
                  style={`background: linear-gradient(135deg, ${
                    ['#f59e0b', '#8b5cf6', '#3b82f6', '#10b981', '#ef4444'][i]
                  }33, ${
                    ['#f59e0b', '#8b5cf6', '#3b82f6', '#10b981', '#ef4444'][i]
                  }11); color: ${['#f59e0b', '#8b5cf6', '#3b82f6', '#10b981', '#ef4444'][i]};`}
                >
                  {p.name.split(' ').map(n => n[0]).join('')}
                </div>
                <div class="patient-info">
                  <div class="patient-name">{p.name}</div>
                  <div class="patient-condition">{p.condition}</div>
                  <div class="patient-meta">
                    <span>FMS: {p.lastFMS}</span>
                    <span>Pain: {p.pain}/10</span>
                  </div>
                </div>
                <div class={`status-dot ${p.status}`}></div>
              </div>
            ))}
          </div>
          <div class="sidebar-footer">
            <div class="connection-status" id="ws-status">
              <div class="connection-dot disconnected" id="ws-dot"></div>
              <span id="ws-text">Pose Engine: Disconnected</span>
              <span id="ws-latency" style="margin-left:auto;color:var(--text-muted)"></span>
            </div>
          </div>
        </aside>

        {/* ── MAIN VIEWER: 3D Skeleton ── */}
        <main class="main-viewer" id="main-viewer">
          <div class="viewer-toolbar">
            <div class="toolbar-title">
              <span class="gold-circle"></span>
              <span>3D Pose Viewer</span>
              <span style="font-size:10px;color:var(--text-muted);font-weight:400" id="current-patient-label">— John Smith</span>
            </div>
            <div class="toolbar-controls">
              <button class="btn" onclick="resetCamera()" title="Reset camera view">
                <i class="fas fa-sync-alt"></i> Reset View
              </button>
              <button class="btn" onclick="toggleAutoRotate()" title="Auto-rotate camera">
                <i class="fas fa-redo"></i> Auto Rotate
              </button>
              <button class="btn" onclick="captureFrame()" title="Capture current frame for assessment">
                <i class="fas fa-camera"></i> Capture
              </button>
            </div>
          </div>
          <div class="three-container" id="three-container">
            <div class="empty-state" id="viewer-empty">
              <div class="icon">🧬</div>
              <h3>3D Skeleton Viewer</h3>
              <p style="font-size:11px">Connect to pose engine to visualize<br/>real-time movement data</p>
            </div>
          </div>
          <div class="viewer-overlay" id="viewer-fps" style="display:none">
            <span id="fps-value">0</span> FPS • <span id="landmark-count">0</span> landmarks
          </div>

          {/* Session Controls */}
          <div class="session-bar">
            <div class="elapsed" id="session-timer">00:00</div>
            <button class="btn" id="btn-start" onclick="toggleRecording()">
              <i class="fas fa-circle" style="color:var(--red);font-size:8px"></i> Start Recording
            </button>
            <button class="btn" id="btn-stop" onclick="stopRecording()" style="display:none">
              <i class="fas fa-stop"></i> Stop
            </button>
            <button class="btn" onclick="captureFrame()">
              <i class="fas fa-camera"></i> Capture Frame
            </button>
            <button class="btn" onclick="resetFilters()">
              <i class="fas fa-filter"></i> Reset Filters
            </button>
            <button class="btn" onclick="requestAssessment()">
              <i class="fas fa-stethoscope"></i> Assess
            </button>
          </div>
        </main>

        {/* ── RIGHT PANEL: Clinical Assessment ── */}
        <aside class="right-panel" id="right-panel">
          <div class="panel-tabs">
            <div class="panel-tab active" data-tab="fms" onclick="switchTab('fms')">FMS</div>
            <div class="panel-tab" data-tab="joints" onclick="switchTab('joints')">Joints</div>
            <div class="panel-tab" data-tab="chiro" onclick="switchTab('chiro')">Chiro</div>
            <div class="panel-tab" data-tab="swarm" onclick="switchTab('swarm')">Swarm</div>
            <div class="panel-tab" data-tab="soap" onclick="switchTab('soap')">SOAP</div>
          </div>
          <div class="panel-content" id="panel-content">
            {/* FMS Tab */}
            <div class="panel-tab-content active" id="tab-fms">
              <div class="panel-section">
                <h3><i class="fas fa-clipboard-check"></i> FMS Scores</h3>
                <div class="fms-grid" id="fms-scores">
                  <div class="fms-item"><div class="score" style="color:var(--text-muted)">-</div><div class="label">Deep Squat</div></div>
                  <div class="fms-item"><div class="score" style="color:var(--text-muted)">-</div><div class="label">Hurdle Step</div></div>
                  <div class="fms-item"><div class="score" style="color:var(--text-muted)">-</div><div class="label">Inline Lunge</div></div>
                  <div class="fms-item"><div class="score" style="color:var(--text-muted)">-</div><div class="label">Shoulder Mob.</div></div>
                  <div class="fms-item"><div class="score" style="color:var(--text-muted)">-</div><div class="label">ASLR</div></div>
                  <div class="fms-item"><div class="score" style="color:var(--text-muted)">-</div><div class="label">TSPU</div></div>
                  <div class="fms-item"><div class="score" style="color:var(--text-muted)">-</div><div class="label">Rotary Stab.</div></div>
                  <div class="fms-item" style="grid-column: span 2">
                    <div class="score" style="color:var(--text-muted);font-size:14px">-</div>
                    <div class="label">Total Score</div>
                  </div>
                </div>
              </div>
              <div class="panel-section">
                <h3><i class="fas fa-chart-line"></i> Session Metrics</h3>
                <div class="metric-row"><span class="metric-label">Movement Quality</span><span class="metric-value" style="color:var(--text-muted)" id="metric-quality">--</span></div>
                <div class="metric-row"><span class="metric-label">Stability Score</span><span class="metric-value" style="color:var(--text-muted)" id="metric-stability">--</span></div>
                <div class="metric-row"><span class="metric-label">Frames Processed</span><span class="metric-value" style="color:var(--text-muted)" id="metric-frames">0</span></div>
                <div class="metric-row"><span class="metric-label">Compensations</span><span class="metric-value" style="color:var(--red)" id="metric-compensations">0</span></div>
              </div>
            </div>

            {/* Joints Tab */}
            <div class="panel-tab-content" id="tab-joints" style="display:none">
              <div class="panel-section">
                <h3><i class="fas fa-bone"></i> Joint Angles</h3>
                <div id="joint-angles">
                  <div class="joint-row"><span class="joint-name">Shoulder Flexion</span><div class="joint-values"><div class="joint-val"><div class="deg" style="color:var(--text-muted)">--</div><div class="side">L</div></div><div class="joint-val"><div class="deg" style="color:var(--text-muted)">--</div><div class="side">R</div></div></div></div>
                  <div class="joint-row"><span class="joint-name">Shoulder Abduction</span><div class="joint-values"><div class="joint-val"><div class="deg" style="color:var(--text-muted)">--</div><div class="side">L</div></div><div class="joint-val"><div class="deg" style="color:var(--text-muted)">--</div><div class="side">R</div></div></div></div>
                  <div class="joint-row"><span class="joint-name">Elbow Flexion</span><div class="joint-values"><div class="joint-val"><div class="deg" style="color:var(--text-muted)">--</div><div class="side">L</div></div><div class="joint-val"><div class="deg" style="color:var(--text-muted)">--</div><div class="side">R</div></div></div></div>
                  <div class="joint-row"><span class="joint-name">Hip Flexion</span><div class="joint-values"><div class="joint-val"><div class="deg" style="color:var(--text-muted)">--</div><div class="side">L</div></div><div class="joint-val"><div class="deg" style="color:var(--text-muted)">--</div><div class="side">R</div></div></div></div>
                  <div class="joint-row"><span class="joint-name">Hip Abduction</span><div class="joint-values"><div class="joint-val"><div class="deg" style="color:var(--text-muted)">--</div><div class="side">L</div></div><div class="joint-val"><div class="deg" style="color:var(--text-muted)">--</div><div class="side">R</div></div></div></div>
                  <div class="joint-row"><span class="joint-name">Knee Flexion</span><div class="joint-values"><div class="joint-val"><div class="deg" style="color:var(--text-muted)">--</div><div class="side">L</div></div><div class="joint-val"><div class="deg" style="color:var(--text-muted)">--</div><div class="side">R</div></div></div></div>
                  <div class="joint-row"><span class="joint-name">Knee Extension</span><div class="joint-values"><div class="joint-val"><div class="deg" style="color:var(--text-muted)">--</div><div class="side">L</div></div><div class="joint-val"><div class="deg" style="color:var(--text-muted)">--</div><div class="side">R</div></div></div></div>
                  <div class="joint-row"><span class="joint-name">Ankle Dorsiflexion</span><div class="joint-values"><div class="joint-val"><div class="deg" style="color:var(--text-muted)">--</div><div class="side">L</div></div><div class="joint-val"><div class="deg" style="color:var(--text-muted)">--</div><div class="side">R</div></div></div></div>
                </div>
              </div>
              <div class="panel-section">
                <h3><i class="fas fa-balance-scale"></i> Asymmetry</h3>
                <div id="asymmetry-indicators">
                  <div class="metric-row"><span class="metric-label">Shoulder Height</span><span class="metric-value" style="color:var(--text-muted)">-- cm</span></div>
                  <div class="metric-row"><span class="metric-label">Hip Height</span><span class="metric-value" style="color:var(--text-muted)">-- cm</span></div>
                  <div class="metric-row"><span class="metric-label">Knee Alignment</span><span class="metric-value" style="color:var(--text-muted)">--</span></div>
                  <div class="metric-row"><span class="metric-label">Pelvic Tilt</span><span class="metric-value" style="color:var(--text-muted)">--°</span></div>
                </div>
              </div>
            </div>

            {/* Chiro Tab */}
            <div class="panel-tab-content" id="tab-chiro" style="display:none">
              <div class="panel-section">
                <h3><i class="fas fa-spine"></i> Postural Analysis</h3>
                <div class="chiropractic-indicator" id="chiro-indicators">
                  <span class="chiro-chip deviation" style="opacity:0.3">Anterior Head Carriage</span>
                  <span class="chiro-chip rotation" style="opacity:0.3">Rounded Shoulders</span>
                  <span class="chiro-chip tilt" style="opacity:0.3">Forward Pelvic Tilt</span>
                  <span class="chiro-chip deviation" style="opacity:0.3">Scoliosis Check</span>
                  <span class="chiro-chip rotation" style="opacity:0.3">Thoracic Kyphosis</span>
                  <span class="chiro-chip tilt" style="opacity:0.3">Lumbar Lordosis</span>
                </div>
              </div>
              <div class="panel-section">
                <h3><i class="fas fa-ruler-combined"></i> Spinal Alignment</h3>
                <div class="metric-row"><span class="metric-label">Cervical Angle</span><span class="metric-value" style="color:var(--text-muted)">--°</span></div>
                <div class="metric-row"><span class="metric-label">Thoracic Angle</span><span class="metric-value" style="color:var(--text-muted)">--°</span></div>
                <div class="metric-row"><span class="metric-label">Lumbar Angle</span><span class="metric-value" style="color:var(--text-muted)">--°</span></div>
                <div class="metric-row"><span class="metric-label">Cobb Angle Est.</span><span class="metric-value" style="color:var(--text-muted)">--°</span></div>
                <div class="metric-row"><span class="metric-label">Plumb Line Deviation</span><span class="metric-value" style="color:var(--text-muted)">-- cm</span></div>
              </div>
            </div>

            {/* Swarm Tab */}
            <div class="panel-tab-content" id="tab-swarm" style="display:none">
              <div class="panel-section">
                <h3><i class="fas fa-brain"></i> Swarm Agent Analysis</h3>
                <div class="swarm-log" id="swarm-log">
                  <div class="swarm-line" style="color:var(--text-muted);text-align:center;padding:20px 0">
                    <i class="fas fa-network-wired" style="font-size:20px;display:block;margin-bottom:8px;opacity:0.3"></i>
                    Swarm agents idle<br/>
                    <span style="font-size:10px">Connect to pose engine to activate<br/>multi-agent clinical analysis</span>
                  </div>
                </div>
              </div>
              <div class="panel-section">
                <h3><i class="fas fa-robot"></i> Active Agents</h3>
                <div class="metric-row"><span class="metric-label">Biomechanics Agent</span><span class="metric-value" style="color:var(--text-muted)">idle</span></div>
                <div class="metric-row"><span class="metric-label">Chiro Agent</span><span class="metric-value" style="color:var(--text-muted)">idle</span></div>
                <div class="metric-row"><span class="metric-label">FMS Agent</span><span class="metric-value" style="color:var(--text-muted)">idle</span></div>
                <div class="metric-row"><span class="metric-label">SOAP Agent</span><span class="metric-value" style="color:var(--text-muted)">idle</span></div>
              </div>
            </div>

            {/* SOAP Tab */}
            <div class="panel-tab-content" id="tab-soap" style="display:none">
              <div class="panel-section">
                <h3><i class="fas fa-file-medical-alt"></i> Quick SOAP Note</h3>
                <div class="soap-preview" id="soap-preview">
                  <div style="color:var(--text-muted);text-align:center;padding:20px 0">
                    <i class="fas fa-file-alt" style="font-size:20px;display:block;margin-bottom:8px;opacity:0.3"></i>
                    No SOAP note generated yet<br/>
                    <span style="font-size:10px">Click "Assess" to generate<br/>a clinical SOAP note</span>
                  </div>
                </div>
              </div>
              <div class="panel-section">
                <button class="btn primary" onclick="generateSOAP()" style="width:100%;justify-content:center;padding:10px">
                  <i class="fas fa-magic"></i> Generate SOAP Note
                </button>
                <button class="btn" onclick="copySOAP()" style="width:100%;justify-content:center;margin-top:8px">
                  <i class="fas fa-copy"></i> Copy to Clipboard
                </button>
              </div>
            </div>
          </div>
        </aside>
      </div>

      {/* ── Core Application Script ── */}
      <script dangerouslySetInnerHTML={{ __html: coreScript }} />
    </Layout>
  )
}

// ── Core JavaScript (injected into page) ──
const coreScript = `
(function() {
  'use strict';

  // =========================================================================
  // STATE
  // =========================================================================
  const state = {
    ws: null,
    wsConnected: false,
    wsLatency: 0,
    lastPing: 0,
    recording: false,
    sessionStart: null,
    sessionTimer: null,
    sessionFrames: 0,
    activePatientId: '1',
    activeTab: 'fms',
    // 3D
    scene: null,
    camera: null,
    renderer: null,
    skeletonGroup: null,
    autoRotate: false,
    rotateAngle: 0,
    // Clinical data
    currentSkeleton: null,
    currentAssessment: null,
    fmsScores: {},
    jointAngles: {},
    swarmOutput: [],
    soapNote: null,
  };

  // Keypoint connections (MediaPipe-style 33-landmark skeleton)
  const SKELETON_CONNECTIONS = [
    [0, 1], [0, 2], [1, 3], [2, 4],           // face
    [5, 6],                                     // shoulders
    [5, 7], [7, 9],                             // left arm
    [6, 8], [8, 10],                            // right arm
    [5, 11], [6, 12],                           // torso sides
    [11, 12],                                   // hips
    [11, 13], [13, 15],                         // left leg
    [12, 14], [14, 16],                         // right leg
    [15, 17], [15, 19], [15, 21],               // left foot
    [16, 18], [16, 20], [16, 22],               // right foot
    [17, 19], [18, 20],                         // feet
    [11, 23], [12, 24], [23, 24],               // torso
    [23, 25], [25, 27], [27, 29], [29, 31],     // left leg detail
    [24, 26], [26, 28], [28, 30], [30, 32],     // right leg detail
  ];

  // Keypoint colors (golden scheme)
  const KEYPOINT_COLORS = {};
  for (let i = 0; i < 33; i++) {
    const hue = (i * 30) % 360;
    KEYPOINT_COLORS[i] = 'hsl(' + hue + ', 70%, 60%)';
  }
  // Override major joints with gold
  [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16].forEach(i => {
    KEYPOINT_COLORS[i] = '#f59e0b';
  });

  // =========================================================================
  // DOM REFS
  // =========================================================================
  const $ = id => document.getElementById(id);
  const qs = sel => document.querySelector(sel);
  const qsa = sel => document.querySelectorAll(sel);

  // =========================================================================
  // WEBSOCKET
  // =========================================================================
  const WS_URL = 'wss://pablodd1--pose-engine-ws-serve.modal.run/ws';
  let reconnectTimer = null;
  let pingInterval = null;

  function connectWS() {
    if (state.ws && (state.ws.readyState === WebSocket.OPEN || state.ws.readyState === WebSocket.CONNECTING)) return;

    updateWSStatus('connecting', 'Connecting...');
    state.ws = new WebSocket(WS_URL);

    state.ws.onopen = () => {
      console.log('[Portal] WebSocket connected to pose engine');
      state.wsConnected = true;
      updateWSStatus('connected', 'Pose Engine: Connected');
      state.lastPing = Date.now();
      // Send initial ping to discover capabilities
      sendWS({ cmd: 'ping' });
    };

    state.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleWSMessage(data);
        // Update latency
        state.wsLatency = Date.now() - state.lastPing;
        updateWSLatency();
      } catch (e) {
        // non-JSON messages ignored
      }
    };

    state.ws.onerror = (err) => {
      console.warn('[Portal] WS error:', err);
      updateWSStatus('disconnected', 'Pose Engine: Error');
    };

    state.ws.onclose = () => {
      console.log('[Portal] WS disconnected');
      state.wsConnected = false;
      updateWSStatus('disconnected', 'Pose Engine: Disconnected');
      if (reconnectTimer) clearTimeout(reconnectTimer);
      reconnectTimer = setTimeout(connectWS, 3000);
    };
  }

  function sendWS(msg) {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.lastPing = Date.now();
      state.ws.send(JSON.stringify(msg));
    }
  }

  function handleWSMessage(data) {
    // Pong with capabilities
    if (data.type === 'pong' || data.capabilities) {
      console.log('[Portal] Pose engine capabilities:', data.capabilities || data);
      notify('Pose engine connected • ' + ((data.capabilities && data.capabilities.models) || 'ready'));
      return;
    }

    // Skeleton data (from frame)
    if (data.type === 'skeleton' || data.skeleton || data.landmarks || data.keypoints) {
      const kps = data.keypoints || data.landmarks || (data.skeleton && data.skeleton.landmarks) || [];
      state.currentSkeleton = kps;
      state.sessionFrames++;
      updateMetric('metric-frames', state.sessionFrames);
      update3DSkeleton(kps);
      updateJointsFromSkeleton(kps);
      updateFPS();
      if ($('viewer-empty')) $('viewer-empty').style.display = 'none';
      if ($('viewer-fps')) $('viewer-fps').style.display = 'block';
    }

    // Handle skeleton response from frame cmd (different format)
    if (data.persons && data.persons.length > 0) {
      const kps = data.persons[0].keypoints || data.persons[0].landmarks || [];
      state.currentSkeleton = kps;
      state.sessionFrames++;
      updateMetric('metric-frames', state.sessionFrames);
      update3DSkeleton(kps);
      updateJointsFromSkeleton(kps);
      updateFPS();
      if ($('viewer-empty')) $('viewer-empty').style.display = 'none';
      if ($('viewer-fps')) $('viewer-fps').style.display = 'block';
      if (data.fps) updateFPSValue(data.fps);
    }

    // Clinical assessment
    if (data.type === 'clinical_assessment' || data.clinical_assessment || data.assessment) {
      const assessment = data.clinical_assessment || data.assessment || data;
      state.currentAssessment = assessment;
      updateClinicalPanel(assessment);
      notify('Clinical assessment received');
    }

    // FMS scores
    if (data.fms_scores || data.fms) {
      updateFMSScores(data.fms_scores || data.fms);
    }

    // Swarm agent output
    if (data.swarm_output || data.swarm || data.agent_output) {
      const swarm = data.swarm_output || data.swarm || data.agent_output;
      updateSwarmOutput(swarm);
    }
  }

  function updateWSStatus(status, text) {
    const dot = $('ws-dot');
    const txt = $('ws-text');
    if (dot) { dot.className = 'connection-dot ' + status; }
    if (txt) { txt.textContent = text; }
  }

  function updateWSLatency() {
    const el = $('ws-latency');
    if (el) el.textContent = state.wsLatency + 'ms';
  }

  // =========================================================================
  // 3D SKELETON RENDERER (Three.js)
  // =========================================================================
  function initThreeJS() {
    const container = $('three-container');
    if (!container || !window.THREE) return;

    const THREE = window.THREE;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0e17);
    scene.fog = new THREE.Fog(0x0a0e17, 3, 8);

    const camera = new THREE.PerspectiveCamera(55, container.clientWidth / container.clientHeight, 0.1, 20);
    camera.position.set(0, 1.2, 4.5);
    camera.lookAt(0, 0.8, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);

    // Lighting
    const ambient = new THREE.AmbientLight(0x304060, 0.6);
    scene.add(ambient);
    const keyLight = new THREE.DirectionalLight(0xf5e6d3, 1.0);
    keyLight.position.set(2, 4, 3);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.set(512, 512);
    scene.add(keyLight);
    const fillLight = new THREE.DirectionalLight(0x4488cc, 0.3);
    fillLight.position.set(-2, 1, -1);
    scene.add(fillLight);
    const rimLight = new THREE.DirectionalLight(0xf59e0b, 0.4);
    rimLight.position.set(0, 0.5, -2);
    scene.add(rimLight);

    // Ground grid
    const grid = new THREE.GridHelper(3, 30, 0x1a3050, 0x0f1a2a);
    grid.position.y = -0.4;
    scene.add(grid);

    // Circular platform
    const platformGeo = new THREE.CylinderGeometry(0.6, 0.65, 0.03, 32);
    const platformMat = new THREE.MeshStandardMaterial({ 
      color: 0x1a2035, 
      roughness: 0.6, 
      metalness: 0.3,
      emissive: 0x0a0e17,
      emissiveIntensity: 0.1
    });
    const platform = new THREE.Mesh(platformGeo, platformMat);
    platform.position.y = -0.41;
    platform.receiveShadow = true;
    scene.add(platform);

    // Skeleton group
    const skeletonGroup = new THREE.Group();
    scene.add(skeletonGroup);

    state.scene = scene;
    state.camera = camera;
    state.renderer = renderer;
    state.skeletonGroup = skeletonGroup;
    state.container = container;

    // Remove empty state
    const empty = $('viewer-empty');
    if (empty) empty.style.display = 'none';
    $('viewer-fps').style.display = 'block';

    // Animation loop
    function animate() {
      requestAnimationFrame(animate);
      if (state.autoRotate && state.camera) {
        state.rotateAngle += 0.002;
        const radius = 4.5;
        state.camera.position.x = Math.sin(state.rotateAngle) * radius;
        state.camera.position.z = Math.cos(state.rotateAngle) * radius;
        state.camera.lookAt(0, 0.8, 0);
      }
      if (state.renderer && state.scene && state.camera) {
        state.renderer.render(state.scene, state.camera);
      }
    }
    animate();

    // Resize handler
    window.addEventListener('resize', () => {
      if (!state.container || !state.camera || !state.renderer) return;
      state.camera.aspect = state.container.clientWidth / state.container.clientHeight;
      state.camera.updateProjectionMatrix();
      state.renderer.setSize(state.container.clientWidth, state.container.clientHeight);
    });
  }

  function update3DSkeleton(keypoints) {
    if (!state.skeletonGroup || !window.THREE) return;
    const THREE = window.THREE;

    // Clear previous skeleton
    while (state.skeletonGroup.children.length) {
      state.skeletonGroup.remove(state.skeletonGroup.children[0]);
    }

    if (!keypoints || !keypoints.length) return;

    // Map keypoints (handle both object format with x,y,z and array format)
    const kps = keypoints.map((kp, i) => {
      if (typeof kp.x !== 'undefined') {
        return { x: kp.x, y: kp.y, z: kp.z || 0, confidence: kp.confidence || kp.visibility || 0.8, id: kp.id || i };
      }
      if (Array.isArray(kp)) {
        return { x: kp[0], y: kp[1], z: kp[2] || 0, confidence: kp[3] || 0.8, id: i };
      }
      return null;
    }).filter(Boolean);

    const jointMap = {};
    const minConf = 0.3;

    // Create joint spheres
    for (const kp of kps) {
      if (kp.confidence < minConf) continue;
      const color = KEYPOINT_COLORS[kp.id] || '#ffffff';
      const mat = new THREE.MeshStandardMaterial({
        color: color,
        emissive: color,
        emissiveIntensity: 0.4,
        roughness: 0.3,
        metalness: 0.1
      });
      const mesh = new THREE.Mesh(new THREE.SphereGeometry(0.03, 12, 12), mat);
      // Convert normalized coordinates to 3D space
      mesh.position.set(
        (kp.x - 0.5) * 2.2,
        (1.0 - kp.y) * 1.8,
        (kp.z || 0) * 1.5
      );
      mesh.castShadow = true;
      state.skeletonGroup.add(mesh);
      jointMap[kp.id] = mesh.position;
    }

    // Draw bones
    const boneMat = new THREE.MeshStandardMaterial({
      color: 0xf59e0b,
      emissive: 0xf59e0b,
      emissiveIntensity: 0.2,
      roughness: 0.4,
      metalness: 0.3,
      transparent: true,
      opacity: 0.7
    });

    for (const [i, j] of SKELETON_CONNECTIONS) {
      const a = jointMap[i];
      const b = jointMap[j];
      if (!a || !b) continue;

      const mid = new THREE.Vector3().addVectors(a, b).multiplyScalar(0.5);
      const dir = new THREE.Vector3().subVectors(b, a);
      const len = dir.length();
      if (len < 0.001) continue;
      dir.normalize();

      const bone = new THREE.Mesh(
        new THREE.CylinderGeometry(0.012, 0.012, len, 6),
        boneMat.clone()
      );
      bone.position.copy(mid);
      bone.quaternion.setFromUnitVectors(
        new THREE.Vector3(0, 1, 0),
        dir
      );
      bone.castShadow = true;
      state.skeletonGroup.add(bone);
    }

    // Head sphere
    if (jointMap[0]) {
      const headMat = new THREE.MeshStandardMaterial({
        color: 0xf59e0b,
        emissive: 0xf59e0b,
        emissiveIntensity: 0.15,
        roughness: 0.2,
        metalness: 0.1,
        transparent: true,
        opacity: 0.25
      });
      const head = new THREE.Mesh(new THREE.SphereGeometry(0.09, 16, 16), headMat);
      head.position.copy(jointMap[0]);
      head.position.y += 0.03;
      state.skeletonGroup.add(head);
    }
  }

  // =========================================================================
  // PATIENT SELECTION
  // =========================================================================
  document.addEventListener('click', function(e) {
    const item = e.target.closest('.patient-item');
    if (!item) return;
    const patientId = item.dataset.patientId;
    if (!patientId) return;

    state.activePatientId = patientId;
    qsa('.patient-item').forEach(el => el.classList.remove('active'));
    item.classList.add('active');

    const name = item.querySelector('.patient-name').textContent;
    const lbl = $('current-patient-label');
    if (lbl) lbl.textContent = '— ' + name;

    // Reset clinical data for new patient
    resetClinicalData();
    notify('Selected patient: ' + name);
  });

  function resetClinicalData() {
    state.currentSkeleton = null;
    state.currentAssessment = null;
    state.fmsScores = {};
    state.jointAngles = {};
    state.swarmOutput = [];
    state.soapNote = null;
    state.sessionFrames = 0;
    updateMetric('metric-frames', 0);
    updateMetric('metric-quality', '--');
    updateMetric('metric-stability', '--');
    updateMetric('metric-compensations', 0);
    // Reset FMS scores
    qsa('#fms-scores .score').forEach(el => { el.textContent = '-'; el.style.color = 'var(--text-muted)'; });
    // Reset joint angles
    qsa('#joint-angles .deg').forEach(el => { el.textContent = '--'; el.style.color = 'var(--text-muted)'; });
    // Reset asymmetry
    qsa('#asymmetry-indicators .metric-value').forEach(el => { el.textContent = '--'; el.style.color = 'var(--text-muted)'; });
    // Reset chiro indicators
    qsa('#chiro-indicators .chiro-chip').forEach(el => { el.style.opacity = '0.3'; });
    // Reset SOAP
    const soap = $('soap-preview');
    if (soap) soap.innerHTML = '<div style="color:var(--text-muted);text-align:center;padding:20px 0"><i class="fas fa-file-alt" style="font-size:20px;display:block;margin-bottom:8px;opacity:0.3"></i>No SOAP note generated yet<br/><span style="font-size:10px">Click "Assess" to generate<br/>a clinical SOAP note</span></div>';
    // Reset swarm
    const swarm = $('swarm-log');
    if (swarm) swarm.innerHTML = '<div class="swarm-line" style="color:var(--text-muted);text-align:center;padding:20px 0"><i class="fas fa-network-wired" style="font-size:20px;display:block;margin-bottom:8px;opacity:0.3"></i>Swarm agents idle<br/><span style="font-size:10px">Connect to pose engine to activate<br/>multi-agent clinical analysis</span></div>';
    // Clear skeleton
    if (state.skeletonGroup) {
      while (state.skeletonGroup.children.length) {
        state.skeletonGroup.remove(state.skeletonGroup.children[0]);
      }
    }
  }

  // =========================================================================
  // TAB SWITCHING
  // =========================================================================
  window.switchTab = function(tabName) {
    state.activeTab = tabName;
    qsa('.panel-tab').forEach(t => t.classList.remove('active'));
    qsa('.panel-tab-content').forEach(c => c.style.display = 'none');
    const tab = qs('.panel-tab[data-tab="' + tabName + '"]');
    const content = $('tab-' + tabName);
    if (tab) tab.classList.add('active');
    if (content) content.style.display = 'block';
  };

  // =========================================================================
  // CLINICAL DATA UPDATES
  // =========================================================================
  function updateFMSScores(scores) {
    state.fmsScores = scores;
    const mapping = ['deep_squat', 'hurdle_step', 'inline_lunge', 'shoulder_mobility', 'aslr', 'tspu', 'rotary_stability'];
    const items = qsa('#fms-scores .fms-item');
    let total = 0;
    mapping.forEach((key, i) => {
      const score = scores[key] || scores[i] || null;
      if (items[i]) {
        const scoreEl = items[i].querySelector('.score');
        if (scoreEl) {
          scoreEl.textContent = score !== null ? score : '-';
          scoreEl.className = 'score ' + (score >= 3 ? 'good' : score >= 2 ? 'warn' : 'bad');
        }
      }
      if (typeof score === 'number') total += score;
    });
    // Total
    const totalEl = items[items.length - 1]?.querySelector('.score');
    if (totalEl && mapping.some((k, i) => scores[k] !== undefined)) {
      totalEl.textContent = total;
      totalEl.className = 'score ' + (total >= 14 ? 'good' : total >= 10 ? 'warn' : 'bad');
    }
  }

  function updateClinicalPanel(assessment) {
    // Movement quality
    if (assessment.movement_quality_score !== undefined) {
      updateMetric('metric-quality', Math.round(assessment.movement_quality_score) + '%');
    }
    // Stability
    if (assessment.stability_score !== undefined) {
      updateMetric('metric-stability', Math.round(assessment.stability_score) + '%');
    }
    // Compensations
    if (assessment.detected_compensations) {
      const count = Array.isArray(assessment.detected_compensations) ? assessment.detected_compensations.length : assessment.compensation_detected ? 1 : 0;
      updateMetric('metric-compensations', count);
    }
    // FMS
    if (assessment.fms_scores || assessment.fms) {
      updateFMSScores(assessment.fms_scores || assessment.fms);
    }
    // Joint angles
    if (assessment.joint_angles) {
      updateJointAngles(assessment.joint_angles);
    }
    // Chiro
    if (assessment.posture || assessment.spinal_alignment) {
      updateChiroAnalysis(assessment);
    }
  }

  function updateJointAngles(angles) {
    state.jointAngles = angles;
    const angleMap = {
      'shoulder_flexion': 0, 'shoulder_abduction': 1, 'elbow_flexion': 2,
      'hip_flexion': 3, 'hip_abduction': 4, 'knee_flexion': 5,
      'knee_extension': 6, 'ankle_dorsiflexion': 7
    };
    const rows = qsa('#joint-angles .joint-row');
    angles.forEach(angle => {
      const name = angle.joint_name?.toLowerCase().replace(/ /g, '_');
      const idx = angleMap[name];
      if (idx !== undefined && rows[idx]) {
        const degs = rows[idx].querySelectorAll('.deg');
        if (degs[0]) { degs[0].textContent = angle.left_angle !== undefined ? Math.round(angle.left_angle) + '°' : '--'; degs[0].style.color = getAngleColor(angle.left_angle, angle.normal_range); }
        if (degs[1]) { degs[1].textContent = angle.right_angle !== undefined ? Math.round(angle.right_angle) + '°' : '--'; degs[1].style.color = getAngleColor(angle.right_angle, angle.normal_range); }
      }
    });
    // Asymmetry
    if (angles.some(a => a.bilateral_difference !== undefined)) {
      updateAsymmetryIndicators(angles);
    }
  }

  function getAngleColor(value, range) {
    if (value === undefined || value === null) return 'var(--text-muted)';
    if (!range) return '#e2e8f0';
    const [min, max] = range;
    if (value >= min && value <= max) return '#10b981';
    if (value >= min * 0.7 && value <= max * 1.2) return '#f59e0b';
    return '#ef4444';
  }

  function updateJointsFromSkeleton(keypoints) {
    // Simple angle estimation from keypoints
    // In production, this comes from the pose engine; here we estimate on-the-fly
    if (!keypoints || keypoints.length < 17) return;
    
    const kps = [];
    keypoints.forEach((kp, i) => {
      if (typeof kp.x !== 'undefined') kps.push(kp);
      else if (Array.isArray(kp)) kps.push({ x: kp[0], y: kp[1], z: kp[2] || 0, id: i });
      else kps.push(null);
    });

    function getPos(idx) {
      if (!kps[idx]) return null;
      return { x: kps[idx].x, y: kps[idx].y, z: kps[idx].z || 0 };
    }

    function calcAngle(a, b, c) {
      if (!a || !b || !c) return null;
      const ba = { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
      const bc = { x: c.x - b.x, y: c.y - b.y, z: c.z - b.z };
      const dot = ba.x * bc.x + ba.y * bc.y + ba.z * bc.z;
      const magA = Math.sqrt(ba.x*ba.x + ba.y*ba.y + ba.z*ba.z);
      const magC = Math.sqrt(bc.x*bc.x + bc.y*bc.y + bc.z*bc.z);
      if (magA < 0.001 || magC < 0.001) return null;
      return Math.round(Math.acos(Math.max(-1, Math.min(1, dot / (magA * magC)))) * 180 / Math.PI);
    }

    // MediaPipe landmarks: 11=left_shoulder, 13=left_elbow, 15=left_wrist, 12=right_shoulder, 14=right_elbow, 16=right_wrist
    // 23=left_hip, 25=left_knee, 27=left_ankle, 24=right_hip, 26=right_knee, 28=right_ankle
    const angleMap = {
      'shoulder_flexion': 0, 'shoulder_abduction': 1, 'elbow_flexion': 2,
      'hip_flexion': 3, 'hip_abduction': 4, 'knee_flexion': 5,
      'knee_extension': 6, 'ankle_dorsiflexion': 7
    };
    const rows = qsa('#joint-angles .joint-row');
    const angles = [];

    // Elbow: shoulder-elbow-wrist
    const lElbow = calcAngle(getPos(11), getPos(13), getPos(15));
    const rElbow = calcAngle(getPos(12), getPos(14), getPos(16));
    if (rows[2]) {
      const degs = rows[2].querySelectorAll('.deg');
      if (degs[0] && lElbow !== null) { degs[0].textContent = lElbow + '°'; degs[0].style.color = '#f59e0b'; }
      if (degs[1] && rElbow !== null) { degs[1].textContent = rElbow + '°'; degs[1].style.color = '#f59e0b'; }
    }

    // Knee: hip-knee-ankle
    const lKnee = calcAngle(getPos(23), getPos(25), getPos(27));
    const rKnee = calcAngle(getPos(24), getPos(26), getPos(28));
    if (rows[5]) {
      const degs = rows[5].querySelectorAll('.deg');
      if (degs[0] && lKnee !== null) { degs[0].textContent = lKnee + '°'; degs[0].style.color = '#f59e0b'; }
      if (degs[1] && rKnee !== null) { degs[1].textContent = rKnee + '°'; degs[1].style.color = '#f59e0b'; }
    }
  }

  function updateAsymmetryIndicators(angles) {
    const rows = qsa('#asymmetry-indicators .metric-row');
    angles.forEach(angle => {
      if (angle.joint_name?.toLowerCase().includes('shoulder')) {
        if (rows[0]) { const v = rows[0].querySelector('.metric-value'); v.textContent = (angle.bilateral_difference || 0).toFixed(1) + ' cm'; v.style.color = '#f59e0b'; }
      }
      if (angle.joint_name?.toLowerCase().includes('hip')) {
        if (rows[1]) { const v = rows[1].querySelector('.metric-value'); v.textContent = (angle.bilateral_difference || 0).toFixed(1) + ' cm'; v.style.color = '#f59e0b'; }
      }
      if (angle.joint_name?.toLowerCase().includes('knee')) {
        if (rows[2]) { const v = rows[2].querySelector('.metric-value'); v.textContent = (angle.bilateral_difference || 0).toFixed(1) + '°'; v.style.color = '#f59e0b'; }
      }
    });
  }

  function updateChiroAnalysis(assessment) {
    const posture = assessment.posture || assessment;
    const indicators = qsa('#chiro-indicators .chiro-chip');
    if (posture.anterior_head) indicators[0].style.opacity = '1';
    if (posture.rounded_shoulders) indicators[1].style.opacity = '1';
    if (posture.forward_pelvic_tilt) indicators[2].style.opacity = '1';
    if (posture.scoliosis_indicator) indicators[3].style.opacity = '1';
    if (posture.thoracic_kyphosis) indicators[4].style.opacity = '1';
    if (posture.lumbar_lordosis) indicators[5].style.opacity = '1';

    // Update spinal metrics
    const chiroRows = qsa('#tab-chiro .panel-section:last-child .metric-row');
    if (posture.spine && chiroRows.length >= 5) {
      if (chiroRows[0]) chiroRows[0].querySelector('.metric-value').textContent = (posture.spine.cervical || '--') + '°';
      if (chiroRows[1]) chiroRows[1].querySelector('.metric-value').textContent = (posture.spine.thoracic || '--') + '°';
      if (chiroRows[2]) chiroRows[2].querySelector('.metric-value').textContent = (posture.spine.lumbar || '--') + '°';
      if (chiroRows[3]) chiroRows[3].querySelector('.metric-value').textContent = (posture.spine.cobb || '--') + '°';
      if (chiroRows[4]) chiroRows[4].querySelector('.metric-value').textContent = (posture.plumb_deviation || '--') + ' cm';
    }
  }

  function updateSwarmOutput(swarm) {
    state.swarmOutput = Array.isArray(swarm) ? swarm : [swarm];
    const log = $('swarm-log');
    if (!log) return;
    log.innerHTML = state.swarmOutput.map(entry => {
      const agent = entry.agent || entry.name || 'Agent';
      const finding = entry.finding || entry.output || JSON.stringify(entry);
      return '<div class="swarm-line"><span class="agent">[' + agent + ']</span> <span class="finding">' + escapeHtml(String(finding)) + '</span></div>';
    }).join('');
  }

  function updateMetric(id, value) {
    const el = $(id);
    if (el) el.textContent = value;
  }

  let fpsCount = 0;
  let fpsLast = Date.now();
  function updateFPS() {
    fpsCount++;
    const now = Date.now();
    if (now - fpsLast >= 1000) {
      updateFPSValue(fpsCount);
      fpsCount = 0;
      fpsLast = now;
    }
  }
  function updateFPSValue(fps) {
    const el = $('fps-value');
    if (el) el.textContent = fps;
  }

  // =========================================================================
  // SESSION CONTROLS
  // =========================================================================
  window.toggleRecording = function() {
    if (!state.recording) {
      startRecording();
    }
  };

  function startRecording() {
    state.recording = true;
    state.sessionStart = Date.now();
    state.sessionFrames = 0;
    $('btn-start').style.display = 'none';
    $('btn-stop').style.display = '';
    const timer = $('session-timer');
    timer.textContent = '00:00';
    timer.classList.add('recording');
    state.sessionTimer = setInterval(() => {
      const elapsed = Math.floor((Date.now() - state.sessionStart) / 1000);
      const mins = String(Math.floor(elapsed / 60)).padStart(2, '0');
      const secs = String(elapsed % 60).padStart(2, '0');
      $('session-timer').textContent = mins + ':' + secs;
    }, 200);
    notify('Recording started');
  }

  window.stopRecording = function() {
    state.recording = false;
    $('btn-start').style.display = '';
    $('btn-stop').style.display = 'none';
    const timer = $('session-timer');
    timer.classList.remove('recording');
    if (state.sessionTimer) clearInterval(state.sessionTimer);
    state.sessionTimer = null;
    notify('Recording stopped • ' + state.sessionFrames + ' frames captured');
  };

  window.captureFrame = function() {
    if (!state.currentSkeleton) {
      notify('No skeleton data to capture', 'warn');
      return;
    }
    notify('Frame captured • ' + state.currentSkeleton.length + ' landmarks');
  };

  window.resetFilters = function() {
    resetClinicalData();
    notify('Filters reset');
  };

  window.requestAssessment = function() {
    if (!state.wsConnected) {
      notify('Pose engine not connected', 'warn');
      return;
    }
    // Send assess command (would require a frame to be sent; for demo we just send the command)
    sendWS({ cmd: 'assess', patient_id: state.activePatientId });
    notify('Assessment requested...', 'info');
  };

  window.generateSOAP = function() {
    if (!state.currentAssessment) {
      // Generate mock SOAP for demo
      const mockSOAP = {
        subjective: 'Patient reports ' + (qs('.patient-item.active .patient-condition')?.textContent || 'condition') + ' with pain level ' + (state.fmsScores.pain || '3') + '/10.',
        objective: 'Real-time pose analysis completed. ' + (state.sessionFrames || 0) + ' frames analyzed. Joint angles within functional ranges with noted asymmetries.',
        assessment: 'Functional movement assessment reveals movement pattern deficits consistent with clinical presentation. FMS total score: ' + ((Object.values(state.fmsScores).reduce((a, b) => a + (typeof b === 'number' ? b : 0), 0)) || 'pending') + '/21.',
        plan: '1. Continue prescribed exercise program\n2. Progress to phase 2 exercises as tolerated\n3. Re-assess in 2 weeks\n4. Monitor pain levels during activity'
      };
      updateSOAPPreview(mockSOAP);
      return;
    }
    updateSOAPPreview(state.currentAssessment);
  };

  function updateSOAPPreview(assessment) {
    state.soapNote = assessment;
    const soap = $('soap-preview');
    if (!soap) return;
    soap.innerHTML = 
      '<div class="soap-s"><b>S - Subjective:</b><br/>' + escapeHtml(assessment.subjective || assessment.chief_complaint || 'No subjective data') + '</div>' +
      '<div class="soap-o"><b>O - Objective:</b><br/>' + escapeHtml(assessment.objective || assessment.clinical_notes || assessment.objective_findings || 'No objective data') + '</div>' +
      '<div class="soap-a"><b>A - Assessment:</b><br/>' + escapeHtml(assessment.assessment || assessment.assessment_summary || 'Pending analysis') + '</div>' +
      '<div class="soap-p"><b>P - Plan:</b><br/>' + escapeHtml(assessment.plan || 'See exercise prescriptions') + '</div>';
    switchTab('soap');
  }

  window.copySOAP = function() {
    if (!state.soapNote) {
      notify('No SOAP note to copy', 'warn');
      return;
    }
    const s = state.soapNote;
    const text = 'SOAP NOTE\\n=======\\n\\nS: ' + (s.subjective || s.chief_complaint || '') + '\\n\\nO: ' + (s.objective || s.clinical_notes || s.objective_findings || '') + '\\n\\nA: ' + (s.assessment || s.assessment_summary || '') + '\\n\\nP: ' + (s.plan || '') + '\\n';
    navigator.clipboard.writeText(text).then(() => notify('SOAP note copied to clipboard'));
  };

  // =========================================================================
  // CAMERA CONTROLS
  // =========================================================================
  window.resetCamera = function() {
    if (state.camera) {
      state.camera.position.set(0, 1.2, 4.5);
      state.camera.lookAt(0, 0.8, 0);
      state.rotateAngle = 0;
    }
  };

  window.toggleAutoRotate = function() {
    state.autoRotate = !state.autoRotate;
    state.rotateAngle = 0;
    notify('Auto-rotate: ' + (state.autoRotate ? 'ON' : 'OFF'));
  };

  // =========================================================================
  // NOTIFICATIONS
  // =========================================================================
  function notify(msg, type) {
    type = type || 'info';
    const icon = type === 'warn' ? '⚠️' : type === 'error' ? '❌' : '✅';
    const notif = document.createElement('div');
    notif.className = 'notification';
    notif.innerHTML = '<span>' + icon + '</span> ' + msg;
    document.body.appendChild(notif);
    setTimeout(() => {
      notif.style.opacity = '0';
      notif.style.transition = 'opacity 0.3s';
      setTimeout(() => notif.remove(), 300);
    }, 3000);
  }

  function escapeHtml(str) {
    if (!str) return '';
    return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  // =========================================================================
  // MOUSE CONTROLS for 3D viewer
  // =========================================================================
  let isDragging = false;
  let prevMouse = { x: 0, y: 0 };

  document.addEventListener('DOMContentLoaded', function() {
    const container = $('three-container');
    if (!container) return;

    container.addEventListener('mousedown', function(e) {
      isDragging = true;
      prevMouse = { x: e.clientX, y: e.clientY };
    });

    window.addEventListener('mousemove', function(e) {
      if (!isDragging || !state.camera) return;
      const dx = e.clientX - prevMouse.x;
      const dy = e.clientY - prevMouse.y;
      prevMouse = { x: e.clientX, y: e.clientY };
      state.rotateAngle -= dx * 0.005;
      if (state.camera) {
        const radius = Math.sqrt(state.camera.position.x ** 2 + state.camera.position.z ** 2);
        state.camera.position.x = Math.sin(state.rotateAngle) * radius;
        state.camera.position.z = Math.cos(state.rotateAngle) * radius;
        state.camera.position.y = Math.max(0.3, Math.min(3, state.camera.position.y - dy * 0.01));
        state.camera.lookAt(0, 0.8, 0);
      }
    });

    window.addEventListener('mouseup', function() {
      isDragging = false;
    });

    // Scroll zoom
    container.addEventListener('wheel', function(e) {
      if (!state.camera) return;
      e.preventDefault();
      const dir = new window.THREE.Vector3().subVectors(
        new window.THREE.Vector3(0, 0.8, 0),
        state.camera.position
      ).normalize();
      const zoom = e.deltaY > 0 ? 0.2 : -0.2;
      state.camera.position.addScaledVector(dir, zoom);
    });
  });

  // =========================================================================
  // INIT
  // =========================================================================
  function init() {
    // Initialize 3D scene
    if (window.THREE) {
      initThreeJS();
    } else {
      // THREE.js loads async, wait for it
      const check = setInterval(() => {
        if (window.THREE) {
          clearInterval(check);
          initThreeJS();
        }
      }, 100);
    }

    // Connect WebSocket
    connectWS();

    // Periodic ping for latency measurement
    setInterval(() => {
      if (state.wsConnected) {
        sendWS({ cmd: 'ping' });
      }
    }, 10000);

    console.log('[ProviderPortal] Initialized');
  }

  // Start when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
`

export default ProviderPortal
