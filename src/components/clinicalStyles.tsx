export const clinicalStyles = `
  :root {
    --deep: #020617;
    --void: #030b1a;
    --panel: rgba(3,11,26,.92);
    --card: rgba(5,17,38,.84);
    --glass: rgba(15,30,55,.72);
    --border: rgba(96,165,250,.14);
    --border-glow: rgba(96,165,250,.32);
    --text: #e0e7ff;
    --muted: #7b8fbb;
    --blue: #3b82f6;
    --blue2: #60a5fa;
    --blue3: #93bbfd;
    --blue-glow: rgba(59,130,246,.24);
    --cyan: #22d3ee;
    --green: #34d399;
    --gold: #f59e0b;
    --red: #f87171;
    --pink: #f472b6;
    --radius: 10px;
    --radius-sm: 6px;
    --font: 'Inter', 'SF Pro Display', -apple-system, system-ui, sans-serif;
    --mono: 'JetBrains Mono', 'Fira Code', monospace;
  }

  *, *::before, *::after { box-sizing: border-box; }

  /* ================================================================
     BACKGROUND LAYERS
     ================================================================ */
  body {
    margin: 0;
    min-height: 100vh;
    background: var(--deep);
    color: var(--text);
    font-family: var(--font);
    -webkit-font-smoothing: antialiased;
    overflow-x: hidden;
  }

  /* Hex grid background */
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    z-index: 0;
    background-image:
      url("data:image/svg+xml,%3Csvg width='60' height='52' viewBox='0 0 60 52' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M30 0L60 15v22L30 52 0 37V15z' fill='none' stroke='rgba(96,165,250,.06)' stroke-width='0.5'/%3E%3C/svg%3E");
    background-size: 60px 52px;
    pointer-events: none;
  }

  /* Radial vignette */
  body::after {
    content: '';
    position: fixed;
    inset: 0;
    z-index: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(59,130,246,.06), transparent 60%),
                radial-gradient(ellipse at 80% 80%, rgba(34,211,238,.04), transparent 50%);
    pointer-events: none;
  }

  /* Scan line overlay */
  .scan-lines {
    position: fixed;
    inset: 0;
    z-index: 1;
    pointer-events: none;
    background: repeating-linear-gradient(
      0deg,
      transparent,
      transparent 2px,
      rgba(96,165,250,.012) 2px,
      rgba(96,165,250,.012) 4px
    );
  }

  @keyframes fadeIn { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:translateY(0); } }
  @keyframes pulseGlow { 0%,100% { box-shadow: 0 0 8px var(--blue-glow); } 50% { box-shadow: 0 0 20px rgba(96,165,250,.18), 0 0 40px rgba(59,130,246,.06); } }
  @keyframes scanPulse { 0% { opacity:.3; } 50% { opacity:.7; } 100% { opacity:.3; } }
  @keyframes dataFlow { 0% { background-position:0 0; } 100% { background-position:0 200%; } }
  @keyframes borderPulse { 0%,100% { border-color: rgba(96,165,250,.14); } 50% { border-color: rgba(96,165,250,.32); } }
  @keyframes slideRight { from { width:0; } to { width:100%; } }
  @keyframes skeletonGlow { 0%,100% { filter: drop-shadow(0 0 8px rgba(59,130,246,.5)); } 50% { filter: drop-shadow(0 0 18px rgba(96,165,250,.7)); } }

  /* ================================================================
     SHELL — Main container
     ================================================================ */
  .clinical-shell {
    position: relative;
    z-index: 2;
    min-height: 100vh;
    padding: 16px 18px 32px;
  }

  /* ================================================================
     TOP BAR — HUD style
     ================================================================ */
  .clinical-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 16px;
    margin-bottom: 18px;
    padding: 14px 18px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    animation: fadeIn .35s ease-out;
  }

  .clinical-top h1 {
    margin: 0;
    font-size: .95rem;
    font-weight: 700;
    letter-spacing: .02em;
    background: linear-gradient(135deg, var(--blue2), var(--cyan));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  .clinical-top p {
    margin: 3px 0 0;
    font-size: .72rem;
    color: var(--muted);
    letter-spacing: .01em;
  }

  /* ================================================================
     NAV — Holographic tabs
     ================================================================ */
  .clinical-nav {
    display: flex;
    gap: 4px;
    flex-wrap: wrap;
  }

  .clinical-nav a {
    position: relative;
    color: var(--muted);
    text-decoration: none;
    padding: 7px 12px;
    border-radius: var(--radius-sm);
    font-size: .72rem;
    font-weight: 500;
    letter-spacing: .03em;
    border: 1px solid transparent;
    transition: all .22s ease;
    overflow: hidden;
  }

  .clinical-nav a::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(59,130,246,.08), rgba(34,211,238,.04));
    opacity: 0;
    transition: opacity .22s ease;
  }

  .clinical-nav a:hover {
    color: #fff;
    border-color: var(--border-glow);
    background: rgba(59,130,246,.08);
  }

  .clinical-nav a:hover::before { opacity: 1; }

  /* ================================================================
     GRID — 12-column responsive
     ================================================================ */
  .clinical-grid {
    display: grid;
    grid-template-columns: repeat(12, 1fr);
    gap: 12px;
    animation: fadeIn .38s ease-out both;
  }

  .span-3 { grid-column: span 3; }
  .span-4 { grid-column: span 4; }
  .span-5 { grid-column: span 5; }
  .span-6 { grid-column: span 6; }
  .span-7 { grid-column: span 7; }
  .span-8 { grid-column: span 8; }
  .span-9 { grid-column: span 9; }
  .span-12 { grid-column: span 12; }

  /* ================================================================
     CARDS — Holographic glass panels
     ================================================================ */
  .clinical-card {
    position: relative;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px;
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    transition: border-color .3s ease, box-shadow .3s ease;
    overflow: hidden;
  }

  .clinical-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(96,165,250,.18), transparent);
  }

  .clinical-card:hover {
    border-color: var(--border-glow);
    box-shadow: 0 0 24px rgba(59,130,246,.06), inset 0 0 32px rgba(59,130,246,.02);
  }

  /* Active/recording card */
  .clinical-card.live {
    animation: pulseGlow 2s ease-in-out infinite;
    border-color: var(--border-glow);
  }

  .clinical-card h2, .clinical-card h3 {
    margin: 0 0 10px;
    font-size: .78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .12em;
    color: var(--blue3);
  }

  /* ================================================================
     METRICS — HUD data rows
     ================================================================ */
  .metric {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    padding: 9px 0;
    border-bottom: 1px solid rgba(96,165,250,.06);
    font-size: .76rem;
    font-family: var(--mono);
  }

  .metric:last-child { border-bottom: 0; }

  .metric span:first-child {
    color: var(--muted);
    font-family: var(--font);
    font-size: .7rem;
    letter-spacing: .03em;
  }

  .metric strong {
    color: #fff;
    font-weight: 500;
    position: relative;
  }

  .metric strong::after {
    content: '';
    display: inline-block;
    width: 5px;
    height: 5px;
    border-radius: 50%;
    margin-left: 6px;
    background: var(--green);
    opacity: .8;
    animation: scanPulse 2s ease-in-out infinite;
  }

  .metric.live strong::after { background: var(--cyan); animation-duration: 1s; }

  /* Phase colors */
  .phase-heel_strike { color: var(--blue3); }
  .phase-midstance { color: var(--blue2); }
  .phase-toe_off { color: var(--gold); }
  .phase-swing { color: var(--cyan); }

  /* ================================================================
     PILLS / BADGES
     ================================================================ */
  .pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 10px;
    border-radius: 999px;
    border: 1px solid var(--border);
    color: var(--blue3);
    background: rgba(59,130,246,.08);
    font-size: .68rem;
    font-family: var(--mono);
    letter-spacing: .04em;
  }

  .pill.active {
    border-color: var(--cyan);
    color: var(--cyan);
    background: rgba(34,211,238,.08);
    animation: scanPulse 2s ease-in-out infinite;
  }

  .pill.warn {
    border-color: var(--gold);
    color: var(--gold);
    background: rgba(245,158,11,.08);
  }

  /* ================================================================
     BUTTONS — Cyber controls
     ================================================================ */
  .clinical-btn {
    position: relative;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    color: var(--blue3);
    background: rgba(59,130,246,.06);
    border: 1px solid var(--border);
    padding: 8px 14px;
    border-radius: var(--radius-sm);
    font-size: .72rem;
    font-weight: 500;
    font-family: var(--font);
    cursor: pointer;
    transition: all .2s ease;
    overflow: hidden;
    letter-spacing: .03em;
  }

  .clinical-btn::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(59,130,246,.1), transparent);
    opacity: 0;
    transition: opacity .2s ease;
  }

  .clinical-btn:hover {
    border-color: var(--border-glow);
    color: #fff;
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(59,130,246,.08);
  }

  .clinical-btn:hover::before { opacity: 1; }

  .clinical-btn:active { transform: translateY(0); }

  .clinical-btn:disabled {
    opacity: .35;
    cursor: not-allowed;
    transform: none;
  }

  .clinical-btn.primary {
    background: linear-gradient(135deg, rgba(59,130,246,.18), rgba(34,211,238,.06));
    border-color: var(--border-glow);
    color: #fff;
  }

  .clinical-btn.danger {
    color: var(--red);
    border-color: rgba(248,113,113,.2);
    background: rgba(248,113,113,.06);
  }

  .clinical-btn.danger:hover {
    border-color: rgba(248,113,113,.4);
    box-shadow: 0 4px 16px rgba(248,113,113,.08);
  }

  /* ================================================================
     SKELETON CANVAS — The 3D viewport
     ================================================================ */
  .skeleton-canvas {
    width: 100%;
    aspect-ratio: 4/3;
    min-height: 280px;
    border-radius: 8px;
    background: var(--void);
    border: 1px solid var(--border);
    box-shadow: inset 0 0 60px rgba(3,11,26,.8);
    animation: skeletonGlow 3s ease-in-out infinite;
  }

  .skeleton-viewport {
    position: relative;
    background: #000;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid var(--border);
  }

  .skeleton-viewport video {
    width: 100%;
    display: block;
  }

  .skeleton-viewport canvas {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
  }

  /* HUD overlay corners */
  .viewport-hud {
    position: absolute;
    pointer-events: none;
    z-index: 5;
  }

  .viewport-hud.top-left { top: 8px; left: 8px; }
  .viewport-hud.top-right { top: 8px; right: 8px; }
  .viewport-hud.bottom-left { bottom: 8px; left: 8px; }
  .viewport-hud.bottom-right { bottom: 8px; right: 8px; }

  .hud-label {
    font-family: var(--mono);
    font-size: 9px;
    color: var(--blue3);
    background: rgba(2,6,23,.72);
    border: 1px solid rgba(96,165,250,.12);
    padding: 4px 8px;
    border-radius: 4px;
    letter-spacing: .08em;
    backdrop-filter: blur(8px);
  }

  .hud-label .dot {
    display: inline-block;
    width: 4px;
    height: 4px;
    border-radius: 50%;
    background: var(--cyan);
    margin-right: 4px;
    animation: scanPulse 1.5s ease-in-out infinite;
  }

  /* ================================================================
     DATA STREAM — Animated metrics bar
     ================================================================ */
  .data-stream {
    display: flex;
    gap: 16px;
    padding: 10px 14px;
    background: rgba(5,17,38,.6);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    font-family: var(--mono);
    font-size: .66rem;
    color: var(--muted);
    overflow-x: auto;
  }

  .data-stream .stream-item {
    display: flex;
    align-items: center;
    gap: 6px;
    white-space: nowrap;
  }

  .data-stream .stream-value {
    color: var(--blue2);
    font-weight: 500;
  }

  .waveform-bar {
    display: flex;
    align-items: flex-end;
    gap: 1px;
    height: 18px;
  }

  .waveform-bar .bar-slice {
    width: 2px;
    background: var(--blue2);
    border-radius: 1px;
    animation: dataFlow 1.2s ease-in-out infinite;
  }

  /* ================================================================
     TABLES
     ================================================================ */
  .clinical-table {
    width: 100%;
    border-collapse: collapse;
    font-size: .74rem;
  }

  .clinical-table th {
    text-align: left;
    padding: 9px 10px;
    color: var(--muted);
    font-weight: 500;
    font-size: .66rem;
    letter-spacing: .08em;
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
  }

  .clinical-table td {
    padding: 9px 10px;
    border-bottom: 1px solid rgba(96,165,250,.04);
    color: var(--text);
  }

  .clinical-table tr:hover td {
    background: rgba(59,130,246,.03);
  }

  /* ================================================================
     INPUTS
     ================================================================ */
  .clinical-input, .clinical-select {
    width: 100%;
    background: rgba(5,17,38,.6);
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: var(--radius-sm);
    padding: 9px 12px;
    font-size: .76rem;
    font-family: var(--font);
    transition: border-color .2s ease, box-shadow .2s ease;
  }

  .clinical-input:focus, .clinical-select:focus {
    outline: none;
    border-color: var(--border-glow);
    box-shadow: 0 0 0 3px rgba(59,130,246,.08);
  }

  .clinical-controls {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
  }

  /* ================================================================
     PROGRESS / BARS
     ================================================================ */
  .bar {
    height: 6px;
    background: rgba(96,165,250,.08);
    border-radius: 999px;
    overflow: hidden;
  }

  .bar > span {
    display: block;
    height: 100%;
    background: linear-gradient(90deg, var(--blue), var(--cyan));
    border-radius: 999px;
    transition: width .6s cubic-bezier(0.4, 0, 0.2, 1);
  }

  .bar.warn > span { background: linear-gradient(90deg, var(--gold), var(--red)); }

  .heat-row {
    display: grid;
    grid-template-columns: 90px 1fr 38px;
    gap: 8px;
    align-items: center;
    margin: 8px 0;
    font-size: .73rem;
    color: var(--muted);
  }

  /* ================================================================
     EXERCISE CARDS
     ================================================================ */
  .exercise-card {
    display: grid;
    grid-template-columns: 92px 1fr;
    gap: 12px;
  }

  .exercise-card img {
    width: 92px;
    height: 74px;
    object-fit: cover;
    border-radius: 7px;
    border: 1px solid var(--border);
  }

  .exercise-card h3 {
    text-transform: none;
    letter-spacing: 0;
    font-size: .88rem;
    margin-bottom: 4px;
    color: var(--text);
  }

  .muted { color: var(--muted); }

  /* ================================================================
     PHOTO TIMELINE — Scrollable captures
     ================================================================ */
  .photo-scroll {
    max-height: 420px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .photo-scroll img {
    border-radius: 6px;
    border: 1px solid var(--border);
    transition: border-color .2s ease;
  }

  .photo-scroll img:hover { border-color: var(--border-glow); }

  /* ================================================================
     MOBILE
     ================================================================ */
  @media (max-width: 980px) {
    .clinical-grid { grid-template-columns: 1fr; }
    .span-3, .span-4, .span-5, .span-6, .span-7, .span-8, .span-9, .span-12 { grid-column: span 1; }
    .clinical-top { flex-direction: column; align-items: flex-start; }
    .clinical-nav { gap: 2px; }
    .clinical-controls { grid-template-columns: 1fr 1fr; }
    .data-stream { flex-wrap: wrap; }
  }

  /* Scrollbar styling */
  ::-webkit-scrollbar { width: 4px; height: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: rgba(96,165,250,.16); border-radius: 2px; }
  ::-webkit-scrollbar-thumb:hover { background: rgba(96,165,250,.28); }
`;

export function ClinicalLayout({ title, subtitle, children }: { title: string; subtitle: string; children: any }) {
  return (
    <html lang="en">
      <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>{title} - IMW PhysioMotion</title>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet" />
        <style>{clinicalStyles}</style>
      </head>
      <body>
        <div class="scan-lines"></div>
        <main class="clinical-shell">
          <div class="clinical-top">
            <div>
              <h1>{title}</h1>
              <p>{subtitle}</p>
            </div>
            <nav class="clinical-nav">
              <a href="/provider">Provider</a>
              <a href="/gait">Gait</a>
              <a href="/muscle">Muscle</a>
              <a href="/clinical-tests">Tests</a>
              <a href="/exercises">Exercises</a>
              <a href="/progress">Progress</a>
              <a href="/reports">Reports</a>
            </nav>
          </div>
          {children}
        </main>
      </body>
    </html>
  );
}
