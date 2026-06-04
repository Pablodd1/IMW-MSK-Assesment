export const clinicalStyles = `
  :root { --navy:#0a1628; --panel:rgba(13,27,49,.86); --card:rgba(26,32,53,.76); --border:rgba(148,163,184,.16); --text:#e2e8f0; --muted:#94a3b8; --blue:#3b82f6; --blue2:#60a5fa; --gold:#f59e0b; --green:#22c55e; --red:#ef4444; }
  * { box-sizing:border-box; }
  body { margin:0; min-height:100vh; background:var(--navy); color:var(--text); font-family:Inter,ui-sans-serif,system-ui,sans-serif; letter-spacing:0; }
  @keyframes fadeIn { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
  .clinical-shell { min-height:100vh; padding:18px; background:radial-gradient(circle at 20% 0%, rgba(59,130,246,.13), transparent 32%), var(--navy); }
  .clinical-top { display:flex; justify-content:space-between; gap:12px; align-items:center; margin-bottom:14px; }
  .clinical-top h1 { margin:0; color:var(--gold); font-size:1.05rem; }
  .clinical-top p { margin:4px 0 0; color:var(--muted); font-size:.78rem; }
  .clinical-nav { display:flex; gap:8px; flex-wrap:wrap; }
  .clinical-nav a, .clinical-btn { color:#dbeafe; text-decoration:none; border:1px solid rgba(96,165,250,.28); background:rgba(59,130,246,.12); padding:7px 10px; border-radius:7px; font-size:.75rem; cursor:pointer; transition:all .18s ease; }
  .clinical-nav a:hover, .clinical-btn:hover { border-color:var(--gold); color:#fff; transform:translateY(-1px); }
  .clinical-grid { display:grid; grid-template-columns:repeat(12,1fr); gap:12px; animation:fadeIn .28s ease-out both; }
  .clinical-card { background:var(--card); border:1px solid var(--border); border-radius:8px; padding:14px; backdrop-filter:blur(14px); box-shadow:0 18px 50px rgba(0,0,0,.18); }
  .clinical-card h2, .clinical-card h3 { margin:0 0 10px; color:var(--gold); font-size:.82rem; text-transform:uppercase; letter-spacing:.08em; }
  .span-3 { grid-column:span 3; } .span-4 { grid-column:span 4; } .span-5 { grid-column:span 5; } .span-6 { grid-column:span 6; } .span-7 { grid-column:span 7; } .span-8 { grid-column:span 8; } .span-12 { grid-column:span 12; }
  .metric { display:flex; justify-content:space-between; gap:12px; padding:8px 0; border-bottom:1px solid rgba(148,163,184,.08); font-size:.78rem; }
  .metric:last-child { border-bottom:0; }
  .metric span:first-child { color:var(--muted); }
  .metric strong { color:#fff; }
  .pill { display:inline-flex; align-items:center; gap:6px; padding:4px 8px; border-radius:999px; border:1px solid var(--border); color:#bfdbfe; background:rgba(59,130,246,.1); font-size:.7rem; }
  .phase-heel_strike { color:#bfdbfe; } .phase-midstance { color:#60a5fa; } .phase-toe_off { color:#fcd34d; } .phase-swing { color:#86efac; }
  .clinical-table { width:100%; border-collapse:collapse; font-size:.76rem; }
  .clinical-table th, .clinical-table td { text-align:left; padding:8px; border-bottom:1px solid rgba(148,163,184,.1); }
  .clinical-table th { color:var(--muted); font-weight:600; }
  .clinical-input, .clinical-select { width:100%; background:rgba(10,22,40,.76); border:1px solid var(--border); color:var(--text); border-radius:7px; padding:8px 10px; font-size:.78rem; }
  .clinical-controls { display:grid; grid-template-columns:repeat(4,1fr); gap:8px; }
  .skeleton-canvas { width:100%; aspect-ratio:4/3; min-height:260px; border-radius:8px; background:#050a16; border:1px solid rgba(96,165,250,.18); }
  .heat-row { display:grid; grid-template-columns:90px 1fr 38px; gap:8px; align-items:center; margin:8px 0; font-size:.75rem; color:var(--muted); }
  .bar { height:8px; background:rgba(148,163,184,.16); border-radius:999px; overflow:hidden; }
  .bar > span { display:block; height:100%; background:linear-gradient(90deg,var(--red),var(--gold),var(--green)); }
  .exercise-card { display:grid; grid-template-columns:92px 1fr; gap:12px; }
  .exercise-card img { width:92px; height:74px; object-fit:cover; border-radius:7px; border:1px solid var(--border); }
  .exercise-card h3 { text-transform:none; letter-spacing:0; font-size:.9rem; margin-bottom:4px; }
  .muted { color:var(--muted); }
  @media (max-width: 980px) { .clinical-grid { grid-template-columns:1fr; } .span-3,.span-4,.span-5,.span-6,.span-7,.span-8,.span-12 { grid-column:span 1; } .clinical-top { align-items:flex-start; flex-direction:column; } .clinical-controls { grid-template-columns:1fr 1fr; } }
`;

export function ClinicalLayout({ title, subtitle, children }: { title: string; subtitle: string; children: any }) {
  return (
    <html lang="en">
      <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>{title} - IMW PhysioMotion</title>
        <style>{clinicalStyles}</style>
      </head>
      <body>
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

