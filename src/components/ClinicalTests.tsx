import { ClinicalLayout } from './clinicalStyles.js';
import { type ClinicalTestResult } from '../utils/clinical.js';

const tests: ClinicalTestResult[] = [
  { test: 'Y-Balance Test', score: 91, maxScore: 100, findings: ['Anterior reach symmetric', 'Posterolateral reach mildly limited left'], measurements: { anterior: 92, posteromedial: 94, posterolateral: 87 } },
  { test: 'Overhead Squat Assessment', score: 2, maxScore: 3, findings: ['Mild forward trunk lean', 'No pain response'], measurements: { kneeValgusIndex: 0.04 } },
  { test: 'Single Leg Stance', score: 2, maxScore: 3, findings: ['Mild sway after 15 seconds'], measurements: { swayCmEstimate: 11 } },
  { test: 'Spine ROM', score: 2, maxScore: 3, findings: ['Thoracic rotation limited'], measurements: { combinedRomDeg: 34 } },
  { test: 'Apley Scratch Shoulder Mobility', score: 3, maxScore: 3, findings: ['Symmetric shoulder reach'], measurements: { wristHeightAsymmetry: 8 } },
];

export function ClinicalTests({ results = tests }: { results?: ClinicalTestResult[] }) {
  return (
    <ClinicalLayout title="Clinical Test Selection" subtitle="Y-Balance, OHSA, single-leg stance, spine ROM, and shoulder mobility screens.">
      <section class="clinical-grid">
        <div class="clinical-card span-4">
          <h2>Test Panel</h2>
          {results.map((result) => (
            <label class="metric" style="cursor:pointer">
              <span><input type="checkbox" checked /> {result.test}</span>
              <strong>{result.score}/{result.maxScore}</strong>
            </label>
          ))}
          <button class="clinical-btn" style="width:100%;margin-top:12px" onclick="runSelectedTests()">Run Selected Tests</button>
        </div>
        <div class="clinical-card span-8">
          <h2>Results</h2>
          <table class="clinical-table">
            <thead><tr><th>Test</th><th>Score</th><th>Findings</th><th>Measurements</th></tr></thead>
            <tbody id="testRows">
              {results.map((result) => (
                <tr>
                  <td>{result.test}</td>
                  <td>{result.score}/{result.maxScore}</td>
                  <td>{result.findings.join('; ')}</td>
                  <td>{Object.entries(result.measurements).map(([k, v]) => `${k}: ${v}`).join(', ')}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
      <script dangerouslySetInnerHTML={{ __html: `function runSelectedTests(){ alert('Selected clinical tests queued for the next assess command.'); }` }} />
    </ClinicalLayout>
  );
}

