import { ClinicalLayout } from './clinicalStyles.js';

export function ReportGenerator() {
  return (
    <ClinicalLayout title="Clinical Report Generator" subtitle="PDF report with findings, SOAP, billing codes, exercise plan, and visit comparison.">
      <section class="clinical-grid">
        <div class="clinical-card span-5">
          <h2>Report Inputs</h2>
          <div class="metric"><span>Patient</span><strong>John Smith</strong></div>
          <div class="metric"><span>Date</span><strong>{new Date().toLocaleDateString()}</strong></div>
          <div class="metric"><span>Clinician</span><strong>Innovate Medical Wellness</strong></div>
          <div class="metric"><span>Included sections</span><strong>FMS, gait, MMT, ROM, SOAP, codes, exercises</strong></div>
          <button class="clinical-btn" onclick="generatePdf()" style="width:100%;margin-top:12px">Generate PDF</button>
        </div>
        <div class="clinical-card span-7">
          <h2>Preview</h2>
          <div id="reportPreview" style="background:#fff;color:#111827;border-radius:8px;padding:20px;min-height:420px">
            <h2 style="margin:0;color:#0a1628">Innovate Medical Wellness</h2>
            <p style="margin:4px 0 16px;color:#374151">IMW-MSK PhysioMotion Clinical Movement Report</p>
            <h3>Patient: John Smith</h3>
            <p><strong>Objective:</strong> FMS 16/21, gait cadence 108 spm, stride length 72 cm, pelvic tilt 3 deg, MMT left hip abduction 3/5.</p>
            <p><strong>Assessment:</strong> Improving lower-quarter control with residual hip abductor weakness and mild gait asymmetry.</p>
            <p><strong>Plan:</strong> Continue progressive strengthening, balance retraining, and gait mechanics program.</p>
            <p><strong>ICD-10:</strong> M25.50, R26.89. <strong>CPT:</strong> 97110, 97112, 97750.</p>
          </div>
        </div>
      </section>
      <script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js"></script>
      <script dangerouslySetInnerHTML={{ __html: reportScript }} />
    </ClinicalLayout>
  );
}

const reportScript = `
function generatePdf(){
  const jsPDF = window.jspdf && window.jspdf.jsPDF;
  if(!jsPDF){ window.print(); return; }
  const doc = new jsPDF();
  doc.setTextColor(10,22,40); doc.setFontSize(16); doc.text('Innovate Medical Wellness', 14, 18);
  doc.setTextColor(59,130,246); doc.setFontSize(12); doc.text('IMW-MSK PhysioMotion Clinical Movement Report', 14, 27);
  doc.setTextColor(17,24,39); doc.setFontSize(10);
  const lines = [
    'Patient: John Smith',
    'Date: ' + new Date().toLocaleDateString(),
    'Clinician: Innovate Medical Wellness',
    '',
    'FMS: 16/21. ROM: lumbar flexion 58 deg. Gait: cadence 108 spm, stride 72 cm.',
    'MMT: left hip abduction 3/5; shoulder and ankle groups 4-5/5.',
    'SOAP Assessment: Improving lower-quarter control with residual hip abductor weakness.',
    'Plan: strengthening, balance retraining, gait mechanics, home exercise progression.',
    'ICD-10: M25.50, R26.89. CPT: 97110, 97112, 97750.',
    'Exercise Prescription: side-lying hip abduction, short foot drill, single-leg reach.'
  ];
  let y=40; lines.forEach(line=>{ doc.text(line,14,y,{maxWidth:180}); y += line ? 8 : 5; });
  doc.save('imw-physiomotion-clinical-report.pdf');
}
`;

