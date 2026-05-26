/**
 * Boxer3D Multi-Agent API Route
 * Integrates the Python agent pipeline into the Node.js/Express backend.
 */
import { Router, Request, Response } from 'express';
import { execFileSync, exec } from 'child_process';
import path from 'path';
import fs from 'fs';

const POSE_ENGINE_DIR = path.resolve(process.cwd(), 'pose-engine');
const PYTHON = 'python3';

export function createAgentApi(): Router {
  const router = Router();

  // Run full Boxer3D pipeline
  router.post('/api/boxer3d/analyze', async (req: Request, res: Response) => {
    try {
      const { videoPath, patientName, model } = req.body;
      if (!videoPath) return res.status(400).json({ error: 'videoPath required' });

      const outputDir = path.join(POSE_ENGINE_DIR, 'output', Date.now().toString());
      fs.mkdirSync(outputDir, { recursive: true });

      const stdout = execFileSync(PYTHON, [
        path.join(POSE_ENGINE_DIR, 'orchestrator.py'),
        videoPath,
        '-o', outputDir,
        '-p', patientName || 'Patient',
        '-m', model || 'yolo11n-pose'
      ], { timeout: 300000, cwd: POSE_ENGINE_DIR, maxBuffer: 10 * 1024 * 1024 }).toString();
      console.log(`[Boxer3D] Done:`, stdout.slice(0, 200));

      const reportPath = path.join(outputDir, 'report_output.json');
      const summaryPath = path.join(outputDir, 'patient_summary.txt');

      const report = fs.existsSync(reportPath) 
        ? JSON.parse(fs.readFileSync(reportPath, 'utf-8'))
        : null;
      const summary = fs.existsSync(summaryPath)
        ? fs.readFileSync(summaryPath, 'utf-8')
        : null;

      res.json({
        status: 'complete',
        outputDir,
        report,
        summary,
        agents: ['pose_agent', 'biomechanics_agent', 'report_agent']
      });
    } catch (err: any) {
      res.status(500).json({ error: err.message, stderr: err.stderr?.toString() });
    }
  });

  // Health check
  router.get('/api/boxer3d/status', (req: Request, res: Response) => {
    const agents = [
      { name: 'pose_agent', file: 'pose_agent.py', exists: fs.existsSync(path.join(POSE_ENGINE_DIR, 'pose_agent.py')) },
      { name: 'biomechanics_agent', file: 'biomechanics_agent.py', exists: fs.existsSync(path.join(POSE_ENGINE_DIR, 'biomechanics_agent.py')) },
      { name: 'report_agent', file: 'report_agent.py', exists: fs.existsSync(path.join(POSE_ENGINE_DIR, 'report_agent.py')) },
      { name: 'orchestrator', file: 'orchestrator.py', exists: fs.existsSync(path.join(POSE_ENGINE_DIR, 'orchestrator.py')) },
    ];
    res.json({ status: 'online', version: '3.0.0', agents });
  });

  return router;
}
