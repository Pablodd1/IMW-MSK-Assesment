export default function handler(req: any, res: any) {
  res.status(200).json({
    success: true,
    model: {
      name: 'phi4-mini:3.8b / deepseek-r1:7b',
      description: 'Boxer3D multi-agent MSK assessment',
      specialists: [
        { id: 'upper-extremity', name: 'Upper Extremity & Spine Specialist', persona: 'orthopedic surgeon' },
        { id: 'lower-extremity', name: 'Lower Extremity & Gait Specialist', persona: 'sports medicine physician' },
        { id: 'spine-core', name: 'Spine, Core & Posture Specialist', persona: 'chiropractic and rehabilitation physician' },
        { id: 'auditor', name: 'Clinical Auditor & Scribe', persona: 'senior physiatrist' }
      ]
    }
  })
}
