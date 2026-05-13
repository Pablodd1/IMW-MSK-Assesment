export default function handler(req: any, res: any) {
  const exercises = [
    { id: 'core_01', name: 'Dead Bug', category: 'core', difficulty: 'beginner' },
    { id: 'knee_01', name: 'Heel Slides', category: 'knee', difficulty: 'beginner' },
    { id: 'quad_01', name: 'Quad Sets', category: 'knee', difficulty: 'beginner' },
    { id: 'shoulder_01', name: 'Sleeper Stretch', category: 'shoulder', difficulty: 'beginner' },
    { id: 'scap_01', name: 'Scapular Retraction', category: 'shoulder', difficulty: 'beginner' },
    { id: 'glute_01', name: 'Glute Bridge', category: 'hip', difficulty: 'beginner' },
    { id: 'hip_01', name: '90/90 Hip Stretch', category: 'hip', difficulty: 'intermediate' },
    { id: 'rotator_01', name: 'External Rotation with Band', category: 'shoulder', difficulty: 'intermediate' },
    { id: 'posture_01', name: 'Thoracic Extension', category: 'posture', difficulty: 'beginner' },
    { id: 'func_01', name: 'Sit to Stand', category: 'functional', difficulty: 'beginner' }
  ]
  const category = req.query.category
  const filtered = category ? exercises.filter(e => e.category === category) : exercises
  res.status(200).json({ success: true, data: filtered })
}
