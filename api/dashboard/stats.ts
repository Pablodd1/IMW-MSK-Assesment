export default function handler(req: any, res: any) {
  const stats = {
    total_patients: 3,
    active_patients: 2,
    total_assessments: 3,
    total_exercises: 10,
    recent_activity: []
  }
  res.status(200).json({ success: true, data: stats })
}
