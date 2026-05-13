export default function handler(req: any, res: any) {
  res.status(200).json({
    success: true,
    status: 'healthy',
    ts: new Date().toISOString()
  })
}
