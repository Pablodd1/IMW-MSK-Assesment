export default function handler(req: any, res: any) {
  res.status(200).json({ ok: true, msg: 'Test endpoint works', ts: new Date().toISOString() })
}
