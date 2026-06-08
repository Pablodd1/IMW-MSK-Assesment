import type { VercelRequest, VercelResponse } from '@vercel/node';

export default function handler(req: VercelRequest, res: VercelResponse) {
  res.json({
    success: true,
    status: 'ok',
    timestamp: new Date().toISOString(),
    route: req.url,
  });
}
