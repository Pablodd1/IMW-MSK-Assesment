import type { VercelRequest, VercelResponse } from '@vercel/node';

export default function handler(req: VercelRequest, res: VercelResponse) {
  res.json({
    success: true,
    route: '/capture',
    message: 'Movement Photo Capture — Live',
    timestamp: new Date().toISOString(),
  });
}
