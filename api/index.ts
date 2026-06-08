import type { VercelRequest, VercelResponse } from '@vercel/node';
import app from '../dist/src/index.js';

/**
 * Vercel Serverless Adapter for Hono
 * Bridges Vercel's Node.js request/response to Hono's fetch API.
 * This ensures ALL routes defined in src/index.ts work on Vercel.
 */

export default async function handler(req: VercelRequest, res: VercelResponse) {
  const url = new URL(
    req.url || '/',
    `http://${req.headers.host || 'localhost'}`
  );

  // Build headers
  const headers = new Headers();
  for (const [key, value] of Object.entries(req.headers)) {
    if (value === undefined) continue;
    if (Array.isArray(value)) {
      for (const v of value) headers.append(key, v);
    } else {
      headers.append(key, String(value));
    }
  }

  // Build body
  let body: BodyInit | undefined = undefined;
  if (req.method !== 'GET' && req.method !== 'HEAD') {
    if (typeof req.body === 'string') {
      body = req.body;
    } else if (Buffer.isBuffer(req.body)) {
      body = new Uint8Array(req.body);
    } else if (req.body !== undefined && req.body !== null) {
      body = JSON.stringify(req.body);
      if (!headers.has('content-type')) {
        headers.set('content-type', 'application/json');
      }
    }
  }

  // Build Web Request
  const request = new Request(url.toString(), {
    method: req.method,
    headers,
    body,
  });

  try {
    const response = await app.fetch(request);

    // Copy status
    res.status(response.status);

    // Copy headers
    response.headers.forEach((value: string, key: string) => {
      res.setHeader(key, value);
    });

    // Send body
    const responseBody = await response.text();
    res.send(responseBody);
  } catch (err: any) {
    console.error('[VERCEL ADAPTER] Error:', err);
    res.status(500).json({
      success: false,
      error: 'Internal server error',
      message: err.message,
    });
  }
}
