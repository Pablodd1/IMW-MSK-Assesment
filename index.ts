import { serve } from '@hono/node-server';
import app from './src/index.js';
import * as dotenv from 'dotenv';

// Load environment variables
dotenv.config();

const port = process.env.PORT ? parseInt(process.env.PORT, 10) : 3001;

console.log(`[PHYSIOMOTION] Starting Unified API Server...`);
console.log(`[PHYSIOMOTION] Port: ${port}`);
console.log(`[PHYSIOMOTION] Mode: ${process.env.NODE_ENV || 'development'}`);

serve({
  fetch: app.fetch,
  port
}, (info) => {
  console.log(`[PHYSIOMOTION] Server running at http://localhost:${info.port}`);
});

export default app;