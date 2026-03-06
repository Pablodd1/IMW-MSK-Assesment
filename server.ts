import { serve } from '@hono/node-server';
import app from './src/index';

const port = process.env.PORT ? parseInt(process.env.PORT, 10) : 3000;

console.log(`Server starting on port ${port}...`);

serve({
  fetch: app.fetch,
  port
});
