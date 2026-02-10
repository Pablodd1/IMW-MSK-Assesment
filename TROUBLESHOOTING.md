# Troubleshooting Vercel Deployment

## Common Errors & Solutions

### Error: `TypeError: Cannot read properties of undefined (reading 'DB')`
**Cause:** The application is trying to access `c.env.DB` in an environment where the D1 database binding is missing (e.g., Vercel Node.js runtime).
**Solution:** This code must run on Cloudflare Workers. See `VERCEL-SETUP-GUIDE.md` for the Hybrid Deployment strategy.

### Error: `404 Not Found` on API Routes
**Cause:**
1. The Vercel deployment is serving static files, and no serverless function is handling `/api`.
2. The `rewrites` in `vercel.json` are missing or incorrect.
**Solution:** Check `vercel.json` and ensure it proxies `/api/*` to your running Cloudflare Worker URL.

### Error: `Command "wrangler" not found`
**Cause:** You might be trying to run `npm run deploy` inside Vercel's build pipeline.
**Solution:** Vercel should only run `npm run build`. Do not try to run Wrangler commands (deploy, dev) inside Vercel.

### Error: `500 Internal Server Error` (Database/Storage)
**Cause:** If you migrated logic to Vercel but kept D1 calls, they will fail.
**Solution:** You must verify that your backend logic is running on the platform it was designed for (Cloudflare).

### Issues with "PhysioMotion-Salt"
**Cause:** The audit flagged a hardcoded salt.
**Solution:** If you changed this to an env var, make sure you added it to your Cloudflare Secrets (`wrangler secret put PASSWORD_SALT`).
