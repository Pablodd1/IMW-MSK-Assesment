# Vercel Setup Guide (Hybrid Deployment)

Since your application uses Cloudflare D1 (Database) and R2 (Storage), the most efficient way to use Vercel is for the **Frontend Only**, while keeping the **Backend on Cloudflare**.

## Step 1: Deploy Backend to Cloudflare
Ensure your backend is running on Cloudflare Workers.
1.  Run `npm run deploy:prod` to deploy your worker.
2.  Note the production URL (e.g., `https://physiomotion.yourname.workers.dev`).

## Step 2: Configure Vercel Project
1.  **Build Command:** `npm run build`
2.  **Output Directory:** `dist`
3.  **Install Command:** `npm install`

## Step 3: Configure API Rewrites
To make your frontend talk to the Cloudflare backend, you need to rewrite API requests.

**Option A: Using `vercel.json` (Already Created)**
We have created a `vercel.json` file in your project root. You must edit it to point to your Cloudflare URL.

```json
{
  "rewrites": [
    {
      "source": "/api/:path*",
      "destination": "https://YOUR-CLOUDFLARE-WORKER.workers.dev/api/:path*"
    },
    {
      "source": "/(.*)",
      "destination": "/index.html"
    }
  ]
}
```
*Action:* Update `vercel.json` with your actual Cloudflare Worker URL.

**Option B: Environment Variables (If using Next.js/Middleware)**
Since this is a Vite app, Option A is the standard way.

## Step 4: Environment Variables on Vercel
Even for a frontend deployment, you might need public environment variables.
- Go to Vercel Project Settings > Environment Variables.
- Add any `VITE_` prefixed variables if your frontend needs them.

## Step 5: Fix Hardcoded Salt (Security)
Before deploying, fix the security issue found in the audit:
1.  Open `src/index.tsx`.
2.  Replace `'physiomotion-salt-2025'` with `c.env.PASSWORD_SALT` (backend) or `import.meta.env.VITE_PASSWORD_SALT` (frontend - but salts shouldn't be on frontend).
    *   *Note: Since the auth logic is in the backend (Cloudflare), you should set this secret in Cloudflare, not Vercel.*
    *   Command: `npx wrangler secret put PASSWORD_SALT`
