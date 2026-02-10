# Final Deployment Checklist

## 🛑 Pre-Flight Checks
- [ ] **Read the Report:** Reviewed `PRE-DEPLOYMENT-REPORT.md` and understood the architectural constraints.
- [ ] **Backend Status:** Cloudflare Worker is deployed and verified working (`npm run deploy:prod`).
- [ ] **URL:** Captured the Cloudflare Worker URL (e.g., `https://physiomotion...`).
- [ ] **Vercel Config:** Updated `vercel.json` with the Cloudflare URL in the `rewrites` section.
- [ ] **Security:** Rotated the hardcoded salt if possible (optional for MVP, required for production).

## 🚀 Deployment Steps
1.  [ ] **Push Code:** Commit and push changes to GitHub.
2.  [ ] **Import to Vercel:** Import the repository in Vercel.
3.  [ ] **Configure Build:**
    - Framework Preset: Vite
    - Build Command: `npm run build`
    - Output Directory: `dist`
4.  [ ] **Environment:** Add any frontend-specific env vars (if any).
5.  [ ] **Deploy:** Click Deploy.

## 🧪 Post-Deployment Verification
- [ ] **Frontend Load:** Visit the Vercel URL. Does the homepage load?
- [ ] **API Check:** Open DevTools Network tab. Are requests to `/api/*` succeeding (200 OK)?
- [ ] **Auth Check:** Try to Login/Register.
- [ ] **Performance:** Verify that the redirects/rewrites aren't adding too much latency.
