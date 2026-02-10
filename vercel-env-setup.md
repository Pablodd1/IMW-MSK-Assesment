# Vercel Environment Setup Guide

## ⚠️ CRITICAL ARCHITECTURE WARNING ⚠️
This application is currently architected for **Cloudflare Workers** using **D1 Database** and **R2 Storage** bindings.
The code heavily relies on `c.env.DB` and `c.env.R2` which are specific to the Cloudflare Workers runtime.

**Deploying this code directly to Vercel will result in runtime errors.**
The Vercel Node.js or Edge runtime does not automatically provide `c.env.DB` bindings.

## Migration Paths

### Option 1: Hybrid Deployment (Recommended for Immediate Action)
* **Frontend:** Deploy the static assets (HTML/JS/CSS) to Vercel.
* **Backend:** Keep the API (`/api/*`) on Cloudflare Workers.
* **Configuration:** Configure `vercel.json` to rewrite `/api/*` requests to your Cloudflare Worker URL.

### Option 2: Full Migration to Vercel
* **Database:** You must migrate from D1 to a Vercel-supported database (e.g., Vercel Postgres, Supabase, Neon).
* **Storage:** You can keep R2 but must access it via the S3-compatible API using the `aws-sdk` or similar, instead of the native `R2Bucket` binding.
* **Code Refactor:** You will need to rewrite all `c.env.DB.prepare(...)` calls to use a standard SQL client or ORM.

## Environment Variables for Option 2 (Full Migration)

If you choose to refactor and migrate, you will need to set the following Environment Variables in your Vercel Project Settings:

| Variable | Description | Example |
|----------|-------------|---------|
| `DATABASE_URL` | Connection string for your PostgreSQL/MySQL database | `postgres://user:pass@host:5432/db` |
| `R2_ACCESS_KEY_ID` | Access Key ID for R2 (S3 API) | `738...` |
| `R2_SECRET_ACCESS_KEY` | Secret Access Key for R2 (S3 API) | `a8d...` |
| `R2_BUCKET_NAME` | Your bucket name | `webapp-videos` |
| `R2_ENDPOINT` | Your account-specific R2 endpoint | `https://<account_id>.r2.cloudflarestorage.com` |

## Next Steps
Please refer to `PRE-DEPLOYMENT-REPORT.md` for a detailed analysis of the necessary changes.
