# Deployment SOP

## Overview
This application is a full-stack Hono + Node application deployed on Vercel.
The database is Neon PostgreSQL accessed via `@neondatabase/serverless`.

## Local Development
1. Ensure you have Node v20.x (`nvm use 20`).
2. Install dependencies: `npm install`
3. Set environment variables: copy `.env.example` to `.env` and fill in `DATABASE_URL`, `JWT_SECRET`, `AUTH_SECRET`.
4. Run locally using Vite: `npm run dev`

## Deployment to Vercel
1. Connect the GitHub repository to your Vercel project.
2. Vercel Settings Configuration:
   - **Framework Preset**: Vite
   - **Root Directory**: `./` (Root)
   - **Build Command**: `npm run build`
   - **Install Command**: `npm install` (default)
   - **Output Directory**: `dist`
   - **Node Version**: 20.x
3. Add Environment Variables in Vercel UI (Project Settings -> Environment Variables):
   - `DATABASE_URL` (production and preview)
   - `JWT_SECRET` (production and preview)
   - `AUTH_SECRET` (production and preview)

## Verification / Health Endpoints
- **System Health**: Visit `/api/healthz` to verify database connectivity. Expected response:
  ```json
  {
    "status": "ok",
    "timestamp": "2024-05-18...",
    "db": "connected",
    "environment": "production",
    "version": "1.0.0"
  }
  ```
- **Core Flows**:
  - `POST /api/auth/login`: Test authentication flow.
  - `GET /api/patients`: Test reading from the database.
  - `POST /api/patients`: Test creating a new record.

## Future Changes SOP
- **Branching Model**: Use feature branches off `main`. Once tested locally, squash merge into `main`.
- **Pre-Commit**: Always ensure `npm run typecheck` passes before committing. Do not commit build artifacts or secrets.
- **Dependency Drift**: Do not install packages without testing `npm run build` locally. Use only `npm` to avoid lockfile conflicts.
