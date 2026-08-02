# IMW-MSK Assessment — PhysioMotion v2.0

**AI-Powered Medical Movement Assessment Platform**
*Real-time 3D joint tracking · Multi-agent AI analysis · SOAP note generation · CPT billing*

---

## 🏥 What It Does

### Core Features
| Feature | Status | Description |
|---------|--------|-------------|
| **Patient Management** | ✅ Live | Full CRUD, medical history, insurance, emergency contacts |
| **Movement Assessments** | ✅ Live | Initial, progress, discharge, athletic performance types |
| **Exercise Library** | ✅ Live | 50+ PT exercises with descriptions, cues, contraindications |
| **Exercise Prescriptions** | ✅ Live | Sets/reps/frequency, clinical reasoning, compliance tracking |
| **Session Tracking** | ✅ Live | Patient completes sessions, pain levels, adherence scoring |
| **SOAP Note Generation** | ✅ Live | Auto-generates clinical notes from assessment data |
| **Dashboard Analytics** | ✅ Live | Patient counts, active patients, recent activity feed |
| **Progress Tracking** | ✅ Live | Pain trends, functional scores, adherence rates |
| **MSK Analysis Pipeline** | ✅ Live | Multi-agent swarm: body-region specialists + vector RAG |
| **AI Specialist Swarm** | ✅ Live | 4 specialists (Upper/Lower/Spine/Auditor) via Ollama |
| **Boxer3D Pipeline** | ✅ Live | Python agents: pose detection → biomechanics → report |
| **3D Skeleton Rendering** | ✅ Live | Three.js real-time 33-joint MediaPipe overlay |
| **Camera Integration** | ✅ Live | Femto Mega, Azure Kinect, Orbbec, Standard Webcam |
| **JWT Authentication** | ✅ Live | Bcrypt hashing, role-based access, demo mode for dev |
| **CPT/ICD-10 Billing** | ✅ Schema ready | Codes table, billable events, Medicare 8-minute rule |

### Frontend Pages (public/)
- `index.html` — Landing page
- `login.html` — Clinician authentication
- `dashboard.html` — Real-time 3D tracking + analytics
- `patients.html` — Patient records & search
- `assessment.html` — Movement assessment workflow
- `exercises.html` — Exercise library browser
- `intake.html` — New patient intake form
- `boxer3d.html` — Boxer3D pipeline interface

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FRONTEND (Static HTML)                   │
│  Three.js + TensorFlow.js + MediaPipe Pose → 33 joints      │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP/JSON
┌───────────────────────────▼─────────────────────────────────┐
│                  BACKEND (Hono + TypeScript)                 │
│                  Deployed on Vercel Serverless                │
│                                                              │
│  Routes:                                                     │
│  ├── /patients, /assessments, /exercises, /prescriptions    │
│  ├── /msk-analysis (Multi-agent swarm)                      │
│  ├── /ai (Ollama specialist agents)                         │
│  ├── /dashboard/stats, /patients/:id/progress               │
│  └── /auth/login, /auth/me                                  │
└───────────────────────────┬─────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
┌──────────────────────┐    ┌──────────────────────────┐
│    Supabase (pg)     │    │     Ollama (local)        │
│  11 tables, HIPAA    │    │  llama3.1, qwen, gemma   │
│  audit logging       │    │  AI specialist analysis   │
└──────────────────────┘    └──────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Node.js 20.x
- Supabase project (PostgreSQL)
- Ollama running locally (for AI analysis)

### Local Development
```bash
cd IMW-MSK-Assesment
npm install
cp .env.example .env    # Edit with your credentials
npm run dev             # Starts on port 3001
```

### Environment Variables (.env)
```env
# Database — Supabase PostgreSQL
DATABASE_URL=postgresql://postgres:[PASSWORD]@db.[PROJECT].supabase.co:5432/postgres

# AI — Gemini for cloud, Ollama for local
GEMINI_API_KEY=your_gemini_key
OLLAMA_HOST=http://localhost:11434

# Auth
JWT_SECRET=your-32+character-jwt-secret
AUTH_SECRET=your-32+character-auth-secret

# App
NODE_ENV=production
PORT=3001
DEMO_MODE=true           # Set false when Supabase connected
```

### Database Setup
```bash
# Initialize Supabase tables
DATABASE_URL="postgresql://..." node scripts/init-db.js
```
This creates all 11 tables: clinics, users, patients, medical_history, assessments, movement_tests, movement_analysis, exercises, prescribed_exercises, exercise_sessions, cpt_codes, billable_events.

---

## 📡 Deployment

### Vercel (Current)
- **Project:** `imw-msk-assesment`
- **URL:** `https://imw-msk-assesment.vercel.app`
- **Health:** `GET /api/health` → `{"status":"healthy"}`
- **Node:** 20.x, runs via `tsx index.ts`

```bash
# Deploy
npx vercel --prod
```

### Environment Variables in Vercel
Set in Vercel Dashboard → Project → Settings → Environment Variables:
- `DATABASE_URL` — Supabase PostgreSQL connection string
- `GEMINI_API_KEY` — For cloud AI (already set)
- `JWT_SECRET` — JWT signing secret
- `NODE_ENV` — `production`
- `DEMO_MODE` — `false` (when Supabase connected)

---

## 🔌 Supabase Setup

### Connection
- **Project URL:** `https://swromegqjbakdiftzwum.supabase.co`
- **Database:** PostgreSQL 15+
- **Password:** Set in `.env` as `DATABASE_URL`

### Connection String Format
```
postgresql://postgres:[PASSWORD]@db.swromegqjbakdiftzwum.supabase.co:5432/postgres
```
> **Note:** Direct connection uses IPv6. If IPv6 is unavailable (WSL), use the Supabase pooler:
> ```
> postgresql://postgres.swromegqjbakdiftzwum:[PASSWORD]@aws-0-us-east-1.pooler.supabase.com:6543/postgres
> ```
> Pooler must be enabled in Supabase Dashboard → Project Settings → Database.

### Tables Created
| Table | Purpose |
|-------|---------|
| `clinics` | Multi-tenant clinic records |
| `users` | Clinicians, admins, staff |
| `patients` | Demographics, insurance, portal access |
| `medical_history` | Conditions, medications, surgeries, pain |
| `assessments` | SOAP notes, scores, billing, 8-minute rule |
| `movement_tests` | Individual tests with camera data |
| `movement_analysis` | Joint angles, ROM, asymmetry, AI analysis |
| `exercises` | 50+ exercise library with cues & CPT codes |
| `prescribed_exercises` | Home exercise programs, compliance tracking |
| `exercise_sessions` | Patient completions, form quality, pain |
| `cpt_codes` | Billing codes with Medicare reimbursement |
| `billable_events` | Billing records linked to assessments |

---

## 🤖 AI Pipelines

### 1. MSK Analysis (Multi-Agent Swarm)
`POST /msk-analysis`
- Accepts MediaPipe pose landmarks (33 points)
- Routes to body-region specialist agents
- Vector RAG for clinical knowledge retrieval
- Cryptographic integrity verification
- Returns structured assessment with ICD-10/CPT codes

### 2. AI Specialist Swarm (Ollama)
`POST /ai/analyze`
- 4 specialist agents analyze same pose data:
  - Upper Extremity & Spine
  - Lower Extremity & Gait
  - Spine, Core & Posture
  - Clinical Auditor & Scribe
- Auditor reconciles findings into final SOAP note

### 3. Boxer3D Pipeline (Python)
- `pose_agent.py` — YOLO pose detection
- `biomechanics_agent.py` — Joint angle + ROM analysis
- `report_agent.py` — Clinical report generation
- `orchestrator.py` — Pipeline coordinator

---

## 📊 API Reference

### Authentication
All patient/assessment routes require `Authorization: Bearer <token>` header.

### Key Endpoints
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/dashboard/stats` | Dashboard analytics |
| `GET` | `/patients` | List all patients |
| `GET` | `/patients/:id` | Patient detail |
| `POST` | `/patients` | Create patient |
| `PUT` | `/patients/:id` | Update patient |
| `GET` | `/patients/:id/medical-history` | Medical history |
| `POST` | `/patients/:id/medical-history` | Add medical history |
| `GET` | `/patients/:id/assessments` | Patient assessments |
| `GET` | `/patients/:id/prescriptions` | Exercise prescriptions |
| `GET` | `/patients/:id/sessions` | Exercise sessions |
| `GET` | `/patients/:id/progress` | Progress metrics |
| `POST` | `/assessments` | Create assessment |
| `GET` | `/assessments` | List all assessments |
| `GET` | `/assessments/:id` | Assessment detail |
| `POST` | `/assessments/:id/tests` | Add movement test |
| `POST` | `/assessments/:id/generate-note` | Generate SOAP note |
| `GET` | `/exercises` | Exercise library |
| `POST` | `/prescriptions` | Prescribe exercise |
| `POST` | `/exercise-sessions` | Record session |
| `POST` | `/auth/login` | Login |
| `GET` | `/auth/me` | Current user |
| `POST` | `/msk-analysis` | Run MSK pipeline |
| `POST` | `/ai/analyze` | AI specialist analysis |

---

## ⚠️ Current Limitations & Next Steps

This is a clinical prototype. It must not be represented as medical-grade or used for clinical decision-making until the validation, safety, and compliance gates in the [medical-grade capability review](docs/MSK_3D_MEDICAL_GRADE_REVIEW_2026-05-04.md) are complete.

### 🔴 Critical
1. **Supabase connection** — App runs in demo/mock mode. Pooler connection needs troubleshooting (IPv6 unreachable from WSL, pooler says "tenant not found"). Need to:
   - Enable pooler in Supabase Dashboard settings
   - Or deploy from environment with IPv6
   - Add `DATABASE_URL` to Vercel env vars

2. **Custom domain** — No domain assigned. Options: `imwmsk.com`, subdomain of `buildinginnovation.us` or `medicalbillingmb.com`

### 🟡 Should Do
3. **Replace mock data with real DB queries** — `src/index.ts` uses `MOCK_PATIENTS` instead of PostgreSQL
4. **Dedicated route files** — Patient/assessment logic is inline in `index.ts`; separate routes exist in `src/routes/` but aren't all wired in
5. **Auth hardening** — Demo mode accepts any credentials; production auth needs JWT + bcrypt with real users table
6. **Frontend modernization** — Static HTML pages work but a React/Vite frontend would enable real-time WebSocket tracking

### 🟢 Nice to Have
7. **CI/CD** — GitHub Actions for automated Vercel deploys on push
8. **Monitoring** — Sentry/LogRocket for error tracking
9. **PDF export** — Assessment reports as downloadable PDFs
10. **Mobile PWA** — Progressive Web App for tablet use in clinic

---

## 🔒 Security

- Bcrypt password hashing (12 rounds)
- JWT authentication
- HIPAA-compliant audit logging (schema ready)
- Zod input validation (routes)
- CORS restricted to configured origins
- Cryptographic integrity for movement data

---

**Built for Innovate Medical Wellness**
`v2.0.0-demo` · Deployed on Vercel · PostgreSQL on Supabase
