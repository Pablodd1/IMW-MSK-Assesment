# IMW-MSK Assessment App — Comprehensive Security & HIPAA Compliance Code Review
**Date:** 2026-06-02  
**Reviewer:** Automated Security Analysis  
**Scope:** src/index.ts, src/routes/*.ts, src/middleware/*.ts, src/lib/database.ts, src/utils/*.ts, src/database.ts, src/env.ts  
**Lines Reviewed:** ~6,200+ TypeScript  

---

## EXECUTIVE SUMMARY

The IMW-MSK codebase is a Hono-based Node.js/TypeScript medical platform with AI-powered pose detection. It exhibits a **mixed security posture**: several well-intentioned security and HIPAA controls exist (audit logging, input sanitization, bcrypt hashing, session timeout logic, role-based middleware), but **critical gaps remain** that would prevent HIPAA compliance and create significant attack surface in production. The most severe issues are **hardcoded cryptographic secrets**, **demo-mode authentication bypasses**, **inconsistent authorization enforcement**, **missing rate limiting on sensitive endpoints**, and **Telegram notification leakage of PHI**.

**Overall Risk Rating: HIGH** — Not production-ready for PHI without remediation.

---

## SEVERITY LEGEND

| Severity | Description |
|----------|-------------|
| **CRITICAL** | Immediate exploitability; data breach or system compromise likely |
| **HIGH** | Significant security/HIPAA impact; should be fixed before production |
| **MEDIUM** | Moderate risk; exploitable under specific conditions |
| **LOW** | Minor issue; defense-in-depth improvement |
| **INFO** | Recommendation or best-practice gap |

---

## 1. AUTHENTICATION & AUTHORIZATION VULNERABILITIES

### 1.1 CRITICAL — Demo Mode Authentication Bypass in Production
- **File:** `src/index.ts:417-426`, `src/middleware/auth.ts:116-121`
- **Issue:** The `/auth/login` route and `authMiddleware` both contain a development demo bypass that activates when `NODE_ENV !== 'production'`. However, `env.ts` defaults `DEMO_MODE` to `true`, and `src/index.ts:508` exposes whether the current session is in demo mode. More critically, the `authMiddleware` accepts `demo-token-12345` in non-production environments. If `NODE_ENV` is misconfigured (e.g., unset, or set to `staging`, `dev`, etc.), this bypass remains active. Railway/Vercel deployments often use `NODE_ENV=production` but local/preview builds may not.
- **Impact:** Unauthenticated attackers can access all protected routes using the hardcoded `demo-token-12345`.
- **Fix:** Remove all demo bypass code from auth middleware. Use a separate debug flag (e.g., `ALLOW_DEMO_AUTH=false` by default) that must be explicitly enabled, and log a prominent warning on startup if enabled.

### 1.2 HIGH — Custom JWT Implementation with Weak Defaults
- **File:** `src/middleware/auth.ts:19-102`
- **Issue:** The application implements its own JWT encoding/decoding using `btoa`/`atob` and `crypto.subtle.sign`, rather than a well-vetted library (e.g., `jose`, `jsonwebtoken`). While HMAC-SHA384 is acceptable, the implementation lacks:
  - Key ID (`kid`) headers for key rotation
  - Proper Base64URL encoding (uses standard Base64 with `replace` hacks)
  - Audience (`aud`) or Issuer (`iss`) validation
  - The expiration is hardcoded to 24 hours (`86400`) with no refresh token mechanism
- **Impact:** Custom crypto is error-prone; token forgery or parsing vulnerabilities could exist. No token revocation capability.
- **Fix:** Replace with `jose` library. Add `aud`, `iss`, `jti` claims. Implement refresh tokens with a separate secure cookie. Store token metadata server-side for revocation.

### 1.3 HIGH — Missing Authorization on Core Endpoints
- **File:** `src/index.ts:351-355`, `src/index.ts:357-361`, `src/routes/ai-analysis.ts:108`, `src/routes/ai-analysis.ts:217`
- **Issue:** `/exercises` (GET), `/exercises/:id` (GET), `/ai/analyze-swarm`, and `/ai/analyze` are **completely unauthenticated**. Anyone can list exercises and trigger expensive AI analysis endpoints without credentials. The `/knowledge/search`, `/knowledge/conditions`, `/knowledge/rehab/:condition`, etc. endpoints are also unprotected.
- **Impact:** Data exposure (exercise library, medical protocols), resource exhaustion via AI endpoints, potential PHI inference from analysis queries.
- **Fix:** Apply `authMiddleware` to all non-public endpoints. Make `/exercises` and knowledge base routes require authentication. Consider role-based access for AI analysis.

### 1.4 HIGH — Patient Portal Uses Insecure Token Construction
- **File:** `src/routes/portal.ts:71-76`, `src/routes/portal.ts:96-119`
- **Issue:** The patient portal creates tokens via `btoa(JSON.stringify({...}))` with no cryptographic signature. Any client can forge a patient token by Base64-encoding a JSON object. The `patientAuth` middleware only checks `payload.exp < Date.now()` and `payload.type === 'patient'`.
- **Impact:** Complete authentication bypass for patient portal — any patient ID can be impersonated.
- **Fix:** Use the same JWT library as clinician auth (or `jose`) with proper HMAC signing. Store patient session state server-side.

### 1.5 MEDIUM — No Token Revocation / Logout is Client-Side Only
- **File:** `src/routes/auth.ts:303-317`
- **Issue:** The `/auth/logout` endpoint only logs an audit event and tells the client to clear the token. The token remains valid until expiration.
- **Impact:** Stolen tokens cannot be invalidated. HIPAA requires the ability to terminate sessions.
- **Fix:** Maintain a token denylist (Redis/database) or use short-lived access tokens + refresh tokens.

### 1.6 MEDIUM — Role Checks Are Inconsistent
- **File:** `src/index.ts`, `src/routes/*.ts`
- **Issue:** While `requireRole('admin')` exists, it is only applied to the `admin` router. Many routes that should be admin-only (e.g., system settings, user management) are reachable by any authenticated user. For example, `/auth/register` (`src/index.ts:467`) allows self-registration with arbitrary roles because `src/index.ts` does not apply validation middleware or role checks.
- **Impact:** Privilege escalation — any user can create admin accounts.
- **Fix:** Apply `requireRole('admin')` to registration and user management endpoints. Validate role assignments server-side.

---

## 2. SQL INJECTION & DATA EXPOSURE RISKS

### 2.1 HIGH — Dynamic SQL Concatenation in Admin Routes
- **File:** `src/routes/admin.ts:30-61`, `src/routes/admin.ts:385-426`, `src/routes/assessments.ts:29-63`, `src/routes/billing.ts:220-267`
- **Issue:** Multiple routes build `whereClause` strings via concatenation with parameterized query placeholders (`$${params.length}`). While this uses parameterized queries for *values*, the column names and some structural elements are concatenated. In `admin.ts:30-61`, `role` and `status` are directly concatenated into the WHERE clause as values (safe), but in `admin.ts:232-234`, `setFields` are built from user-provided field names without strict allowlisting validation — `allowedFields` is defined but `updates[field]` values are not sanitized for SQL keywords.
- **Impact:** If any route allows user-controlled column names in `setFields` or `whereClause`, SQL injection is possible.
- **Fix:** Strictly validate all column names against an explicit allowlist before concatenation. Never concatenate user input into SQL structure.

### 2.2 MEDIUM — Patient Search Uses ILIKE Without Input Length Limits
- **File:** `src/routes/patients.ts:41-44`, `src/routes/exercises.ts:56-59`
- **Issue:** The `search` parameter is directly used in `ILIKE` with `%${search}%`. While parameterized queries prevent SQL injection, there is no length limit on `search`, allowing extremely long strings that could cause query performance degradation or denial of service.
- **Impact:** DoS via expensive wildcard searches on large patient tables.
- **Fix:** Cap `search` length (e.g., 100 characters) and add query timeout.

### 2.3 MEDIUM — Hardcoded Fallback to Mock Data Exposes All Patients
- **File:** `src/lib/database.ts:104`, `src/lib/database.ts:114-115`
- **Issue:** When Supabase is unavailable (`isSupabaseEnabled() === false`), the system falls back to `MOCK_PATIENTS`, an in-memory array containing full mock patient records. `listPatients()` returns all mock patients without filtering by clinician or clinic.
- **Impact:** In fallback mode, any authenticated user sees all mock patients, violating the Minimum Necessary standard.
- **Fix:** Remove mock data from production paths. Return empty arrays or 503 errors when the database is unavailable.

### 2.4 LOW — Database Connection String Exposed in `env.ts` Default
- **File:** `src/env.ts:4`
- **Issue:** `DATABASE_URL` defaults to `postgresql://localhost:5432/imw_msk_demo`. While not a credential leak, it reveals database naming conventions.
- **Fix:** Default to empty string; require explicit configuration.

---

## 3. HIPAA COMPLIANCE GAPS

### 3.1 CRITICAL — Telegram Notifications Transmit PHI Unencrypted
- **File:** `src/index.ts:517-551`
- **Issue:** The `/notify/telegram` endpoint accepts `patientName` and `message` from the request body and sends them via Telegram Bot API with `parse_mode: 'HTML'`. There is no encryption, no Business Associate Agreement (BAA) with Telegram, no opt-in consent tracking, and no redaction of PHI. Telegram messages are stored on Telegram's servers without HIPAA guarantees.
- **Impact:** **Direct HIPAA violation** — transmitting PHI to an unauthorized third party without a BAA.
- **Fix:** Remove Telegram notification integration for PHI. If notifications are required, use HIPAA-compliant services (e.g., AWS SNS with BAA, Twilio with BAA, or secure in-app notifications). If retained for non-PHI alerts only, add explicit PHI redaction and a prominent compliance warning.

### 3.2 HIGH — Audit Logging Silent Failures
- **File:** `src/middleware/hipaa.ts:113-146`, `src/routes/auth.ts:27-62`
- **Issue:** The `hipaaLogger.audit()` function catches all errors silently (`catch (e) { /* silent fail */ }`). The auth route's `logAudit()` also silently fails. HIPAA requires that audit logs be **reliable** and **tamper-evident**. Silent failures mean audit gaps go undetected.
- **Impact:** Missing audit records during security incidents. Cannot prove compliance during an investigation.
- **Fix:** Audit logging failures must be treated as critical errors. Alert administrators immediately (e.g., via separate monitoring channel) and optionally fail-safe (block the operation if audit logging is down, or queue audits for retry).

### 3.3 HIGH — No Encryption at Rest for Database
- **File:** `src/database.ts:25-31`, `src/lib/supabase.ts`
- **Issue:** The PostgreSQL connection uses SSL with `rejectUnauthorized: false` in non-development environments. This disables certificate validation, making the connection vulnerable to MITM attacks. There is no mention of database-level encryption (TDE) or column-level encryption for PHI fields (SSN, DOB, diagnosis, etc.).
- **Impact:** PHI is vulnerable to eavesdropping and unauthorized access if the database server or backups are compromised.
- **Fix:** Enable `rejectUnauthorized: true` with proper CA certificates. Implement column-level encryption for highly sensitive fields (SSN, insurance policy numbers, diagnoses). Ensure Supabase encryption at rest is enabled.

### 3.4 HIGH — Video Upload Metadata-Only with No Real Storage Security
- **File:** `src/routes/videos.ts:60-93`
- **Issue:** Videos are accepted via `formData` but only metadata is stored in the database. The comment states "Video storage is currently metadata-only on Railway. In production, integrate with S3 or local persistent storage." There is no encryption, no access control on playback URLs, and no retention policy. Medical video recordings are high-sensitivity PHI.
- **Impact:** PHI video data is either discarded (data loss) or stored insecurely.
- **Fix:** Implement secure video storage with encryption at rest (e.g., S3 with SSE-KMS). Generate signed, time-limited URLs for playback. Implement retention policies per HIPAA requirements.

### 3.5 MEDIUM — Session Timeout Logic Queries Wrong Table
- **File:** `src/middleware/hipaa.ts:234-277`
- **Issue:** `sessionTimeout` queries `users` table (`SELECT last_activity_at FROM users WHERE id = $1`), but the auth system inserts into `users` table while the `clinician` object may reference a different schema. The `last_activity_at` column is updated in `auth.ts` but not consistently across all routes. Also, `autoLogoutWarning` (`hipaa.ts:311-339`) queries the same table but the middleware is never actually applied in `src/index.ts`.
- **Impact:** Session timeout may not work correctly, leading to stale sessions exceeding HIPAA's 15-minute idle timeout.
- **Fix:** Ensure `sessionTimeout` middleware is applied globally. Verify the table/column names match the actual schema. Add a global middleware that updates `last_activity_at` on every authenticated request.

### 3.6 MEDIUM — Insufficient PHI Redaction in Logs
- **File:** `src/middleware/hipaa.ts:46-76`
- **Issue:** `redactSensitiveData` redacts values if keys contain sensitive substrings. However, the SENSITIVE_FIELDS list does not include `portal_password_hash`, `video_url`, `recording_url`, `skeleton_data`, `landmarks`, or `movementTest`. Console error messages in routes frequently log raw error objects (`error.message`) which may contain SQL snippets with embedded PHI.
- **Impact:** PHI may leak into application logs, which are often sent to centralized logging systems without HIPAA controls.
- **Fix:** Expand SENSITIVE_FIELDS. Sanitize all error messages before logging. Route all logs through `hipaaLogger` with strict redaction.

### 3.7 MEDIUM — No Consent or Authorization Tracking
- **File:** Global
- **Issue:** There is no mechanism to track patient consent for data processing, video recording, AI analysis, or third-party AI model usage (Ollama, Gemini, Kimi, etc.). HIPAA requires documented authorization for uses beyond TPO (Treatment, Payment, Healthcare Operations).
- **Impact:** Using patient data for AI training or analysis without consent is a HIPAA violation.
- **Fix:** Add a `patient_consents` table tracking consent types (video, AI analysis, data sharing, research). Verify consent before processing.

### 3.8 LOW — Health Endpoint Leaks Configuration Details
- **File:** `src/index.ts:170-195`
- **Issue:** `/health` exposes `version`, `commit`, `features`, `ai` provider details, and whether `telegram_notifications` are configured. This aids attackers in fingerprinting the application.
- **Fix:** Minimize health endpoint output. Move detailed diagnostics behind an authenticated `/health/detailed` endpoint.

---

## 4. RATE LIMITING EFFECTIVENESS

### 4.1 HIGH — Rate Limiting is In-Memory Only and Not Applied Globally
- **File:** `src/middleware/rateLimit.ts`, `src/index.ts`
- **Issue:** Rate limiting uses an in-memory `Map` (`rateLimits`, `slidingWindows`). In a serverless or multi-worker environment (Vercel, Railway), each instance maintains its own counter, making rate limits trivial to bypass by distributing requests across instances. Furthermore, **no rate limiter is actually applied in `src/index.ts`** — the pre-configured limiters (`authRateLimit`, `apiRateLimit`, etc.) are defined but never mounted.
- **Impact:** Brute force attacks on auth, credential stuffing, and DoS are effectively unmitigated in production.
- **Fix:** Mount rate limiters on all routes. Use a shared store (Redis, Cloudflare KV, or PostgreSQL) for rate limit counters in production. Apply `authRateLimit` to `/auth/login` and `/auth/register`.

### 4.2 MEDIUM — Auth Rate Limit is Too Permissive
- **File:** `src/middleware/rateLimit.ts:40-44`
- **Issue:** Auth rate limit allows 5 attempts per 15 minutes. While reasonable, there is no progressive delay or CAPTCHA challenge. The IP extraction relies on `CF-Connecting-IP` and `X-Forwarded-For`, which can be spoofed if the app is not behind Cloudflare or a trusted proxy.
- **Impact:** Slightly elevated brute-force risk if IP headers are untrusted.
- **Fix:** Implement progressive delays (exponential backoff) and account-level lockout (already partially done in `auth.ts`). Validate IP headers against a trusted proxy list.

---

## 5. INPUT VALIDATION

### 5.1 HIGH — Zod Validation Bypassed in `src/index.ts`
- **File:** `src/index.ts:118-165`, `src/index.ts:213-217`, `src/index.ts:274-286`
- **Issue:** Routes in `src/index.ts` (e.g., `/assessments/enhanced`, `/patients`, `/assessments`) use `authMiddleware` but **do not apply the Zod validation middleware** (`validate(schema)`). Only the sub-routers (`auth.ts`, `exercises.ts`) apply validation. The main app routes accept arbitrary JSON bodies.
- **Impact:** Malformed data, injection attacks, and type confusion can reach the database layer.
- **Fix:** Apply Zod validation middleware to all POST/PUT routes in `src/index.ts`. Consider migrating all routes to sub-routers for consistency.

### 5.2 MEDIUM — `sanitizeString` is Reversible and Insufficient
- **File:** `src/utils/security.ts:7-27`, `src/middleware/validation.ts:372-378`
- **Issue:** `sanitizeString` escapes HTML entities but does not prevent all XSS vectors. It allows through attributes, style tags, and other vectors. The validation middleware's `sanitizeInput` only removes angle brackets, `javascript:`, and event handlers — a very weak sanitizer.
- **Impact:** Stored XSS is possible if data is rendered in contexts other than HTML text content (e.g., JSON, attributes, CSS).
- **Fix:** Use a robust sanitization library like `DOMPurify` (server-side) for HTML contexts. For non-HTML contexts, validate structure rather than attempting to sanitize.

### 5.3 MEDIUM — Patient ID Type Confusion (string vs number)
- **File:** `src/lib/database.ts:107-116`, `src/index.ts:206-211`
- **Issue:** `getPatient(id: string)` accepts a string, but `src/index.ts` passes the raw URL parameter (string). However, `src/routes/patients.ts` parses it as `parseInt(c.req.param('id'))`. The Supabase queries use `.eq('id', id)` — if the database expects UUIDs or integers, type mismatches could cause errors or unexpected behavior.
- **Impact:** Inconsistent behavior between mock data and Supabase paths. Potential for IDOR if string coercion bypasses filters.
- **Fix:** Standardize patient ID type (UUID or auto-increment integer) across the entire application.

### 5.4 LOW — Missing File Upload Validation Beyond Type/Size
- **File:** `src/routes/videos.ts:18-105`
- **Issue:** Video uploads check MIME type and size but do not scan for malware, verify magic numbers, or enforce naming conventions. The `storageKey` uses `Math.random()` which is not cryptographically secure.
- **Fix:** Verify file magic numbers. Use `crypto.randomUUID()` for storage keys. Scan uploads with an antivirus/Content Disarm and Reconstruction (CDR) service.

---

## 6. API DESIGN ISSUES

### 6.1 HIGH — Inconsistent Route Architecture
- **File:** `src/index.ts`, `src/routes/*.ts`
- **Issue:** Some routes are defined directly in `src/index.ts` (patients, assessments, auth, exercises) while others are in sub-routers (`admin.ts`, `billing.ts`, `portal.ts`). The main app imports both `mskRouter` and `aiRoutes` but also defines its own overlapping `/ai/*` and patient/assessment routes. This creates duplication and maintenance risk.
- **Impact:** Security controls (auth, validation, rate limiting, audit) are applied inconsistently.
- **Fix:** Consolidate all routes into sub-routers under a consistent `/api/v1/` prefix. Apply global middleware (auth, rate limit, audit, security headers) uniformly.

### 6.2 HIGH — Boxer3D Agent Route Uses Express Import in Hono App
- **File:** `src/routes/boxer3d-agents.ts:1-9`
- **Issue:** This file imports `Router` from `express` and `child_process` to execute Python scripts via `execFileSync`. It is never integrated into the Hono app (`src/index.ts` does not import it), but it represents a dangerous pattern: executing arbitrary system commands with user-controlled input (`videoPath`, `patientName`).
- **Impact:** Command injection if `videoPath` or `patientName` contains shell metacharacters. `execFileSync` is safer than `exec` but still executes external binaries.
- **Fix:** If this route is needed, strictly validate `videoPath` against an allowlist, sanitize all arguments, and run the Python process in a sandboxed container with minimal privileges. Remove Express imports.

### 6.3 MEDIUM — AI Analysis Routes Accept Unbounded Input
- **File:** `src/routes/ai-analysis.ts:40-48`, `src/routes/ai-analysis.ts:125-127`
- **Issue:** The `/ai/analyze-swarm` and `/ai/analyze` endpoints accept `keypoints` arrays and arbitrary `assessmentData` objects without size limits or schema validation. These are forwarded to a local Ollama instance.
- **Impact:** DoS via huge payloads. Prompt injection if `assessmentData` contains malicious instructions.
- **Fix:** Add payload size limits and strict Zod schemas for AI endpoints. Sanitize data before including in prompts.

### 6.4 MEDIUM — CORS Allows Wildcard in Development
- **File:** `src/index.ts:22-28`
- **Issue:** `cors({ origin: '*' })` is used in non-production environments. If `NODE_ENV` is unset or misconfigured, this applies to production.
- **Impact:** Cross-origin attacks from malicious websites if credentials are inadvertently included.
- **Fix:** Always use an explicit allowlist. Reject requests with `Origin` not in the allowlist, even in development.

### 6.5 LOW — Version Endpoint Exposes Internal Details
- **File:** `src/index.ts:170-195`
- **Issue:** As noted in 3.8, the health endpoint exposes too much information.
- **Fix:** Redact sensitive configuration from public health checks.

---

## 7. PERFORMANCE CONCERNS

### 7.1 HIGH — No Query Result Caching
- **File:** `src/routes/exercises.ts:11-12`, Global
- **Issue:** A `CACHE_TTL` constant is defined but never used. Every request hits the database directly. High-frequency endpoints like `/exercises`, `/patients`, and `/dashboard/stats` will create unnecessary load.
- **Impact:** Database saturation under load. Slow response times.
- **Fix:** Implement Redis or in-memory LRU caching for frequently accessed, slowly changing data (exercises, CPT codes, normative ROM tables).

### 7.2 HIGH — In-Memory Rate Limit Map Grows Unbounded
- **File:** `src/middleware/rateLimit.ts:14-24`
- **Issue:** The `rateLimits` Map stores an entry per IP/clinician. With a 1-minute cleanup interval, a high-traffic site could accumulate millions of entries between cleanups, causing memory exhaustion.
- **Impact:** Memory leak leading to application crash.
- **Fix:** Use a TTL-backed data structure or limit Map size (LRU eviction). Move to Redis in production.

### 7.3 MEDIUM — N+1 Query Patterns
- **File:** `src/routes/patients.ts:330-363`, `src/routes/assessments.ts:83-131`
- **Issue:** `patients.get('/:id/assessments')` and `assessments.get('/:id')` execute separate queries for the main record and related data (tests, prescriptions). Under load, this creates N+1 query behavior.
- **Impact:** Increased database round-trips and latency.
- **Fix:** Use JOINs to fetch related data in a single query, or use a data loader pattern.

### 7.4 MEDIUM — AI Analysis is Synchronous and Blocking
- **File:** `src/routes/ai-analysis.ts:108-214`
- **Issue:** The swarm analysis runs specialist agents sequentially (`for (const spec of activeSpecialists) { ... await ollamaChat(...) }`). Each call may take seconds. The request handler blocks until all agents complete.
- **Impact:** Request timeout, server thread exhaustion, poor user experience.
- **Fix:** Make AI analysis asynchronous — queue the job and return a job ID. Use WebSockets or polling for results.

### 7.5 LOW — Database Connection Pool May Be Undersized
- **File:** `src/database.ts:28`
- **Issue:** Pool `max` is set to 20. Under high concurrency with blocking AI calls, connections may be exhausted.
- **Fix:** Monitor connection usage. Increase pool size or implement connection pooling per tenant.

---

## 8. CRYPTOGRAPHY & SECRETS MANAGEMENT

### 8.1 CRITICAL — Hardcoded Cryptographic Secrets
- **File:** `src/utils/crypto-vault.ts:21-25`, `src/middleware/auth.ts:123`
- **Issue:** `crypto-vault.ts` contains **hardcoded fallback secrets**:
  - `hmacSecret: 'physiomotion-medical-integrity-key-2026'`
  - `encryptionPassword: 'physiomotion-vault-aes256-key-2026'`
  If `HMAC_SECRET` or `ENCRYPTION_PASSWORD` environment variables are unset, these defaults are used. Additionally, `auth.ts` falls back to `process.env.JWT_SECRET || process.env.AUTH_SECRET` with no default, but `env.ts` sets `JWT_SECRET` default to `'dev-secret-change-in-production'`.
- **Impact:** Attackers who obtain the source code can forge HMAC signatures, decrypt embeddings, and forge JWT tokens.
- **Fix:** **Never use hardcoded secrets.** Require environment variables at startup and crash with a clear error if they are missing or too short (minimum 256 bits / 32 bytes for HMAC, 32 chars for JWT).

### 8.2 HIGH — JWT Secret Default is Weak
- **File:** `src/env.ts:5`
- **Issue:** `JWT_SECRET` defaults to `'dev-secret-change-in-production'` which is a low-entropy, predictable string.
- **Impact:** JWT token forgery if the default is not overridden.
- **Fix:** Remove the default. Enforce a minimum length (e.g., 32 random bytes) and validate on startup.

### 8.3 MEDIUM — bcrypt Salt Rounds is Acceptable but Not Configurable
- **File:** `src/middleware/auth.ts:9`
- **Issue:** `SALT_ROUNDS = 12` is reasonable but not configurable per environment.
- **Fix:** Make salt rounds an environment variable with a sensible default.

---

## 9. DEPENDENCY & SUPPLY CHAIN

### 9.1 MEDIUM — Outdated or Unnecessary Dependencies
- **File:** `package.json`
- **Issue:** `three` and `@types/three` are listed but appear unused in the backend. `dotenv` is listed but Hono doesn't require it (Node 20 has `--env-file`). `@types/express` is listed but the app uses Hono. Outdated dependencies increase attack surface.
- **Impact:** Larger bundle size, potential vulnerabilities in unused packages.
- **Fix:** Audit and remove unused dependencies. Run `npm audit` regularly.

### 9.2 LOW — `mockD1` is Exported as Production Helper
- **File:** `src/utils/db-helpers.ts:4-18`
- **Issue:** `mockD1` is a no-op mock that returns empty results. It is used in `auth.ts` for `sessionActivity` and `secureAuth` middleware. If the Cloudflare `c.env.DB` binding is missing, the mock silently succeeds, making database updates appear to work when they don't.
- **Fix:** Throw an error when the real database is unavailable. Do not silently fallback to mock in production.

---

## 10. REMEDIATION ROADMAP

### Immediate (Before Production Deployment)
1. **Remove hardcoded secrets** from `crypto-vault.ts` and `env.ts`. Require environment variables.
2. **Remove demo auth bypass** from `authMiddleware` and `/auth/login`.
3. **Apply authentication** to `/exercises`, `/ai/*`, and `/knowledge/*` endpoints.
4. **Fix patient portal tokens** to use proper JWT signing.
5. **Remove or secure Telegram integration** — do not send PHI.
6. **Mount rate limiters** globally and use a shared store.
7. **Apply Zod validation** to all POST/PUT routes in `src/index.ts`.

### Short-Term (Within 30 Days)
8. Replace custom JWT with `jose` library; implement refresh tokens.
9. Fix SSL `rejectUnauthorized: false` in database connections.
10. Implement column-level encryption for SSN, insurance, diagnosis fields.
11. Add patient consent tracking.
12. Secure video storage with encryption and signed URLs.
13. Remove `mockD1` fallback from production code paths.
14. Consolidate routes and apply middleware consistently.

### Medium-Term (Within 90 Days)
15. Implement Redis-backed caching and rate limiting.
16. Make AI analysis asynchronous with job queuing.
17. Add comprehensive audit log alerting and tamper-evident storage.
18. Conduct penetration testing and HIPAA risk assessment.
19. Implement MFA for clinician accounts.
20. Add automated vulnerability scanning (SAST/DAST) to CI/CD.

---

## APPENDIX: FILE-BY-FILE RISK SUMMARY

| File | Lines | Risk Level | Key Issues |
|------|-------|------------|------------|
| `src/index.ts` | 566 | **HIGH** | Unprotected routes, demo bypass, no validation, Telegram PHI leak, CORS wildcard |
| `src/middleware/auth.ts` | 231 | **HIGH** | Custom JWT, demo token bypass, hardcoded fallback in env.ts |
| `src/middleware/hipaa.ts` | 345 | **MEDIUM** | Silent audit failures, session timeout querying wrong table, incomplete redaction |
| `src/middleware/rateLimit.ts` | 208 | **HIGH** | In-memory only, never mounted, unbounded growth |
| `src/middleware/validation.ts` | 408 | **MEDIUM** | Weak `sanitizeInput`, good Zod schemas but underutilized |
| `src/routes/admin.ts` | 580 | **MEDIUM** | SQL structure concatenation (values are parameterized) |
| `src/routes/ai-analysis.ts` | 311 | **HIGH** | Unauthenticated, no input limits, blocking sync calls |
| `src/routes/assessments.ts` | 530 | **MEDIUM** | Good XSS sanitization, no validation middleware |
| `src/routes/auth.ts` | 527 | **MEDIUM** | Good login lockout, but logout is client-side only |
| `src/routes/billing.ts` | 498 | **LOW** | Generally well-structured, minor SQL concatenation |
| `src/routes/boxer3d-agents.ts` | 68 | **CRITICAL** | Command injection risk, Express import, user input to shell |
| `src/routes/exercises.ts` | 551 | **MEDIUM** | No auth on GET routes, search length unbounded |
| `src/routes/msk-analysis.ts` | 300 | **MEDIUM** | Good integrity checks, hardcoded crypto secrets |
| `src/routes/patients.ts` | 534 | **MEDIUM** | Good sanitization, no validation middleware on PUT |
| `src/routes/portal.ts` | 387 | **HIGH** | Completely insecure patient tokens |
| `src/routes/tests.ts` | 199 | **LOW** | Generally acceptable |
| `src/routes/videos.ts` | 276 | **HIGH** | Metadata-only storage, no real security, random() for keys |
| `src/lib/database.ts` | 414 | **MEDIUM** | Mock data fallback exposes all patients |
| `src/utils/crypto-vault.ts` | 433 | **CRITICAL** | Hardcoded HMAC and AES passwords |
| `src/utils/security.ts` | 168 | **MEDIUM** | Insufficient XSS sanitization |
| `src/database.ts` | 109 | **MEDIUM** | SSL validation disabled |
| `src/env.ts` | 28 | **HIGH** | Weak JWT default, demo mode default true |

---

*End of Review*
