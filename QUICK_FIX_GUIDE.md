# PhysioMotion - Quick Fix Guide

**🚨 Critical Issues That Need Immediate Attention**

---

## 🔴 TOP 5 MUST FIX BEFORE PRODUCTION

### 1. Add Authentication (CRITICAL - HIPAA VIOLATION)

**Problem:** Anyone can access any patient data without login

**Quick Fix (Temporary):**
```typescript
// Add to src/index.tsx
import { jwt } from 'hono/jwt'

// Simple password check (replace with proper auth)
const authMiddleware = async (c: any, next: any) => {
  const authHeader = c.req.header('Authorization')
  
  if (!authHeader || authHeader !== 'Bearer TEMP_PASSWORD_123') {
    return c.json({ error: 'Unauthorized' }, 401)
  }
  
  await next()
}

// Protect all API routes
app.use('/api/*', authMiddleware)
```

**Proper Solution:** Implement JWT-based auth with Cloudflare Access or Auth0

---

### 2. Add Input Validation (CRITICAL - SECURITY RISK)

**Problem:** No validation of user inputs (SQL injection risk, data corruption)

**Quick Fix:**
```bash
npm install zod
```

```typescript
// Add to src/index.tsx
import { z } from 'zod'

const patientSchema = z.object({
  first_name: z.string().min(1).max(100),
  last_name: z.string().min(1).max(100),
  date_of_birth: z.string().regex(/^\d{4}-\d{2}-\d{2}$/),
  email: z.string().email().optional(),
  gender: z.enum(['male', 'female', 'other', 'prefer_not_to_say'])
})

// In POST /api/patients:
app.post('/api/patients', async (c) => {
  try {
    const data = await c.req.json()
    const validated = patientSchema.parse(data) // Throws if invalid
    // ... rest of code
  } catch (error) {
    if (error instanceof z.ZodError) {
      return c.json({ success: false, error: 'Invalid input', details: error.errors }, 400)
    }
    return c.json({ success: false, error: error.message }, 500)
  }
})
```

---

### 3. Configure CORS (CRITICAL - SECURITY)

**Problem:** CORS allows all origins (security risk)

**Quick Fix:**
```typescript
// src/index.tsx:10
// REPLACE:
app.use('/api/*', cors())

// WITH:
app.use('/api/*', cors({
  origin: process.env.ALLOWED_ORIGINS?.split(',') || 'https://yourdomain.com',
  allowMethods: ['GET', 'POST', 'PUT', 'DELETE'],
  credentials: true
}))
```

**Add to wrangler.toml:**
```toml
[env.production]
vars = { ALLOWED_ORIGINS = "https://yourdomain.com,https://app.yourdomain.com" }
```

---

### 4. Remove PHI from Console Logs (CRITICAL - HIPAA VIOLATION)

**Problem:** Patient data logged to browser console

**Quick Find & Replace:**
```javascript
// Find all instances of:
console.log('✅ Assessment created:', ASSESSMENT_STATE.assessmentId);
console.log('Patient:', patient);

// Replace with:
if (window.location.hostname === 'localhost') {
  console.log('✅ Assessment created:', ASSESSMENT_STATE.assessmentId);
}
```

**Or create utility:**
```javascript
// Add to assessment-workflow.js
const isDev = () => window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'

const devLog = (...args) => {
  if (isDev()) {
    console.log(...args)
  }
}

// Use: devLog('Patient data:', patient) instead of console.log
```

---

### 5. Add Global Error Handler (HIGH PRIORITY)

**Problem:** Errors crash the app or show cryptic messages

**Quick Fix:**
```typescript
// Add to src/index.tsx (at bottom, before export)
app.onError((err, c) => {
  console.error('Error:', err)
  
  // Don't expose internal errors in production
  const isDev = c.env.ENVIRONMENT === 'development'
  
  return c.json({ 
    success: false,
    error: isDev ? err.message : 'An error occurred',
    ...(isDev && { stack: err.stack })
  }, 500)
})

app.notFound((c) => {
  return c.json({ success: false, error: 'Route not found' }, 404)
})
```

---

## 🟠 QUICK WINS (30 MIN FIXES)

### Fix #1: Hardcoded Clinician IDs
```typescript
// Find: assessment.clinician_id || 1
// Replace with: throw new Error('clinician_id required')

// Or better: Get from auth context
const clinicianId = c.get('userId') // After auth is implemented
```

### Fix #2: Add HTTP Status Validation
```typescript
// Add to all frontend fetch calls:
const response = await fetch('/api/patients')

if (!response.ok) {
  throw new Error(`HTTP ${response.status}: ${response.statusText}`)
}

const result = await response.json()

if (!result.success) {
  throw new Error(result.error || 'Request failed')
}
```

### Fix #3: Add Loading States
```html
<!-- Add to all async operations -->
<button id="saveBtn" onclick="saveData()">
  Save
</button>

<script>
async function saveData() {
  const btn = document.getElementById('saveBtn')
  btn.disabled = true
  btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Saving...'
  
  try {
    await fetch(...)
    showSuccess('Saved!')
  } catch (error) {
    showError('Save failed: ' + error.message)
  } finally {
    btn.disabled = false
    btn.innerHTML = 'Save'
  }
}
</script>
```

### Fix #4: Add Confirmation Dialogs
```javascript
function startNewAssessment() {
  if (confirm('This will discard current progress. Continue?')) {
    // proceed
  }
}

function deletePatient(id) {
  if (confirm('Delete this patient? This cannot be undone.')) {
    // proceed
  }
}
```

### Fix #5: Fix Empty States Flash
```javascript
// Don't show empty state until data is loaded
// WRONG:
<div id="emptyState">No data</div>

// RIGHT:
<div id="emptyState" class="hidden">No data</div>

// Then show only if actually empty:
if (data.length === 0) {
  emptyState.classList.remove('hidden')
}
```

---

## 🛠️ TESTING CHECKLIST

### Before Every Commit:
- [ ] No console.log() with PHI
- [ ] No hardcoded IDs (1, '1', etc.)
- [ ] Error handling on all fetch() calls
- [ ] Input validation on all forms
- [ ] XSS protection (escapeHtml) on all dynamic content
- [ ] Loading states on all buttons
- [ ] Success/error notifications

### Before Deployment:
```bash
# Run database migrations
npm run db:migrate:prod

# Type check
npx tsc --noEmit

# Build
npm run build

# Test critical paths manually:
# 1. Create patient
# 2. Start assessment
# 3. View patient list
# 4. Check all navigation links
```

---

## 📋 COMMON PATTERNS

### Proper API Endpoint Pattern
```typescript
app.post('/api/resource', async (c) => {
  try {
    // 1. Validate input
    const data = validateInput(await c.req.json())
    
    // 2. Check authorization
    const userId = c.get('userId')
    if (!userId) throw new Error('Unauthorized')
    
    // 3. Perform operation
    const result = await c.env.DB.prepare(`...`).bind(...).run()
    
    // 4. Return success
    return c.json({ success: true, data: result })
    
  } catch (error: any) {
    console.error('Error in POST /api/resource:', error)
    return c.json({ success: false, error: error.message }, 500)
  }
})
```

### Proper Frontend Fetch Pattern
```javascript
async function fetchData() {
  const loading = document.getElementById('loading')
  const error = document.getElementById('error')
  
  try {
    loading.style.display = 'block'
    error.style.display = 'none'
    
    const response = await fetch('/api/data')
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`)
    }
    
    const result = await response.json()
    
    if (!result.success) {
      throw new Error(result.error || 'Request failed')
    }
    
    // Process result.data
    displayData(result.data)
    showSuccess('Data loaded')
    
  } catch (err) {
    console.error('Fetch error:', err)
    error.textContent = err.message
    error.style.display = 'block'
    showError('Failed to load data')
    
  } finally {
    loading.style.display = 'none'
  }
}
```

### Proper XSS Protection Pattern
```javascript
function escapeHtml(text) {
  if (!text) return text
  const div = document.createElement('div')
  div.textContent = text
  return div.innerHTML
}

// Use in templates:
element.innerHTML = `
  <div class="name">${escapeHtml(user.name)}</div>
  <div class="email">${escapeHtml(user.email)}</div>
`
```

---

## 🚫 THINGS TO AVOID

### DON'T:
```javascript
// ❌ No error handling
const data = await fetch('/api/data').then(r => r.json())

// ❌ Hardcoded values
const clinicianId = 1

// ❌ No input validation
const name = req.body.name

// ❌ Direct HTML injection
element.innerHTML = user.name

// ❌ No loading state
function save() {
  fetch('/api/save', ...)
}

// ❌ Silent failures
try { ... } catch (e) { /* nothing */ }

// ❌ PHI in logs
console.log('Patient:', patient)
```

### DO:
```javascript
// ✅ Proper error handling
try {
  const response = await fetch('/api/data')
  if (!response.ok) throw new Error(`HTTP ${response.status}`)
  const result = await response.json()
  if (!result.success) throw new Error(result.error)
  return result.data
} catch (error) {
  console.error('Error:', error)
  showError(error.message)
  throw error
}

// ✅ Get from context
const clinicianId = c.get('userId')

// ✅ Validate input
const validated = schema.parse(input)

// ✅ Escape HTML
element.innerHTML = escapeHtml(user.name)

// ✅ Show loading state
async function save() {
  button.disabled = true
  try { await fetch(...) } finally { button.disabled = false }
}

// ✅ Handle and display errors
try { ... } catch (e) { showError(e.message) }

// ✅ No PHI in production logs
if (isDev()) console.log('Debug:', data)
```

---

## 📞 EMERGENCY FIXES

### If Production is Down:

1. **Check Cloudflare dashboard** for errors
2. **Rollback to previous version:**
   ```bash
   wrangler rollback --message "Emergency rollback"
   ```

3. **Check database:**
   ```bash
   npm run db:console:prod
   SELECT 1; -- Test connectivity
   ```

4. **Enable debug mode temporarily:**
   ```toml
   # wrangler.toml
   [env.production]
   vars = { DEBUG = "true" }
   ```

5. **Monitor logs:**
   ```bash
   wrangler tail --env production
   ```

---

## 🎯 PRIORITY ORDER

Fix in this order:

1. **Week 1:** Authentication + Input Validation + CORS
2. **Week 2:** Remove PHI logs + Global error handler + Rate limiting
3. **Week 3:** CSP headers + Missing endpoints + Pagination
4. **Week 4:** Tests + Documentation + Monitoring

---

**Remember:** These are quick fixes. For production, follow the full recommendations in `COMPREHENSIVE_AUDIT_REPORT.md`

---

Last Updated: February 5, 2026
