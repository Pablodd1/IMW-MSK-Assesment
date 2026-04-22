# PhysioMotion Environment Setup

## Vercel + Supabase Configuration

### 1. Supabase Setup

1. **Create Supabase Project**
   - Go to [supabase.com](https://supabase.com)
   - Create new project: `physiomotion`

2. **Run SQL Schema**
   - Open Supabase Dashboard → SQL Editor
   - Copy content from `supabase/schema.sql`
   - Run to create all tables

3. **Get Credentials**
   - Settings → API
   - Copy `Project URL`
   - Copy `anon public` key

### 2. Vercel Setup

1. **Import Project**
   - Go to [vercel.com](https://vercel.com)
   - Import GitHub repo: `Pablodd1/IMW-MSK-Assesment`

2. **Environment Variables**
   Add these in Vercel Dashboard → Settings → Environment Variables:

| Variable | Value | Example |
|----------|-------|---------|
| `VITE_SUPABASE_URL` | Your Supabase URL | `https://xxxxx.supabase.co` |
| `VITE_SUPABASE_ANON_KEY` | Your anon key | `eyJhbGciOiJIUzI1NiIs...` |
| `JWT_SECRET` | 32+ char random string | (generate with `openssl rand -base64 32`) |
| `ALLOWED_ORIGINS` | Your Vercel URLs | `https://imw-msk.vercel.app,localhost:3000` |

3. **Deploy**
   - Vercel auto-deploys on push to main

### 3. Development Local

Create `.env.local`:
```env
VITE_SUPABASE_URL=https://xxxxx.supabase.co
VITE_SUPABASE_ANON_KEY=your_anon_key
JWT_SECRET=your_32_char_secret
ALLOWED_ORIGINS=http://localhost:3000
```

### 4. Test with Demo

Demo credentials are seeded:
- **Email**: `demo@physiomotion.com`
- **Password**: `demo123` (hashed with bcrypt)

### 5. Troubleshooting

| Issue | Solution |
|-------|----------|
| CORS errors | Add origin to `ALLOWED_ORIGINS` |
| Auth fails | Verify Supabase URL and key |
| DB errors | Run `schema.sql` in Supabase |
| Build fails | Clear Vercel cache and redeploy |

## 🚀 Quick Test

```bash
# Health check
curl https://your-app.vercel.app/api/health

# Should return: {"status":"ok"}
```