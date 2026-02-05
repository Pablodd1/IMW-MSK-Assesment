# PhysioMotion Authentication System

## Overview
Complete clinician credentialing and authentication system with demo mode for fast testing and development.

## Features

### 1. **Clinician Login System**
- Email/password authentication
- Remember me functionality (localStorage vs sessionStorage)
- Secure password hashing with SHA-256 + salt
- Session management across all pages

### 2. **Registration System**
- Full clinician registration with professional credentials:
  - Basic info (name, email, password)
  - Professional details (title, license number, state, NPI)
  - Clinic information
- Password confirmation validation
- Email uniqueness validation

### 3. **Demo Mode** ⭐
Quick access for testing without credentials:
- **"Skip to Demo Mode"** button on both login and registration pages
- Instant access with demo session (no database required)
- Demo badge displayed in navigation
- Perfect for presentations and testing

### 4. **Protected Routes**
All pages require authentication:
- Login redirect if not authenticated
- Session validation on page load
- Logout functionality across all pages

## Usage

### Demo Mode (Recommended for Testing)
1. Navigate to any page (auto-redirects to login)
2. Click **"Skip to Demo Mode"** button
3. Instant access with demo credentials

### Login with Credentials
**Demo Account:**
- Email: `demo@physiomotion.com`
- Password: `demo123`

### Registration
1. Navigate to `/register` or click "Register here" on login page
2. Fill in required information
3. Professional fields (license, NPI) are optional
4. Create account and login

## API Endpoints

### Authentication
- `POST /api/auth/register` - Register new clinician
- `POST /api/auth/login` - Login clinician
- `GET /api/auth/profile/:id` - Get clinician profile

## Files

### Frontend
- `/public/static/login.html` - Login page with demo mode
- `/public/static/register.html` - Registration page
- `/public/static/auth.js` - Shared authentication utilities

### Backend
- `/src/index.tsx` - Authentication API routes
- Password hashing functions (hashPassword, verifyPassword)

### Database
- `/migrations/0005_add_demo_clinician.sql` - Demo clinician account

## Session Storage

### LocalStorage (Remember Me)
```javascript
localStorage.setItem('clinician_session', JSON.stringify(session))
```

### SessionStorage (Default)
```javascript
sessionStorage.setItem('clinician_session', JSON.stringify(session))
```

### Session Data Structure
```javascript
{
  id: number,
  email: string,
  first_name: string,
  last_name: string,
  title: string,
  role: string,
  is_demo: boolean // true for demo mode
}
```

## Protected Pages
All pages check authentication and display clinician info:
- Main dashboard (`/`)
- Patients list (`/patients`)
- Patient intake (`/intake`)
- Movement assessment (`/assessment`)

## Navigation Features
All authenticated pages show:
- Clinician name and title
- Demo badge (if in demo mode)
- Logout button

## Security Notes

⚠️ **Current Implementation**
- Uses SHA-256 password hashing (suitable for demo/development)
- For production: implement bcrypt or Argon2

⚠️ **Demo Mode**
- Demo sessions bypass database authentication
- Perfect for testing, presentations, and fast development
- No credentials needed - instant access

## Development Tips

### Enable Demo Mode Anywhere
```javascript
const demoSession = {
  id: 0,
  email: 'demo@physiomotion.com',
  first_name: 'Demo',
  last_name: 'Clinician',
  title: 'DPT',
  role: 'clinician',
  is_demo: true
};
sessionStorage.setItem('clinician_session', JSON.stringify(demoSession));
```

### Check Auth in New Pages
```html
<script src="/static/auth.js"></script>
<script>
  checkAuth(); // Redirects if not logged in
  displayClinicianInfo(); // Shows name in nav
</script>
```

## Database Schema
Clinicians table includes:
- Authentication (email, password_hash)
- Personal info (first_name, last_name)
- Professional credentials (title, license_number, license_state, NPI)
- Contact (phone, clinic_name)
- Role & status (role, active)
- Timestamps (created_at, last_login)

## Future Enhancements
- [ ] Implement bcrypt password hashing
- [ ] Add JWT tokens for API authentication
- [ ] Password reset functionality
- [ ] Two-factor authentication (2FA)
- [ ] Session timeout/refresh
- [ ] Role-based access control (RBAC)
- [ ] Audit logging for authentication events
