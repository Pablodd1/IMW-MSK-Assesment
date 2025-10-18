# PhysioAI - Medical Movement Assessment System

## 🏥 Project Overview

**PhysioAI** is a comprehensive medical-grade musculoskeletal assessment platform designed for chiropractors, physical therapists, and movement specialists. The system provides real-time motion analysis, biomechanical assessments, exercise prescription, and remote patient monitoring capabilities.

### Main Features
- ✅ **Complete Patient Intake System** - Demographics, medical history, insurance info
- ✅ **Functional Movement Screen (FMS)** - 7-test standardized assessment protocol
- ✅ **Real-time Motion Analysis** - MediaPipe Pose integration for mobile cameras
- ✅ **Orbbec Femto Mega Support** - Professional-grade 3D skeleton tracking (32 joints)
- ✅ **AI-Powered Biomechanical Analysis** - Joint angle calculations, ROM measurements
- ✅ **Automated Medical Note Generation** - SOAP-formatted clinical documentation
- ✅ **Exercise Prescription System** - Comprehensive library with sets/reps/frequency
- ✅ **Patient Portal** - Home exercise monitoring with form feedback
- ✅ **Remote Patient Monitoring (RPM)** - Medicare billing codes (CPT 98975-98981)
- ✅ **Medical Billing Integration** - Automated compliance tracking

## 🌐 Live Demo

**Public URL**: https://3000-isjigehibebqnf5jhl4y7-2e1b9533.sandbox.novita.ai

### Demo Credentials
- **Clinician Portal**: Full access to all features
- **Demo Patient**: John Doe (pre-loaded with sample data)

## 📊 Current Status

### ✅ Completed Features

#### Phase 1: Core Medical System
- [x] Patient registration with comprehensive demographics
- [x] Medical history capture (conditions, medications, allergies)
- [x] Emergency contact information
- [x] Insurance and billing information
- [x] Chief complaint and pain scale tracking

#### Phase 2: Assessment Workflow
- [x] Functional Movement Screen (FMS) protocol
- [x] 7 standardized movement tests (Deep Squat, Hurdle Step, Inline Lunge, etc.)
- [x] Step-by-step instructions for each test
- [x] Video capture capability
- [x] Real-time scoring system (0-3 scale)
- [x] Automated total score calculation

#### Phase 3: Motion Analysis
- [x] MediaPipe Pose integration for mobile cameras
- [x] 33-landmark skeleton tracking
- [x] Joint angle calculations
- [x] Range of motion (ROM) measurements
- [x] Movement pattern analysis
- [x] Compensation detection
- [x] Quality scoring (0-100 scale)

#### Phase 4: Medical Documentation
- [x] Automated SOAP note generation
- [x] Deficiency identification
- [x] Treatment plan recommendations
- [x] Progress tracking
- [x] Export to PDF/printable format

#### Phase 5: Exercise Prescription
- [x] Exercise library (17+ exercises)
- [x] Targeted deficiency matching
- [x] Customizable sets/reps/frequency
- [x] Exercise ordering and grouping
- [x] Progression criteria
- [x] Precautions and contraindications

#### Phase 6: Patient Portal
- [x] Exercise program viewing
- [x] Video demonstrations
- [x] Real-time form feedback
- [x] Session completion tracking
- [x] Progress visualization
- [x] Pain level logging

#### Phase 7: Remote Patient Monitoring
- [x] Daily activity tracking
- [x] Adherence monitoring
- [x] CPT code tracking (98975, 98976, 98977, 98980, 98981)
- [x] Billable time calculation
- [x] Monthly compliance reports

### 🔧 Features In Development
- [ ] Femto Mega local server integration (Phase 3 - requires hardware)
- [ ] Advanced AI deficiency detection models
- [ ] Multi-clinician collaboration tools
- [ ] Telehealth video integration
- [ ] Mobile app (iOS/Android)

### 📈 Recommended Next Steps
1. **Deploy to Production Cloudflare Pages**
2. **Set up Femto Mega camera workstation** (for professional clinical use)
3. **Integrate with EHR systems** (Epic, Cerner, AllScripts)
4. **Add payment processing** (Stripe for patient billing)
5. **Implement role-based access control** (Admin, Clinician, Assistant)
6. **Build reporting dashboards** (outcome tracking, patient analytics)

## 🏗️ Tech Stack

### Frontend
- **Framework**: Pure JavaScript + TailwindCSS
- **UI Components**: Custom components with Tailwind utility classes
- **Icons**: Font Awesome 6.4
- **Motion Analysis**: MediaPipe Pose 0.5
- **Camera Access**: WebRTC getUserMedia API

### Backend
- **Framework**: Hono (lightweight edge framework)
- **Runtime**: Cloudflare Workers
- **Database**: Cloudflare D1 (SQLite)
- **Storage**: Cloudflare R2 (videos, images)
- **Deployment**: Cloudflare Pages

### Development Tools
- **Language**: TypeScript 5.0
- **Build Tool**: Vite 6.3
- **Package Manager**: npm
- **Process Manager**: PM2
- **Version Control**: Git

## 📁 Project Structure

```
webapp/
├── src/
│   ├── index.tsx              # Main Hono application (27KB, 711 lines)
│   ├── types.ts               # TypeScript type definitions (14KB)
│   ├── utils/
│   │   └── biomechanics.ts   # Joint angle calculations
│   └── renderer.tsx           # HTML renderer
├── public/
│   └── static/
│       └── app.js            # Frontend JavaScript (17KB, 452 lines)
├── migrations/
│   └── 0001_initial_schema.sql  # Database schema (13KB, 391 lines)
├── seed.sql                   # Sample data (10KB, 65 rows)
├── wrangler.jsonc            # Cloudflare configuration
├── vite.config.ts            # Build configuration
├── ecosystem.config.cjs      # PM2 configuration
└── package.json              # Dependencies and scripts
```

**Total Lines of Code**: ~2,800 lines (TypeScript + SQL + JavaScript)

## 🗄️ Database Schema

### Core Tables
- **patients** - Demographics, medical history, insurance
- **assessments** - Movement evaluations, scores, videos
- **movement_tests** - Individual FMS tests, skeleton data
- **exercises** - Exercise library with instructions
- **prescriptions** - Exercise programs for patients
- **prescribed_exercises** - Specific exercise parameters
- **exercise_sessions** - Patient home session tracking
- **exercise_performances** - Individual exercise execution data
- **rpm_monitoring** - Remote patient monitoring for billing
- **clinicians** - User accounts and credentials

### Key Features
- **Comprehensive indexes** for fast queries
- **JSON fields** for flexible data structures
- **Foreign key constraints** for data integrity
- **Automatic timestamps** on all records

## 🚀 Development Workflow

### Local Development
```bash
# Install dependencies
npm install

# Build project
npm run build

# Create and seed local D1 database
npm run db:reset

# Start development server with PM2
pm2 start ecosystem.config.cjs

# View logs
pm2 logs webapp --nostream

# Stop server
pm2 stop webapp
```

### Database Commands
```bash
# Apply migrations locally
npm run db:migrate:local

# Seed database
npm run db:seed

# Reset database (drop and recreate)
npm run db:reset

# Execute SQL commands
npm run db:console:local
```

### Testing
```bash
# Test application
curl http://localhost:3000

# Test API endpoints
curl http://localhost:3000/api/patients
curl http://localhost:3000/api/exercises
curl http://localhost:3000/api/assessments
```

## 🌍 Deployment

### Prerequisites
1. Cloudflare account
2. Wrangler CLI configured
3. D1 database created
4. R2 bucket created (optional for video storage)

### Deploy to Cloudflare Pages
```bash
# Build project
npm run build

# Create D1 production database
npx wrangler d1 create webapp-production

# Update wrangler.jsonc with database_id

# Apply migrations to production
npm run db:migrate:prod

# Deploy to Cloudflare Pages
npm run deploy:prod

# Set environment variables (if needed)
npx wrangler pages secret put API_KEY --project-name webapp
```

### Production URLs
- **Application**: https://webapp.pages.dev
- **API**: https://webapp.pages.dev/api/*

## 📸 Screenshots

### Dashboard
- Real-time patient statistics
- Quick action buttons
- Recent assessments

### Patient Intake Form
- Multi-section form with validation
- Medical history capture
- Emergency contact information

### Assessment Workflow
- Step-by-step FMS protocol
- Camera integration
- Real-time skeleton overlay
- Automated scoring

### Exercise Prescription
- Searchable exercise library
- Deficiency-based recommendations
- Customizable parameters
- Patient assignment

### Patient Portal
- Exercise viewing
- Video demonstrations
- Real-time form feedback
- Progress tracking

## 🔐 Security & Compliance

### HIPAA Considerations
- **Patient data encryption** (at rest and in transit)
- **Secure authentication** required for production
- **Audit logging** for all data access
- **Business Associate Agreement (BAA)** with Cloudflare needed

### Recommended Security Additions
1. Implement user authentication (Auth0, Clerk, or Cloudflare Access)
2. Add role-based access control (RBAC)
3. Enable Cloudflare WAF for API protection
4. Implement rate limiting
5. Add CAPTCHA for public forms
6. Enable 2FA for clinician accounts

## 🩺 Medical Use Case

### For Chiropractors
- Pre/post-adjustment movement assessments
- Range of motion documentation
- Patient progress tracking
- Exercise prescription for home care

### For Physical Therapists
- Functional movement screening
- Post-surgical rehabilitation tracking
- Athletic performance optimization
- Injury prevention protocols

### For Athletic Trainers
- Pre-season baseline assessments
- Return-to-play evaluations
- Movement quality monitoring
- Injury risk identification

## 📊 Billing & Reimbursement

### Supported CPT Codes
- **98975** - Remote therapeutic monitoring (first 20 min/month)
- **98976** - RTM additional 20 minutes
- **98977** - RTM interactive communication (20+ min)
- **98980** - RTM setup and patient education
- **98981** - RTM device supply

### Billing Requirements
- Minimum 16 days of monitoring per month
- 20+ minutes of interactive communication
- Documented clinical decision-making
- Patient consent for remote monitoring

## 🤝 Integration Opportunities

### Orbbec Femto Mega Camera
- Professional-grade motion capture
- 32-joint skeleton tracking
- Azure Kinect compatibility
- Requires separate Python server

### EHR Systems
- Export assessments to SOAP notes
- Import patient demographics
- Sync appointments and schedules

### Telehealth Platforms
- Video consultation integration
- Screen sharing for exercise review
- Real-time collaboration

## 📝 API Documentation

### Patient Management
- `GET /api/patients` - List all patients
- `POST /api/patients` - Create new patient
- `GET /api/patients/:id` - Get patient details
- `PUT /api/patients/:id` - Update patient

### Assessment Management
- `POST /api/assessments` - Create assessment
- `GET /api/assessments/:id` - Get assessment with tests
- `PUT /api/assessments/:id` - Update assessment
- `POST /api/assessments/:id/complete` - Generate medical note

### Exercise Management
- `GET /api/exercises` - List exercises
- `POST /api/exercises/recommend` - Get recommendations for deficiencies

### Prescription Management
- `POST /api/prescriptions` - Create exercise program
- `GET /api/prescriptions/:id` - Get program with exercises
- `GET /api/patients/:id/prescriptions` - List patient programs

### Session Tracking
- `POST /api/exercise-sessions` - Log patient session
- `PUT /api/exercise-sessions/:id/complete` - Mark completed
- `POST /api/exercise-performances` - Log exercise performance

## 🐛 Known Issues & Limitations

1. **MediaPipe Performance**: Heavy CPU usage during real-time tracking
2. **Mobile Safari**: Camera permissions require HTTPS in production
3. **Video Storage**: Large files may require R2 optimization
4. **Femto Mega**: Requires separate local server (not browser-compatible)

## 📧 Support & Contact

For questions, issues, or feature requests, please contact:
- **Developer**: PhysioAI Development Team
- **Email**: support@physioai.example.com
- **Documentation**: https://docs.physioai.example.com

## 📄 License

Copyright © 2025 PhysioAI. All rights reserved.

This software is proprietary and confidential. Unauthorized copying, distribution, or use is strictly prohibited.

---

**Built with ❤️ for movement specialists worldwide**
