-- IMW-MSK PhysioMotion clinical platform phases 4-9
-- Supabase project: swromegqjbakdiftzwum.supabase.co

create table if not exists gait_sessions (
  id uuid primary key default gen_random_uuid(),
  patient_id uuid references patients(id) on delete cascade,
  assessment_id uuid references assessments(id) on delete set null,
  clinician_id uuid references clinicians(id) on delete set null,
  treadmill_mode boolean not null default false,
  phase_summary jsonb not null default '{}'::jsonb,
  stride_length_cm numeric,
  cadence_spm numeric,
  step_width_cm numeric,
  single_support_pct numeric,
  double_support_pct numeric,
  pronation text check (pronation in ('neutral', 'pronation', 'supination')),
  pelvic_tilt_deg numeric,
  arm_swing_symmetry_pct numeric,
  raw_keypoints jsonb,
  created_at timestamptz not null default now()
);

create table if not exists muscle_assessments (
  id uuid primary key default gen_random_uuid(),
  patient_id uuid references patients(id) on delete cascade,
  assessment_id uuid references assessments(id) on delete set null,
  clinician_id uuid references clinicians(id) on delete set null,
  joint text not null,
  movement text not null,
  side text not null check (side in ('left', 'right', 'bilateral')),
  muscle_group text not null,
  mmt_grade numeric not null check (mmt_grade >= 0 and mmt_grade <= 5),
  force_estimate_n numeric,
  fma_upper_extremity integer,
  fma_lower_extremity integer,
  fma_total integer,
  notes text,
  created_at timestamptz not null default now()
);

create table if not exists exercise_prescriptions (
  id uuid primary key default gen_random_uuid(),
  patient_id uuid references patients(id) on delete cascade,
  assessment_id uuid references assessments(id) on delete set null,
  clinician_id uuid references clinicians(id) on delete set null,
  exercise_id text not null,
  exercise_name text not null,
  diagnosis text,
  body_region text,
  difficulty text,
  equipment_needed text[] not null default '{}',
  sets integer not null default 3,
  reps text not null default '10',
  frequency text not null default 'daily',
  progression jsonb not null default '{}'::jsonb,
  cpt_codes text[] not null default '{}',
  media_url text,
  status text not null default 'active' check (status in ('draft', 'active', 'completed', 'paused')),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists progress_metrics (
  id uuid primary key default gen_random_uuid(),
  patient_id uuid references patients(id) on delete cascade,
  assessment_id uuid references assessments(id) on delete set null,
  session_date timestamptz not null default now(),
  metric_type text not null,
  metric_name text not null,
  baseline_value numeric,
  current_value numeric not null,
  unit text,
  improvement_pct numeric,
  goal_value numeric,
  goal_progress_pct numeric,
  comparison jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists idx_gait_sessions_patient_created on gait_sessions(patient_id, created_at desc);
create index if not exists idx_muscle_assessments_patient_created on muscle_assessments(patient_id, created_at desc);
create index if not exists idx_exercise_prescriptions_patient_status on exercise_prescriptions(patient_id, status);
create index if not exists idx_progress_metrics_patient_date on progress_metrics(patient_id, session_date desc);

alter table gait_sessions enable row level security;
alter table muscle_assessments enable row level security;
alter table exercise_prescriptions enable row level security;
alter table progress_metrics enable row level security;

do $$
begin
  if not exists (select 1 from pg_policies where schemaname = 'public' and tablename = 'gait_sessions' and policyname = 'Authenticated clinical gait access') then
    create policy "Authenticated clinical gait access" on gait_sessions
      for all using (auth.role() = 'authenticated') with check (auth.role() = 'authenticated');
  end if;
  if not exists (select 1 from pg_policies where schemaname = 'public' and tablename = 'muscle_assessments' and policyname = 'Authenticated clinical muscle access') then
    create policy "Authenticated clinical muscle access" on muscle_assessments
      for all using (auth.role() = 'authenticated') with check (auth.role() = 'authenticated');
  end if;
  if not exists (select 1 from pg_policies where schemaname = 'public' and tablename = 'exercise_prescriptions' and policyname = 'Authenticated exercise prescription access') then
    create policy "Authenticated exercise prescription access" on exercise_prescriptions
      for all using (auth.role() = 'authenticated') with check (auth.role() = 'authenticated');
  end if;
  if not exists (select 1 from pg_policies where schemaname = 'public' and tablename = 'progress_metrics' and policyname = 'Authenticated progress metric access') then
    create policy "Authenticated progress metric access" on progress_metrics
      for all using (auth.role() = 'authenticated') with check (auth.role() = 'authenticated');
  end if;
end $$;
