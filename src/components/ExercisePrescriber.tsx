import { ClinicalLayout } from './clinicalStyles.js';
import { EXERCISE_LIBRARY_CLINICAL } from '../utils/clinical.js';

export function ExercisePrescriber() {
  return (
    <ClinicalLayout title="Exercise Prescriber" subtitle="AI-assisted exercise plans with filters, dosage, progression, and CPT mapping.">
      <section class="clinical-grid">
        <div class="clinical-card span-12">
          <h2>Filters</h2>
          <div class="clinical-controls">
            <select class="clinical-select" id="regionFilter"><option value="">All regions</option><option>hip</option><option>knee</option><option>ankle</option><option>shoulder</option><option>spine</option><option>balance</option></select>
            <select class="clinical-select" id="difficultyFilter"><option value="">Any difficulty</option><option>beginner</option><option>intermediate</option></select>
            <select class="clinical-select" id="equipmentFilter"><option value="">Any equipment</option><option>none</option><option>mat</option><option>chair</option><option>wall</option></select>
            <input class="clinical-input" id="diagnosisFilter" placeholder="Diagnosis or finding" />
          </div>
        </div>
        <div class="span-12 clinical-grid" id="exerciseGrid">
          {EXERCISE_LIBRARY_CLINICAL.map((exercise) => (
            <div class="clinical-card span-6 exercise-card" data-region={exercise.region} data-difficulty={exercise.difficulty} data-equipment={exercise.equipment.join(',')} data-diagnosis={exercise.diagnosis.join(',')}>
              <img src={exercise.media} alt={`${exercise.name} reference`} />
              <div>
                <h3>{exercise.name}</h3>
                <p class="muted" style="margin:0 0 6px;font-size:.76rem">{exercise.diagnosis.join(', ')}</p>
                <div class="metric"><span>Dosage</span><strong>{exercise.sets} sets, {exercise.reps}, {exercise.frequency}</strong></div>
                <div class="metric"><span>Equipment</span><strong>{exercise.equipment.join(', ')}</strong></div>
                <div class="metric"><span>CPT</span><strong>{exercise.cpt.join(', ')}</strong></div>
              </div>
            </div>
          ))}
        </div>
      </section>
      <script dangerouslySetInnerHTML={{ __html: exerciseScript }} />
    </ClinicalLayout>
  );
}

const exerciseScript = `
(function(){
  const ids=['regionFilter','difficultyFilter','equipmentFilter','diagnosisFilter'];
  function filter(){
    const region=document.getElementById('regionFilter').value;
    const difficulty=document.getElementById('difficultyFilter').value;
    const equipment=document.getElementById('equipmentFilter').value;
    const dx=document.getElementById('diagnosisFilter').value.toLowerCase();
    document.querySelectorAll('#exerciseGrid .exercise-card').forEach(card=>{
      const okRegion=!region||card.dataset.region===region;
      const okDifficulty=!difficulty||card.dataset.difficulty===difficulty;
      const okEquipment=!equipment||card.dataset.equipment.includes(equipment);
      const okDx=!dx||card.dataset.diagnosis.toLowerCase().includes(dx)||card.textContent.toLowerCase().includes(dx);
      card.style.display=okRegion&&okDifficulty&&okEquipment&&okDx?'grid':'none';
    });
  }
  ids.forEach(id=>document.getElementById(id).addEventListener('input',filter));
})();
`;

