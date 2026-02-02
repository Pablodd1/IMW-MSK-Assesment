// PhysioMotion - Intake Form Workflow
// Multi-step patient intake with validation and review

let currentStep = 1;
let formData = {};
let createdPatientId = null;

// ============================================================================
// STEP NAVIGATION
// ============================================================================

function nextStep(step) {
  // Validate current step before proceeding
  if (!validateCurrentStep()) {
    return;
  }

  // Save current step data
  saveStepData();

  // Hide current step
  document.getElementById(`step${currentStep}`).classList.add('hidden');
  document.getElementById(`step${currentStep}`).classList.remove('active');
  document.getElementById(`progressStep${currentStep}`).classList.remove('active');
  document.getElementById(`progressStep${currentStep}`).classList.add('completed');

  // Show next step
  currentStep = step;
  document.getElementById(`step${step}`).classList.remove('hidden');
  document.getElementById(`step${step}`).classList.add('active');
  document.getElementById(`progressStep${step}`).classList.add('active');

  // If moving to review step, generate review content
  if (step === 4) {
    generateReviewContent();
  }

  // Scroll to top
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function prevStep(step) {
  // Save current step data
  saveStepData();

  // Hide current step
  document.getElementById(`step${currentStep}`).classList.add('hidden');
  document.getElementById(`step${currentStep}`).classList.remove('active');
  document.getElementById(`progressStep${currentStep}`).classList.remove('active');

  // Show previous step
  currentStep = step;
  document.getElementById(`step${step}`).classList.remove('hidden');
  document.getElementById(`step${step}`).classList.add('active');
  document.getElementById(`progressStep${step}`).classList.add('active');
  document.getElementById(`progressStep${step}`).classList.remove('completed');

  // Scroll to top
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ============================================================================
// VALIDATION
// ============================================================================

function validateCurrentStep() {
  const stepElement = document.getElementById(`step${currentStep}`);
  const requiredFields = stepElement.querySelectorAll('[required]');

  let isValid = true;
  let firstInvalidField = null;

  requiredFields.forEach(field => {
    if (field.type === 'radio') {
      // Check if at least one radio in the group is selected
      const radioGroup = document.getElementsByName(field.name);
      const isRadioChecked = Array.from(radioGroup).some(radio => radio.checked);

      if (!isRadioChecked) {
        isValid = false;
        if (!firstInvalidField) {
          firstInvalidField = field;
        }
        // Highlight the radio group container
        const container = field.closest('.assessment-card');
        if (container) {
          container.parentElement.classList.add('border-2', 'border-red-500', 'rounded-lg', 'p-2');
        }
      }
    } else {
      if (!field.value.trim()) {
        isValid = false;
        field.classList.add('border-red-500');
        if (!firstInvalidField) {
          firstInvalidField = field;
        }
      } else {
        field.classList.remove('border-red-500');
      }
    }
  });

  if (!isValid) {
    alert('Please fill in all required fields marked with *');
    if (firstInvalidField) {
      firstInvalidField.focus();
    }
    return false;
  }

  return true;
}

// ============================================================================
// DATA MANAGEMENT
// ============================================================================

function saveStepData() {
  const stepElement = document.getElementById(`step${currentStep}`);
  const inputs = stepElement.querySelectorAll('input, select, textarea');

  inputs.forEach(input => {
    if (input.type === 'radio') {
      if (input.checked) {
        formData[input.name] = input.value;
      }
    } else if (input.type === 'checkbox') {
      formData[input.name] = input.checked;
    } else {
      formData[input.name] = input.value;
    }
  });

  console.log('💾 Saved step data:', formData);
}

// ============================================================================
// REVIEW CONTENT GENERATION
// ============================================================================

function generateReviewContent() {
  const reviewContent = document.getElementById('reviewContent');

  const assessmentReasonLabels = {
    'pre_surgery': 'Pre-Surgery Assessment',
    'post_surgery': 'Post-Surgery Recovery',
    'injury_recovery': 'Injury Recovery',
    'athletic_performance': 'Athletic Performance',
    'general_wellness': 'General Wellness'
  };

  const activityLevelLabels = {
    'sedentary': 'Sedentary',
    'light': 'Light Activity',
    'moderate': 'Moderate Activity',
    'active': 'Active',
    'very_active': 'Very Active'
  };

  reviewContent.innerHTML = `
    <!-- Personal Information -->
    <div class="bg-blue-50 border-l-4 border-blue-500 p-4 rounded">
      <h3 class="font-bold text-lg mb-3 text-blue-900">
        <i class="fas fa-user mr-2"></i>Personal Information
      </h3>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
        <div><span class="font-semibold">Name:</span> ${formData.first_name} ${formData.last_name}</div>
        <div><span class="font-semibold">Date of Birth:</span> ${formData.date_of_birth}</div>
        <div><span class="font-semibold">Gender:</span> ${formData.gender}</div>
        <div><span class="font-semibold">Email:</span> ${formData.email}</div>
        <div><span class="font-semibold">Phone:</span> ${formData.phone}</div>
        ${formData.address_line1 ? `<div><span class="font-semibold">Address:</span> ${formData.address_line1}, ${formData.city}, ${formData.state} ${formData.zip_code}</div>` : ''}
      </div>
    </div>

    <!-- Emergency Contact -->
    ${formData.emergency_contact_name ? `
    <div class="bg-yellow-50 border-l-4 border-yellow-500 p-4 rounded">
      <h3 class="font-bold text-lg mb-3 text-yellow-900">
        <i class="fas fa-phone mr-2"></i>Emergency Contact
      </h3>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
        <div><span class="font-semibold">Name:</span> ${formData.emergency_contact_name}</div>
        <div><span class="font-semibold">Phone:</span> ${formData.emergency_contact_phone || 'Not provided'}</div>
      </div>
    </div>
    ` : ''}

    <!-- Medical Information -->
    <div class="bg-green-50 border-l-4 border-green-500 p-4 rounded">
      <h3 class="font-bold text-lg mb-3 text-green-900">
        <i class="fas fa-notes-medical mr-2"></i>Medical Information
      </h3>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
        ${formData.height_cm ? `<div><span class="font-semibold">Height:</span> ${formData.height_cm} cm</div>` : ''}
        ${formData.weight_kg ? `<div><span class="font-semibold">Weight:</span> ${formData.weight_kg} kg</div>` : ''}
        ${formData.insurance_provider ? `<div><span class="font-semibold">Insurance:</span> ${formData.insurance_provider}</div>` : ''}
        <div><span class="font-semibold">Pain Level:</span> ${formData.pain_scale || 0}/10</div>
        <div><span class="font-semibold">Activity Level:</span> ${activityLevelLabels[formData.activity_level] || formData.activity_level}</div>
      </div>
      ${formData.chief_complaint ? `
      <div class="mt-3">
        <span class="font-semibold">Chief Complaint:</span>
        <p class="text-gray-700 mt-1">${formData.chief_complaint}</p>
      </div>
      ` : ''}
    </div>

    <!-- Assessment Reason -->
    <div class="bg-purple-50 border-l-4 border-purple-500 p-4 rounded">
      <h3 class="font-bold text-lg mb-3 text-purple-900">
        <i class="fas fa-clipboard-check mr-2"></i>Assessment Reason
      </h3>
      <div class="text-sm">
        <span class="inline-block px-3 py-1 bg-purple-200 text-purple-800 rounded-full font-semibold">
          ${assessmentReasonLabels[formData.assessment_reason] || formData.assessment_reason}
        </span>
      </div>
    </div>

    <!-- Action Notice -->
    <div class="bg-blue-100 border border-blue-300 p-4 rounded-lg">
      <p class="text-blue-900 font-semibold mb-2">
        <i class="fas fa-info-circle mr-2"></i>Next Steps
      </p>
      <p class="text-blue-800 text-sm">
        After creating this patient record, you'll be able to:
      </p>
      <ul class="list-disc list-inside text-blue-800 text-sm mt-2 space-y-1">
        <li>Start a movement assessment immediately</li>
        <li>Capture biomechanical data using camera</li>
        <li>Generate AI-powered analysis and recommendations</li>
        <li>Create personalized exercise prescriptions</li>
      </ul>
    </div>
  `;
}

// ============================================================================
// FORM SUBMISSION
// ============================================================================

document.getElementById('intakeForm').addEventListener('submit', async (e) => {
  e.preventDefault();

  const submitBtn = document.getElementById('submitBtn');
  const originalText = submitBtn.innerHTML;

  try {
    // Disable button and show loading
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Creating Patient...';

    // Save final step data
    saveStepData();

    console.log('📤 Submitting patient data:', formData);

    // Submit to API
    const response = await axios.post('/api/patients', formData);

    if (response.data.success) {
      createdPatientId = response.data.data.id;
      console.log('✅ Patient created with ID:', createdPatientId);

      // Hide form
      document.getElementById('intakeForm').style.display = 'none';

      // Show success message
      const successMessage = document.getElementById('successMessage');
      document.getElementById('patientName').textContent = `${formData.first_name} ${formData.last_name}`;
      successMessage.classList.remove('hidden');

      // Scroll to success message
      successMessage.scrollIntoView({ behavior: 'smooth', block: 'center' });

    } else {
      throw new Error(response.data.error || 'Failed to create patient');
    }

  } catch (error) {
    console.error('❌ Error creating patient:', error);
    alert('Error creating patient: ' + (error.response?.data?.error || error.message));

    // Re-enable button
    submitBtn.disabled = false;
    submitBtn.innerHTML = originalText;
  }
});

// ============================================================================
// ASSESSMENT REDIRECTION
// ============================================================================

function startAssessment() {
  if (createdPatientId) {
    window.location.href = `/static/assessment?patient_id=${createdPatientId}`;
  } else {
    alert('No patient ID available. Please try again.');
  }
}

// ============================================================================
// TRANSCRIPTION HANDLER
// ============================================================================

function toggleTranscription(targetId) {
  const btn = document.getElementById('transcribeBtn');
  const btnText = document.getElementById('transcribeBtnText');
  const service = window.PhysioMotion.TranscriptionService;

  if (service.isTranscribing) {
    service.stop();
    btn.classList.remove('bg-red-100', 'text-red-700');
    btn.classList.add('bg-gray-100', 'text-gray-700');
    btnText.textContent = 'Transcribe Session';
  } else {
    service.start(targetId);
    btn.classList.remove('bg-gray-100', 'text-gray-700');
    btn.classList.add('bg-red-100', 'text-red-700');
    btnText.textContent = 'Stop Transcribing';
  }
}

// ============================================================================
// ASSESSMENT CARD SELECTION STYLING
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
  // Add click handlers for assessment cards
  const assessmentCards = document.querySelectorAll('.assessment-card');

  assessmentCards.forEach(card => {
    card.addEventListener('click', () => {
      // Remove selected class from all cards
      assessmentCards.forEach(c => c.classList.remove('selected-card'));

      // Add selected class to clicked card
      card.classList.add('selected-card');
    });
  });

  console.log('✅ Intake workflow initialized');
});
