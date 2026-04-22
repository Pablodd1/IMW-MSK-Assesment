#!/usr/bin/env node

// Qwen Model Integration for IMW-MSK-Assessment
// This script connects the local Qwen model to provide AI-powered analysis

const { spawn } = require('child_process');
const readline = require('readline');

class QwenMedicalAI {
  constructor() {
    this.modelName = 'qwen2.5-medical';
  }

  async analyzeAssessment(assessmentData) {
    const prompt = `
You are a medical AI assistant analyzing musculoskeletal assessments. 
Based on the following assessment data, provide:
1. Key findings summary
2. Clinical recommendations
3. Risk factors identified
4. Progression suggestions

Assessment Data:
${JSON.stringify(assessmentData, null, 2)}

Please provide a concise medical analysis:`;

    return this.runQuery(prompt);
  }

  async runQuery(prompt) {
    return new Promise((resolve, reject) => {
      const ollama = spawn('ollama', ['run', this.modelName, prompt]);
      let output = '';
      let error = '';

      ollama.stdout.on('data', (data) => {
        output += data.toString();
      });

      ollama.stderr.on('data', (data) => {
        error += data.toString();
      });

      ollama.on('close', (code) => {
        if (code === 0) {
          resolve(output.trim());
        } else {
          reject(new Error(`Ollama process exited with code ${code}: ${error}`));
        }
      });

      ollama.on('error', (err) => {
        reject(err);
      });
    });
  }
}

// Test the integration
async function testIntegration() {
  const ai = new QwenMedicalAI();
  
  const sampleAssessment = {
    patient_name: "John Doe",
    assessment_type: "Initial Evaluation",
    chief_complaint: "Lower back pain radiating to right leg, pain level 7/10",
    tests: {
      lumbar_rom: { flexion: 45, normal: 60, status: "limited" },
      straight_leg_raise: { right: 45, normal: 80, status: "limited" }
    },
    pain_scale: 7,
    functional_status: "Limited household ambulation"
  };

  try {
    console.log("🤖 Analyzing assessment with Qwen Medical AI...\n");
    const analysis = await ai.analyzeAssessment(sampleAssessment);
    console.log("📋 AI Analysis Results:");
    console.log("=".repeat(50));
    console.log(analysis);
    console.log("=".repeat(50));
  } catch (error) {
    console.error("❌ Error:", error.message);
  }
}

// Run test if called directly
if (require.main === module) {
  testIntegration();
}

module.exports = QwenMedicalAI;
