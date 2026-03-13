// Gemini Integration for Medical Analysis
import { GoogleGenerativeAI } from '@google/generative-ai';

export class MedicalAIAnalysis {
  private genAI: GoogleGenerativeAI;
  private model: any;

  constructor(apiKey: string) {
    this.genAI = new GoogleGenerativeAI(apiKey);
    this.model = this.genAI.getGenerativeModel({ model: 'gemini-1.5-flash' });
  }

  async analyzeBiomechanics(jointData: any, patientContext: any): Promise<any> {
    const prompt = `
      You are a medical biomechanics expert analyzing movement data for a physiatrist.
      Analyze the following joint data and provide clinical insights:

      Joint Angles: ${JSON.stringify(jointData.joint_angles)}
      Detected Compensations: ${JSON.stringify(jointData.detected_compensations)}
      Patient Context: ${JSON.stringify(patientContext)}

      Provide:
      1. Clinical interpretation of findings
      2. Specific recommendations for physiotherapy
      3. Risk assessment for injury
      4. Exercise prescription suggestions
      5. Follow-up timeline

      Format as JSON with keys: interpretation, recommendations, risk_level, exercises, follow_up
    `;

    const result = await this.model.generateContent(prompt);
    return JSON.parse(result.response.text());
  }

  async generateProgressReport(patientId: number, historicalData: any[]): Promise<string> {
    const prompt = `
      Generate a medical progress report for patient ${patientId} based on ${historicalData.length} assessments.
      
      Historical Data: ${JSON.stringify(historicalData.slice(-5))}
      
      Include:
      - Trend analysis (improving/stable/declining)
      - Key metrics comparison
      - Clinical summary
      - Next steps recommendations
      
      Format as markdown for physician review.
    `;

    const result = await this.model.generateContent(prompt);
    return result.response.text();
  }
}

// Export for global access
export const medicalAI = {
  instance: null as MedicalAIAnalysis | null,
  init(apiKey: string) {
    this.instance = new MedicalAIAnalysis(apiKey);
  }
};
