import { Hono } from 'hono';
import { cors } from 'hono/cors';

const aiRoutes = new Hono();

// Enable CORS for AI routes
aiRoutes.use('/*', cors());

// Qwen Medical AI Integration
aiRoutes.post('/analyze', async (c) => {
  try {
    const { assessmentData } = await c.req.json();
    
    if (!assessmentData) {
      return c.json({ 
        success: false, 
        error: 'Assessment data is required' 
      }, 400);
    }

    console.log('Analyzing assessment with Qwen Medical AI');
    
    // For demo purposes, return a mock AI analysis
    // In production, this would call the actual Qwen model
    const mockAnalysis = {
      summary: "Patient presents with classic signs of lumbar radiculopathy with positive straight leg raise test.",
      recommendations: [
        "Continue with lumbar stabilization exercises",
        "Consider MRI if no improvement in 2-4 weeks",
        "Monitor for red flags (bowel/bladder changes, progressive weakness)"
      ],
      risk_factors: [
        "Prolonged sitting occupation",
        "Reduced straight leg raise range",
        "Pain radiation pattern"
      ],
      progression: [
        "Week 1-2: Pain management and gentle mobility",
        "Week 3-4: Core strengthening progression",
        "Week 5-6: Functional movement patterns"
      ],
      confidence: 0.87,
      model: "qwen2.5-medical (demo mode)"
    };

    return c.json({
      success: true,
      analysis: mockAnalysis,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('AI Analysis Error:', error);
    return c.json({ 
      success: false, 
      error: 'Failed to analyze assessment' 
    }, 500);
  }
});

// Get AI model info
aiRoutes.get('/info', (c) => {
  return c.json({
    success: true,
    model: {
      name: "Qwen 2.5 Medical",
      version: "7B-Instruct",
      description: "Specialized medical AI for musculoskeletal assessment analysis",
      capabilities: [
        "Movement pattern analysis",
        "Risk factor identification", 
        "Treatment recommendations",
        "Progression planning"
      ],
      status: "demo mode - production ready"
    },
    timestamp: new Date().toISOString()
  });
});

export default aiRoutes;