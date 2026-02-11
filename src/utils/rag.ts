// Embedded Training RAG (Retrieval-Augmented Generation) Prototype
// This module simulates a RAG system for exercise knowledge retrieval

export interface RAGResult {
  answer: string;
  sources: string[];
  confidence: number;
}

/**
 * Searches the exercise database for relevant information based on a query
 * In a real production system, this would use vector embeddings and a vector database.
 * For this prototype, we use keyword matching against the provided exercises.
 */
export async function queryExerciseKnowledge(db: any, query: string): Promise<RAGResult> {
  const keywords = query.toLowerCase().split(' ').filter(w => w.length > 3);

  if (keywords.length === 0) {
    return {
      answer: "I'm sorry, I couldn't find enough specific information in your query to provide clinical guidance.",
      sources: [],
      confidence: 0
    };
  }

  // Search exercises table for keywords
  let matches: any[] = [];
  try {
    // Construct dynamic query with parameterized LIKE clauses
    // Use LOWER() for case-insensitive matching to match original behavior
    const conditions = keywords.map(() => `(LOWER(name) LIKE ? OR LOWER(description) LIKE ? OR LOWER(instructions) LIKE ?)`).join(' OR ');
    const params = keywords.flatMap(k => [`%${k}%`, `%${k}%`, `%${k}%`]);

    const { results } = await db.prepare(`
      SELECT * FROM exercises WHERE ${conditions} LIMIT 20
    `).bind(...params).all();
    matches = results;
  } catch (e) {
    console.error('RAG Database error:', e);
  }

  if (matches.length === 0) {
    return {
      answer: "Based on our clinical library, I don't have specific training protocols for that movement yet. I recommend focusing on fundamental mobility exercises.",
      sources: [],
      confidence: 0.1
    };
  }

  // Build answer based on matches
  const topMatch = matches[0];
  const answer = `Based on our clinical protocols, for concerns related to "${query}", we recommend focusing on ${topMatch.name}. ${topMatch.description}. To perform this correctly: ${(topMatch.instructions || '').split('\n')[0]}.`;

  return {
    answer,
    sources: matches.slice(0, 3).map(m => m.name),
    confidence: 0.85
  };
}
