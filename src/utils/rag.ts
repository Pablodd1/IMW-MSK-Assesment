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
  let exercises: any[] = [];
  try {
    // Optimize: Use SQL LIKE for filtering instead of fetching all rows
    // Construct dynamic query for keywords
    const conditions = keywords.map(() =>
      `(LOWER(name) LIKE ? OR LOWER(description) LIKE ? OR LOWER(instructions) LIKE ?)`
    );

    const sql = `SELECT * FROM exercises WHERE ${conditions.join(' OR ')} LIMIT 20`;

    // Create parameters array (3 params per keyword)
    const params: string[] = [];
    keywords.forEach(k => {
      const pattern = `%${k}%`;
      params.push(pattern, pattern, pattern);
    });

    // Ensure we don't execute invalid SQL if logic changes
    if (conditions.length > 0) {
        const { results } = await db.prepare(sql).bind(...params).all();
        exercises = results;
    }
  } catch (e) {
    console.error('RAG Database error:', e);
  }

  const matches = exercises;

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
