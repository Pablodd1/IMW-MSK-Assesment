export function validateEnv() {
  const required = ['DATABASE_URL', 'JWT_SECRET'];
  const missing = required.filter(key => !process.env[key]);

  if (missing.length > 0) {
    if (process.env.NODE_ENV !== 'test') {
      throw new Error(`Missing required environment variables: ${missing.join(', ')}`);
    }
  }
}
validateEnv();
