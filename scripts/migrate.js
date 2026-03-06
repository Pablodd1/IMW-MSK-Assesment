import { Pool } from 'pg';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

if (!process.env.DATABASE_URL) {
  console.error("DATABASE_URL is not set");
  process.exit(1);
}

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
});

async function run() {
  const schemaPath = path.join(__dirname, '../sql/schema-pg.sql');
  const seedPath = path.join(__dirname, '../sql/seed-pg.sql');

  try {
    console.log("Applying schema...");
    const schema = fs.readFileSync(schemaPath, 'utf8');
    await pool.query(schema);
    console.log("Schema applied successfully.");

    if (process.argv.includes('--seed')) {
        console.log("Applying seed data...");
        const seed = fs.readFileSync(seedPath, 'utf8');
        await pool.query(seed);
        console.log("Seed data applied successfully.");
    }
  } catch (e) {
    console.error("Migration failed:", e);
    process.exit(1);
  } finally {
    await pool.end();
  }
}

run();
