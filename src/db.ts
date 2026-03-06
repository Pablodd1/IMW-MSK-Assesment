import { Pool } from '@neondatabase/serverless';

if (!process.env.DATABASE_URL) {
  throw new Error('DATABASE_URL environment variable is not set');
}

export const db = new Pool({ connectionString: process.env.DATABASE_URL });

export class D1Mock {
  pool: Pool;
  constructor(pool: Pool) {
    this.pool = pool;
  }

  prepare(query: string) {
    // Replace ? placeholders with $1, $2, etc., avoiding placeholders inside string literals
    let pgQuery = '';
    let count = 1;
    let inString = false;
    for (let i = 0; i < query.length; i++) {
      const char = query[i];
      if (char === "'") {
        inString = !inString;
        pgQuery += char;
      } else if (char === '?' && !inString) {
        pgQuery += `$${count++}`;
      } else {
        pgQuery += char;
      }
    }

    // Check if it's an INSERT statement without RETURNING
    const isInsert = /^\s*INSERT\s+INTO/i.test(pgQuery);
    if (isInsert && !/RETURNING/i.test(pgQuery)) {
        // Find the end of the query (remove trailing semicolon if any)
        pgQuery = pgQuery.replace(/;\s*$/, '');
        pgQuery += ' RETURNING id';
    }

    return {
      bind: (...args: any[]) => {
        return {
          first: async <T = any>(): Promise<T | null> => {
            const res = await this.pool.query(pgQuery, args);
            return (res.rows[0] as T) || null;
          },
          all: async <T = any>() => {
            const res = await this.pool.query(pgQuery, args);
            return { results: res.rows as T[], success: true, meta: { last_row_id: null } };
          },
          run: async () => {
            const res = await this.pool.query(pgQuery, args);
            let lastId = null;
            if (isInsert && res.rows.length > 0) {
               lastId = res.rows[0].id;
            }
            return { success: true, meta: { last_row_id: lastId } };
          }
        };
      },
      first: async <T = any>(): Promise<T | null> => {
        const res = await this.pool.query(pgQuery);
        return (res.rows[0] as T) || null;
      },
      all: async <T = any>() => {
        const res = await this.pool.query(pgQuery);
        return { results: res.rows as T[], success: true, meta: { last_row_id: null } };
      },
      run: async () => {
        const res = await this.pool.query(pgQuery);
        let lastId = null;
        if (isInsert && res.rows.length > 0) {
           lastId = res.rows[0].id;
        }
        return { success: true, meta: { last_row_id: lastId } };
      }
    };
  }
}

export const mockD1 = new D1Mock(db);
