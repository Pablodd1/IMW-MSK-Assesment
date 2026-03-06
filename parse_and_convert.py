import re
import os
import glob

sql_content = ""
for filepath in sorted(glob.glob('migrations/*.sql')):
    with open(filepath, 'r') as f:
        sql_content += f"\n\n-- File: {filepath}\n\n" + f.read()

# 1. Transform SQLite syntax to Postgres Syntax
pg_sql = sql_content

# Data Types
pg_sql = re.sub(r'INTEGER\s+PRIMARY\s+KEY\s+AUTOINCREMENT', 'SERIAL PRIMARY KEY', pg_sql, flags=re.IGNORECASE)
pg_sql = re.sub(r'\bDATETIME\b', 'TIMESTAMP', pg_sql, flags=re.IGNORECASE)

# Booleans in Schema
# SQLite often uses BOOLEAN DEFAULT FALSE/TRUE (which is valid), but we need to ensure no BOOLEAN DEFAULT 1/0
pg_sql = re.sub(r'BOOLEAN\s+DEFAULT\s+1\b', 'BOOLEAN DEFAULT TRUE', pg_sql, flags=re.IGNORECASE)
pg_sql = re.sub(r'BOOLEAN\s+DEFAULT\s+0\b', 'BOOLEAN DEFAULT FALSE', pg_sql, flags=re.IGNORECASE)

# We shouldn't execute ALTER TABLE duplicates. It's safer to just output the raw parsed PG SQL
# but we MUST fix ALTER TABLE if it's already there? No, running ALTER TABLE ADD COLUMN IF NOT EXISTS
# is not easily supported in Postgres.
# Since we are starting a fresh DB, let's just create a fresh merged schema.
# I'll use the original migration sequence. If the DB is fresh, running `CREATE TABLE` then `ALTER TABLE` works!
# Wait, in Postgres, `ALTER TABLE ... ADD COLUMN` fails if column exists. But since the DB is fresh,
# it will succeed exactly once! This is perfectly safe as long as we drop the database or tables first,
# or we just run the migrations one by one. But the easiest way is to just let the script run all migrations.
# Wait, the `0006_fix_schema_mismatch.sql` file might have duplicated some ALTER TABLE commands, let's check.

with open('sql/schema-pg.sql', 'w') as f:
    f.write(pg_sql)
