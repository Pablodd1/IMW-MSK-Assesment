import re

with open('sql/schema-pg.sql', 'r') as f:
    sql = f.read()

# Since we have duplicate ALTER TABLE, and Postgres throws an error when a column exists,
# we need to remove duplicates manually from the final SQL.
# Line 593: ALTER TABLE patients ADD COLUMN height_cm REAL;
# Line 594: ALTER TABLE patients ADD COLUMN weight_kg REAL;
# Line 598: ALTER TABLE movement_tests ADD COLUMN status TEXT DEFAULT 'pending'
# Actually, the best way to handle Postgres migrations without duplicates is to execute
# the CREATE TABLEs directly with the final columns merged in, rather than chaining ALTER TABLEs.
# But writing a python script to merge ALTER TABLEs is tricky.
# Let's write a simple script that removes duplicated ADD COLUMN lines.

lines = sql.split('\n')
seen_alters = set()
new_lines = []

for line in lines:
    m = re.match(r'^\s*ALTER\s+TABLE\s+(\w+)\s+ADD\s+COLUMN\s+(\w+)', line, re.IGNORECASE)
    if m:
        table_name = m.group(1).lower()
        col_name = m.group(2).lower()
        key = f"{table_name}.{col_name}"
        if key in seen_alters:
            continue
        seen_alters.add(key)
    new_lines.append(line)

with open('sql/schema-pg.sql', 'w') as f:
    f.write('\n'.join(new_lines))
