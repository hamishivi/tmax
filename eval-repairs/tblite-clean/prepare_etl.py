"""Apply the ETL startup repair to a new full Harbor task directory."""

import argparse
import shlex
import shutil
from pathlib import Path

STARTUP = """#!/bin/bash
set -e

# Serialize initialization and any verifier command with other startup calls.
# The long-lived server must not inherit this lock.
(
  flock -x 9
  echo "Starting PostgreSQL..."
  if ! su - postgres -c "pg_isready -h 127.0.0.1 -p 5432 -q" 9>&-; then
    su - postgres -c "/usr/lib/postgresql/*/bin/pg_ctl -D /var/lib/postgresql/data -l /var/log/postgresql/logfile -w start" 9>&-
  fi

  for database in legacy_db new_db; do
    exists=$(su - postgres -c "psql -h 127.0.0.1 -p 5432 -d postgres -v ON_ERROR_STOP=1 -tAc \\\"SELECT 1 FROM pg_database WHERE datname='$database'\\\"" 9>&-)
    if [ "$exists" != "1" ]; then
      su - postgres -c "psql -h 127.0.0.1 -p 5432 -d postgres -v ON_ERROR_STOP=1 -c \\\"CREATE DATABASE $database;\\\"" 9>&-
    fi
  done
  echo "PostgreSQL started and databases created"

  export LEGACY_DB_URL="postgresql://postgres:@localhost:5432/legacy_db"
  export NEW_DB_URL="postgresql://postgres:@localhost:5432/new_db"
  python /workspace/scripts/setup_databases.py 9>&-
  echo "Database schemas created"
  python /workspace/scripts/populate_legacy_data.py 100 9>&-
  echo "Test data populated - ready for migration"

  # Run verification before another initializer can reset its database tables.
  if [ "$#" -gt 0 ]; then
    "$@"
  fi
) 9>/tmp/etl-initialization.lock

# Preserve the original container startup behavior when no command was supplied.
if [ "$#" -eq 0 ]; then
  exec tail -f /dev/null
fi
"""


def prepare(source, output):
    startup = (source / "environment/start_postgres.sh").read_text()
    for old, count in (("pg_isready -q", 1), ("psql -c", 2)):
        if startup.count(old) != count:
            raise ValueError(f"Unexpected upstream ETL startup: {old}")
    verifier = (source / "tests/test.sh").read_text()
    prefix = "#!/bin/bash\n\n# Start the application in the background\n/workspace/start.sh &\nAPP_PID=$!\n\n"
    if not verifier.startswith(prefix):
        raise ValueError("Unexpected upstream ETL verifier")
    # Startup errors remain visible and produce no misleading model reward.
    verifier = (
        "#!/bin/bash\nset -e\n/workspace/start_postgres.sh bash -c " + shlex.quote(verifier[len(prefix) :]) + "\n"
    )
    shutil.copytree(source, output)
    (output / "environment/start_postgres.sh").write_text(STARTUP)
    (output / "tests/test.sh").write_text(verifier)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.source, args.output)
