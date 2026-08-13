#!/usr/bin/env bash
# Execute assigned contemporaneous RQ1 blocks only after explicit acknowledgement.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WORKERS="${RQ1_FACTORIAL_WORKERS:-2}"
WORKER="${1:?worker id required}"

if [[ "${RQ1_FACTORIAL_EXECUTE:-}" != "YES" ]]; then
  echo "Dry guard: set RQ1_FACTORIAL_EXECUTE=YES to permit live API runs." >&2
  echo "No experiment was started." >&2
  exit 2
fi
if [[ ! "$WORKERS" =~ ^[0-9]+$ ]] || (( WORKERS < 1 )); then
  echo "RQ1_FACTORIAL_WORKERS must be a positive integer" >&2
  exit 2
fi
if [[ ! "$WORKER" =~ ^[0-9]+$ ]] || (( WORKER < 0 || WORKER >= WORKERS )); then
  echo "worker id must be in 0..$((WORKERS - 1))" >&2
  exit 2
fi

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

block=0
for archive_index in $(seq 0 4); do
  for continuation_index in $(seq 0 1); do
    if (( block % WORKERS == WORKER )); then
      uv run python "$ROOT/scripts/run_rq1_contemporaneous_factorial.py" \
        run-block \
        --archive-index "$archive_index" \
        --continuation-index "$continuation_index" \
        --execute \
        --ack RQ1-CF-2026-08-13
    fi
    block=$((block + 1))
  done
done
