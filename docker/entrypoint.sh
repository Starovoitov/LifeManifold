#!/usr/bin/env bash
# Bind-mounted repo at /workspace. The image venv lives outside that mount so
# the host .venv (Windows or a broken symlink tree) is not what the scripts run.
set -euo pipefail

cd /workspace

if [[ -f /workspace/.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source <(sed 's/\r$//' /workspace/.env)
  set +a
fi

if ! /workspace/.venv/bin/python -c "import numpy" >/dev/null 2>&1; then
  echo "Copying the Linux environment into the Docker volume (first start only)..."
  mkdir -p /workspace/.venv
  cp -a /opt/lifemanifold/.venv/. /workspace/.venv/
fi

export PATH="/workspace/.venv/bin:${PATH}"
export VIRTUAL_ENV="/workspace/.venv"
export UV_PROJECT_ENVIRONMENT="/workspace/.venv"
export UV_LINK_MODE=copy
export PYTHONUNBUFFERED=1
export PYTHONPATH="/workspace${PYTHONPATH:+:$PYTHONPATH}"

if [[ $# -eq 0 ]]; then
  exec bash
fi
exec "$@"
