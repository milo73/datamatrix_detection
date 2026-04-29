#!/usr/bin/env bash
#
# Install / start the datamatrix-detector container on an Azure RHEL 9 VM
# that already has Docker Engine and the Compose plugin running.
#
# Usage:
#   bash scripts/install-rhel.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Verifying Docker is available..."
if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: 'docker' not found on PATH. Install Docker Engine first." >&2
  exit 1
fi
if ! docker compose version >/dev/null 2>&1; then
  echo "ERROR: 'docker compose' plugin not available." >&2
  echo "       Install the docker-compose-plugin package (or upgrade Docker Engine)." >&2
  exit 1
fi
if ! docker ps >/dev/null 2>&1; then
  echo "ERROR: cannot run 'docker ps' as user '$(whoami)'." >&2
  echo "       Either:" >&2
  echo "         - add this user to the docker group:" >&2
  echo "             sudo usermod -aG docker \$USER" >&2
  echo "           and then start a new login shell, or" >&2
  echo "         - re-run this script with sudo." >&2
  exit 1
fi

echo "==> Building image and starting container..."
docker compose up -d --build

echo "==> Container status:"
docker compose ps

PUBLIC_IP="$(curl -fsS --max-time 5 https://api.ipify.org 2>/dev/null || echo '<vm-public-ip>')"
cat <<EOF

Done.

Web UI:   http://${PUBLIC_IP}:8501

If you can't reach it, you still need to:
  1. Open the port in firewalld:
       sudo firewall-cmd --add-port=8501/tcp --permanent
       sudo firewall-cmd --reload
  2. Open the port in the Azure NSG (inbound rule, source = your IP, dest port 8501).

Batch CLI (PDFs go in ./data on the host, reports come back to the same folder):
       docker compose exec app python detector_batch.py /data/<your-folder>

EOF
