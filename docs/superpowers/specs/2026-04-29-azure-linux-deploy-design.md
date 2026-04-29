# Azure Linux (RHEL 9) Docker Deployment

**Date:** 2026-04-29
**Branch:** `azure-linux-deploy`
**Status:** Design — awaiting user approval before implementation plan

## Goal

Make the existing PDF DataMatrix/QR detector (Streamlit web UI + CLI) installable and runnable on an Azure VM running RHEL 9.7, using Docker. The host already has Docker Engine and the Compose plugin running. No changes to Windows/macOS workflows.

## Out of scope

- HTTPS / TLS termination (HTTP only on port 8501).
- Authentication or multi-tenancy (the deployment is intended for a trusted network).
- CI image publishing to a registry — the image is built locally on the VM via `docker compose build`.
- Persisting Streamlit-uploaded PDFs across container restarts (web uploads remain ephemeral; persistence is only provided for the batch CLI workflow).
- Switching `opencv-python` to `opencv-python-headless`. We install the minimum GUI runtime libs (`libgl1`, `libglib2.0-0`) inside the image instead, so the existing `requirements.txt` keeps working unchanged on Windows/macOS.

## Requirements

1. **Web UI** reachable at `http://<vm-ip>:8501` from outside the VM.
2. **Batch CLI** runnable against host-side folders without copying files into the container.
3. **Survives reboots** — the container restarts automatically.
4. **One-shot install** — a script that gets a fresh-cloned repo to a running container in a single command, plus documented manual steps that do the same.
5. **Doesn't disturb the existing Docker workload** on the VM (no port conflicts, no name collisions, no shared networks).

## Architecture

A single Docker image based on `python:3.12-slim`, run as a single Compose service. The host's `./data` directory is bind-mounted into the container at `/data`, giving the batch CLI a stable path that is also accessible from the host.

```
┌──────────────────────────── Azure VM (RHEL 9.7) ────────────────────────────┐
│                                                                              │
│  ┌──────────────────────────────────────┐    other existing                 │
│  │  Container: datamatrix-detector      │    Docker container(s)            │
│  │  - python:3.12-slim                  │    (untouched)                    │
│  │  - libdmtx0b, libzbar0               │                                   │
│  │  - libgl1, libglib2.0-0              │                                   │
│  │  - app code, requirements.txt        │                                   │
│  │  - streamlit on 0.0.0.0:8501         │                                   │
│  │           │                          │                                   │
│  │           │  bind mount              │                                   │
│  │           ▼                          │                                   │
│  │     /data  ◄──►  ./data on host      │                                   │
│  └────────┬─────────────────────────────┘                                   │
│           │ port 8501 ► host 8501                                            │
│  ─────────┴───────────────────────────────────────────────────              │
│           firewalld (port 8501/tcp open)                                     │
└────────────────────────────┬─────────────────────────────────────────────────┘
                             │
                Azure NSG (inbound 8501/tcp from your IP)
                             │
                          End user
```

## File-by-file design

### `Dockerfile` (new)

```dockerfile
FROM python:3.12-slim

# System libs:
#   libdmtx0b     -> required by pylibdmtx (DataMatrix decode)
#   libzbar0      -> required by pyzbar (QR decode)
#   libgl1, libglib2.0-0 -> opencv-python runtime deps on a server (no GUI)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libdmtx0b \
        libzbar0 \
        libgl1 \
        libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code (.dockerignore excludes venv, __pycache__, *.pdf, .git)
COPY . .

EXPOSE 8501

# Bind to 0.0.0.0 so the port is reachable from the host
CMD ["python", "-m", "streamlit", "run", "app_web.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
```

Notes:
- `python:3.12-slim` matches the `requirements.txt` "Python 3.12+ compatibility" comment.
- `--server.headless=true` prevents Streamlit from trying to open a browser at startup and from prompting for an email on first run.

### `docker-compose.yml` (new)

```yaml
services:
  app:
    build: .
    image: datamatrix-detector:latest
    container_name: datamatrix-detector
    ports:
      - "8501:8501"
    volumes:
      - ./data:/data
    restart: unless-stopped
```

Notes:
- `container_name` is set explicitly to avoid collision with other containers on the host.
- Default Compose project network is fine; no `networks:` block needed.
- `restart: unless-stopped` means the container comes back after `reboot` and after `docker daemon` restarts, but stays down if the user explicitly `docker compose stop`s it.

### `.dockerignore` (new)

```
.git
.gitignore
__pycache__
*.pyc
venv
.venv
*.pdf
docs
debug_corners
qr_detection_results
*.log
.DS_Store
start_web.bat
start_web.cmd
start_web.ps1
```

PDFs are excluded so the 19 MB sample PDF doesn't bloat every image build. The Windows launcher scripts are excluded because they're not used in the container.

### `data/.gitkeep` (new)

Empty file. Ensures the bind-mount target directory exists immediately after a fresh clone (Docker would otherwise create it as `root`-owned, which causes permission surprises later).

### `scripts/install-rhel.sh` (new)

```bash
#!/usr/bin/env bash
set -euo pipefail

# Install/run the datamatrix-detector container on an Azure RHEL 9 VM
# that already has Docker Engine + Compose plugin.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Verifying Docker is available..."
if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: 'docker' not found. Install Docker Engine first." >&2
  exit 1
fi
if ! docker compose version >/dev/null 2>&1; then
  echo "ERROR: 'docker compose' plugin not found. Install docker-compose-plugin." >&2
  exit 1
fi
if ! docker ps >/dev/null 2>&1; then
  echo "ERROR: cannot run 'docker ps' as user '$(whoami)'." >&2
  echo "       Add the user to the docker group (sudo usermod -aG docker \$USER)" >&2
  echo "       and start a new login shell, or re-run this script with sudo." >&2
  exit 1
fi

echo "==> Building and starting container..."
docker compose up -d --build

echo "==> Container status:"
docker compose ps

PUBLIC_IP="$(curl -fsS https://api.ipify.org 2>/dev/null || echo '<vm-public-ip>')"
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
```

Notes:
- Script is idempotent: re-running rebuilds and restarts.
- It does NOT try to install Docker (per requirement: Docker is already running).
- It does NOT modify firewalld or the NSG — those are out-of-band concerns and the user may have stricter policies. The script prints exactly the commands needed.
- Using `set -euo pipefail` so partial failures stop the script with a non-zero exit code.

### `README.md` (modified)

A new section "Deploy on an Azure RHEL 9 VM" inserted after the existing "macOS / Linux" section. Structure:

1. **Prerequisites** — VM with Docker Engine + Compose plugin, port 8501 free.
2. **Fast path** — `git clone …`, `cd …`, `bash scripts/install-rhel.sh`.
3. **Manual steps** — `docker compose up -d --build`, then the firewalld + NSG commands.
4. **Using the web UI** — `http://<vm-ip>:8501`.
5. **Using the batch CLI** — `docker compose exec app python detector_batch.py /data/<folder>`.
6. **Logs / lifecycle** — `docker compose logs -f`, `docker compose restart`, `docker compose down`.
7. **Security note** — HTTP only, no auth; restrict NSG source to known IPs.

### `.gitignore` (modified)

Add:
```
# Docker bind-mount data (keep .gitkeep)
data/*
!data/.gitkeep
```

So that PDFs and reports placed in `./data` for batch processing don't get accidentally committed.

## Data flow

**Web UI flow** — unchanged from current behavior. Streamlit accepts an upload, writes a tempfile inside the container's tmpfs, runs `analyze_pdf_for_codes`, returns results to the browser. Uploads do not persist after the container is restarted; this is acceptable per the "out of scope" list.

**Batch CLI flow:**
1. User drops PDFs into `./data/jobs/` on the host.
2. User runs `docker compose exec app python detector_batch.py /data/jobs`.
3. `detector_batch.py` writes `*_results.csv` and `*_results.json` next to the inputs at `/data/jobs/`, which is the same directory on the host.
4. User reads/copies the reports from the host filesystem.

## Error handling

- **Build failure** (e.g. apt repo unreachable, pip install fails) → `docker compose up -d --build` exits non-zero; the install script propagates the error via `set -e`.
- **Port already in use** (8501 occupied by something else after all) → Compose surfaces the bind error; user changes the host port mapping in `docker-compose.yml` or stops the conflicting service. Documented in README troubleshooting bullet.
- **Cannot reach web UI from outside VM** → almost always firewalld or NSG. README "Manual steps" section lists both. Install script's final message also lists them.
- **`docker ps` fails for the invoking user** → install script detects this up front and prints the `usermod -aG docker` remediation.

## Testing strategy

The deployment is configuration code, not application code. Verification is operational rather than unit-testable:

1. **Image build** — `docker compose build` completes without errors on the target VM.
2. **Container starts** — `docker compose up -d` and `docker compose ps` shows status `running` with no restart loop.
3. **Web UI smoke test** — `curl -fsS http://localhost:8501/_stcore/health` returns HTTP 200 (Streamlit's built-in health endpoint).
4. **Web UI functional test** — upload one of the existing sample PDFs through the browser, confirm detection results appear.
5. **Batch CLI smoke test** — copy a sample PDF into `./data/`, run `docker compose exec app python detector_batch.py /data`, verify a `*_results.csv` is written next to it.
6. **Reboot test** — `sudo reboot` the VM, confirm the container comes back automatically and the web UI is reachable.
7. **No collision check** — confirm the existing Docker container on the VM is still running and unaffected.

These will be turned into a test plan in the implementation plan that follows.

## Risks and trade-offs

| Risk | Mitigation |
|------|------------|
| HTTP-only on a public VM is exposed to anyone who finds the IP. | NSG source-IP restriction is the primary control. README states this explicitly. |
| Image size (~700 MB-ish with opencv + numpy + libdmtx) | Acceptable; not a constraint. Multi-stage build deferred until proven necessary. |
| Bind-mount permissions (root in container vs. non-root on host) | `data/.gitkeep` ensures the dir exists with the host user's ownership before the container ever touches it. If permission issues still appear in practice, follow-up is a `--user` flag on the service. |
| `requirements.txt` pinning is loose (`>=`); a future upstream release could break the image build. | Out of scope for this change. If it becomes a problem, a follow-up is to `pip freeze` the working set into `requirements.lock.txt` or pin in `requirements.txt`. |

## Branch & commit plan

- Branch: `azure-linux-deploy` (already created from `main` at `766ddda`).
- Commit 1: this design document.
- Commit 2+: implementation, sequenced per the implementation plan that follows brainstorming.
