# Azure Linux (RHEL 9) Docker Deployment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing Streamlit + CLI app installable and runnable as a Docker container on an Azure RHEL 9.7 VM that already has Docker Engine + Compose plugin running.

**Architecture:** Single `python:3.12-slim` image, single Compose service named `datamatrix-detector` listening on host port 8501, with a `./data` host bind-mount at `/data` for batch CLI input/output. A RHEL bootstrap script verifies Docker availability and runs `docker compose up -d --build`. Windows/macOS workflows untouched.

**Tech Stack:** Docker, docker-compose, Bash, Python 3.12, Streamlit, libdmtx, zbar, OpenCV.

**Spec:** [`docs/superpowers/specs/2026-04-29-azure-linux-deploy-design.md`](../specs/2026-04-29-azure-linux-deploy-design.md)

**Branch:** `azure-linux-deploy` (already created from `main` at `766ddda`; commit `f2174d4` contains the design doc).

---

## Pre-flight notes for the implementer

This plan is **infrastructure / configuration code, not application code.** There is no `pytest` test suite to extend. Verification is operational: build the image, run the container, hit the health endpoint, exec the batch CLI. Each task includes the exact commands to run and the expected output.

**Local development environment:**
- macOS arm64 with Docker Desktop. The verification builds and runs locally for fast feedback.
- The local image will be `linux/arm64`. The Azure VM is `linux/amd64`. This is fine: the VM rebuilds its own image (`docker compose up -d --build` in `install-rhel.sh`), and `python:3.12-slim` plus all apt packages are multi-arch.
- Before starting, **make sure Docker Desktop is running** (`docker ps` should not error). If it isn't, open Docker Desktop and wait for the whale to settle.

**Existing files this plan reads but does not modify:**
- `requirements.txt` — already has all Python deps the image needs.
- `app_web.py` — Streamlit entry point (unchanged).
- `detector.py`, `detector_batch.py` — core code (unchanged).

**Existing files this plan modifies:**
- `README.md` — adds one new section.
- `.gitignore` — adds two lines.

---

## Task 1: Create `.dockerignore`

**Files:**
- Create: `.dockerignore`

A clean build context keeps the image small and reproducible. Without `.dockerignore`, the 19 MB sample PDF, the local `venv/`, and the `.git` directory all get sent to the Docker daemon on every build.

- [ ] **Step 1: Create `.dockerignore`**

Write the file `/Users/milovandiest/datamatrix_detection/datamatrix_detection/.dockerignore` with this content:

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

- [ ] **Step 2: Verify the file was written**

Run: `cat .dockerignore | wc -l`
Expected: `14`

- [ ] **Step 3: Commit**

```bash
git add .dockerignore
git commit -m "build: add .dockerignore for slim Docker build context"
```

---

## Task 2: Create the `Dockerfile`

**Files:**
- Create: `Dockerfile`

The image is `python:3.12-slim` (matches `requirements.txt` "Python 3.12+" requirement) plus the four apt packages needed for `pylibdmtx` (`libdmtx0b`), `pyzbar` (`libzbar0`), and headless OpenCV (`libgl1`, `libglib2.0-0`). Streamlit binds to `0.0.0.0:8501` so the port is reachable through the Compose port mapping.

- [ ] **Step 1: Create `Dockerfile`**

Write the file `/Users/milovandiest/datamatrix_detection/datamatrix_detection/Dockerfile` with this content:

```dockerfile
FROM python:3.12-slim

# System libs:
#   libdmtx0b     -> required by pylibdmtx (DataMatrix decode)
#   libzbar0      -> required by pyzbar (QR decode)
#   libgl1, libglib2.0-0 -> opencv-python runtime deps on a headless server
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libdmtx0b \
        libzbar0 \
        libgl1 \
        libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so this layer caches across code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code (.dockerignore excludes venv, __pycache__, *.pdf, .git, docs, ...)
COPY . .

EXPOSE 8501

# --server.address=0.0.0.0  -> reachable from outside the container
# --server.headless=true    -> no email prompt on first run, no browser auto-open
CMD ["python", "-m", "streamlit", "run", "app_web.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
```

- [ ] **Step 2: Build the image**

Run: `docker build -t datamatrix-detector:test .`

Expected: Build completes with a final line like `Successfully tagged datamatrix-detector:test` (or `naming to docker.io/library/datamatrix-detector:test` on newer Docker). First build takes 3–8 minutes (apt + pip downloads). Subsequent rebuilds are seconds because of layer caching.

If the build fails on `pip install`, look for the exact missing package — most likely a system lib (`libdmtx0b` / `libzbar0`) was misspelled in the apt step.

- [ ] **Step 3: Verify the image exists**

Run: `docker image ls datamatrix-detector:test --format '{{.Repository}}:{{.Tag}} {{.Size}}'`
Expected: one line, e.g. `datamatrix-detector:test 720MB` (size may be 600 MB–900 MB depending on platform).

- [ ] **Step 4: Smoke-test imports inside the image**

Run:
```bash
docker run --rm datamatrix-detector:test python -c "import pylibdmtx.pylibdmtx; import pyzbar.pyzbar; import cv2; import streamlit; from detector import analyze_pdf_for_codes; print('ok')"
```
Expected: prints `ok` and exits 0. If `cv2` errors with `libGL.so.1: cannot open shared object file`, the `libgl1` apt step is wrong.

- [ ] **Step 5: Commit**

```bash
git add Dockerfile
git commit -m "build: add Dockerfile based on python:3.12-slim"
```

---

## Task 3: Add the `data/` bind-mount target and update `.gitignore`

**Files:**
- Create: `data/.gitkeep`
- Modify: `.gitignore`

The bind-mount target needs to exist on the host before `docker compose up` runs, otherwise Docker creates `./data` as `root`-owned, which causes permission surprises later. The `.gitkeep` placeholder ensures the directory survives a fresh clone. The `.gitignore` update keeps user-dropped PDFs and reports out of commits.

- [ ] **Step 1: Create the data directory and placeholder**

Run: `mkdir -p data && touch data/.gitkeep`

- [ ] **Step 2: Append to `.gitignore`**

Edit `.gitignore`. Find the existing `# PDF test files` block at the top:
```
# PDF test files
*.pdf
```

After the existing `# OS` block at the bottom (the last two lines are `.DS_Store` and `Thumbs.db`), append:

```

# Docker bind-mount data (keep .gitkeep)
data/*
!data/.gitkeep
```

(Note the leading blank line for readability.)

- [ ] **Step 3: Verify the .gitkeep is tracked but other data files are ignored**

Run: `touch data/dummy.pdf && git status --short data/`

Expected output (exact):
```
A  data/.gitkeep
```

The `dummy.pdf` should NOT appear because it matches both `*.pdf` and `data/*`.

Then clean up: `rm data/dummy.pdf`

- [ ] **Step 4: Commit**

```bash
git add .gitignore data/.gitkeep
git commit -m "build: add ./data bind-mount target with gitignore exclusion"
```

---

## Task 4: Create `docker-compose.yml` and verify the running stack

**Files:**
- Create: `docker-compose.yml`

Single service, explicit container name (so it can't collide with the other container on the VM), host port 8501, the `./data` bind-mount, and `restart: unless-stopped` for reboot recovery.

- [ ] **Step 1: Create `docker-compose.yml`**

Write the file `/Users/milovandiest/datamatrix_detection/datamatrix_detection/docker-compose.yml` with this content:

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

- [ ] **Step 2: Validate the compose file syntax**

Run: `docker compose config`
Expected: prints the resolved compose config (with `name:` derived from the directory) and exits 0. If it errors, fix the YAML.

- [ ] **Step 3: Bring the stack up (builds the `datamatrix-detector:latest` image)**

Run: `docker compose up -d --build`
Expected: ends with a line like `Container datamatrix-detector  Started`. The build itself reuses cached layers from Task 2 if the source files haven't changed, so it's fast.

- [ ] **Step 4: Confirm the container is running**

Run: `docker compose ps`
Expected: one row, `datamatrix-detector`, status `running` (or `Up <duration>` on older Docker), port mapping `0.0.0.0:8501->8501/tcp`.

- [ ] **Step 5: Hit the Streamlit health endpoint**

Wait ~5 seconds for Streamlit to finish starting, then run:
```bash
curl -fsS http://localhost:8501/_stcore/health
```
Expected: `ok` (single word, no trailing newline). If it errors with "connection refused", run `docker compose logs app | tail -30` and look for Streamlit's startup banner.

- [ ] **Step 6: Verify the bind-mount is wired correctly**

Run: `docker compose exec app sh -c 'ls -la /data && touch /data/__roundtrip && ls -la data/__roundtrip 2>/dev/null || true'`

(That's running inside the container, so the second `ls` checking the host path will fail — expected. The point is `/data` is writable.)

Then on the host: `ls -la data/__roundtrip`
Expected: the file exists on the host, proving the bind-mount round-trips.

Clean up: `rm data/__roundtrip`

- [ ] **Step 7: Tear down**

Run: `docker compose down`
Expected: `Container datamatrix-detector  Removed`.

- [ ] **Step 8: Commit**

```bash
git add docker-compose.yml
git commit -m "build: add docker-compose with bind-mount and restart policy"
```

---

## Task 5: Create the RHEL bootstrap script `scripts/install-rhel.sh`

**Files:**
- Create: `scripts/install-rhel.sh`

The script assumes Docker is already running on the VM (per user's environment). It verifies prerequisites, runs `docker compose up -d --build`, and prints the URL plus the firewalld + Azure NSG commands the user still needs to run out-of-band.

- [ ] **Step 1: Create `scripts/install-rhel.sh`**

Run: `mkdir -p scripts`

Then write the file `/Users/milovandiest/datamatrix_detection/datamatrix_detection/scripts/install-rhel.sh` with this content:

```bash
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
```

- [ ] **Step 2: Make the script executable**

Run: `chmod +x scripts/install-rhel.sh`

- [ ] **Step 3: Bash syntax check (lint without running)**

Run: `bash -n scripts/install-rhel.sh`
Expected: no output, exit code 0.

- [ ] **Step 4: Verify the script exits cleanly when prerequisites are present**

Run: `bash scripts/install-rhel.sh`

Expected: builds (or reuses cached layers), starts the container, prints `==> Container status:` followed by a row, then the "Done." block with a public IP and instructions. Exit code 0.

If the public-IP fetch fails (no internet on test host), the URL will read `http://<vm-public-ip>:8501` — that's the documented fallback, not an error.

- [ ] **Step 5: Tear down**

Run: `docker compose down`

- [ ] **Step 6: Commit**

```bash
git add scripts/install-rhel.sh
git commit -m "build: add RHEL 9 bootstrap script for Azure VM deploy"
```

---

## Task 6: Document the deployment in `README.md`

**Files:**
- Modify: `README.md`

The new section goes after the existing "macOS / Linux" block and before "Command Line (all platforms)". Heading level matches the existing `###` style.

- [ ] **Step 1: Locate the insertion point in README.md**

Run: `grep -n "^### Command Line" README.md`
Expected: a single line, something like `60:### Command Line (all platforms)`. Note the line number.

- [ ] **Step 2: Insert the new section**

Insert the following block **immediately before** the `### Command Line (all platforms)` line, separated from the macOS/Linux section above by a blank line and a `---` divider (the existing pattern in this README).

```markdown
---

### Azure VM (RHEL 9) — Docker

Deploy on an Azure VM running RHEL 9.7 (or compatible). Assumes the VM already has Docker Engine and the Compose plugin installed and running.

**Fast path:**

```bash
git clone https://github.com/milo73/datamatrix_detection.git
cd datamatrix_detection
bash scripts/install-rhel.sh
```

The script verifies Docker is available, builds the image, starts the container with auto-restart on reboot, and prints the URL.

**Manual steps (if you'd rather see what's happening):**

```bash
git clone https://github.com/milo73/datamatrix_detection.git
cd datamatrix_detection
docker compose up -d --build
docker compose ps
```

**Open the port** (the script does NOT do this for you):

```bash
# 1. firewalld on the VM
sudo firewall-cmd --add-port=8501/tcp --permanent
sudo firewall-cmd --reload

# 2. Azure NSG: inbound rule, source = your IP, destination port = 8501/tcp
#    (do this in the Azure portal, or with `az network nsg rule create`)
```

Then open `http://<vm-public-ip>:8501` in a browser.

**Batch CLI on the VM:**

Drop PDFs into the `./data/` folder on the host, then:

```bash
docker compose exec app python detector_batch.py /data/<your-folder>
```

The reports (`*_results.csv`, `*_results.json`) are written next to the PDFs and are visible from the host filesystem.

**Lifecycle:**

```bash
docker compose logs -f         # follow logs
docker compose restart         # restart container
docker compose down            # stop and remove
docker compose up -d --build   # rebuild after pulling new code
```

**Security note:** This deployment is HTTP-only with no authentication. It is intended for trusted networks. Restrict the Azure NSG inbound rule to known source IPs; do not leave port 8501 open to `0.0.0.0/0` long-term.
```

Use the `Edit` tool with the `### Command Line (all platforms)` line plus a couple of preceding lines as `old_string` so the edit is unambiguous, and `new_string` is the new section followed by the original `### Command Line` block.

- [ ] **Step 3: Verify markdown structure**

Run: `grep -n "^### " README.md`
Expected: shows `### Azure VM (RHEL 9) — Docker` listed between `### macOS / Linux` and `### Command Line (all platforms)`.

- [ ] **Step 4: Verify no broken section divider**

Run: `grep -c "^---$" README.md`
Expected: a count one higher than before the change (the new section adds exactly one `---` divider).

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: add Azure RHEL 9 Docker deployment section"
```

---

## Task 7: End-to-end integration verification

**Files:** none modified.

A clean, full-stack run-through that mirrors what a user does on a fresh clone, to catch anything the per-task verifications missed. No commit at the end; this is purely a check.

- [ ] **Step 1: Tear down anything left running**

Run: `docker compose down 2>&1 || true`
Expected: either "no resource found" or a clean `Removed` message.

- [ ] **Step 2: Cold build + run, end to end**

Run: `bash scripts/install-rhel.sh`
Expected: build completes, container starts, "Done." block printed.

- [ ] **Step 3: Confirm web UI health**

Run: `curl -fsS http://localhost:8501/_stcore/health`
Expected: `ok`.

- [ ] **Step 4: Smoke-test the batch CLI on a real PDF**

The repo has `OpGroen-modified.pdf` (315 KB) sitting at the repo root. Copy it into `./data/`:

```bash
mkdir -p data/smoke
cp OpGroen-modified.pdf data/smoke/
docker compose exec app python detector_batch.py /data/smoke
```

Expected: `detector_batch.py` runs without traceback, prints a summary line, and writes `data/smoke/*_results.csv` (and `*_results.json`).

Verify on the host:
```bash
ls data/smoke/
```
Expected output includes both the original PDF and at least one `*_results.csv` file.

- [ ] **Step 5: Clean up the smoke test artifacts**

```bash
rm -rf data/smoke
```

- [ ] **Step 6: Verify auto-restart works (proxy for "survives reboot")**

```bash
docker kill datamatrix-detector
sleep 3
docker compose ps
```
Expected: the row shows `datamatrix-detector` back up (status `running` or `Up <few seconds>`), because of `restart: unless-stopped`.

- [ ] **Step 7: Tear down for good**

```bash
docker compose down
```

- [ ] **Step 8: Update the design doc's testing section if anything diverged**

If any of steps 1–7 surfaced a behavior or fix not captured in `docs/superpowers/specs/2026-04-29-azure-linux-deploy-design.md`, update the spec inline and commit. If nothing diverged, skip this step.

- [ ] **Step 9: Final state check**

Run: `git status`
Expected: clean working tree on `azure-linux-deploy`.

Run: `git log --oneline main..HEAD`
Expected (in order, oldest at bottom):
```
<hash>  docs: add Azure RHEL 9 Docker deployment section
<hash>  build: add RHEL 9 bootstrap script for Azure VM deploy
<hash>  build: add docker-compose with bind-mount and restart policy
<hash>  build: add ./data bind-mount target with gitignore exclusion
<hash>  build: add Dockerfile based on python:3.12-slim
<hash>  build: add .dockerignore for slim Docker build context
<hash>  docs: design for Azure RHEL 9 Docker deployment
```

Six implementation commits + the design commit, totaling seven on top of `main`.

---

## Operational verification on the actual VM (out of band)

The above tasks all happen locally on macOS arm64. The final verification — that the same plan works on the actual `linux/amd64` RHEL 9.7 VM with the existing other Docker container — is operational and not part of this plan's checkboxes. Steps for the user when they're ready:

1. Push the branch and pull on the VM (or `scp` the working tree).
2. From the repo dir on the VM: `bash scripts/install-rhel.sh`.
3. Confirm `docker ps` still shows the *other* existing container alongside `datamatrix-detector`.
4. Hit `http://<vm-ip>:8501` from the user's browser after the firewalld + NSG rules are in place.
5. Reboot the VM (`sudo reboot`); after it comes back, confirm `docker ps` shows `datamatrix-detector` running again.

If any of these surfaces an issue, file it back into the spec and re-run the affected task.
