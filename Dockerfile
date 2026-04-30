FROM python:3.12-slim-bookworm

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
