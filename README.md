# PDF QR Code & DataMatrix Detector

Detect and decode QR codes and DataMatrix codes in PDF documents. Optimized for finding small codes in document corners — ideal for invoice and document quality control workflows.

---

## Quick Start

### Windows — double-click to run

**Download the project, then double-click `start_web.bat`.**

That's it. The script will:
1. Check that Python is installed
2. Create a virtual environment automatically
3. Install all dependencies
4. Open the web interface in your browser

No command prompt needed. If you do need to run it manually:

```cmd
start_web.bat
```

> **Tip**: If you see "Access Denied" / "Toegang geweigerd", right-click `start_web.bat` and choose **Run as administrator**. The script also tries a `--user` install automatically as a fallback.

---

### macOS / Linux

```bash
# Create virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start the web interface
python -m streamlit run app_web.py
```

Or use the included setup script (macOS with Homebrew):

```bash
bash quick_setup.sh
```

---

### Command Line (all platforms)

```bash
# Activate your virtual environment first, then:
python detector.py document.pdf

# Fast check — detect presence only, no decoding
python detector.py document.pdf --detect-only

# Process multiple PDFs in a folder
python detector_batch.py /path/to/folder
```

See [Batch Processing](#batch-processing) and [CLI Reference](#cli-reference) for details.

---

## Features

- **Web interface** — drag-and-drop PDF upload, interactive results table, downloadable reports
- **Dual detection** — `pylibdmtx` for DataMatrix, `pyzbar` for QR codes, OpenCV as fallback
- **Smart page skipping** — automatically skips even pages and blank pages (configurable)
- **Corner-focused** — scans the top-right corner region (size configurable)
- **Two modes** — full decode (reads values) or detection-only (faster, presence check)
- **Advanced preprocessing** — 8 image enhancement techniques × 4 scales for difficult codes
- **Batch processing** — process entire folders, generates CSV + JSON reports
- **Debug tools** — extract corner images from specific pages for inspection

---

## Installation

### Requirements

- **Python** 3.7 – 3.13
- **System libraries** — see per-platform instructions below

> **Python 3.12+ note**: `pylibdmtx` internally uses `distutils`, which was removed in Python 3.12. Installing `setuptools` (already in `requirements.txt`) provides the backport. If you still hit issues, run `python fix_quick.py`.

### Windows

`start_web.bat` handles everything automatically. For a manual install:

```cmd
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
```

If pip gives "Access Denied" / "Toegang geweigerd":

| Option | Command |
|--------|---------|
| Install for current user (no admin) | `python -m pip install --user -r requirements.txt` |
| Run as Administrator | Right-click CMD → *Run as administrator* |
| Reinstall Python | Re-run installer, choose **"Install for current user"** |

If you get **DLL errors** after install:
- [Visual C++ 2015-2022 Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) — fixes most `pylibdmtx` / `pyzbar` DLL issues
- [Visual C++ 2013 Redistributable](https://aka.ms/highdpimfc2013x64enu) — fixes `libzbar-64.dll` missing errors

### macOS

```bash
brew install libdmtx zbar
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Apple Silicon (M1/M2/M3)**:
```bash
export PATH="/opt/homebrew/bin:$PATH"
export DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH"
```

### Ubuntu / Debian

```bash
sudo apt-get install libdmtx0b libzbar0
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Web Interface

The web interface (`app_web.py`) is the easiest way to use the tool.

**Start it:**

| Platform | Command |
|----------|---------|
| Windows | Double-click `start_web.bat` |
| macOS/Linux | `python -m streamlit run app_web.py` |

**Opens at** `http://localhost:8501`

**Features:**
- Drag-and-drop PDF upload
- Configurable settings: DPI, corner ratio, detection mode
- Interactive decoded values table
- Download reports as JSON, CSV, or TXT

---

## CLI Reference

### Single PDF

```bash
python detector.py <pdf_file> [dpi] [corner_ratio] [options]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `pdf_file` | required | Path to PDF |
| `dpi` | 300 | Render resolution (300–600 recommended) |
| `corner_ratio` | 0.2 | Corner scan area as fraction of page (0.1–0.3) |

| Flag | Description |
|------|-------------|
| `--detect-only` | Check presence only — no decoding, ~3× faster |
| `--skip-white` | Process all odd pages, disable blank-page detection |
| `--debug` | Show preprocessing detail and save debug images |
| `--extract-corners P1,P2` | Save corner images for specific pages |

**Common recipes:**

```bash
# Standard analysis
python detector.py invoice.pdf

# Small or difficult codes
python detector.py invoice.pdf 500 0.15 --skip-white

# Fast validation — just check if codes are present
python detector.py invoice.pdf --detect-only --skip-white

# Debug a specific problem page
python detector.py invoice.pdf --extract-corners 377,379
python detector.py invoice.pdf 600 0.1 --debug
```

**DPI guide:**

| DPI | Use case |
|-----|----------|
| 300 | Standard codes (default) |
| 400 | Small codes (< 1 cm²) |
| 500–600 | Very small DataMatrix codes |

### Understanding the output

```
Page  375: ✓ DATAMATRIX Found! Value: [83065676] (pylibdmtx_inverted_scale_1.5)
Page  376: Skipped (even page)
Page  377: ✗ NO CODE FOUND - Missing DataMatrix/QR!
Page  389: Skipped (white/blank page)
```

Summary at the end:

```
Pages WITH codes:    238  (DataMatrix: 230, QR: 8)
Pages WITHOUT codes:   2  → [377, 379]
```

---

## Batch Processing

Process all PDFs in a folder:

```bash
python detector_batch.py /path/to/folder

# Detection-only (fastest)
python detector_batch.py /path/to/folder --detect-only

# Include subfolders
python detector_batch.py /path/to/folder --recursive
```

Generates in the output folder:
- `batch_summary_<timestamp>.csv` — one row per PDF, Excel-compatible
- `batch_results_<timestamp>.json` — full detail
- `missing_codes_report_<timestamp>.txt` — pages without codes

See [BATCH_README.md](BATCH_README.md) for the full reference.

---

## Performance Guide

| Mode | Flags | Speed | Use case |
|------|-------|-------|----------|
| Fastest | `--detect-only --skip-white` | ~0.3 s/page | Quick validation |
| Fast | `--detect-only` | ~0.5 s/page | Find missing codes |
| Standard | *(default)* | ~2 s/page | Full analysis |
| Thorough | `600 --debug` | ~4 s/page | Difficult/small codes |

**Recommended workflow for large batches:**

```bash
# 1. Fast pass to identify problem files
python detector_batch.py folder --detect-only

# 2. Re-scan problem files at higher quality
python detector_batch.py problem_files --dpi 500 --skip-white
```

---

## Troubleshooting

### Codes not detected

| Symptom | Fix |
|---------|-----|
| Known-good pages fail | Try `--skip-white` (page wrongly flagged as blank) |
| Detection unreliable | Increase DPI: `500` or `600` |
| Code is far from corner | Increase corner ratio: `0.3` |
| Code is very close to corner | Decrease corner ratio: `0.1` |
| Only some codes found | Run `--debug` to see which method succeeds |

```bash
# Verify libraries are installed
python -c "import pylibdmtx; print('pylibdmtx OK')"
python -c "import pyzbar; print('pyzbar OK')"
python -c "import fitz; print('PyMuPDF OK')"
python -c "import cv2; print('OpenCV OK')"
```

### Common errors

**"No detection libraries available"**
```bash
pip install pylibdmtx pyzbar
```

**"ImportError: libdmtx.so.0: cannot open shared object file"** (Linux/macOS)
```bash
sudo apt-get install libdmtx0b   # Ubuntu/Debian
brew install libdmtx              # macOS
```

**"distutils" / "ModuleNotFoundError" on Python 3.12+**
```bash
pip install setuptools
# or run the included fix script:
python fix_quick.py
```

**Codes in a different corner**

Edit the `detect_codes_in_top_right_corner` function in `detector.py`:
```python
start_x = w - corner_width  # top-right (default)
start_x = 0                 # top-left
```

---

## Technical Details

### Detection pipeline

1. PDF page → rasterised image (configurable DPI)
2. Extract corner region (configurable size)
3. Apply 8 preprocessing variants: grayscale, inverted, binary threshold, inverted binary, Otsu, adaptive, CLAHE, sharpening
4. Test each variant at 4 scales: 1.0×, 1.5×, 2.0×, 3.0×
5. Try each result with pylibdmtx → pyzbar → OpenCV; return on first hit

### Library comparison

| Library | QR | DataMatrix | Speed |
|---------|----|------------|-------|
| pylibdmtx | — | best | medium |
| pyzbar | best | good | fast |
| OpenCV | good | — | fast |

### File structure

```
project/
├── app_web.py           # Streamlit web interface
├── detector.py          # Single-PDF detector (CLI)
├── detector_batch.py    # Batch processor (CLI)
├── fix_quick.py         # One-command Python 3.13 fix
├── fix_python313.py     # Detailed Python 3.13 fix script
├── requirements.txt     # Python dependencies
├── start_web.bat        # Windows one-click launcher
├── quick_setup.sh       # macOS/Linux setup script
├── README.md            # This file
├── BATCH_README.md      # Batch processing reference
└── debug_corners/       # Created by --extract-corners
```

---

**Version**: 2.2.0 | **Python**: 3.7 – 3.13 | **Platform**: Windows, macOS, Linux
