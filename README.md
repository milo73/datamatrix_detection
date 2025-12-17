# PDF QR Code & DataMatrix Detector

A powerful Python script for detecting and decoding QR codes and DataMatrix codes in PDF documents, optimized for finding small codes in document corners.

## 📋 Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Command Line Options](#command-line-options)
- [Understanding the Output](#understanding-the-output)
- [Performance Guide](#performance-guide)
- [Troubleshooting](#troubleshooting)
- [Common Issues & Solutions](#common-issues--solutions)
- [Examples](#examples)
- [Technical Details](#technical-details)

## ✨ Features

- **Dual Detection Libraries**: Uses `pylibdmtx` for superior DataMatrix detection and `pyzbar` for QR codes
- **Smart Page Processing**: Automatically skips even pages and blank pages (configurable)
- **Corner-Focused Detection**: Optimized for codes in top-right corners (configurable region size)
- **Multiple Detection Modes**: 
  - Full decode mode (reads code content)
  - Detection-only mode (faster, verifies presence only)
- **Advanced Preprocessing**: 8 different image preprocessing techniques for difficult codes
- **Multi-Scale Detection**: Tests multiple image scales (1.0x, 1.5x, 2.0x, 3.0x)
- **Debug Features**: Extract and save corner regions for manual inspection
- **Comprehensive Reporting**: Detailed analysis with decoded values and statistics

## 📦 Requirements

### Python Version
- Python 3.7, 3.8, 3.9, 3.10, 3.11

### Python Libraries
```bash
pip install pylibdmtx pyzbar PyMuPDF opencv-python numpy
```

### System Dependencies

#### Ubuntu/Debian
```bash
# For pylibdmtx (DataMatrix detection)
sudo apt-get update
sudo apt-get install libdmtx0b

# For pyzbar (QR code detection)
sudo apt-get install libzbar0
```

#### macOS
```bash
# Using Homebrew (install Homebrew first from https://brew.sh if needed)
brew install libdmtx
brew install zbar

# Verify installation
brew list libdmtx
brew list zbar
```

**macOS Troubleshooting**:
- If `brew install` fails, try `brew update && brew upgrade` first
- For M1/M2 Macs, Homebrew installs to `/opt/homebrew/`. You may need to add to PATH:
  ```bash
  export PATH="/opt/homebrew/bin:$PATH"
  ```
- If Python can't find the libraries, try:
  ```bash
  export DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH"
  ```

#### Windows

**Option 1: Using pip (Recommended)**
```cmd
# Install Python libraries (includes pre-built binaries for Windows)
pip install pylibdmtx pyzbar PyMuPDF opencv-python numpy
```

**Option 2: Using Conda/Anaconda**
```cmd
conda install -c conda-forge pylibdmtx pyzbar
pip install PyMuPDF opencv-python
```

**Windows Troubleshooting**:

1. **If pylibdmtx fails to import**:
   - The pip package includes pre-built DLLs for Windows
   - If you get "DLL not found" errors, install Visual C++ Redistributable:
     - [Visual C++ 2015-2022 Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)

2. **If pyzbar fails to import**:
   - pyzbar requires `libzbar-64.dll` which is bundled with the pip package
   - If missing, install Visual C++ 2013 Redistributable:
     - [Visual C++ 2013 Redistributable](https://aka.ms/highdpimfc2013x64enu)

3. **For 32-bit Python on Windows**:
   ```cmd
   pip install pylibdmtx[win32]
   ```

4. **Manual DLL installation** (if automatic fails):
   - Download `libdmtx.dll` from [libdmtx releases](https://github.com/dmtx/libdmtx/releases)
   - Place in your Python's `Scripts` folder or add to system PATH

## 🚀 Quick Start

### Option 1: Web Interface (Recommended for Desktop Use)

```bash
# Run the web interface
streamlit run app_web.py

# Opens automatically in browser at http://localhost:8501
# Features:
# - Drag and drop PDF files
# - Interactive results table
# - Configurable settings (DPI, corner ratio, modes)
# - Download reports (JSON, CSV, TXT)
# - View decoded values easily
```

### Option 2: Command Line Interface

```bash
# Analyze a PDF with default settings
python detector.py document.pdf

# Fast detection-only mode (doesn't decode content)
python detector.py document.pdf --detect-only

# Process all odd pages (skip white page detection)
python detector.py document.pdf --skip-white
```

### For Problem PDFs

```bash
# Maximum detection capability
python detector.py document.pdf 500 0.15 --skip-white --debug
```

## 📘 Usage

### Basic Syntax
```bash
python detector.py <pdf_file> [dpi] [corner_ratio] [options]
```

### Positional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `pdf_file` | Required | Path to the PDF file to analyze |
| `dpi` | 300 | Resolution for page extraction (300-600 recommended) |
| `corner_ratio` | 0.2 | Size of corner region to scan (0.1-0.3 = 10%-30% of page) |

### Optional Flags

| Flag | Description |
|------|-------------|
| `--detect-only` | Fast mode - only detects presence, doesn't decode content |
| `--skip-white` | Process ALL odd pages (disables blank page detection) |
| `--debug` | Show detailed debugging information |
| `--extract-corners P1,P2` | Extract and save corner images from specific pages |

## 🎯 Command Line Options

### Detection Modes

#### 1. Full Decode Mode (Default)
Detects and decodes the content of all codes.
```bash
python detector.py invoice.pdf
```
**Use when**: You need to read the actual values in the codes

#### 2. Detection-Only Mode
Faster scanning that only verifies code presence.
```bash
python detector.py invoice.pdf --detect-only
```
**Use when**: You only need to find which pages are missing codes

### Resolution Settings

#### DPI (Dots Per Inch)
Higher DPI improves detection of small codes but increases processing time.

```bash
# Standard (default)
python detector.py file.pdf 300

# High quality for small codes
python detector.py file.pdf 400

# Maximum quality for very small DataMatrix
python detector.py file.pdf 600
```

**Recommendations**:
- 300 DPI: Standard documents with normal-sized codes
- 400 DPI: Small codes (< 1cm²)
- 500-600 DPI: Very small DataMatrix codes

#### Corner Size Ratio
Defines what percentage of the page to scan in the top-right corner.

```bash
# Scan 20% of page (default)
python detector.py file.pdf 300 0.2

# Scan smaller region (10%) for codes very close to corner
python detector.py file.pdf 300 0.1

# Scan larger region (30%) for codes further from corner
python detector.py file.pdf 300 0.3
```

### Page Processing Options

#### Skip White Page Detection
```bash
python detector.py file.pdf --skip-white
```
**Use when**: Pages with small codes are incorrectly detected as blank

#### Debug Mode
```bash
python detector.py file.pdf --debug
```
Shows:
- Detailed detection attempts
- Which preprocessing method succeeded
- Saved debug images location
- Extended error messages

#### Extract Corners for Analysis
```bash
# Extract corners from specific pages
python detector.py file.pdf --extract-corners 377,379,381

# This creates a debug_corners/ folder with:
# - page_377_corner_original.png
# - page_377_corner_enhanced.png
# - page_377_corner_binary.png
# - page_377_corner_inverted.png
```

## 📊 Understanding the Output

### During Processing

#### Success - Code Found and Decoded
```
Page  375: ✓ DATAMATRIX Found! Value: [83065676] (pylibdmtx_inverted_scale_1.5) - Valid 8-digit number
```
- ✓ = Success
- Code type: DATAMATRIX or QRCODE
- Decoded value in brackets
- Detection method used
- Content validation

#### Success - Detection Only
```
Page  375: ✓ DATAMATRIX Detected (not decoded) (pylibdmtx_original_scale_1.0)
```

#### Failure - No Code Found
```
Page  377: ✗ NO CODE FOUND - Missing DataMatrix/QR!
```

#### Skipped Pages
```
Page  376: Skipped (even page)
Page  389: Skipped (white/blank page)
```

### Summary Report

```
==================================================
ANALYSIS SUMMARY
==================================================
PDF File: invoice.pdf
Total pages: 500
Settings: DPI=300, Corner=20%, Mode=Decode

Page Statistics:
  • Even pages (skipped): 250 pages
  • White/blank pages (skipped): 10 pages
  • Odd content pages (checked): 240 pages
  • Pages WITH codes: 238 pages
    - DataMatrix codes: 230 pages
    - QR codes: 8 pages
  • Pages WITHOUT codes: 2 pages

DECODED VALUES:
--------------------------------------------------
  Page    1 (DataMatrix ): 83065676
  Page    3 (DataMatrix ): 83065677
  Page    5 (QR        ): https://example.com/verify
  ...

⚠️  MISSING CODES - PAGES WITHOUT DETECTION:
--------------------------------------------------
  Pages: [377, 379]
  Total: 2 pages missing codes
```

## ⚡ Performance Guide

### Speed Comparison

| Mode | Options | Speed | Use Case |
|------|---------|-------|----------|
| Fastest | `--detect-only --skip-white` | ~0.3 sec/page | Quick validation |
| Fast | `--detect-only` | ~0.5 sec/page | Finding missing codes |
| Standard | (default) | ~2 sec/page | Full analysis |
| Thorough | `500 DPI --debug` | ~4 sec/page | Difficult codes |

### Performance Tips

1. **For Initial Scanning**:
   ```bash
   # Fast scan to identify problem pages
   python detector.py file.pdf --detect-only --skip-white
   ```

2. **For Problem Pages**:
   ```bash
   # Intensive scan on specific extracted corners
   python detector.py file.pdf --extract-corners 377,379
   ```

3. **For Production Use**:
   ```bash
   # Balance of speed and accuracy
   python detector.py file.pdf 400 0.15 --skip-white
   ```

## 🔧 Troubleshooting

### Codes Not Being Detected

#### 1. Install pylibdmtx for DataMatrix
```bash
# Check if installed
python -c "import pylibdmtx; print('pylibdmtx installed')"

# If not installed
pip install pylibdmtx
```

#### 2. Increase DPI
```bash
# Try progressively higher DPI
python detector.py file.pdf 400
python detector.py file.pdf 500
python detector.py file.pdf 600
```

#### 3. Adjust Corner Size
```bash
# Smaller region if code is very close to corner
python detector.py file.pdf 400 0.1

# Larger region if code is further from corner
python detector.py file.pdf 400 0.3
```

#### 4. Skip White Page Detection
```bash
# If pages are incorrectly marked as blank
python detector.py file.pdf --skip-white
```

#### 5. Debug Specific Pages
```bash
# Extract and examine problem pages
python detector.py file.pdf --extract-corners 377,379

# Check the debug_corners/ folder
ls debug_corners/
```

### Pages Incorrectly Marked as Blank

**Problem**: Pages containing small DataMatrix codes are skipped as "white/blank"

**Solution**:
```bash
python detector.py file.pdf --skip-white
```

### Detection Works but Decoding Fails

**Problem**: Script detects codes but shows "decode failed"

**Solutions**:
1. Increase DPI for better image quality
2. Try different preprocessing (automatic with pylibdmtx)
3. Check if code is damaged or partially obscured

## 🐛 Common Issues & Solutions

### Issue 1: "No detection libraries available"

**Solution**: Install at least one detection library
```bash
pip install pylibdmtx pyzbar
```

### Issue 2: "ImportError: libdmtx.so.0: cannot open shared object file"

**Solution**: Install system library
```bash
# Ubuntu/Debian
sudo apt-get install libdmtx0b

# macOS
brew install libdmtx
```

### Issue 3: Very slow processing

**Solutions**:
```bash
# Use detection-only mode
python detector.py file.pdf --detect-only

# Process specific page range by extracting PDF pages first
# Using pymupdf or other PDF tools
```

### Issue 4: Codes in wrong corner

**Solution**: Modify the script's `detect_codes_in_top_right_corner` function or scan full page:
```python
# Change this line in the script:
start_x = w - corner_width  # Top-right
# To:
start_x = 0  # Top-left
```

## 📚 Examples

### Example 1: Standard Invoice Processing
```bash
# Typical invoice with QR codes in top-right
python detector.py invoice.pdf
```

### Example 2: Document with Small DataMatrix Codes
```bash
# Small DataMatrix codes requiring high DPI
python detector.py document.pdf 500 0.15 --skip-white
```

### Example 3: Quick Validation Check
```bash
# Fast check to find missing codes
python detector.py batch_scan.pdf --detect-only
```

### Example 4: Debugging Specific Pages
```bash
# When pages 377 and 379 are reported as missing codes
python detector.py document.pdf --extract-corners 377,379

# Then try with higher DPI
python detector.py document.pdf 600 0.1 --debug
```

### Example 5: Processing Multiple PDFs
```bash
#!/bin/bash
# Batch processing script
for pdf in *.pdf; do
    echo "Processing $pdf..."
    python detector.py "$pdf" --detect-only --skip-white
done
```

## 🔬 Technical Details

### Detection Pipeline

1. **Page Extraction**: PDF page → Image (configurable DPI)
2. **Region Selection**: Extract top-right corner (configurable size)
3. **Preprocessing**: Apply 8 different techniques:
   - Original grayscale
   - Inverted (negative)
   - Binary threshold (127)
   - Binary threshold inverted
   - Otsu's threshold
   - Adaptive threshold
   - CLAHE enhancement
   - Sharpening filter
4. **Multi-Scale Testing**: Each preprocessed image at 1.0x, 1.5x, 2.0x, 3.0x
5. **Detection Attempts**:
   - pylibdmtx (DataMatrix specialist)
   - pyzbar (QR and DataMatrix)
   - OpenCV (QR only)

### Supported Code Types

| Library | QR Codes | DataMatrix | Other 2D | Speed |
|---------|----------|------------|----------|-------|
| pylibdmtx | ❌ | ✅✅✅ | ❌ | Medium |
| pyzbar | ✅✅✅ | ✅✅ | ✅ | Fast |
| OpenCV | ✅✅ | ❌ | ❌ | Fast |

### File Structure

```
project/
├── app_web.py                  # Streamlit web interface
├── detector.py                 # Main single-file detector
├── detector_batch.py           # Batch processing for multiple PDFs
├── decoder_optimized.py        # Performance-optimized decoder functions
├── integration_examples.py     # Integration examples and benchmarks
├── test_image_generator.py     # Create test images for benchmarking
├── fix_python313.py            # Python 3.13 compatibility fix
├── fix_quick.py                # Quick installation fix script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── BATCH_README.md             # Batch processing documentation
├── CLAUDE.md                   # AI assistant project documentation
├── debug_corners/              # Created when using --extract-corners
│   ├── page_377_corner_original.png
│   ├── page_377_corner_enhanced.png
│   ├── page_377_corner_binary.png
│   └── page_377_corner_inverted.png
└── test_pdfs/                  # Your PDF files
    └── document.pdf
```

## 📄 License

This script is provided as-is for document processing and quality control purposes.

## 🤝 Contributing

Improvements and bug fixes are welcome. Key areas for enhancement:
- Support for other corner positions
- Batch processing improvements
- GUI interface
- Additional barcode formats
- Performance optimizations

## 📮 Support

For issues, questions, or suggestions:
1. Check the troubleshooting section
2. Review common issues
3. Test with --debug flag
4. Extract problem pages for analysis

---

**Version**: 2.0.0  
**Last Updated**: 2024  
**Requirements**: Python 3.7+, pylibdmtx, pyzbar, PyMuPDF, OpenCV