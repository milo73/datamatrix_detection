# CLAUDE.MD - DataMatrix Detection Project

## Project Overview

This is a **PDF QR Code & DataMatrix Detector** - a Python-based tool for detecting and decoding QR codes and DataMatrix codes in PDF documents. The project is optimized for finding small codes in document corners, particularly for invoice and document processing workflows.

**Primary Use Case**: Automated quality control for document batches to verify presence of tracking codes on odd-numbered pages.

## Key Features

- **Dual Detection Libraries**: Uses `pylibdmtx` (best for DataMatrix) and `pyzbar` (best for QR codes)
- **Smart Processing**: Auto-skips even pages and blank pages
- **Corner-Focused**: Optimized for codes in top-right corners (configurable)
- **Advanced Preprocessing**: 8 different image enhancement techniques
- **Multi-Scale Detection**: Tests at 1.0x, 1.5x, 2.0x, 3.0x scales
- **Batch Processing**: Can process entire folders of PDFs
- **Multiple Modes**: Full decode or detection-only (faster)

## Project Architecture

### Core Components

#### 1. `detector.py` - Main Single-File Detector
**Purpose**: Process a single PDF file for QR/DataMatrix codes

**Key Functions**:
- `is_white_page()`: Detects blank pages to skip
- `detect_codes_in_top_right_corner()`: Main detection logic
- `preprocess_for_detection()`: Image enhancement pipeline
- `check_page_for_code()`: Per-page processing
- `analyze_pdf()`: Main entry point

**Detection Pipeline**:
1. PDF page → Image (configurable DPI)
2. Extract top-right corner region
3. Apply 8 preprocessing techniques (grayscale, inverted, binary, etc.)
4. Test each preprocessed image at 4 scales (1.0x, 1.5x, 2.0x, 3.0x)
5. Attempt detection with pylibdmtx, pyzbar, OpenCV
6. Return first successful detection

#### 2. `detector_batch.py` - Batch Processing
**Purpose**: Process multiple PDF files in a folder

**Key Features**:
- Folder scanning with optional recursion
- Pattern matching for file selection
- Individual JSON reports per PDF
- Consolidated CSV and JSON summaries
- Missing codes report generation
- Progress tracking

#### 3. `decoder_optimized.py` - Performance-Optimized Decoder
**Purpose**: Optimized detection functions for speed-critical applications

**Optimizations**:
- Reduced preprocessing steps
- Faster scale testing
- Early exit on detection

#### 4. `integration_examples.py` - Usage Examples
**Purpose**: Demonstrates integration patterns and benchmarks

**Contains**:
- API integration examples
- Performance benchmarking
- Different usage patterns

#### 5. Utility Scripts

- `test_image_generator.py`: Creates test images with codes for testing
- `fix_python313.py`: Compatibility fixes for Python 3.13
- `fix_quick.py`: Quick installation repair script

## Technical Details

### Detection Strategy

The script uses a **layered fallback approach**:

1. **Primary**: `pylibdmtx` - Best for DataMatrix codes
2. **Secondary**: `pyzbar` - Good for QR codes and DataMatrix
3. **Tertiary**: `OpenCV` - QR codes only

### Preprocessing Techniques

Applied in this order:
1. Original grayscale
2. Inverted (negative)
3. Binary threshold (127)
4. Binary threshold inverted
5. Otsu's adaptive threshold
6. Adaptive threshold (Gaussian)
7. CLAHE (Contrast Limited Adaptive Histogram Equalization)
8. Sharpening filter

### Multi-Scale Testing

Each preprocessed image is tested at:
- 1.0x (original size)
- 1.5x (150% scale)
- 2.0x (200% scale)
- 3.0x (300% scale)

This helps detect very small codes that need upscaling.

### Page Processing Logic

**Default Behavior**:
- Process only **odd-numbered pages** (1, 3, 5, 7, ...)
- Skip **even pages** (assume duplex printing with content on odd pages)
- Skip **white/blank pages** (auto-detected using pixel intensity)

**Configurable via flags**:
- `--skip-white`: Disable blank page detection, process all odd pages

### Region of Interest

**Default**: Top-right corner, 20% of page width/height

**Rationale**: Most tracking codes are placed in document headers, right-aligned

**Configurable**: `corner_ratio` parameter (0.1 to 0.3 typical)

## Common Workflows

### Development Workflow

When adding new features or fixing bugs:

1. **Test with sample PDFs**: Keep test PDFs in a `test_pdfs/` folder (not in repo)
2. **Use debug mode**: `--debug` flag shows detailed processing info
3. **Extract corners**: `--extract-corners` to save debug images
4. **Check preprocessing**: Review `debug_corners/` output

### Debugging Detection Failures

If codes aren't being detected:

1. **Extract corner images**:
   ```bash
   python detector.py file.pdf --extract-corners 377,379
   ```

2. **Check debug_corners/ folder** for visual inspection

3. **Increase DPI**:
   ```bash
   python detector.py file.pdf 500
   ```

4. **Adjust corner size**:
   ```bash
   python detector.py file.pdf 400 0.3
   ```

5. **Skip white page detection**:
   ```bash
   python detector.py file.pdf --skip-white
   ```

### Testing Changes

**Quick test**:
```bash
python detector.py test.pdf --detect-only
```

**Full test**:
```bash
python detector.py test.pdf 400 0.15 --debug
```

**Batch test**:
```bash
python detector_batch.py test_pdfs/ --detect-only
```

## Code Modification Guidelines

### When Modifying Detection Logic

**Location**: `detector.py` - `detect_codes_in_top_right_corner()`

**Important**:
- Maintain the preprocessing order (original → inverted → binary → ...)
- Keep scale testing (1.0x, 1.5x, 2.0x, 3.0x)
- Preserve fallback chain: pylibdmtx → pyzbar → OpenCV
- Return detection method name for debugging

### When Adding New Preprocessing

**Location**: `detector.py` - `preprocess_for_detection()`

**Pattern**:
```python
def preprocess_for_detection(img):
    methods = {}

    # Add new method
    methods['my_new_method'] = my_preprocessing_function(img)

    return methods
```

### When Changing Corner Location

**Location**: `detector.py` - `detect_codes_in_top_right_corner()`

**Current logic**:
```python
start_x = w - corner_width  # Top-right
start_y = 0                 # Top
```

**For different corners**:
```python
# Top-left
start_x = 0
start_y = 0

# Bottom-right
start_x = w - corner_width
start_y = h - corner_height

# Bottom-left
start_x = 0
start_y = h - corner_height
```

## Dependencies

### Critical Dependencies

**Python Libraries**:
- `pylibdmtx` - DataMatrix detection (requires libdmtx system library)
- `pyzbar` - QR/barcode detection (requires libzbar system library)
- `PyMuPDF` (fitz) - PDF processing
- `opencv-python` (cv2) - Image processing
- `numpy` - Array operations

**System Libraries**:
- Linux: `libdmtx0b`, `libzbar0`
- macOS: `libdmtx`, `zbar` (via Homebrew)
- Windows: Included in pip packages

### Version Compatibility

**Python**: 3.7, 3.8, 3.9, 3.10, 3.11, 3.12
**Python 3.13**: Use `fix_python313.py` for compatibility fixes

## Performance Characteristics

### Speed Benchmarks

**Detection-only mode**:
- ~0.3-0.5 sec/page
- Skips decoding, just verifies presence

**Full decode mode**:
- ~2-4 sec/page
- Includes value extraction and validation

**Factors affecting speed**:
- DPI (higher = slower but better detection)
- Number of preprocessing methods
- Page complexity
- Code size and quality

### Memory Usage

**Per page**:
- ~10-50 MB depending on DPI
- Released after each page

**Batch processing**:
- Processes files sequentially
- No accumulation of memory

## Known Limitations

1. **Corner-only detection**: Only scans top-right corner by default
2. **Odd pages only**: Assumes duplex printing pattern
3. **No rotation handling**: Codes must be right-side-up
4. **No PDF/A handling**: Some secured PDFs may fail
5. **Single code per page**: Returns first detected code only

## Error Handling

### Common Errors

**"No detection libraries available"**:
- Missing pylibdmtx or pyzbar
- Solution: `pip install pylibdmtx pyzbar`

**"libdmtx.so.0: cannot open shared object file"**:
- Missing system library
- Solution: `sudo apt-get install libdmtx0b` (Linux)

**"Pages incorrectly marked as blank"**:
- White page detection too aggressive
- Solution: Use `--skip-white` flag

**"Code detected but decode failed"**:
- Image quality too low
- Solution: Increase DPI (`400` or `500`)

## Output Formats

### Single File Mode

**Console output**:
```
Page  375: ✓ DATAMATRIX Found! Value: [83065676] (method_name)
Page  377: ✗ NO CODE FOUND
```

**Summary**:
- Page statistics
- Decoded values list
- Missing codes list

### Batch Mode

**Files created**:
1. `{filename}_analysis.json` - Per-file detailed results
2. `batch_summary_{timestamp}.csv` - Excel-compatible summary
3. `missing_codes_report_{timestamp}.txt` - Missing codes list
4. `batch_results_{timestamp}.json` - Complete consolidated results

## Git Workflow

**Branch**: `claude/update-claude-md-Zemsb`
**Remote**: `origin`

**For changes**:
1. Develop on the designated branch
2. Commit with descriptive messages
3. Push with: `git push -u origin claude/update-claude-md-Zemsb`

## Testing Strategy

### Manual Testing

1. **Create test PDFs**: Use `test_image_generator.py`
2. **Test detection**: `python detector.py test.pdf`
3. **Verify output**: Check console and debug images
4. **Batch test**: `python detector_batch.py test_folder/`

### Integration Testing

Use `integration_examples.py` for:
- API integration patterns
- Performance benchmarking
- Edge case testing

## Troubleshooting Tips for Claude

1. **Always read the file before modifying**: Use Read tool first
2. **Check dependencies**: Verify import statements work
3. **Test with debug mode**: Use `--debug` to see detailed output
4. **Preserve detection order**: Don't change preprocessing sequence without testing
5. **Keep backward compatibility**: Maintain CLI argument structure
6. **Document new parameters**: Update README.md and BATCH_README.md

## Quick Reference

### File Purposes
- `detector.py` - Single PDF processing
- `detector_batch.py` - Batch processing
- `decoder_optimized.py` - Performance-optimized version
- `integration_examples.py` - Usage examples
- `test_image_generator.py` - Test data creation
- `fix_python313.py` - Python 3.13 fixes
- `fix_quick.py` - Installation fixes

### Key Parameters
- `dpi`: 300 (default), 400-600 for small codes
- `corner_ratio`: 0.2 (default), 0.1-0.3 typical range
- `--detect-only`: Speed up by 3-5x
- `--skip-white`: Process all odd pages
- `--debug`: Verbose output
- `--extract-corners`: Save debug images

### Detection Success Indicators
- `pylibdmtx`: Best for DataMatrix
- `pyzbar`: Best for QR codes
- `inverted` or `binary`: Often works for low-contrast codes
- `scale_1.5` or `scale_2.0`: Needed for very small codes

---

**Last Updated**: 2025-12-16
**Project Version**: 2.0.0
**Maintained by**: milo73
