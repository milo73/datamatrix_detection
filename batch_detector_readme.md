# Batch PDF QR/DataMatrix Code Detector

A powerful batch processing tool for detecting QR codes and DataMatrix codes across multiple PDF files in a folder. Generates comprehensive reports with CSV exports and detailed JSON analysis.

## 🚀 Quick Start

```bash
# Process all PDFs in current folder
python batch_detector.py .

# Process specific folder
python batch_detector.py /path/to/invoices

# Fast detection mode (no decoding)
python batch_detector.py . --detect-only

# Process all subfolders
python batch_detector.py . --recursive
```

## 📋 Features

### Batch Processing Capabilities
- **Folder Processing**: Analyze all PDF files in a folder
- **Recursive Search**: Option to include subfolders
- **Pattern Matching**: Filter specific PDF files
- **Parallel Results**: Individual and consolidated reports
- **Error Handling**: Continues processing even if some files fail

### Output Formats
- **JSON Reports**: Detailed analysis for each PDF
- **CSV Summary**: Excel-compatible summary table
- **Missing Codes Report**: Text file listing all pages without codes
- **Console Summary**: Real-time progress and final statistics

### Performance Features
- **Detection-Only Mode**: 3-5x faster when decoding isn't needed
- **Skip White Pages**: Option to process all odd pages
- **Configurable DPI**: Balance between speed and accuracy
- **Progress Tracking**: Shows current file being processed

## 📦 Installation

```bash
# Install required libraries
pip install pylibdmtx pyzbar PyMuPDF opencv-python numpy

# Install system dependencies (Ubuntu/Debian)
sudo apt-get install libdmtx0b libzbar0

# macOS
brew install libdmtx zbar
```

## 🎯 Command Line Arguments

```bash
python batch_detector.py <folder> [options]
```

### Required Argument
| Argument | Description |
|----------|-------------|
| `folder` | Path to folder containing PDF files |

### Optional Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--output DIR` | `<folder>/qr_detection_results` | Output directory for results |
| `--dpi N` | 300 | Resolution for page extraction |
| `--corner-size F` | 0.2 | Corner region size (0.1-0.3) |
| `--detect-only` | False | Fast detection without decoding |
| `--skip-white` | False | Skip white page detection |
| `--recursive` | False | Include subfolders |
| `--pattern PAT` | `*.pdf` | File pattern to match |
| `--quiet` | False | Minimal output during processing |

## 📊 Output Files

The script creates a results folder with the following files:

### 1. Individual PDF Reports
```
qr_detection_results/
├── invoice001_analysis.json
├── invoice002_analysis.json
└── invoice003_analysis.json
```

Each JSON file contains:
```json
{
  "filename": "invoice001.pdf",
  "total_pages": 10,
  "checked_pages": [1, 3, 5, 7, 9],
  "pages_with_codes": [1, 3, 5, 7],
  "pages_without_codes": [9],
  "code_values": {
    "1": "83065676",
    "3": "83065677"
  },
  "datamatrix_count": 4,
  "qr_count": 0,
  "processing_time": 3.45
}
```

### 2. Batch Summary CSV
`batch_summary_20240115_143022.csv`

| Filename | Total Pages | Pages Checked | Codes Found | Missing Codes | DataMatrix | QR | Time(s) | Status |
|----------|------------|---------------|-------------|---------------|------------|----|---------|----|
| invoice001.pdf | 10 | 5 | 4 | 1 | 4 | 0 | 3.45 | MISSING: 1 |
| invoice002.pdf | 8 | 4 | 4 | 0 | 4 | 0 | 2.89 | OK |

### 3. Missing Codes Report
`missing_codes_report_20240115_143022.txt`
```
MISSING CODES REPORT
==================================================

File: invoice001.pdf
Missing pages: [9]
Total missing: 1
------------------------------

File: invoice003.pdf
Missing pages: [5, 7, 11]
Total missing: 3
------------------------------
```

### 4. Complete JSON Report
`batch_results_20240115_143022.json`

Contains both summary and detailed results for all files.

## 💡 Usage Examples

### Example 1: Process Invoices Folder
```bash
python batch_detector.py ~/Documents/Invoices --dpi 400
```
Output:
```
==================================================
BATCH PROCESSING: 152 PDF files
==================================================
[1/152] Processing: invoice_2024_001.pdf
  ✓ Processed: 5 pages, Found codes: 5, Missing: 0
[2/152] Processing: invoice_2024_002.pdf
  ✓ Processed: 3 pages, Found codes: 2, Missing: 1
...
```

### Example 2: Quick Validation Check
```bash
python batch_detector.py . --detect-only --recursive
```
Fast detection-only mode for all PDFs in current folder and subfolders.

### Example 3: High-Quality Scan with Custom Output
```bash
python batch_detector.py /scans --dpi 500 --corner-size 0.15 --output /reports/scan_results
```

### Example 4: Process Specific File Pattern
```bash
python batch_detector.py . --pattern "invoice_2024*.pdf" --skip-white
```

## 📈 Performance Optimization

### Speed Comparison

| Mode | Options | Speed per File | Use Case |
|------|---------|---------------|----------|
| Fastest | `--detect-only --skip-white` | 1-3 sec | Quick validation |
| Fast | `--detect-only` | 2-5 sec | Detection without values |
| Standard | (default) | 5-15 sec | Full analysis |
| Thorough | `--dpi 500` | 15-30 sec | Small codes |

### Tips for Large Batches

1. **Start with Detection-Only**:
   ```bash
   python batch_detector.py large_folder --detect-only
   ```
   Quickly identify files with missing codes.

2. **Process Problem Files Separately**:
   ```bash
   # After identifying problem files, process them with higher quality
   python batch_detector.py problem_files --dpi 500 --skip-white
   ```

3. **Use Recursive Carefully**:
   ```bash
   # Test on one subfolder first
   python batch_detector.py subfolder --detect-only
   
   # Then process all
   python batch_detector.py . --recursive
   ```

## 🔍 Understanding the Console Output

### During Processing
```
[15/100] Processing: document_15.pdf
  ✓ Processed: 10 pages, Found codes: 9, Missing: 1
```
- `[15/100]`: Current file number / Total files
- `✓`: Successfully processed
- Pages/Found/Missing: Quick statistics

### Final Summary
```
==================================================
BATCH PROCESSING SUMMARY
==================================================

Files Processed: 100
  ✓ Successful: 98
  ✗ Failed: 2

Pages Analysis:
  • Total pages checked: 450
  • Pages with codes: 440
  • Pages missing codes: 10

Code Types Found:
  • DataMatrix: 430
  • QR Codes: 10

Processing Time:
  • Total: 234.5 seconds
  • Average per file: 2.35 seconds

⚠️ Files with Missing Codes: 5
  • invoice_2024_015.pdf: 2 pages missing
  • invoice_2024_027.pdf: 1 pages missing

✅ Batch processing complete!
📁 Results saved in: ./qr_detection_results
```

## 🛠️ Advanced Usage

### Custom Processing Pipeline

Create a script for specific workflows:

```python
#!/usr/bin/env python3
import subprocess
import json
from pathlib import Path

# Step 1: Quick detection
subprocess.run(['python', 'batch_detector.py', '.', '--detect-only'])

# Step 2: Load results
with open('qr_detection_results/batch_results_*.json') as f:
    results = json.load(f)

# Step 3: Reprocess files with missing codes
problem_files = [r['filename'] for r in results['details'] 
                 if r['pages_without_codes']]

for pdf in problem_files:
    subprocess.run(['python', 'batch_detector.py', pdf, 
                   '--dpi', '500', '--skip-white'])
```

### Integration with Other Tools

Export to Excel:
```python
import pandas as pd
df = pd.read_csv('qr_detection_results/batch_summary_*.csv')
df.to_excel('analysis.xlsx', index=False)
```

## 🐛 Troubleshooting

### Issue: "No PDF files found"
```bash
# Check if PDFs exist
ls *.pdf

# Use recursive if PDFs are in subfolders
python batch_detector.py . --recursive

# Check pattern matching
python batch_detector.py . --pattern "*.PDF"
```

### Issue: Processing is too slow
```bash
# Use detection-only mode
python batch_detector.py folder --detect-only

# Reduce DPI for initial scan
python batch_detector.py folder --dpi 200 --detect-only
```

### Issue: Missing codes on known good files
```bash
# Skip white page detection
python batch_detector.py folder --skip-white

# Increase DPI
python batch_detector.py folder --dpi 500

# Adjust corner size
python batch_detector.py folder --corner-size 0.3
```

## 📊 Analyzing Results

### Using the CSV Output
1. Open in Excel/Google Sheets
2. Filter by "Status" column to find problems
3. Sort by "Missing Codes" to prioritize fixes

### Using the JSON Output
```python
import json

# Load results
with open('batch_results_20240115_143022.json') as f:
    data = json.load(f)

# Find files with most missing codes
problems = sorted(data['summary']['files_with_missing_codes'], 
                 key=lambda x: x['missing_count'], 
                 reverse=True)

for p in problems[:10]:
    print(f"{p['filename']}: {p['missing_pages']}")
```

## 🔄 Return Codes

The script returns different exit codes:
- `0`: All files processed successfully, no missing codes
- `1`: Some pages missing codes or processing errors

Use in shell scripts:
```bash
#!/bin/bash
if python batch_detector.py folder --detect-only; then
    echo "All codes detected!"
else
    echo "Some codes missing, check reports"
fi
```

## 📝 Notes

- **Even pages** are automatically skipped (configurable in code)
- **Results folder** is created automatically
- **Existing results** are not overwritten (timestamped files)
- **Failed files** don't stop the batch process
- **Memory usage** is optimized for large batches

## 🎯 Best Practices

1. **Start Simple**: Run detection-only first
2. **Check Reports**: Review CSV before full processing
3. **Incremental Processing**: Process problem files separately
4. **Archive Results**: Keep timestamped results for comparison
5. **Monitor Progress**: Use verbose mode for large batches

---

**Version**: 1.0.0  
**Supports**: Python 3.7+  
**Libraries**: pylibdmtx, pyzbar, PyMuPDF, OpenCV