#!/usr/bin/env python3
"""
PDF QR Code & DataMatrix Detector - Batch Processing Version
Processes all PDF files in a folder with QR/DataMatrix detection.
Generates individual reports and a consolidated summary.
"""

import sys
import os
import logging
import fitz  # PyMuPDF
import cv2
import numpy as np
from pathlib import Path
from collections import Counter
import json
import csv
from datetime import datetime
import argparse
import time

# Configure logging
logger = logging.getLogger(__name__)

# Try to import detection libraries
try:
    from pyzbar import pyzbar
    PYZBAR_AVAILABLE = True
except ImportError:
    PYZBAR_AVAILABLE = False
    print("Warning: pyzbar not installed. Install with: pip install pyzbar")

try:
    from pylibdmtx import pylibdmtx
    PYLIBDMTX_AVAILABLE = True
except ImportError:
    PYLIBDMTX_AVAILABLE = False
    print("Warning: pylibdmtx not installed. Install with: pip install pylibdmtx")

# Import detection functions from single-file version
def extract_page_as_image(pdf_document, page_num, dpi=300):
    """Extract a PDF page as an image array."""
    page = pdf_document[page_num]
    mat = fitz.Matrix(dpi/72, dpi/72)
    pix = page.get_pixmap(matrix=mat)
    img_data = pix.tobytes("ppm")
    nparr = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image

def is_white_page(image, threshold=0.99):
    """Check if a page is mostly white/blank."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    non_white_pixels = np.sum(gray < 240)
    total_pixels = gray.shape[0] * gray.shape[1]
    non_white_ratio = non_white_pixels / total_pixels
    if non_white_ratio > (1 - threshold):
        return False
    edges = cv2.Canny(gray, 50, 150)
    edge_pixels = np.sum(edges > 0)
    return edge_pixels <= 100

def apply_clahe(gray_image):
    """Apply CLAHE enhancement."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_image)

def sharpen_image(gray_image):
    """Apply sharpening filter."""
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(gray_image, -1, kernel)

def detect_datamatrix_with_pylibdmtx(gray_roi, decode=True, timeout=1000):
    """Detect DataMatrix using pylibdmtx."""
    if not PYLIBDMTX_AVAILABLE:
        return []
    
    detected = []
    try:
        codes = pylibdmtx.decode(gray_roi, timeout=timeout if decode else 100, max_count=5 if decode else 1)
        for code in codes:
            x, y, w, h = code.rect
            if decode:
                try:
                    decoded_data = code.data.decode('utf-8')
                except (UnicodeDecodeError, AttributeError):
                    decoded_data = str(code.data)
            else:
                decoded_data = "DATAMATRIX_DETECTED"
            
            detected.append({
                'type': 'DATAMATRIX',
                'data': decoded_data,
                'position': (x, y, w, h),
                'method': 'pylibdmtx'
            })
    except Exception as e:
        logger.debug(f"pylibdmtx detection failed: {e}")
    return detected

def detect_codes_with_pyzbar(gray_roi, decode=True):
    """Detect codes using pyzbar."""
    if not PYZBAR_AVAILABLE:
        return []
    
    detected = []
    try:
        codes = pyzbar.decode(gray_roi)
        for code in codes:
            x, y, w, h = code.rect
            if decode:
                try:
                    decoded_data = code.data.decode('utf-8')
                except (UnicodeDecodeError, AttributeError):
                    decoded_data = str(code.data)
            else:
                decoded_data = f"{code.type}_DETECTED"
            
            detected.append({
                'type': code.type,
                'data': decoded_data,
                'position': (x, y, w, h),
                'method': 'pyzbar'
            })
    except Exception as e:
        logger.debug(f"pyzbar detection failed: {e}")
    return detected

def detect_codes_in_corner(image, corner_size_ratio=0.2, decode=True):
    """Detect codes in top-right corner."""
    detected_codes = []

    # Validate image input
    if image is None or not hasattr(image, 'shape') or len(image.shape) < 2:
        logger.warning("Invalid image input")
        return detected_codes

    # Validate corner_size_ratio
    if not 0 < corner_size_ratio <= 0.5:
        corner_size_ratio = 0.2

    h, w = image.shape[:2] if len(image.shape) == 3 else image.shape
    
    corner_width = int(w * corner_size_ratio)
    corner_height = int(h * corner_size_ratio)
    start_x = w - corner_width
    start_y = 0
    corner_roi = image[start_y:start_y+corner_height, start_x:start_x+corner_width]
    
    if len(corner_roi.shape) == 3:
        gray_roi = cv2.cvtColor(corner_roi, cv2.COLOR_BGR2GRAY)
    else:
        gray_roi = corner_roi
    
    # Preprocessing methods
    preprocessing_methods = [
        ('original', gray_roi),
        ('inverted', cv2.bitwise_not(gray_roi)),
        ('thresh_127', cv2.threshold(gray_roi, 127, 255, cv2.THRESH_BINARY)[1]),
        ('thresh_otsu', cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
    ]
    
    scales = [1.0, 1.5] if decode else [1.0]
    
    # Try pylibdmtx
    if PYLIBDMTX_AVAILABLE:
        for method_name, processed_img in preprocessing_methods:
            for scale in scales:
                if scale != 1.0:
                    scaled_h = int(processed_img.shape[0] * scale)
                    scaled_w = int(processed_img.shape[1] * scale)
                    if scaled_h > 10 and scaled_w > 10:
                        scaled_img = cv2.resize(processed_img, (scaled_w, scaled_h))
                    else:
                        continue
                else:
                    scaled_img = processed_img
                
                codes = detect_datamatrix_with_pylibdmtx(scaled_img, decode=decode)
                if codes:
                    for code in codes:
                        x, y, w, h = code['position']
                        if scale != 1.0:
                            x, y, w, h = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
                        code['position'] = (start_x + x, start_y + y, w, h)
                        code['method'] = f"{code['method']}_{method_name}_s{scale}"
                    return codes
    
    # Try pyzbar
    if PYZBAR_AVAILABLE and not detected_codes:
        for method_name, processed_img in preprocessing_methods:
            codes = detect_codes_with_pyzbar(processed_img, decode=decode)
            if codes:
                for code in codes:
                    x, y, w, h = code['position']
                    code['position'] = (start_x + x, start_y + y, w, h)
                    code['method'] = f"{code['method']}_{method_name}"
                return codes
    
    return detected_codes

def analyze_single_pdf(pdf_path, dpi=300, corner_size_ratio=0.2, decode=True, skip_white=False, quiet=False):
    """Analyze a single PDF file."""
    try:
        pdf_document = fitz.open(pdf_path)
        total_pages = len(pdf_document)
        
        results = {
            'filename': os.path.basename(pdf_path),
            'path': str(pdf_path),
            'total_pages': total_pages,
            'checked_pages': [],
            'pages_with_codes': [],
            'pages_without_codes': [],
            'code_values': {},
            'datamatrix_count': 0,
            'qr_count': 0,
            'processing_time': 0,
            'error': None
        }
        
        start_time = time.time()
        
        for page_num in range(total_pages):
            page_number = page_num + 1
            
            # Skip even pages
            if page_number % 2 == 0:
                continue
            
            try:
                image = extract_page_as_image(pdf_document, page_num, dpi)
                
                # Check if blank (unless disabled)
                if not skip_white and is_white_page(image):
                    continue
                
                results['checked_pages'].append(page_number)
                
                # Detect codes
                codes = detect_codes_in_corner(image, corner_size_ratio, decode)
                
                if codes:
                    results['pages_with_codes'].append(page_number)
                    for code in codes:
                        if 'DATAMATRIX' in code['type']:
                            results['datamatrix_count'] += 1
                        elif 'QR' in code['type']:
                            results['qr_count'] += 1
                        
                        if decode and code['data'] and 'DETECTED' not in code['data']:
                            results['code_values'][page_number] = code['data']
                else:
                    results['pages_without_codes'].append(page_number)
                    
            except Exception as e:
                results['pages_without_codes'].append(page_number)
        
        pdf_document.close()
        results['processing_time'] = round(time.time() - start_time, 2)
        
        if not quiet:
            print(f"  ✓ Processed: {len(results['checked_pages'])} pages, "
                  f"Found codes: {len(results['pages_with_codes'])}, "
                  f"Missing: {len(results['pages_without_codes'])}")
        
        return results
        
    except Exception as e:
        return {
            'filename': os.path.basename(pdf_path),
            'path': str(pdf_path),
            'error': str(e),
            'processing_time': 0
        }

class BatchProcessor:
    """Batch processor for multiple PDF files."""
    
    def __init__(self, folder_path, output_dir=None, **kwargs):
        self.folder_path = Path(folder_path)
        self.output_dir = Path(output_dir) if output_dir else self.folder_path / "qr_detection_results"
        self.output_dir.mkdir(exist_ok=True)
        
        # Processing options
        self.dpi = kwargs.get('dpi', 300)
        self.corner_size_ratio = kwargs.get('corner_size_ratio', 0.2)
        self.decode = kwargs.get('decode', True)
        self.skip_white = kwargs.get('skip_white', False)
        self.recursive = kwargs.get('recursive', False)
        self.pattern = kwargs.get('pattern', '*.pdf')
        
        # Results storage
        self.results = []
        self.summary = {}
        
    def find_pdf_files(self):
        """Find all PDF files to process."""
        if self.recursive:
            pdf_files = list(self.folder_path.rglob(self.pattern))
        else:
            pdf_files = list(self.folder_path.glob(self.pattern))
        
        return sorted(pdf_files)
    
    def process_files(self, pdf_files, verbose=True):
        """Process all PDF files."""
        total_files = len(pdf_files)
        
        print(f"\n{'='*70}")
        print(f"BATCH PROCESSING: {total_files} PDF files")
        print(f"{'='*70}")
        print(f"Folder: {self.folder_path}")
        print(f"Settings: DPI={self.dpi}, Corner={self.corner_size_ratio*100}%, "
              f"Mode={'Decode' if self.decode else 'Detect-only'}")
        if self.skip_white:
            print("White page detection: DISABLED")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        for idx, pdf_file in enumerate(pdf_files, 1):
            if verbose:
                print(f"[{idx}/{total_files}] Processing: {pdf_file.name}")
            
            result = analyze_single_pdf(
                pdf_file,
                dpi=self.dpi,
                corner_size_ratio=self.corner_size_ratio,
                decode=self.decode,
                skip_white=self.skip_white,
                quiet=not verbose
            )
            
            self.results.append(result)
            
            # Save individual result
            self.save_individual_result(result)
        
        total_time = time.time() - start_time
        
        # Generate summary
        self.generate_summary(total_time)
        
        # Save batch results
        self.save_batch_results()
        
        # Print summary
        self.print_summary()
    
    def save_individual_result(self, result):
        """Save individual PDF analysis result."""
        filename = Path(result['filename']).stem
        result_file = self.output_dir / f"{filename}_analysis.json"
        
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
    
    def generate_summary(self, total_time):
        """Generate batch processing summary."""
        successful = [r for r in self.results if r.get('error') is None]
        failed = [r for r in self.results if r.get('error') is not None]
        
        total_pages_checked = sum(len(r['checked_pages']) for r in successful)
        total_codes_found = sum(len(r['pages_with_codes']) for r in successful)
        total_missing = sum(len(r['pages_without_codes']) for r in successful)
        total_datamatrix = sum(r.get('datamatrix_count', 0) for r in successful)
        total_qr = sum(r.get('qr_count', 0) for r in successful)
        
        self.summary = {
            'timestamp': datetime.now().isoformat(),
            'folder': str(self.folder_path),
            'total_files': len(self.results),
            'successful_files': len(successful),
            'failed_files': len(failed),
            'total_pages_checked': total_pages_checked,
            'total_codes_found': total_codes_found,
            'total_missing_codes': total_missing,
            'total_datamatrix': total_datamatrix,
            'total_qr': total_qr,
            'total_processing_time': round(total_time, 2),
            'average_time_per_file': round(total_time / len(self.results), 2) if self.results else 0,
            'settings': {
                'dpi': self.dpi,
                'corner_size_ratio': self.corner_size_ratio,
                'decode_mode': self.decode,
                'skip_white': self.skip_white
            }
        }
        
        # Files with issues
        self.summary['files_with_missing_codes'] = [
            {
                'filename': r['filename'],
                'missing_pages': r['pages_without_codes'],
                'missing_count': len(r['pages_without_codes'])
            }
            for r in successful if r['pages_without_codes']
        ]
        
        # Failed files
        if failed:
            self.summary['failed_files'] = [
                {'filename': r['filename'], 'error': r['error']}
                for r in failed
            ]
    
    def save_batch_results(self):
        """Save all batch processing results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed JSON report
        json_file = self.output_dir / f"batch_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump({
                'summary': self.summary,
                'details': self.results
            }, f, indent=2)
        
        # Save CSV summary
        csv_file = self.output_dir / f"batch_summary_{timestamp}.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Filename', 'Total Pages', 'Pages Checked', 'Codes Found', 
                           'Missing Codes', 'DataMatrix Count', 'QR Count', 
                           'Processing Time (s)', 'Status'])
            
            for r in self.results:
                if r.get('error'):
                    writer.writerow([r['filename'], 'ERROR', '', '', '', '', '', '', r['error']])
                else:
                    writer.writerow([
                        r['filename'],
                        r['total_pages'],
                        len(r['checked_pages']),
                        len(r['pages_with_codes']),
                        len(r['pages_without_codes']),
                        r.get('datamatrix_count', 0),
                        r.get('qr_count', 0),
                        r['processing_time'],
                        'OK' if not r['pages_without_codes'] else f"MISSING: {len(r['pages_without_codes'])}"
                    ])
        
        # Save missing codes report
        if self.summary['files_with_missing_codes']:
            missing_file = self.output_dir / f"missing_codes_report_{timestamp}.txt"
            with open(missing_file, 'w') as f:
                f.write("MISSING CODES REPORT\n")
                f.write("="*50 + "\n\n")
                
                for item in self.summary['files_with_missing_codes']:
                    f.write(f"File: {item['filename']}\n")
                    f.write(f"Missing pages: {item['missing_pages']}\n")
                    f.write(f"Total missing: {item['missing_count']}\n")
                    f.write("-"*30 + "\n")
        
        print(f"\nResults saved to: {self.output_dir}")
    
    def print_summary(self):
        """Print processing summary."""
        print(f"\n{'='*70}")
        print("BATCH PROCESSING SUMMARY")
        print(f"{'='*70}")
        
        print(f"\nFiles Processed: {self.summary['total_files']}")
        print(f"  ✓ Successful: {self.summary['successful_files']}")
        if self.summary['failed_files']:
            print(f"  ✗ Failed: {self.summary['failed_files']}")
        
        print(f"\nPages Analysis:")
        print(f"  • Total pages checked: {self.summary['total_pages_checked']}")
        print(f"  • Pages with codes: {self.summary['total_codes_found']}")
        print(f"  • Pages missing codes: {self.summary['total_missing_codes']}")
        
        print(f"\nCode Types Found:")
        print(f"  • DataMatrix: {self.summary['total_datamatrix']}")
        print(f"  • QR Codes: {self.summary['total_qr']}")
        
        print(f"\nProcessing Time:")
        print(f"  • Total: {self.summary['total_processing_time']} seconds")
        print(f"  • Average per file: {self.summary['average_time_per_file']} seconds")
        
        if self.summary['files_with_missing_codes']:
            print(f"\n⚠️  Files with Missing Codes: {len(self.summary['files_with_missing_codes'])}")
            for item in self.summary['files_with_missing_codes'][:5]:
                print(f"  • {item['filename']}: {item['missing_count']} pages missing")
            if len(self.summary['files_with_missing_codes']) > 5:
                print(f"  ... and {len(self.summary['files_with_missing_codes'])-5} more")
        
        print(f"\n✅ Batch processing complete!")
        print(f"📁 Results saved in: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(
        description='Batch PDF QR/DataMatrix Code Detector',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all PDFs in current folder
  python batch_detector.py .
  
  # Process specific folder with options
  python batch_detector.py /path/to/pdfs --dpi 400 --detect-only
  
  # Recursive processing
  python batch_detector.py . --recursive --skip-white
  
  # Custom output directory
  python batch_detector.py . --output results_folder
        """
    )
    
    parser.add_argument('folder', help='Folder containing PDF files')
    parser.add_argument('--output', help='Output directory for results')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for page extraction (default: 300)')
    parser.add_argument('--corner-size', type=float, default=0.2, 
                       help='Corner region size ratio (default: 0.2)')
    parser.add_argument('--detect-only', action='store_true', 
                       help='Detection only mode (faster, no decoding)')
    parser.add_argument('--skip-white', action='store_true',
                       help='Skip white page detection')
    parser.add_argument('--recursive', action='store_true',
                       help='Process PDFs in subfolders too')
    parser.add_argument('--pattern', default='*.pdf',
                       help='File pattern to match (default: *.pdf)')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal output during processing')
    
    args = parser.parse_args()
    
    # Check libraries
    if not PYLIBDMTX_AVAILABLE and not PYZBAR_AVAILABLE:
        print("\nERROR: No detection libraries available!")
        print("Install at least one:")
        print("  pip install pylibdmtx")
        print("  pip install pyzbar")
        sys.exit(1)
    
    # Check folder exists
    if not os.path.exists(args.folder):
        print(f"Error: Folder '{args.folder}' not found.")
        sys.exit(1)
    
    # Create batch processor
    processor = BatchProcessor(
        args.folder,
        output_dir=args.output,
        dpi=args.dpi,
        corner_size_ratio=args.corner_size,
        decode=not args.detect_only,
        skip_white=args.skip_white,
        recursive=args.recursive,
        pattern=args.pattern
    )
    
    # Find PDF files
    pdf_files = processor.find_pdf_files()
    
    if not pdf_files:
        print(f"No PDF files found in '{args.folder}'")
        if not args.recursive:
            print("Tip: Use --recursive to search in subfolders")
        sys.exit(1)
    
    # Process files
    processor.process_files(pdf_files, verbose=not args.quiet)
    
    return 0 if not processor.summary.get('files_with_missing_codes') else 1

if __name__ == "__main__":
    sys.exit(main())