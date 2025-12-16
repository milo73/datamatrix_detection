#!/usr/bin/env python3
"""
PDF QR Code & DataMatrix Detector - Enhanced with pylibdmtx
Specifically looks for QR codes and DataMatrix codes in the top-right corner of odd pages.
Uses pylibdmtx for superior DataMatrix detection.
Supports detection-only mode for faster performance.
"""

import sys
import os
import fitz  # PyMuPDF
import cv2
import numpy as np
from pathlib import Path
from collections import Counter

# Try to import both libraries
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
    print("Note: On Ubuntu/Debian: sudo apt-get install libdmtx0b")
    print("      On macOS: brew install libdmtx")
    print("      On Windows: See https://github.com/NaturalHistoryMuseum/pylibdmtx#windows")

def extract_page_as_image(pdf_document, page_num, dpi=300):
    """Extract a PDF page as an image array with high DPI for better small code detection."""
    page = pdf_document[page_num]
    # Use higher DPI for better detection of small codes
    mat = fitz.Matrix(dpi/72, dpi/72)  # 72 is default DPI
    pix = page.get_pixmap(matrix=mat)
    
    # Convert to numpy array for OpenCV
    img_data = pix.tobytes("ppm")
    nparr = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    return image

def is_white_page(image, threshold=0.99):
    """
    Check if a page is mostly white/blank.
    More lenient to avoid false positives on pages with small DataMatrix codes.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Check if there's any significant content
    non_white_pixels = np.sum(gray < 240)
    total_pixels = gray.shape[0] * gray.shape[1]
    
    non_white_ratio = non_white_pixels / total_pixels
    
    # If more than 1% of the page has content, it's not blank
    if non_white_ratio > (1 - threshold):
        return False
    
    # Additional check: look for edges/features
    edges = cv2.Canny(gray, 50, 150)
    edge_pixels = np.sum(edges > 0)
    
    # If there are any significant edges, it's not blank
    if edge_pixels > 100:  # Very low threshold
        return False
    
    return True

def apply_clahe(gray_image):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray_image)

def sharpen_image(gray_image):
    """Apply sharpening to enhance edges."""
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    return cv2.filter2D(gray_image, -1, kernel)

def detect_datamatrix_with_pylibdmtx(gray_roi, decode=True, timeout=1000, max_count=5):
    """
    Detect DataMatrix codes using pylibdmtx.
    
    Args:
        gray_roi: Grayscale image region
        decode: If True, decode the content. If False, only detect presence
        timeout: Timeout in milliseconds for detection
        max_count: Maximum number of codes to detect
    
    Returns:
        List of detected codes with format, data, and position
    """
    if not PYLIBDMTX_AVAILABLE:
        return []
    
    detected = []
    
    try:
        # pylibdmtx decode function
        if decode:
            # Full decode with timeout
            codes = pylibdmtx.decode(gray_roi, timeout=timeout, max_count=max_count)
        else:
            # Quick detection only - use very short timeout
            codes = pylibdmtx.decode(gray_roi, timeout=100, max_count=1)
        
        for code in codes:
            # Extract position
            x, y, w, h = code.rect
            
            # Get data if decoding
            if decode:
                try:
                    decoded_data = code.data.decode('utf-8')
                except:
                    decoded_data = str(code.data)
            else:
                decoded_data = "DATAMATRIX_DETECTED_NOT_DECODED"
            
            detected.append({
                'type': 'DATAMATRIX',
                'data': decoded_data,
                'position': (x, y, w, h),
                'method': 'pylibdmtx'
            })
            
    except Exception as e:
        # Silent fail - detection didn't work
        pass
    
    return detected

def detect_codes_with_pyzbar(gray_roi, decode=True):
    """
    Detect codes using pyzbar (QR codes and DataMatrix).
    
    Args:
        gray_roi: Grayscale image region
        decode: If True, decode the content
    
    Returns:
        List of detected codes
    """
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
                except:
                    decoded_data = str(code.data)
            else:
                decoded_data = f"{code.type}_DETECTED_NOT_DECODED"
            
            detected.append({
                'type': code.type,
                'data': decoded_data,
                'position': (x, y, w, h),
                'method': 'pyzbar'
            })
            
    except Exception:
        pass
    
    return detected

def detect_codes_in_top_right_corner(image, corner_size_ratio=0.2, decode=True, debug=False):
    """
    Focus detection specifically on the top-right corner where codes are expected.
    Enhanced with pylibdmtx for DataMatrix detection.
    
    Args:
        image: OpenCV image array
        corner_size_ratio: Ratio of image dimensions to use for corner (default 0.2 = 20%)
        decode: If True, decode the content. If False, only detect presence
        debug: If True, save debug images
    
    Returns:
        List of detected codes
    """
    detected_codes = []
    h, w = image.shape[:2] if len(image.shape) == 3 else image.shape
    
    # Define top-right corner region
    corner_width = int(w * corner_size_ratio)
    corner_height = int(h * corner_size_ratio)
    
    # Extract top-right corner
    start_x = w - corner_width
    start_y = 0
    corner_roi = image[start_y:start_y+corner_height, start_x:start_x+corner_width]
    
    # Convert to grayscale if needed
    if len(corner_roi.shape) == 3:
        gray_roi = cv2.cvtColor(corner_roi, cv2.COLOR_BGR2GRAY)
    else:
        gray_roi = corner_roi
    
    # Save debug image if requested
    if debug:
        debug_path = f"debug_corner_p{start_x}_{start_y}.png"
        cv2.imwrite(debug_path, gray_roi)
        if debug:
            print(f"    Saved debug image: {debug_path}")
    
    # Preprocessing methods to try
    preprocessing_methods = [
        ('original', gray_roi),
        ('inverted', cv2.bitwise_not(gray_roi)),
        ('thresh_127', cv2.threshold(gray_roi, 127, 255, cv2.THRESH_BINARY)[1]),
        ('thresh_inv', cv2.threshold(gray_roi, 127, 255, cv2.THRESH_BINARY_INV)[1]),
        ('thresh_otsu', cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
        ('adaptive', cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
        ('clahe', apply_clahe(gray_roi)),
        ('sharpened', sharpen_image(gray_roi))
    ]
    
    # Scales to try
    scales = [1.0, 1.5, 2.0] if decode else [1.0]  # Use fewer scales in detection-only mode
    
    # METHOD 1: Try pylibdmtx first (best for DataMatrix)
    if PYLIBDMTX_AVAILABLE:
        for method_name, processed_img in preprocessing_methods[:4 if not decode else 8]:  # Fewer attempts in detect-only
            for scale in scales:
                try:
                    # Scale the image if needed
                    if scale != 1.0:
                        scaled_h = int(processed_img.shape[0] * scale)
                        scaled_w = int(processed_img.shape[1] * scale)
                        if scaled_h > 10 and scaled_w > 10:
                            scaled_img = cv2.resize(processed_img, (scaled_w, scaled_h), 
                                                   interpolation=cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA)
                        else:
                            continue
                    else:
                        scaled_img = processed_img
                    
                    # Try pylibdmtx detection
                    codes = detect_datamatrix_with_pylibdmtx(scaled_img, decode=decode)
                    
                    if codes:
                        # Adjust coordinates back to full image
                        for code in codes:
                            x, y, w, h = code['position']
                            if scale != 1.0:
                                x, y, w, h = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
                            
                            detected_codes.append({
                                'type': 'DATAMATRIX',
                                'data': code['data'],
                                'position': (start_x + x, start_y + y, w, h),
                                'method': f'pylibdmtx_{method_name}_scale_{scale}'
                            })
                        
                        if debug:
                            print(f"    Found DataMatrix with pylibdmtx ({method_name}, scale {scale})")
                        
                        return detected_codes  # Return on first successful detection
                        
                except Exception as e:
                    if debug:
                        print(f"    pylibdmtx error with {method_name} at scale {scale}: {str(e)[:50]}")
    
    # METHOD 2: Try pyzbar if pylibdmtx didn't find anything
    if not detected_codes and PYZBAR_AVAILABLE:
        for method_name, processed_img in preprocessing_methods[:4 if not decode else 8]:
            for scale in scales:
                try:
                    # Scale the image if needed
                    if scale != 1.0:
                        scaled_h = int(processed_img.shape[0] * scale)
                        scaled_w = int(processed_img.shape[1] * scale)
                        if scaled_h > 10 and scaled_w > 10:
                            scaled_img = cv2.resize(processed_img, (scaled_w, scaled_h),
                                                   interpolation=cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA)
                        else:
                            continue
                    else:
                        scaled_img = processed_img
                    
                    # Try pyzbar detection
                    codes = detect_codes_with_pyzbar(scaled_img, decode=decode)
                    
                    if codes:
                        # Adjust coordinates back to full image
                        for code in codes:
                            x, y, w, h = code['position']
                            if scale != 1.0:
                                x, y, w, h = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
                            
                            detected_codes.append({
                                'type': code['type'],
                                'data': code['data'],
                                'position': (start_x + x, start_y + y, w, h),
                                'method': f'pyzbar_{method_name}_scale_{scale}'
                            })
                        
                        if debug:
                            print(f"    Found {code['type']} with pyzbar ({method_name}, scale {scale})")
                        
                        return detected_codes  # Return on first successful detection
                        
                except Exception as e:
                    if debug:
                        print(f"    pyzbar error with {method_name} at scale {scale}: {str(e)[:50]}")
    
    # METHOD 3: OpenCV QR detector (only for QR codes)
    if not detected_codes and decode:  # Only try this in decode mode
        try:
            qr_detector = cv2.QRCodeDetector()
            retval, data, points, _ = qr_detector.detectAndDecode(gray_roi)
            if retval and (data or not decode):
                detected_codes.append({
                    'type': 'QRCODE',
                    'data': data if data else 'QR_DETECTED_NOT_DECODED',
                    'position': (start_x, start_y, corner_width, corner_height),
                    'method': 'opencv_qr'
                })
                if debug:
                    print(f"    Found QR with OpenCV")
        except Exception as e:
            if debug:
                print(f"    OpenCV QR error: {str(e)[:50]}")
    
    return detected_codes

def detect_codes_with_fallback(image, decode=True):
    """
    Try detection on full page as fallback.
    """
    detected_codes = []
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Try pylibdmtx first
    if PYLIBDMTX_AVAILABLE:
        codes = detect_datamatrix_with_pylibdmtx(gray, decode=decode)
        if codes:
            for code in codes:
                code['method'] = 'pylibdmtx_fullpage'
            detected_codes.extend(codes)
    
    # Try pyzbar if no codes found
    if not detected_codes and PYZBAR_AVAILABLE:
        codes = detect_codes_with_pyzbar(gray, decode=decode)
        if codes:
            for code in codes:
                code['method'] = 'pyzbar_fullpage'
            detected_codes.extend(codes)
    
    return detected_codes

def validate_code_content(content, code_type='UNKNOWN'):
    """
    Validate the content of a decoded code.
    """
    if not content or 'NOT_DECODED' in content:
        return "Detection only - not decoded"
    
    if not content:
        return "Empty content"
    
    # Check if it's a pure number (likely invoice/document number)
    if content.isdigit():
        num_digits = len(content)
        if num_digits == 8:
            return f"Valid 8-digit number (likely invoice/doc ID)"
        elif num_digits == 7:
            return f"Valid 7-digit number"
        else:
            return f"Valid number ({num_digits} digits)"
    
    # Check if it's an alphanumeric code
    if content.isalnum():
        return f"Alphanumeric code ({len(content)} chars)"
    
    # Check if it's a URL
    if content.startswith(('http://', 'https://', 'www.')):
        return "URL detected"
    
    # Check if it contains invoice/document patterns
    if any(keyword in content.lower() for keyword in ['invoice', 'factuur', 'document', 'order']):
        return "Document reference"
    
    # Default for any other text
    return f"Text content ({len(content)} chars)"

def analyze_pdf_for_codes(pdf_path, verbose=True, dpi=300, corner_size_ratio=0.2, 
                          decode=True, debug=False, skip_white_check=False):
    """
    Analyze PDF for QR codes and DataMatrix codes in top-right corner of odd pages.
    
    Args:
        pdf_path: Path to the PDF file
        verbose: Print detailed information for each page
        dpi: DPI for page extraction (higher = better for small codes)
        corner_size_ratio: Ratio of page to use for corner detection
        decode: If True, decode content. If False, only detect presence
        debug: Save debug images and show extra information
        skip_white_check: Skip white page detection (process all odd pages)
    
    Returns:
        Dictionary with results
    """
    try:
        pdf_document = fitz.open(pdf_path)
        total_pages = len(pdf_document)
        
        results = {
            'total_pages': total_pages,
            'checked_pages': [],
            'skipped_even_pages': [],
            'skipped_white_pages': [],
            'pages_with_codes': [],
            'pages_without_codes': [],
            'code_details': {},
            'code_values': {},
            'datamatrix_pages': [],
            'qr_pages': []
        }
        
        if verbose:
            print(f"Analyzing PDF: {pdf_path}")
            print(f"Total pages: {total_pages}")
            print(f"Settings: DPI={dpi}, Corner={corner_size_ratio*100}%")
            print(f"Mode: {'Full decode' if decode else 'Detection only (faster)'}")
            print("Target: Top-right corner of odd pages")
            print("Detection libraries:", end="")
            if PYLIBDMTX_AVAILABLE:
                print(" pylibdmtx (DataMatrix)", end="")
            if PYZBAR_AVAILABLE:
                print(" pyzbar (QR/DataMatrix)", end="")
            print()
            if skip_white_check:
                print("White page detection: DISABLED")
            print("-" * 70)
        
        for page_num in range(total_pages):
            page_number = page_num + 1  # 1-based page numbering
            
            # Skip even pages
            if page_number % 2 == 0:
                results['skipped_even_pages'].append(page_number)
                if verbose and debug:
                    print(f"Page {page_number:4d}: Skipped (even page)")
                continue
            
            if verbose:
                print(f"Page {page_number:4d}: ", end="")
            
            try:
                # Extract page as image with specified DPI
                image = extract_page_as_image(pdf_document, page_num, dpi)
                
                # Check if page is mostly white/blank (unless disabled)
                if not skip_white_check and is_white_page(image):
                    results['skipped_white_pages'].append(page_number)
                    if verbose:
                        print("Skipped (white/blank page)")
                    continue
                
                # This page should be checked for codes
                results['checked_pages'].append(page_number)
                
                # Primary detection: top-right corner
                codes = detect_codes_in_top_right_corner(image, corner_size_ratio, decode=decode, debug=debug)
                
                # If no codes found in top-right, try fallback detection
                if not codes:
                    if debug and verbose:
                        print("No code in corner, trying full page...", end=" ")
                    codes = detect_codes_with_fallback(image, decode=decode)
                
                if codes:
                    results['pages_with_codes'].append(page_number)
                    results['code_details'][page_number] = codes
                    
                    # Process each detected code
                    for code in codes:
                        code_type = code.get('type', 'UNKNOWN')
                        decoded_value = code.get('data', '')
                        
                        # Track code types
                        if 'DATAMATRIX' in code_type:
                            if page_number not in results['datamatrix_pages']:
                                results['datamatrix_pages'].append(page_number)
                        elif 'QR' in code_type:
                            if page_number not in results['qr_pages']:
                                results['qr_pages'].append(page_number)
                        
                        # Store decoded values (if in decode mode)
                        if decode and decoded_value and 'NOT_DECODED' not in decoded_value:
                            results['code_values'][page_number] = decoded_value
                        
                        # Print detection results
                        if verbose:
                            if not decode:
                                print(f"✓ {code_type} Detected (not decoded) ({code['method']})")
                            else:
                                validation = validate_code_content(decoded_value, code_type)
                                if 'NOT_DECODED' in decoded_value:
                                    print(f"✓ {code_type} Detected (decode failed) ({code['method']})")
                                else:
                                    print(f"✓ {code_type} Found! Value: [{decoded_value}] ({code['method']}) - {validation}")
                else:
                    results['pages_without_codes'].append(page_number)
                    if verbose:
                        print(f"✗ NO CODE FOUND - Missing {'DataMatrix/QR' if decode else '2D barcode'}!")
                        
            except Exception as e:
                if verbose:
                    print(f"✗ Error: {str(e)[:50]}")
                results['pages_without_codes'].append(page_number)
        
        pdf_document.close()
        return results
        
    except Exception as e:
        print(f"Error opening PDF: {e}")
        return None

def extract_and_save_corners(pdf_path, page_numbers, output_dir="debug_corners", dpi=300, corner_size_ratio=0.2):
    """
    Extract and save top-right corners of specific pages for debugging.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        pdf_document = fitz.open(pdf_path)
        
        for page_num in page_numbers:
            if page_num <= 0 or page_num > len(pdf_document):
                print(f"Page {page_num} out of range")
                continue
                
            print(f"Extracting corner from page {page_num}...")
            
            # Extract page as image
            image = extract_page_as_image(pdf_document, page_num - 1, dpi)
            h, w = image.shape[:2]
            
            # Extract top-right corner
            corner_width = int(w * corner_size_ratio)
            corner_height = int(h * corner_size_ratio)
            start_x = w - corner_width
            start_y = 0
            
            corner_roi = image[start_y:start_y+corner_height, start_x:start_x+corner_width]
            
            # Save original corner
            corner_path = os.path.join(output_dir, f"page_{page_num}_corner_original.png")
            cv2.imwrite(corner_path, corner_roi)
            print(f"  Saved: {corner_path}")
            
            # Save processed versions
            gray_roi = cv2.cvtColor(corner_roi, cv2.COLOR_BGR2GRAY) if len(corner_roi.shape) == 3 else corner_roi
            
            # Enhanced version
            enhanced = apply_clahe(gray_roi)
            enhanced_path = os.path.join(output_dir, f"page_{page_num}_corner_enhanced.png")
            cv2.imwrite(enhanced_path, enhanced)
            
            # Binary version
            _, binary = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary_path = os.path.join(output_dir, f"page_{page_num}_corner_binary.png")
            cv2.imwrite(binary_path, binary)
            
            # Inverted version
            inverted = cv2.bitwise_not(gray_roi)
            inverted_path = os.path.join(output_dir, f"page_{page_num}_corner_inverted.png")
            cv2.imwrite(inverted_path, inverted)
            
            # Try to detect codes in this corner
            print(f"  Testing detection on extracted corner:")
            if PYLIBDMTX_AVAILABLE:
                codes = detect_datamatrix_with_pylibdmtx(gray_roi, decode=True)
                if codes:
                    print(f"    ✓ pylibdmtx found DataMatrix: {codes[0]['data']}")
                else:
                    print(f"    ✗ pylibdmtx found no DataMatrix")
            
            if PYZBAR_AVAILABLE:
                codes = detect_codes_with_pyzbar(gray_roi, decode=True)
                if codes:
                    print(f"    ✓ pyzbar found {codes[0]['type']}: {codes[0]['data']}")
                else:
                    print(f"    ✗ pyzbar found no codes")
            
        pdf_document.close()
        print(f"\nCorner images saved to {output_dir}/")
        
    except Exception as e:
        print(f"Error: {e}")

def main():
    if len(sys.argv) < 2:
        print("PDF QR Code & DataMatrix Detector - Enhanced with pylibdmtx")
        print("="*60)
        print("\nUsage: python pdf_qr_detector.py <pdf_file> [options]")
        print("\nExamples:")
        print("  python pdf_qr_detector.py document.pdf")
        print("  python pdf_qr_detector.py document.pdf --detect-only")
        print("  python pdf_qr_detector.py document.pdf --skip-white")
        print("  python pdf_qr_detector.py document.pdf 400 0.15")
        print("  python pdf_qr_detector.py document.pdf 400 0.15 --debug")
        print("  python pdf_qr_detector.py document.pdf --extract-corners 377,379")
        print("\nPositional Arguments:")
        print("  pdf_file: Path to the PDF file to analyze")
        print("  dpi: Resolution for page extraction (default: 300)")
        print("       Try 400-600 for very small DataMatrix codes")
        print("  corner_size_ratio: Size of corner region (default: 0.2 = 20%)")
        print("       Try 0.1-0.15 for codes very close to corner")
        print("\nOptions:")
        print("  --detect-only: Only detect codes, don't decode (faster)")
        print("  --skip-white: Skip white page detection (process ALL odd pages)")
        print("  --debug: Show detailed debugging information")
        print("  --extract-corners P1,P2: Extract corners of specific pages")
        print("\nLibraries:")
        if not PYLIBDMTX_AVAILABLE:
            print("  ⚠ pylibdmtx NOT installed - DataMatrix detection limited")
            print("    Install: pip install pylibdmtx")
        else:
            print("  ✓ pylibdmtx installed - Enhanced DataMatrix detection")
        if not PYZBAR_AVAILABLE:
            print("  ⚠ pyzbar NOT installed - QR detection limited")
            print("    Install: pip install pyzbar")
        else:
            print("  ✓ pyzbar installed - QR and backup DataMatrix detection")
        sys.exit(1)
    
    # Check if at least one library is available
    if not PYLIBDMTX_AVAILABLE and not PYZBAR_AVAILABLE:
        print("\nERROR: No detection libraries available!")
        print("Install at least one:")
        print("  pip install pylibdmtx  (recommended for DataMatrix)")
        print("  pip install pyzbar     (for QR codes)")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    dpi = 300
    corner_size_ratio = 0.2
    debug = False
    skip_white = False
    decode = True
    extract_corners = None
    
    # Parse optional arguments
    arg_idx = 2
    while arg_idx < len(sys.argv):
        arg = sys.argv[arg_idx]
        
        if arg == '--debug':
            debug = True
        elif arg == '--skip-white':
            skip_white = True
        elif arg == '--detect-only':
            decode = False
        elif arg.startswith('--extract-corners'):
            if '=' in arg:
                pages_str = arg.split('=')[1]
            elif arg_idx + 1 < len(sys.argv):
                arg_idx += 1
                pages_str = sys.argv[arg_idx]
            else:
                print("Error: --extract-corners requires page numbers")
                sys.exit(1)
            extract_corners = [int(p.strip()) for p in pages_str.split(',')]
        elif not arg.startswith('--'):
            # Positional arguments
            try:
                val = float(arg)
                if val > 10:  # Likely DPI
                    dpi = int(val)
                elif val < 1:  # Likely corner ratio
                    corner_size_ratio = val
                else:
                    print(f"Warning: Unclear argument {arg}, ignoring")
            except ValueError:
                print(f"Warning: Invalid argument {arg}, ignoring")
        arg_idx += 1
    
    if not Path(pdf_path).exists():
        print(f"Error: File '{pdf_path}' not found.")
        sys.exit(1)
    
    # If extract corners mode, do that instead
    if extract_corners:
        extract_and_save_corners(pdf_path, extract_corners, dpi=dpi, corner_size_ratio=corner_size_ratio)
        sys.exit(0)
    
    # Analyze the PDF
    results = analyze_pdf_for_codes(pdf_path, verbose=True, dpi=dpi, 
                                   corner_size_ratio=corner_size_ratio, 
                                   decode=decode, debug=debug,
                                   skip_white_check=skip_white)
    
    if results is None:
        sys.exit(1)
    
    # Print detailed summary
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)
    print(f"PDF File: {pdf_path}")
    print(f"Total pages: {results['total_pages']}")
    print(f"Settings: DPI={dpi}, Corner={corner_size_ratio*100}%, Mode={'Decode' if decode else 'Detect-only'}")
    if skip_white:
        print("White page detection: DISABLED")
    print("-"*70)
    
    # Statistics
    print(f"\nPage Statistics:")
    print(f"  • Even pages (skipped): {len(results['skipped_even_pages'])} pages")
    if not skip_white:
        print(f"  • White/blank pages (skipped): {len(results['skipped_white_pages'])} pages")
    print(f"  • Odd content pages (checked): {len(results['checked_pages'])} pages")
    print(f"  • Pages WITH codes: {len(results['pages_with_codes'])} pages")
    if results['datamatrix_pages']:
        print(f"    - DataMatrix codes: {len(results['datamatrix_pages'])} pages")
    if results['qr_pages']:
        print(f"    - QR codes: {len(results['qr_pages'])} pages")
    print(f"  • Pages WITHOUT codes: {len(results['pages_without_codes'])} pages")
    
    # Decoded Values (only if in decode mode)
    if decode and results['code_values']:
        print(f"\n{'='*70}")
        print("DECODED VALUES:")
        print("-"*70)
        for page_num in sorted(results['code_values'].keys()):
            value = results['code_values'][page_num]
            code_type = "DataMatrix" if page_num in results['datamatrix_pages'] else "QR"
            print(f"  Page {page_num:4d} ({code_type:10s}): {value}")
        
        # Check for patterns
        values = list(results['code_values'].values())
        unique_values = set(values)
        if len(unique_values) < len(values):
            print(f"\n  Found {len(values)} codes with {len(unique_values)} unique values")
    
    # Missing Codes Report
    if results['pages_without_codes']:
        print(f"\n{'='*70}")
        print("⚠️  MISSING CODES - PAGES WITHOUT DETECTION:")
        print("-"*70)
        
        missing = sorted(results['pages_without_codes'])
        # Show first 20 pages or all if less
        if len(missing) <= 20:
            print(f"  Pages: {missing}")
        else:
            print(f"  First 20: {missing[:20]}")
            print(f"  ... and {len(missing)-20} more")
        
        print(f"\n  Total: {len(results['pages_without_codes'])} pages missing codes")
        
        # Suggestions based on mode
        if not decode:
            print("\n  Note: Running in detect-only mode (faster but no content)")
            print("  To decode content, run without --detect-only flag")
    
    # Final verdict
    print(f"\n{'='*70}")
    if len(results['pages_without_codes']) == 0:
        print("✅ SUCCESS: All checked pages contain codes!")
        print(f"   Verified {len(results['checked_pages'])} odd pages")
        if decode:
            print(f"   Decoded {len(results['code_values'])} codes successfully")
    else:
        print(f"❌ ISSUES FOUND: {len(results['pages_without_codes'])} page(s) missing codes")
        
        print("\n📝 Troubleshooting:")
        if not PYLIBDMTX_AVAILABLE:
            print("   1. Install pylibdmtx for better DataMatrix detection:")
            print("      pip install pylibdmtx")
        if not skip_white:
            print("   2. Skip white page detection:")
            print(f"      python {sys.argv[0]} {pdf_path} --skip-white")
        if not decode:
            print("   3. Try full decode mode for better detection:")
            print(f"      python {sys.argv[0]} {pdf_path}")
        print("   4. Try higher DPI for small codes:")
        print(f"      python {sys.argv[0]} {pdf_path} 500 --skip-white")
        print("   5. Extract problem pages to examine:")
        missing_sample = ','.join(str(p) for p in results['pages_without_codes'][:3])
        print(f"      python {sys.argv[0]} {pdf_path} --extract-corners {missing_sample}")
    
    print("="*70)
    
    return len(results['pages_without_codes']) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)