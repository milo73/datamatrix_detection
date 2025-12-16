#!/usr/bin/env python3
"""
Create Test DataMatrix Image
============================

Generates test images with DataMatrix codes for benchmarking.
Run this first, then run the benchmark.

Usage:
    python create_test_image.py                    # Creates test.png
    python create_test_image.py custom_name.png    # Custom filename
    python create_test_image.py --multiple         # Create multiple test images
"""

import sys
from pathlib import Path

def create_test_image(filename='test.png', data=b'83065676', size='120x120'):
    """
    Create a test image with DataMatrix code.
    
    Args:
        filename: Output filename
        data: Data to encode (bytes)
        size: DataMatrix size (e.g., '120x120', '32x32')
    """
    try:
        from pylibdmtx.pylibdmtx import encode
        from PIL import Image
        
        print(f"Creating test image: {filename}")
        print(f"  Data: {data.decode('utf-8')}")
        print(f"  Size: {size}")
        
        # Encode the data
        encoded = encode(data, size=size)
        
        # Create PIL image from encoded data
        img = Image.frombytes(
            'RGB',
            (encoded.width, encoded.height),
            encoded.pixels
        )
        
        # Save
        img.save(filename)
        print(f"✓ Created: {filename} ({encoded.width}x{encoded.height})")
        return filename
        
    except ImportError as e:
        print(f"✗ Error: {e}")
        print("\nMissing dependencies. Install with:")
        print("  pip install pylibdmtx Pillow")
        return None
    except Exception as e:
        print(f"✗ Error creating image: {e}")
        return None


def create_test_suite():
    """Create multiple test images with different characteristics"""
    
    print("="*70)
    print("Creating Test Suite")
    print("="*70)
    print()
    
    test_cases = [
        # (filename, data, size, description)
        ('test.png', b'83065676', '120x120', 'Standard 8-digit invoice number'),
        ('test_small.png', b'TEST123', '32x32', 'Small DataMatrix'),
        ('test_large.png', b'INVOICE-2024-001-ABCDEF', '144x144', 'Large DataMatrix'),
        ('test_url.png', b'https://example.com/verify?id=12345', '144x144', 'URL DataMatrix'),
        ('test_simple.png', b'HELLO', '48x48', 'Simple text'),
    ]
    
    created = []
    for filename, data, size, description in test_cases:
        print(f"\n{description}")
        result = create_test_image(filename, data, size)
        if result:
            created.append(result)
    
    print("\n" + "="*70)
    print(f"Created {len(created)} test images")
    print("="*70)
    
    if created:
        print("\nYou can now run benchmarks:")
        for img in created:
            print(f"  python pdf_scanner_integration.py --benchmark {img}")
    
    return created


def create_document_corner_test():
    """Create a test image simulating a document corner with DataMatrix"""
    try:
        from pylibdmtx.pylibdmtx import encode
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        
        print("\nCreating document corner simulation...")
        
        # Create a document-sized image (A4 at 300 DPI = 2480x3508)
        # We'll create a smaller version for testing
        doc_width, doc_height = 1240, 1754  # Half size for faster testing
        
        # Create white document
        doc = Image.new('RGB', (doc_width, doc_height), color='white')
        draw = ImageDraw.Draw(doc)
        
        # Add some document-like content
        # Title
        try:
            font_large = ImageFont.truetype("arial.ttf", 40)
            font_normal = ImageFont.truetype("arial.ttf", 20)
        except (IOError, OSError):
            font_large = ImageFont.load_default()
            font_normal = ImageFont.load_default()
        
        # Add document header
        draw.text((100, 100), "INVOICE", fill='black', font=font_large)
        draw.text((100, 160), "Invoice Number: 83065676", fill='black', font=font_normal)
        draw.text((100, 200), "Date: 2024-01-15", fill='black', font=font_normal)
        
        # Add some lines to simulate content
        for i in range(10):
            y = 300 + i * 40
            draw.text((100, y), f"Line item {i+1}: Product description", fill='black', font=font_normal)
        
        # Generate DataMatrix code
        data = b'83065676'
        encoded = encode(data, size='120x120')
        
        # Create DataMatrix image
        dm_img = Image.frombytes('RGB', (encoded.width, encoded.height), encoded.pixels)
        
        # Paste DataMatrix in top-right corner (typical location)
        margin = 50
        x_pos = doc_width - encoded.width - margin
        y_pos = margin
        
        doc.paste(dm_img, (x_pos, y_pos))
        
        # Save
        filename = 'test_document_corner.png'
        doc.save(filename)
        print(f"✓ Created: {filename}")
        print(f"  Simulates A4 document with DataMatrix in top-right corner")
        print(f"  Size: {doc_width}x{doc_height}")
        print(f"  DataMatrix at: ({x_pos}, {y_pos})")
        
        return filename
        
    except Exception as e:
        print(f"✗ Error creating document: {e}")
        return None


def extract_from_pdf_for_testing(pdf_path, page_num=0, output='test_from_pdf.png'):
    """Extract a page from your actual PDF for testing"""
    try:
        import fitz  # PyMuPDF
        from PIL import Image
        import numpy as np
        
        print(f"\nExtracting page {page_num+1} from: {pdf_path}")
        
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        
        # Render at 300 DPI
        mat = fitz.Matrix(300/72, 300/72)
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Save
        img.save(output)
        doc.close()
        
        print(f"✓ Extracted: {output}")
        print(f"  Size: {pix.width}x{pix.height}")
        return output
        
    except ImportError:
        print("✗ PyMuPDF not installed. Install with: pip install PyMuPDF")
        return None
    except Exception as e:
        print(f"✗ Error extracting from PDF: {e}")
        return None


def main():
    """Main function"""
    print("="*70)
    print("DataMatrix Test Image Generator")
    print("="*70)
    print()
    
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        
        if arg == '--multiple' or arg == '-m':
            # Create test suite
            create_test_suite()
            
        elif arg == '--document' or arg == '-d':
            # Create document simulation
            create_test_image('test.png')  # Standard test
            create_document_corner_test()  # Document corner test
            
        elif arg.endswith('.pdf'):
            # Extract from PDF
            page_num = int(sys.argv[2]) if len(sys.argv) > 2 else 0
            output = sys.argv[3] if len(sys.argv) > 3 else 'test_from_pdf.png'
            result = extract_from_pdf_for_testing(arg, page_num, output)
            if result:
                print(f"\nYou can now benchmark with:")
                print(f"  python pdf_scanner_integration.py --benchmark {result}")
                
        elif arg.endswith('.png') or arg.endswith('.jpg'):
            # Custom filename
            create_test_image(arg)
            print(f"\nYou can now benchmark with:")
            print(f"  python pdf_scanner_integration.py --benchmark {arg}")
            
        else:
            print(f"Unknown option: {arg}")
            print_usage()
    else:
        # Default: create simple test.png
        result = create_test_image('test.png')
        
        if result:
            print(f"\n✓ Test image created successfully!")
            print(f"\nNext steps:")
            print(f"  1. View the image: test.png")
            print(f"  2. Run benchmark:")
            print(f"     python pdf_scanner_integration.py --benchmark test.png")
            print(f"\nOr create more test images:")
            print(f"  python create_test_image.py --multiple")
            print(f"  python create_test_image.py --document")


def print_usage():
    """Print usage information"""
    print("""
Usage:
  python create_test_image.py                    # Create test.png
  python create_test_image.py mytest.png         # Custom filename
  python create_test_image.py --multiple         # Create multiple test images
  python create_test_image.py --document         # Create document corner simulation
  python create_test_image.py input.pdf          # Extract page from PDF
  python create_test_image.py input.pdf 5        # Extract page 6 from PDF

Examples:
  # Quick start
  python create_test_image.py
  python pdf_scanner_integration.py --benchmark test.png
  
  # Test with document corner simulation
  python create_test_image.py --document
  python pdf_scanner_integration.py --benchmark test_document_corner.png
  
  # Extract from your actual PDF
  python create_test_image.py your_invoice.pdf 0
  python pdf_scanner_integration.py --benchmark test_from_pdf.png
    """)


if __name__ == '__main__':
    main()
