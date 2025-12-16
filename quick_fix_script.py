#!/usr/bin/env python3
"""
Quick Fix for pylibdmtx on Python 3.13
======================================

This script automatically fixes the distutils import error.
Just run: python quick_fix.py

It will:
1. Install required dependencies
2. Patch the wrapper.py file
3. Verify everything works
"""

import sys
import subprocess
from pathlib import Path

def quick_fix():
    """One-command fix for Python 3.13 compatibility"""
    
    print("="*70)
    print("QUICK FIX: pylibdmtx Python 3.13 Compatibility")
    print("="*70)
    print()
    
    # Step 1: Install setuptools (easiest fix)
    print("Step 1: Installing setuptools (includes distutils backport)...")
    try:
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', '--upgrade', 'setuptools'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("✓ setuptools installed")
    except:
        print("⚠ Could not install setuptools automatically")
        print("  Run manually: pip install setuptools")
    
    # Step 2: Install packaging (better alternative)
    print("\nStep 2: Installing packaging module...")
    try:
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', 'packaging'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("✓ packaging installed")
    except:
        print("⚠ Could not install packaging")
    
    # Step 3: Test if it works
    print("\nStep 3: Testing pylibdmtx import...")
    try:
        # Clear cache
        for mod in list(sys.modules.keys()):
            if 'pylibdmtx' in mod:
                del sys.modules[mod]
        
        from pylibdmtx import pylibdmtx
        print("✓ pylibdmtx imports successfully!")
        
        # Test decode function
        import numpy as np
        test_img = np.ones((50, 50, 3), dtype=np.uint8) * 255
        pylibdmtx.decode(test_img, timeout=50)
        print("✓ decode() function works!")
        
        print("\n" + "="*70)
        print("SUCCESS! You can now run your benchmark:")
        print("  python pdf_scanner_integration.py --benchmark test.png")
        print("="*70)
        return True
        
    except Exception as e:
        print(f"✗ Still getting error: {e}")
        print("\nTrying advanced patch...")
        return patch_wrapper_file()

def patch_wrapper_file():
    """Advanced patching if quick fix doesn't work"""
    try:
        import pylibdmtx
        wrapper_path = Path(pylibdmtx.__file__).parent / 'wrapper.py'
        
        print(f"\nPatching: {wrapper_path}")
        
        # Backup
        backup_path = Path(str(wrapper_path) + '.backup')
        if not backup_path.exists():
            import shutil
            shutil.copy2(wrapper_path, backup_path)
            print(f"✓ Backup created")
        
        # Read and patch
        with open(wrapper_path, 'r') as f:
            content = f.read()
        
        if 'packaging.version' in content:
            print("✓ Already patched!")
            return True
        
        old = "from distutils.version import LooseVersion"
        new = """try:
    from packaging.version import Version as LooseVersion
except ImportError:
    try:
        from distutils.version import LooseVersion
    except:
        class LooseVersion:
            def __init__(self, v):
                self.vstring = str(v)
            def __str__(self):
                return self.vstring"""
        
        if old in content:
            content = content.replace(old, new)
            with open(wrapper_path, 'w') as f:
                f.write(content)
            print("✓ Patched successfully!")
            
            # Test again
            for mod in list(sys.modules.keys()):
                if 'pylibdmtx' in mod:
                    del sys.modules[mod]
            
            from pylibdmtx import pylibdmtx
            print("✓ Import works after patch!")
            
            print("\n" + "="*70)
            print("SUCCESS! Fixed with advanced patch")
            print("="*70)
            return True
        else:
            print("✗ Could not find import to patch")
            return False
            
    except Exception as e:
        print(f"✗ Advanced patch failed: {e}")
        print("\nManual fix required:")
        print("1. Run: pip install setuptools")
        print("2. Or see: pylibdmtx_py313_fix.py for detailed instructions")
        return False

if __name__ == '__main__':
    success = quick_fix()
    sys.exit(0 if success else 1)
