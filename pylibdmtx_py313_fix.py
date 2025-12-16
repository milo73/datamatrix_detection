"""
Python 3.13 Compatibility Fix for pylibdmtx
============================================

The issue: pylibdmtx uses 'distutils' which was removed in Python 3.12+
Solution: Patch the wrapper.py file to use packaging.version instead

Three methods to fix:
1. Quick Fix - Install compatibility package
2. Manual Patch - Replace distutils in wrapper.py  
3. Forked Version - Use patched pylibdmtx
"""

# ============================================================================
# METHOD 1: QUICK FIX - Install setuptools (includes distutils backport)
# ============================================================================

"""
QUICKEST SOLUTION (30 seconds):

Run this in your terminal:
    pip install setuptools

That's it! setuptools includes a distutils backport for Python 3.12+.
Then re-run your benchmark:
    python pdf_scanner_integration.py --benchmark test.png

This is the easiest temporary fix while we wait for upstream to update.
"""

# ============================================================================
# METHOD 2: MANUAL PATCH - Replace distutils in your installation
# ============================================================================

"""
PERMANENT FIX (2 minutes):

Step 1: Find your pylibdmtx installation
The error shows it's at:
C:\\Users\\milo.van.diest\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python313\\site-packages\\pylibdmtx\\wrapper.py

Step 2: Open wrapper.py in a text editor

Step 3: Find this line (around line 8):
    from distutils.version import LooseVersion

Step 4: Replace it with:
    try:
        from packaging.version import Version as LooseVersion
    except ImportError:
        # Fallback for older Python
        try:
            from distutils.version import LooseVersion
        except ImportError:
            # Simple fallback if nothing available
            class LooseVersion:
                def __init__(self, vstring):
                    self.vstring = str(vstring)
                def __str__(self):
                    return self.vstring

Step 5: Install packaging if needed:
    pip install packaging

Step 6: Save and test:
    python pdf_scanner_integration.py --benchmark test.png
"""

# ============================================================================
# METHOD 3: AUTOMATED PATCHER SCRIPT
# ============================================================================

import sys
import os
from pathlib import Path

def find_pylibdmtx_wrapper():
    """Find the wrapper.py file in your pylibdmtx installation"""
    try:
        import pylibdmtx
        pylibdmtx_path = Path(pylibdmtx.__file__).parent
        wrapper_path = pylibdmtx_path / 'wrapper.py'
        return wrapper_path if wrapper_path.exists() else None
    except ImportError:
        return None

def backup_file(filepath):
    """Create backup before modifying"""
    backup_path = Path(str(filepath) + '.backup')
    if not backup_path.exists():
        import shutil
        shutil.copy2(filepath, backup_path)
        print(f"✓ Created backup: {backup_path}")
        return True
    else:
        print(f"ℹ Backup already exists: {backup_path}")
        return True

def patch_wrapper_file(wrapper_path):
    """Patch the wrapper.py file to fix distutils import"""
    
    print(f"\nPatching: {wrapper_path}")
    
    # Read original content
    with open(wrapper_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already patched
    if 'packaging.version' in content:
        print("✓ File already patched!")
        return True
    
    # Find and replace the distutils import
    old_import = "from distutils.version import LooseVersion"
    
    new_import = """try:
    # Python 3.12+ - distutils was removed
    from packaging.version import Version as LooseVersion
except ImportError:
    # Fallback for older environments
    try:
        from distutils.version import LooseVersion
    except ImportError:
        # Simple fallback implementation
        class LooseVersion:
            def __init__(self, vstring):
                self.vstring = str(vstring)
                # Parse version for comparison
                parts = []
                for part in str(vstring).split('.'):
                    try:
                        parts.append(int(part))
                    except ValueError:
                        parts.append(part)
                self._parts = parts
            
            def __str__(self):
                return self.vstring
            
            def __lt__(self, other):
                if not isinstance(other, LooseVersion):
                    other = LooseVersion(str(other))
                return self._parts < other._parts
            
            def __le__(self, other):
                return self < other or self == other
            
            def __eq__(self, other):
                if not isinstance(other, LooseVersion):
                    other = LooseVersion(str(other))
                return self._parts == other._parts
            
            def __ge__(self, other):
                return not self < other
            
            def __gt__(self, other):
                return not self <= other"""
    
    if old_import in content:
        # Replace the import
        patched_content = content.replace(old_import, new_import)
        
        # Write patched content
        with open(wrapper_path, 'w', encoding='utf-8') as f:
            f.write(patched_content)
        
        print("✓ Successfully patched wrapper.py!")
        print("\nChanges made:")
        print("  - Replaced distutils.version import")
        print("  - Added packaging.version as primary (Python 3.12+)")
        print("  - Added distutils fallback (older Python)")
        print("  - Added custom LooseVersion fallback")
        return True
    else:
        print(f"✗ Could not find import statement to replace")
        print(f"  Looking for: {old_import}")
        return False

def install_packaging():
    """Install packaging module if not present"""
    try:
        import packaging
        print("✓ packaging module already installed")
        return True
    except ImportError:
        print("\nInstalling packaging module...")
        import subprocess
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'packaging'])
            print("✓ Successfully installed packaging")
            return True
        except subprocess.CalledProcessError:
            print("✗ Failed to install packaging")
            print("  Please run manually: pip install packaging")
            return False

def verify_fix():
    """Verify the fix works"""
    print("\nVerifying fix...")
    try:
        # Clear any cached imports
        if 'pylibdmtx' in sys.modules:
            del sys.modules['pylibdmtx']
        if 'pylibdmtx.wrapper' in sys.modules:
            del sys.modules['pylibdmtx.wrapper']
        if 'pylibdmtx.pylibdmtx' in sys.modules:
            del sys.modules['pylibdmtx.pylibdmtx']
        
        # Try importing
        from pylibdmtx import pylibdmtx
        print("✓ pylibdmtx imports successfully!")
        
        # Try basic functionality
        from PIL import Image
        import numpy as np
        
        # Create a simple test image
        test_img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        result = pylibdmtx.decode(test_img, timeout=100)
        print("✓ decode() function works!")
        
        return True
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return False

def main():
    """Main patcher function"""
    print("="*70)
    print("Python 3.13 Compatibility Patcher for pylibdmtx")
    print("="*70)
    
    # Check Python version
    print(f"\nPython version: {sys.version}")
    py_version = sys.version_info
    
    if py_version < (3, 12):
        print("ℹ Your Python version doesn't need this patch")
        print("  (distutils is still available in Python < 3.12)")
        return
    
    # Find wrapper.py
    print("\nStep 1: Locating pylibdmtx installation...")
    wrapper_path = find_pylibdmtx_wrapper()
    
    if not wrapper_path:
        print("✗ Could not find pylibdmtx installation")
        print("  Please install it first: pip install pylibdmtx")
        return
    
    print(f"✓ Found: {wrapper_path}")
    
    # Backup
    print("\nStep 2: Creating backup...")
    if not backup_file(wrapper_path):
        print("✗ Could not create backup")
        return
    
    # Install packaging
    print("\nStep 3: Checking dependencies...")
    install_packaging()
    
    # Patch file
    print("\nStep 4: Patching wrapper.py...")
    if not patch_wrapper_file(wrapper_path):
        print("\n✗ Patching failed!")
        print("  Try METHOD 1 (Quick Fix) or METHOD 2 (Manual Patch)")
        return
    
    # Verify
    print("\nStep 5: Verifying fix...")
    if verify_fix():
        print("\n" + "="*70)
        print("SUCCESS! pylibdmtx is now Python 3.13 compatible")
        print("="*70)
        print("\nYou can now run:")
        print("  python pdf_scanner_integration.py --benchmark test.png")
    else:
        print("\n" + "="*70)
        print("PARTIAL SUCCESS - Patched but verification failed")
        print("="*70)
        print("\nThe file was patched but there might be other issues.")
        print("Try restarting your Python session and test again.")

if __name__ == '__main__':
    main()

# ============================================================================
# METHOD 4: INSTALL PATCHED VERSION FROM ALTERNATIVE FORK
# ============================================================================

"""
ALTERNATIVE: Use a modern fork

There's a modernized fork called 'pydmtxlib' that's Python 3.13 compatible:

    pip uninstall pylibdmtx
    pip install pydmtxlib

However, this might require updating your import statements from:
    from pylibdmtx.pylibdmtx import decode
to:
    from pydmtxlib import decode

So the patcher above is probably easier for your existing scripts.
"""

# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================

"""
HOW TO USE THIS PATCHER:

Option A - Run the automated patcher:
    python pylibdmtx_py313_fix.py

Option B - Quick fix (recommended):
    pip install setuptools
    # Then run your benchmark again

Option C - Manual patch:
    1. Find wrapper.py location from error message
    2. Open in text editor
    3. Replace the distutils import (see METHOD 2 above)
    4. Save and test

After applying ANY of these fixes, you should be able to run:
    python pdf_scanner_integration.py --benchmark test.png
"""
