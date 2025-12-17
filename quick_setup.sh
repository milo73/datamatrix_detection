#!/bin/bash
# Quick setup script voor DataMatrix Detection

echo "Installing system libraries..."
brew install libdmtx zbar

echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing Python packages..."
pip install pylibdmtx pyzbar PyMuPDF opencv-python numpy

echo ""
echo "Setup complete! To use the scripts:"
echo "1. Activate environment: source venv/bin/activate"
echo "2. Run detector: python detector.py your_file.pdf"
echo "3. Deactivate when done: deactivate"
