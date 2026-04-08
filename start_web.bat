@echo off
title PDF DataMatrix Scanner - Startup
echo ============================================
echo   PDF DataMatrix ^& QR Code Scanner
echo   Startup Check
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [FAIL] Python is not installed or not in PATH.
    echo        Download from https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PYVER=%%i
echo [OK]   %PYVER%

:: Check if virtual environment exists, create if not
if not exist "venv\Scripts\activate.bat" (
    echo.
    echo [INFO] Virtual environment not found. Creating one...
    python -m venv venv
    if errorlevel 1 (
        echo [FAIL] Could not create virtual environment.
        pause
        exit /b 1
    )
    echo [OK]   Virtual environment created.
)

:: Activate virtual environment
call venv\Scripts\activate.bat
echo [OK]   Virtual environment activated.

:: Install/check dependencies
echo.
echo Checking dependencies...
pip install -q -r requirements.txt >nul 2>&1
if errorlevel 1 (
    echo [FAIL] Could not install dependencies from requirements.txt.
    echo        Try running: pip install -r requirements.txt
    pause
    exit /b 1
)
echo [OK]   All dependencies installed.

:: Check critical packages
echo.
echo Verifying critical packages...

python -c "import setuptools" >nul 2>&1
if errorlevel 1 (
    echo [FAIL] setuptools - run: pip install setuptools
    pause
    exit /b 1
)
echo [OK]   setuptools

python -c "import pylibdmtx" >nul 2>&1
if errorlevel 1 (
    echo [FAIL] pylibdmtx - run: pip install pylibdmtx
    echo        Also ensure libdmtx is installed on your system.
    pause
    exit /b 1
)
echo [OK]   pylibdmtx

python -c "import pyzbar" >nul 2>&1
if errorlevel 1 (
    echo [FAIL] pyzbar - run: pip install pyzbar
    echo        Also ensure zbar DLL is installed on your system.
    pause
    exit /b 1
)
echo [OK]   pyzbar

python -c "import fitz" >nul 2>&1
if errorlevel 1 (
    echo [FAIL] PyMuPDF - run: pip install PyMuPDF
    pause
    exit /b 1
)
echo [OK]   PyMuPDF

python -c "import cv2" >nul 2>&1
if errorlevel 1 (
    echo [FAIL] opencv-python - run: pip install opencv-python
    pause
    exit /b 1
)
echo [OK]   opencv-python

python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo [FAIL] streamlit - run: pip install streamlit
    pause
    exit /b 1
)
echo [OK]   streamlit

:: Check that app_web.py exists
if not exist "app_web.py" (
    echo.
    echo [FAIL] app_web.py not found in current directory.
    echo        Make sure you run this batch file from the project folder.
    pause
    exit /b 1
)

:: All checks passed - start the app
echo.
echo ============================================
echo   All checks passed! Starting web app...
echo   The browser will open automatically.
echo   Press Ctrl+C in this window to stop.
echo ============================================
echo.
streamlit run app_web.py
