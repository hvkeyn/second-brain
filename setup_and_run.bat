@echo off
chcp 65001 >nul
REM ============================================
REM  Second Brain - Setup and Run Script
REM  Automatically installs dependencies and runs the application
REM ============================================

echo ========================================
echo   Second Brain - Setup and Launch
echo ========================================
echo.

REM Check if Python is installed
echo [1/6] Checking for Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo Please install Python 3.9 or higher from https://www.python.org/
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Python found: %PYTHON_VERSION%
echo.

REM Check required files
echo [2/6] Checking required files...
set MISSING_FILES=0

if not exist "SecondBrainBackend.py" (
    echo [ERROR] SecondBrainBackend.py not found!
    set MISSING_FILES=1
)

if not exist "SecondBrainFrontend.py" (
    echo [ERROR] SecondBrainFrontend.py not found!
    set MISSING_FILES=1
)

if not exist "config.json" (
    echo [ERROR] config.json not found!
    set MISSING_FILES=1
)

if not exist "requirements.txt" (
    echo [ERROR] requirements.txt not found!
    set MISSING_FILES=1
)

if %MISSING_FILES%==1 (
    echo.
    echo [ERROR] Missing required files! Please download all project files.
    pause
    exit /b 1
)

echo [OK] All required files present
echo.

REM Create or check virtual environment
echo [3/6] Checking virtual environment...
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment!
        echo Try: python -m pip install --upgrade pip
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment already exists
)
echo.

REM Activate virtual environment
echo [4/6] Activating virtual environment...
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment activation script not found!
    echo Try deleting the 'venv' folder and running this script again.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment!
    pause
    exit /b 1
)
echo [OK] Virtual environment activated
echo.

REM Upgrade pip
echo [5/6] Upgrading pip...
python -m pip install --upgrade pip --quiet
if errorlevel 1 (
    echo [WARNING] Failed to upgrade pip, continuing...
) else (
    echo [OK] pip upgraded
)
echo.

REM Install dependencies
echo [6/6] Installing dependencies...
echo This may take several minutes on first run (downloading ~2-3 GB)...
echo.

pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [WARNING] Some dependencies may have failed to install
    echo Common fixes:
    echo   - Check your internet connection
    echo   - Try running the script again
    echo   - Manually install: pip install -r requirements.txt
    echo.
    echo Press any key to try launching anyway...
    pause >nul
) else (
    echo.
    echo [OK] All dependencies installed successfully
)

echo.
echo ========================================
echo   Launching Second Brain...
echo ========================================
echo.

REM Launch application
python SecondBrainFrontend.py

echo.
echo ========================================
echo   Second Brain closed
echo ========================================
echo.
echo To run again, simply execute: setup_and_run.bat
echo.
pause

