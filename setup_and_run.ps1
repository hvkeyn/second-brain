# ============================================
#  Second Brain - Setup and Run Script
#  Automatically installs dependencies and runs the application
# ============================================

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Second Brain - Setup and Launch" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
Write-Host "[1/6] Checking for Python..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "[OK] Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python not found!" -ForegroundColor Red
    Write-Host "Please install Python 3.9 or higher from https://www.python.org/" -ForegroundColor Red
    Write-Host "Make sure to check 'Add Python to PATH' during installation" -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

# Check Python version
$versionString = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
$versionParts = $versionString.Split('.')
$major = [int]$versionParts[0]
$minor = [int]$versionParts[1]

if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 9)) {
    Write-Host "[ERROR] Python 3.9 or higher is required. Current version: $versionString" -ForegroundColor Red
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""

# Check required files
Write-Host "[2/6] Checking required files..." -ForegroundColor Yellow
$missingFiles = $false

$requiredFiles = @(
    "SecondBrainBackend.py",
    "SecondBrainFrontend.py",
    "config.json",
    "requirements.txt"
)

foreach ($file in $requiredFiles) {
    if (-Not (Test-Path $file)) {
        Write-Host "[ERROR] $file not found!" -ForegroundColor Red
        $missingFiles = $true
    }
}

if ($missingFiles) {
    Write-Host ""
    Write-Host "[ERROR] Missing required files! Please download all project files." -ForegroundColor Red
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "[OK] All required files present" -ForegroundColor Green
Write-Host ""

# Create or check virtual environment
Write-Host "[3/6] Checking virtual environment..." -ForegroundColor Yellow
if (-Not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to create virtual environment!" -ForegroundColor Red
        Write-Host "Try: python -m pip install --upgrade pip" -ForegroundColor Yellow
        Write-Host ""
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host "[OK] Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "[OK] Virtual environment already exists" -ForegroundColor Green
}

Write-Host ""

# Activate virtual environment
Write-Host "[4/6] Activating virtual environment..." -ForegroundColor Yellow
if (-Not (Test-Path "venv\Scripts\Activate.ps1")) {
    Write-Host "[ERROR] Virtual environment activation script not found!" -ForegroundColor Red
    Write-Host "Try deleting the 'venv' folder and running this script again." -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

& ".\venv\Scripts\Activate.ps1"
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to activate virtual environment!" -ForegroundColor Red
    Write-Host "You may need to run: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "[OK] Virtual environment activated" -ForegroundColor Green
Write-Host ""

# Upgrade pip
Write-Host "[5/6] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet 2>&1 | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] pip upgraded" -ForegroundColor Green
} else {
    Write-Host "[WARNING] Failed to upgrade pip, continuing..." -ForegroundColor Yellow
}
Write-Host ""

# Install dependencies
Write-Host "[6/6] Installing dependencies..." -ForegroundColor Yellow
Write-Host "This may take several minutes on first run (downloading ~2-3 GB)..." -ForegroundColor Cyan
Write-Host ""

pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[WARNING] Some dependencies may have failed to install" -ForegroundColor Yellow
    Write-Host "Common fixes:" -ForegroundColor Cyan
    Write-Host "  - Check your internet connection" -ForegroundColor White
    Write-Host "  - Try running the script again" -ForegroundColor White
    Write-Host "  - Manually install: pip install -r requirements.txt" -ForegroundColor White
    Write-Host ""
    Write-Host "Press any key to try launching anyway..." -ForegroundColor Yellow
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
} else {
    Write-Host ""
    Write-Host "[OK] All dependencies installed successfully" -ForegroundColor Green
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Launching Second Brain..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Launch application
if (Test-Path "SecondBrainFrontend.py") {
    python SecondBrainFrontend.py
} else {
    Write-Host "[ERROR] SecondBrainFrontend.py not found!" -ForegroundColor Red
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Second Brain closed" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To run again, simply execute: .\setup_and_run.ps1" -ForegroundColor White
Write-Host ""
Read-Host "Press Enter to exit"

