#!/bin/bash
# ============================================
#  Second Brain - Setup and Run Script
#  Automatically installs dependencies and runs the application
# ============================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Second Brain - Setup and Launch${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# Check if Python is installed
echo -e "${YELLOW}[1/6] Checking for Python...${NC}"
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo -e "${RED}[ERROR] Python not found!${NC}"
        echo -e "${RED}Please install Python 3.9 or higher${NC}"
        echo -e "${YELLOW}  - Ubuntu/Debian: sudo apt install python3 python3-venv python3-pip${NC}"
        echo -e "${YELLOW}  - macOS: brew install python3${NC}"
        echo ""
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
echo -e "${GREEN}[OK] Python found: $PYTHON_VERSION${NC}"

# Check Python version
VERSION_CHECK=$($PYTHON_CMD -c 'import sys; print(1 if sys.version_info >= (3, 9) else 0)')
if [ "$VERSION_CHECK" -eq "0" ]; then
    echo -e "${RED}[ERROR] Python 3.9 or higher is required${NC}"
    echo ""
    exit 1
fi

echo ""

# Check required files
echo -e "${YELLOW}[2/6] Checking required files...${NC}"
MISSING_FILES=0

for file in "SecondBrainBackend.py" "SecondBrainFrontend.py" "config.json" "requirements.txt"; do
    if [ ! -f "$file" ]; then
        echo -e "${RED}[ERROR] $file not found!${NC}"
        MISSING_FILES=1
    fi
done

if [ $MISSING_FILES -eq 1 ]; then
    echo ""
    echo -e "${RED}[ERROR] Missing required files! Please download all project files.${NC}"
    echo ""
    exit 1
fi

echo -e "${GREEN}[OK] All required files present${NC}"
echo ""

# Create or check virtual environment
echo -e "${YELLOW}[3/6] Checking virtual environment...${NC}"
if [ ! -d "venv" ]; then
    echo -e "${CYAN}Creating virtual environment...${NC}"
    $PYTHON_CMD -m venv venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}[ERROR] Failed to create virtual environment!${NC}"
        echo -e "${YELLOW}Try: $PYTHON_CMD -m pip install --upgrade pip${NC}"
        echo ""
        exit 1
    fi
    echo -e "${GREEN}[OK] Virtual environment created${NC}"
else
    echo -e "${GREEN}[OK] Virtual environment already exists${NC}"
fi

echo ""

# Activate virtual environment
echo -e "${YELLOW}[4/6] Activating virtual environment...${NC}"
if [ ! -f "venv/bin/activate" ]; then
    echo -e "${RED}[ERROR] Virtual environment activation script not found!${NC}"
    echo -e "${YELLOW}Try deleting the 'venv' folder and running this script again.${NC}"
    echo ""
    exit 1
fi

source venv/bin/activate
if [ $? -ne 0 ]; then
    echo -e "${RED}[ERROR] Failed to activate virtual environment!${NC}"
    echo ""
    exit 1
fi
echo -e "${GREEN}[OK] Virtual environment activated${NC}"
echo ""

# Upgrade pip
echo -e "${YELLOW}[5/6] Upgrading pip...${NC}"
pip install --upgrade pip --quiet > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}[OK] pip upgraded${NC}"
else
    echo -e "${YELLOW}[WARNING] Failed to upgrade pip, continuing...${NC}"
fi
echo ""

# Install dependencies
echo -e "${YELLOW}[6/6] Installing dependencies...${NC}"
echo -e "${CYAN}This may take several minutes on first run (downloading ~2-3 GB)...${NC}"
echo ""

pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo ""
    echo -e "${YELLOW}[WARNING] Some dependencies may have failed to install${NC}"
    echo -e "${CYAN}Common fixes:${NC}"
    echo -e "${WHITE}  - Check your internet connection${NC}"
    echo -e "${WHITE}  - Try running the script again${NC}"
    echo -e "${WHITE}  - Manually install: pip install -r requirements.txt${NC}"
    echo ""
    echo -e "${YELLOW}Press any key to try launching anyway...${NC}"
    read -n 1 -s -r
else
    echo ""
    echo -e "${GREEN}[OK] All dependencies installed successfully${NC}"
fi

echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Launching Second Brain...${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# Launch application
if [ -f "SecondBrainFrontend.py" ]; then
    python SecondBrainFrontend.py
else
    echo -e "${RED}[ERROR] SecondBrainFrontend.py not found!${NC}"
    echo ""
    exit 1
fi

echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Second Brain closed${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""
echo -e "${WHITE}To run again, simply execute: ./setup_and_run.sh${NC}"
echo ""

