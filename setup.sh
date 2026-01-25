#!/bin/bash
# Complete Setup and Deployment Script
# For Hydroponic Agriculture CNN-PSO ML Project

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║   HYDROPONIC AGRICULTURE ML OPTIMIZATION - AUTOMATED SETUP                 ║"
echo "║   CNN-PSO Hybrid Deep Learning System                                      ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "[1/7] Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found. Please install Python 3.8+${NC}"
    exit 1
fi
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"

# Create virtual environment
echo ""
echo "[2/7] Creating virtual environment..."
if [ -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment already exists${NC}"
else
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo ""
echo "[3/7] Activating virtual environment..."
source venv/bin/activate 2>/dev/null || . venv/Scripts/activate 2>/dev/null
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Upgrade pip
echo ""
echo "[4/7] Upgrading pip..."
python -m pip install --upgrade pip -q
echo -e "${GREEN}✓ Pip upgraded${NC}"

# Install requirements
echo ""
echo "[5/7] Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt -q
    echo -e "${GREEN}✓ All dependencies installed${NC}"
else
    echo -e "${RED}✗ requirements.txt not found${NC}"
    exit 1
fi

# Create necessary directories
echo ""
echo "[6/7] Creating project directories..."
mkdir -p models results logs templates static data
echo -e "${GREEN}✓ Directories created${NC}"

# Check for data
echo ""
echo "[7/7] Checking for IoT dataset..."
if [ -f "IoTData-Raw.csv" ]; then
    RECORDS=$(tail -1 IoTData-Raw.csv | wc -l)
    echo -e "${GREEN}✓ IoTData-Raw.csv found${NC}"
else
    echo -e "${YELLOW}⚠ IoTData-Raw.csv not found in current directory${NC}"
    echo "  Please ensure IoTData-Raw.csv is in the project root directory"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                    SETUP COMPLETED SUCCESSFULLY                            ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📋 NEXT STEPS:"
echo ""
echo "1. TRAIN THE MODEL:"
echo "   python train.py"
echo ""
echo "2. START WEB INTERFACE:"
echo "   python flask_app.py"
echo ""
echo "3. OPEN IN BROWSER:"
echo "   http://localhost:5000"
echo ""
echo "4. VIEW RESULTS:"
echo "   - Check 'results/' directory for visualization graphs"
echo "   - Check 'models/' directory for trained model files"
echo ""
echo -e "${GREEN}Ready to begin training!${NC}"
echo ""
