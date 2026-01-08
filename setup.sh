#!/bin/bash
# Quick setup script for UDS LayoutLM

echo "================================================"
echo "UDS LayoutLM - Setup Script"
echo "================================================"

# Check Python
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.8+"
    exit 1
fi

echo "✅ Python found: $(python --version)"

# Create virtual environment (optional)
read -p "Create virtual environment? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python -m venv venv
    source venv/bin/activate || . venv/Scripts/activate
    echo "✅ Virtual environment created and activated"
fi

# Install dependencies
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Check Poppler
if command -v pdftoppm &> /dev/null; then
    echo "✅ Poppler found"
else
    echo "⚠️  Poppler not found. Install with:"
    echo "   - Windows: choco install poppler"
    echo "   - Linux: sudo apt-get install poppler-utils"
    echo "   - macOS: brew install poppler"
fi

# Check Tesseract
if command -v tesseract &> /dev/null; then
    echo "✅ Tesseract found: $(tesseract --version | head -n1)"
else
    echo "⚠️  Tesseract not found. Install with:"
    echo "   - Windows: choco install tesseract"
    echo "   - Linux: sudo apt-get install tesseract-ocr"
    echo "   - macOS: brew install tesseract"
fi

echo ""
echo "================================================"
echo "Setup complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Place PDFs in data/raw_pdfs/"
echo "2. Run: python scripts/export_for_labeling.py --process-pdfs ./data/raw_pdfs"
echo "3. Start Label Studio and begin labeling"
echo ""
echo "See README.md for detailed instructions."
