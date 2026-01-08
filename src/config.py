"""Configuration settings."""

import os
from pathlib import Path

# Base directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Windows tool paths
TESSERACT_PATH = r"C:\Users\jloya\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\poppler\Library\bin"

# Check alternate poppler structure
if not os.path.exists(os.path.join(POPPLER_PATH, "pdftoppm.exe")):
    POPPLER_PATH = r"C:\poppler\bin"

# Verify paths exist
def verify_dependencies():
    """Check if required tools are available."""
    issues = []
    
    if not os.path.exists(TESSERACT_PATH):
        issues.append(f"Tesseract not found at: {TESSERACT_PATH}")
    
    pdftoppm = os.path.join(POPPLER_PATH, "pdftoppm.exe")
    if not os.path.exists(pdftoppm):
        issues.append(f"Poppler (pdftoppm.exe) not found at: {POPPLER_PATH}")
    
    return issues