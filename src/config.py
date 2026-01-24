"""Configuration settings."""

import os
from pathlib import Path
from typing import Literal

# Base directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# ==============================================================================
# OCR Configuration
# ==============================================================================

# OCR Backend: "paddleocr" (recommended) or "tesseract" (legacy)
OCR_BACKEND: Literal["paddleocr", "tesseract"] = os.environ.get("OCR_BACKEND", "paddleocr").lower()

# PaddleOCR settings
PADDLEOCR_USE_GPU = os.environ.get("PADDLEOCR_USE_GPU", "true").lower() == "true"
PADDLEOCR_LANG = os.environ.get("PADDLEOCR_LANG", "en")

# Tesseract settings (legacy fallback)
TESSERACT_PATH = os.environ.get(
    "TESSERACT_PATH",
    r"C:\Users\jloya\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
)

# Poppler settings (for PDF to image conversion)
POPPLER_PATH = os.environ.get("POPPLER_PATH", r"C:\poppler\Library\bin")

# Check alternate poppler structure
if os.name == 'nt' and not os.path.exists(os.path.join(POPPLER_PATH, "pdftoppm.exe")):
    POPPLER_PATH = r"C:\poppler\bin"

# OCR confidence threshold (0-100 for tesseract, 0-1 for paddleocr internally)
OCR_CONFIDENCE_THRESHOLD = int(os.environ.get("OCR_CONFIDENCE_THRESHOLD", "30"))


def get_ocr_config() -> dict:
    """Get OCR configuration dictionary."""
    return {
        "ocr_backend": OCR_BACKEND,
        "use_gpu": PADDLEOCR_USE_GPU,
        "lang": PADDLEOCR_LANG,
        "confidence_threshold": OCR_CONFIDENCE_THRESHOLD,
        "tesseract_path": TESSERACT_PATH if OCR_BACKEND == "tesseract" else None,
        "poppler_path": POPPLER_PATH if os.name == 'nt' else None,
    }


# Verify paths exist
def verify_dependencies():
    """Check if required tools are available."""
    issues = []
    
    if OCR_BACKEND == "tesseract":
        if os.name == 'nt' and not os.path.exists(TESSERACT_PATH):
            issues.append(f"Tesseract not found at: {TESSERACT_PATH}")
    elif OCR_BACKEND == "paddleocr":
        try:
            import paddle
            if PADDLEOCR_USE_GPU and not paddle.is_compiled_with_cuda():
                issues.append("PaddleOCR GPU requested but paddlepaddle-gpu not installed")
        except ImportError:
            issues.append("PaddlePaddle not installed. Run: pip install paddlepaddle==3.2.0 paddleocr==2.7.0.3")
    
    if os.name == 'nt':
        pdftoppm = os.path.join(POPPLER_PATH, "pdftoppm.exe")
        if not os.path.exists(pdftoppm):
            issues.append(f"Poppler (pdftoppm.exe) not found at: {POPPLER_PATH}")
    
    return issues