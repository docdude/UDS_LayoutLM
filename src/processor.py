"""PDF processing and OCR utilities."""

import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json

from pdf2image import convert_from_path
from PIL import Image
import pytesseract


# Configure Tesseract path for Windows
TESSERACT_PATH = r"C:\Users\jloya\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Poppler path for Windows
POPPLER_PATH = r"C:\poppler\Library\bin"
if not os.path.exists(POPPLER_PATH):
    # Try alternate structure
    POPPLER_PATH = r"C:\poppler\bin"


@dataclass
class ProcessedPage:
    """A processed document page."""
    image: Image.Image
    words: List[str]
    boxes: List[List[int]]  # Normalized 0-1000
    page_num: int
    source_file: str
    raw_boxes: List[List[int]]  # Original pixel coordinates


def find_poppler_path() -> Optional[str]:
    """Find Poppler installation on Windows."""
    if os.name != 'nt':  # Not Windows
        return None
    
    # Check common installation paths
    common_paths = [
        r"C:\poppler\Library\bin",
        r"C:\poppler\bin",
        r"C:\Program Files\poppler\Library\bin",
        r"C:\Program Files\poppler\bin",
        r"C:\Program Files (x86)\poppler\Library\bin",
        r"C:\tools\poppler\Library\bin",
    ]
    
    for path in common_paths:
        pdftoppm = os.path.join(path, "pdftoppm.exe")
        if os.path.exists(pdftoppm):
            return path
    
    return None


class PDFProcessor:
    """Process PDFs for LayoutLMv3."""
    
    def __init__(
        self, 
        dpi: int = 300, 
        confidence_threshold: int = 30,
        poppler_path: Optional[str] = None,
        tesseract_path: Optional[str] = None
    ):
        self.dpi = dpi
        self.confidence_threshold = confidence_threshold
        
        # Set up Poppler
        self.poppler_path = poppler_path or find_poppler_path()
        if self.poppler_path:
            print(f"Using Poppler from: {self.poppler_path}")
        elif os.name == 'nt':  # Only warn on Windows
            print("⚠️  Poppler path not found - PDF conversion may fail")
        
        # Set up Tesseract
        if tesseract_path and os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            print(f"Using Tesseract from: {tesseract_path}")
        elif os.path.exists(TESSERACT_PATH):
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
            print(f"Using Tesseract from: {TESSERACT_PATH}")
    
    def process_pdf(self, pdf_path: str) -> List[ProcessedPage]:
        """Process all pages of a PDF."""
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        # Convert PDF to images
        convert_kwargs = {"dpi": self.dpi}
        if self.poppler_path:
            convert_kwargs["poppler_path"] = self.poppler_path
        
        try:
            images = convert_from_path(str(pdf_path), **convert_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"Failed to convert PDF '{pdf_path}': {e}\n"
                f"Poppler path: {self.poppler_path}\n"
                "Make sure Poppler is installed correctly."
            ) from e
        
        pages = []
        for page_num, image in enumerate(images):
            words, boxes, raw_boxes = self._ocr_with_boxes(image)
            pages.append(ProcessedPage(
                image=image,
                words=words,
                boxes=boxes,
                page_num=page_num,
                source_file=str(pdf_path),
                raw_boxes=raw_boxes
            ))
        
        return pages
    
    def process_image(self, image_path: str) -> ProcessedPage:
        """Process a single image."""
        image = Image.open(image_path).convert("RGB")
        words, boxes, raw_boxes = self._ocr_with_boxes(image)
        
        return ProcessedPage(
            image=image,
            words=words,
            boxes=boxes,
            page_num=0,
            source_file=image_path,
            raw_boxes=raw_boxes
        )
    
    def _ocr_with_boxes(
        self, image: Image.Image
    ) -> Tuple[List[str], List[List[int]], List[List[int]]]:
        """Extract text with normalized bounding boxes."""
        try:
            ocr_data = pytesseract.image_to_data(
                image, output_type=pytesseract.Output.DICT
            )
        except Exception as e:
            raise RuntimeError(
                f"OCR failed: {e}\n"
                f"Tesseract cmd: {pytesseract.pytesseract.tesseract_cmd}\n"
                "Make sure Tesseract is installed correctly."
            ) from e
        
        width, height = image.size
        words, boxes, raw_boxes = [], [], []
        
        for i in range(len(ocr_data["text"])):
            text = ocr_data["text"][i].strip()
            conf = ocr_data["conf"][i]
            
            # Handle confidence as string or int
            if isinstance(conf, str):
                try:
                    conf = int(conf)
                except ValueError:
                    conf = 0
            
            if text and conf >= self.confidence_threshold:
                left = ocr_data["left"][i]
                top = ocr_data["top"][i]
                w = ocr_data["width"][i]
                h = ocr_data["height"][i]
                
                # Raw pixel coordinates
                raw_box = [left, top, left + w, top + h]
                raw_boxes.append(raw_box)
                
                # Normalized to 0-1000 (LayoutLMv3 format)
                norm_box = [
                    int(1000 * left / width),
                    int(1000 * top / height),
                    int(1000 * (left + w) / width),
                    int(1000 * (top + h) / height)
                ]
                # Clamp values
                norm_box = [max(0, min(1000, v)) for v in norm_box]
                
                words.append(text)
                boxes.append(norm_box)
        
        return words, boxes, raw_boxes
    
    def save_for_labeling(
        self, 
        pdf_path: str, 
        output_dir: str,
        image_format: str = "png"
    ) -> Dict:
        """Save processed PDF for Label Studio annotation."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        pages = self.process_pdf(pdf_path)
        pdf_name = Path(pdf_path).stem
        
        tasks = []
        for page in pages:
            # Save image
            img_filename = f"{pdf_name}_page{page.page_num}.{image_format}"
            img_path = output_dir / img_filename
            page.image.save(str(img_path))
            
            # Create Label Studio task
            task = {
                "data": {
                    "image": str(img_path.absolute()),
                    "ocr": [
                        {
                            "text": word,
                            "bbox": box,
                            "id": idx
                        }
                        for idx, (word, box) in enumerate(zip(page.words, page.boxes))
                    ]
                },
                "predictions": []
            }
            tasks.append(task)
        
        # Save tasks JSON
        tasks_file = output_dir / f"{pdf_name}_tasks.json"
        with open(tasks_file, "w") as f:
            json.dump(tasks, f, indent=2)
        
        return {
            "tasks_file": str(tasks_file),
            "num_pages": len(pages),
            "total_words": sum(len(p.words) for p in pages)
        }
    
    def batch_process(
        self, 
        input_dir: str, 
        output_dir: str,
        extensions: List[str] = [".pdf", ".PDF"]
    ) -> List[Dict]:
        """Process all PDFs in a directory."""
        input_dir = Path(input_dir)
        results = []
        
        files = []
        for ext in extensions:
            files.extend(input_dir.glob(f"*{ext}"))
        
        print(f"Found {len(files)} PDF files to process")
        
        for pdf_file in files:
            print(f"Processing: {pdf_file.name}...", end=" ", flush=True)
            try:
                result = self.save_for_labeling(str(pdf_file), output_dir)
                result["file"] = str(pdf_file)
                result["status"] = "success"
                print(f"✅ {result['num_pages']} pages, {result['total_words']} words")
            except Exception as e:
                result = {
                    "file": str(pdf_file),
                    "status": "error",
                    "error": str(e)
                }
                print(f"❌ {e}")
            results.append(result)
        
        return results