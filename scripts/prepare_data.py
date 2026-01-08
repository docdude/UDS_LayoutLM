"""Prepare PDF documents for labeling."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.processor import PDFProcessor


def main():
    parser = argparse.ArgumentParser(description="Prepare PDFs for labeling")
    parser.add_argument("input", help="Input PDF file or directory")
    parser.add_argument("--output", default="./data/processed", help="Output directory")
    parser.add_argument("--dpi", type=int, default=300, help="OCR DPI")
    
    args = parser.parse_args()
    
    processor = PDFProcessor(dpi=args.dpi)
    input_path = Path(args.input)
    
    if input_path.is_file():
        result = processor.save_for_labeling(str(input_path), args.output)
        print(f"Processed: {result}")
    else:
        results = processor.batch_process(str(input_path), args.output)
        success = sum(1 for r in results if r["status"] == "success")
        print(f"Processed {success}/{len(results)} files")


if __name__ == "__main__":
    main()