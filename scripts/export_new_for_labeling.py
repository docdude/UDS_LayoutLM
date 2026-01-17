"""Export new PDFs to Label Studio format with model pre-annotations."""
import os
import sys
import json
import shutil
from pathlib import Path
from collections import defaultdict

import torch
from PIL import Image
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.labels_crc_triage import ID2LABEL
from src.processor import PDFProcessor


def export_for_label_studio(
    input_dir: str = "data/raw_pdfs/new_crc_pdf",
    output_dir: str = "data/new_crc_for_labeling",
    model_path: str = "models/crc_triage/final_model",
    confidence_threshold: float = 0.5,
    limit: int = None
):
    """
    Export new PDFs to Label Studio format with pre-annotations.
    
    Creates:
    - Images in output_dir/images/
    - Label Studio import JSON with OCR and pre-annotations
    """
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Export New PDFs for Label Studio")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
    processor = LayoutLMv3Processor.from_pretrained(model_path, apply_ocr=False)
    model.eval()
    
    # PDF processor
    pdf_processor = PDFProcessor()
    
    # Get PDF files
    pdf_files = sorted(input_path.glob("*.pdf"))
    if limit:
        pdf_files = pdf_files[:limit]
    
    print(f"\nProcessing {len(pdf_files)} PDFs...")
    print("-" * 60)
    
    tasks = []
    
    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        
        try:
            pages = pdf_processor.process_pdf(str(pdf_path))
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        
        for page in pages:
            if not page.words:
                continue
            
            # Save image
            image_filename = f"{pdf_path.stem}_page{page.page_num}.png"
            image_path = images_dir / image_filename
            page.image.save(image_path)
            
            # Build OCR data in Label Studio format (0-1000 scale)
            ocr_data = []
            for i, (word, box) in enumerate(zip(page.words, page.boxes)):
                ocr_data.append({
                    "id": i,
                    "text": word,
                    "bbox": box  # Already 0-1000
                })
            
            # Run inference for pre-annotations
            encoding = processor(
                page.image,
                page.words,
                boxes=page.boxes,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                outputs = model(**encoding)
                predictions = torch.argmax(outputs.logits, dim=-1)
                probs = torch.softmax(outputs.logits, dim=-1)
            
            # Extract predictions
            word_ids = encoding.word_ids()
            seen_words = set()
            predictions_list = []
            
            for idx, pred in enumerate(predictions[0].tolist()):
                word_id = word_ids[idx]
                if word_id is None or word_id in seen_words:
                    continue
                seen_words.add(word_id)
                
                label = ID2LABEL[pred]
                confidence = probs[0, idx, pred].item()
                
                if label != "O" and confidence >= confidence_threshold:
                    # Get box for this word
                    box = page.boxes[word_id]
                    
                    # Convert from 0-1000 to 0-100 for Label Studio
                    x = box[0] / 10
                    y = box[1] / 10
                    width = (box[2] - box[0]) / 10
                    height = (box[3] - box[1]) / 10
                    
                    # Remove B- prefix for label name
                    label_name = label[2:] if label.startswith("B-") else label
                    
                    predictions_list.append({
                        "label": label_name,
                        "text": page.words[word_id],
                        "confidence": round(confidence, 3),
                        "x": round(x, 2),
                        "y": round(y, 2),
                        "width": round(width, 2),
                        "height": round(height, 2)
                    })
            
            # Create Label Studio task
            task = {
                "data": {
                    "image": f"/data/local-files/?d=images/{image_filename}",
                    "ocr": ocr_data
                },
                "predictions": [{
                    "model_version": "crc_triage_v1",
                    "result": [
                        {
                            "id": f"pred_{i}",
                            "type": "rectanglelabels",
                            "from_name": "label",
                            "to_name": "image",
                            "original_width": page.image.width,
                            "original_height": page.image.height,
                            "value": {
                                "x": p["x"],
                                "y": p["y"],
                                "width": p["width"],
                                "height": p["height"],
                                "rotation": 0,
                                "rectanglelabels": [p["label"]]
                            }
                        }
                        for i, p in enumerate(predictions_list)
                    ]
                }] if predictions_list else []
            }
            
            tasks.append(task)
            print(f"  Page {page.page_num}: {len(predictions_list)} pre-annotations")
    
    # Save Label Studio import file
    import_file = output_path / "label_studio_import.json"
    with open(import_file, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"Export complete!")
    print(f"  Tasks created: {len(tasks)}")
    print(f"  Images saved to: {images_dir}")
    print(f"  Import file: {import_file}")
    print()
    print("To import into Label Studio:")
    print(f"  1. Copy {images_dir} to your Label Studio data directory")
    print(f"  2. Import {import_file} into your project")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export PDFs to Label Studio")
    parser.add_argument("--input", type=str, default="data/raw_pdfs/new_crc_pdf",
                        help="Input PDF directory")
    parser.add_argument("--output", type=str, default="data/new_crc_for_labeling",
                        help="Output directory")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of PDFs")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Confidence threshold for pre-annotations")
    
    args = parser.parse_args()
    
    export_for_label_studio(
        input_dir=args.input,
        output_dir=args.output,
        limit=args.limit,
        confidence_threshold=args.threshold
    )
