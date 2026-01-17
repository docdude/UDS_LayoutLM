"""Inference script for CRC triage model on new PDFs."""
import os
import sys
import json
from pathlib import Path
from collections import defaultdict

import torch
from PIL import Image
from pdf2image import convert_from_path
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
import pytesseract

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.labels_crc_triage import ID2LABEL, LABEL2ID


def pdf_to_images(pdf_path: str, dpi: int = 300) -> list:
    """Convert PDF to list of PIL Images."""
    return convert_from_path(pdf_path, dpi=dpi)


def run_ocr(image: Image.Image) -> dict:
    """Run OCR on image and return words with bboxes."""
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    
    words = []
    boxes = []
    
    img_width, img_height = image.size
    
    for i, word in enumerate(ocr_data['text']):
        word = word.strip()
        if not word:
            continue
        
        conf = int(ocr_data['conf'][i])
        if conf < 30:  # Skip low confidence
            continue
        
        x = ocr_data['left'][i]
        y = ocr_data['top'][i]
        w = ocr_data['width'][i]
        h = ocr_data['height'][i]
        
        # Normalize to 0-1000
        box = [
            int(1000 * x / img_width),
            int(1000 * y / img_height),
            int(1000 * (x + w) / img_width),
            int(1000 * (y + h) / img_height)
        ]
        
        words.append(word)
        boxes.append(box)
    
    return {"words": words, "boxes": boxes}


def run_inference(
    model,
    processor,
    image: Image.Image,
    words: list,
    boxes: list,
    confidence_threshold: float = 0.5
) -> list:
    """Run inference on a single image."""
    
    # Prepare inputs
    encoding = processor(
        image,
        words,
        boxes=boxes,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    
    # Move to device
    device = next(model.parameters()).device
    for k, v in encoding.items():
        if isinstance(v, torch.Tensor):
            encoding[k] = v.to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(**encoding)
    
    # Get predictions
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    predictions = torch.argmax(logits, dim=-1)
    
    # Extract entities
    entities = []
    word_ids = encoding.word_ids()
    
    seen_words = set()
    for idx, (pred, prob) in enumerate(zip(predictions[0], probs[0])):
        word_idx = word_ids[idx]
        if word_idx is None or word_idx in seen_words:
            continue
        seen_words.add(word_idx)
        
        pred_id = pred.item()
        confidence = prob[pred_id].item()
        label = ID2LABEL[pred_id]
        
        if label != "O" and confidence >= confidence_threshold:
            entities.append({
                "text": words[word_idx],
                "label": label,
                "confidence": round(confidence, 3),
                "box": boxes[word_idx]
            })
    
    return entities


def process_pdf(pdf_path: str, model, processor, confidence_threshold: float = 0.5) -> dict:
    """Process a PDF and extract entities."""
    
    print(f"Processing: {os.path.basename(pdf_path)}")
    
    # Convert PDF to images
    images = pdf_to_images(pdf_path)
    
    all_entities = []
    
    for page_num, image in enumerate(images):
        # Run OCR
        ocr_result = run_ocr(image)
        words = ocr_result["words"]
        boxes = ocr_result["boxes"]
        
        if not words:
            continue
        
        # Run inference
        entities = run_inference(
            model, processor, image, words, boxes, confidence_threshold
        )
        
        # Add page info
        for ent in entities:
            ent["page"] = page_num
        
        all_entities.extend(entities)
    
    # Group by label type
    by_label = defaultdict(list)
    for ent in all_entities:
        by_label[ent["label"]].append(ent)
    
    return {
        "file": os.path.basename(pdf_path),
        "entities": all_entities,
        "by_label": dict(by_label),
        "summary": {label: len(ents) for label, ents in by_label.items()}
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="CRC Triage Inference")
    parser.add_argument("--input", type=str, default="data/raw_pdfs/new_crc_pdf",
                        help="Input folder with PDFs")
    parser.add_argument("--model", type=str, default="models/crc_triage/final_model",
                        help="Path to trained model")
    parser.add_argument("--output", type=str, default="data/inference_results.json",
                        help="Output JSON file")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Confidence threshold")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of PDFs to process")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CRC Triage Model - Inference")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model from {args.model}...")
    model = LayoutLMv3ForTokenClassification.from_pretrained(args.model)
    processor = LayoutLMv3Processor.from_pretrained(args.model, apply_ocr=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"Using device: {device}")
    
    # Get PDF files
    input_path = Path(args.input)
    pdf_files = list(input_path.glob("*.pdf"))
    
    if args.limit:
        pdf_files = pdf_files[:args.limit]
    
    print(f"\nFound {len(pdf_files)} PDFs to process")
    print("-" * 60)
    
    # Process each PDF
    results = []
    for pdf_path in pdf_files:
        try:
            result = process_pdf(str(pdf_path), model, processor, args.threshold)
            results.append(result)
            
            # Print summary
            print(f"  Found: {result['summary']}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"file": pdf_path.name, "error": str(e)})
    
    # Save results
    print("\n" + "=" * 60)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_path}")
    
    # Print overall summary
    print("\nOverall Summary:")
    total_by_label = defaultdict(int)
    for r in results:
        if "summary" in r:
            for label, count in r["summary"].items():
                total_by_label[label] += count
    
    for label, count in sorted(total_by_label.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
