"""Inference on new unseen CRC PDFs."""
import torch
import json
import sys
from pathlib import Path
from collections import defaultdict

from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
from src.labels_crc_triage import ID2LABEL
from src.processor import PDFProcessor


def run_inference_on_pdf(
    pdf_path: str,
    model,
    processor,
    pdf_processor,
    confidence_threshold: float = 0.3
):
    """Run inference on a PDF file."""
    
    print(f"Processing: {Path(pdf_path).name}")
    
    # Process PDF pages
    try:
        pages = pdf_processor.process_pdf(pdf_path)
    except Exception as e:
        print(f"  ERROR processing PDF: {e}")
        return {"file": Path(pdf_path).name, "error": str(e)}
    
    all_entities = []
    
    for page in pages:
        if not page.words:
            continue
        
        # Process with LayoutLMv3
        encoding = processor(
            page.image,
            page.words,
            boxes=page.boxes,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )
        
        # Run inference
        with torch.no_grad():
            outputs = model(**encoding)
            predictions = torch.argmax(outputs.logits, dim=-1)
            probs = torch.softmax(outputs.logits, dim=-1)
        
        # Extract entities
        word_ids = encoding.word_ids()
        seen_words = set()
        
        for idx, pred in enumerate(predictions[0].tolist()):
            word_id = word_ids[idx]
            if word_id is None or word_id in seen_words:
                continue
            seen_words.add(word_id)
            
            label = ID2LABEL[pred]
            confidence = probs[0, idx, pred].item()
            
            if label != "O" and confidence >= confidence_threshold:
                all_entities.append({
                    "label": label,
                    "text": page.words[word_id],
                    "confidence": round(confidence, 3),
                    "page": page.page_num
                })
    
    # Group by label
    by_label = defaultdict(list)
    for e in all_entities:
        by_label[e["label"]].append(e)
    
    return {
        "file": Path(pdf_path).name,
        "entities": all_entities,
        "by_label": {k: v for k, v in by_label.items()},
        "summary": {label: len(ents) for label, ents in by_label.items()}
    }


def main():
    input_dir = Path("data/raw_pdfs/new_crc_pdf")
    model_path = "models/crc_triage/final_model"
    output_path = Path("data/new_crc_inference_results.json")
    
    # Number of PDFs to process
    limit = None
    if len(sys.argv) > 1:
        limit = int(sys.argv[1])
    
    print("=" * 60)
    print("CRC Triage Model - Inference on NEW Unseen PDFs")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
    processor = LayoutLMv3Processor.from_pretrained(model_path)
    model.eval()
    
    # Initialize PDF processor
    pdf_processor = PDFProcessor()
    
    # Get PDF files
    pdf_files = sorted(input_dir.glob("*.pdf"))
    if limit:
        pdf_files = pdf_files[:limit]
    
    print(f"\nProcessing {len(pdf_files)} PDFs from {input_dir}")
    print("-" * 60)
    
    results = []
    total_by_label = defaultdict(int)
    
    for pdf_path in pdf_files:
        result = run_inference_on_pdf(
            str(pdf_path), model, processor, pdf_processor
        )
        results.append(result)
        
        if "summary" in result:
            print(f"  Found: {result['summary']}")
            for label, count in result["summary"].items():
                total_by_label[label] += count
    
    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"Results saved to: {output_path}")
    print("\nOverall Summary:")
    for label, count in sorted(total_by_label.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
