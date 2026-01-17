"""Quick inference test for CRC triage model on processed images."""
import torch
from pathlib import Path
from PIL import Image
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
from src.labels_crc_triage import ID2LABEL
from src.processor import PDFProcessor
import json
import sys


def run_inference(image_path: str, model_path: str = "models/crc_triage/final_model", confidence_threshold: float = 0.3):
    """Run inference on a single image."""
    
    # Load model and processor
    print(f"Loading model from {model_path}...")
    model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
    processor = LayoutLMv3Processor.from_pretrained(model_path)
    model.eval()
    
    # Process document
    print(f"Processing {image_path}...")
    doc_processor = PDFProcessor()
    
    # Get OCR results
    result = doc_processor.process_image(image_path)
    words = result.words
    boxes = result.boxes  # Already normalized to 0-1000
    image = result.image
    
    print(f"  Found {len(words)} words")
    
    # Process with LayoutLMv3
    encoding = processor(
        image,
        words,
        boxes=boxes,
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
    entities = []
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
            entities.append({
                "label": label,
                "text": words[word_id],
                "confidence": round(confidence, 3),
                "box": boxes[word_id]
            })
    
    return entities


def main():
    # Find test images
    processed_dir = Path("data/processed")
    images = list(processed_dir.glob("*.png"))
    
    if not images:
        print("No images found in data/processed/")
        sys.exit(1)
    
    # Model path
    model_path = "models/crc_triage/final_model"
    
    print(f"\n{'='*60}")
    print("CRC Triage Model - Inference Test")
    print(f"{'='*60}\n")
    
    # Test on multiple images
    num_to_test = 5
    if len(sys.argv) > 1:
        num_to_test = int(sys.argv[1])
    
    test_images = images[:num_to_test]
    
    all_results = []
    
    for img_path in test_images:
        print(f"\n{'-'*60}")
        try:
            entities = run_inference(str(img_path), model_path)
            
            print(f"\nExtracted {len(entities)} entities from {img_path.name}:")
            
            # Group by label
            by_label = {}
            for e in entities:
                label = e['label']
                if label not in by_label:
                    by_label[label] = []
                by_label[label].append(e)
            
            for label, ents in sorted(by_label.items()):
                texts = [e['text'] for e in ents]
                print(f"  {label}:")
                for e in ents[:3]:  # Show first 3
                    print(f"    - \"{e['text']}\" (conf: {e['confidence']:.2f})")
                if len(ents) > 3:
                    print(f"    ... and {len(ents)-3} more")
            
            all_results.append({
                "file": img_path.name,
                "entities": entities,
                "summary": {label: len(ents) for label, ents in by_label.items()}
            })
            
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results.append({"file": img_path.name, "error": str(e)})
    
    # Save results
    output_path = Path("data/crc_triage_inference_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
