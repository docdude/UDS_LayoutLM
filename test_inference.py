"""Quick inference test on a sample document."""
import argparse
import torch
from pathlib import Path
from PIL import Image
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
import importlib
from src.processor import PDFProcessor
import json


def run_inference(image_path: str, model_path: str = "models/crc_triage/final_model", labels_module: str = "src.labels_crc_triage"):
    """Run inference on a single image."""
    
    # Load labels
    labels = importlib.import_module(labels_module)
    ID2LABEL = labels.ID2LABEL
    
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
    
    current_entity = None
    current_words = []
    
    for idx, (pred, word_id) in enumerate(zip(predictions[0].tolist(), word_ids)):
        if word_id is None:
            continue
            
        label = ID2LABEL[pred]
        confidence = probs[0, idx, pred].item()
        
        if label.startswith("B-"):
            # Save previous entity
            if current_entity:
                entities.append({
                    "label": current_entity,
                    "text": " ".join(current_words),
                    "confidence": confidence
                })
            # Start new entity
            current_entity = label[2:]  # Remove "B-"
            current_words = [words[word_id]]
        elif label.startswith("I-") and current_entity == label[2:]:
            # Continue entity
            current_words.append(words[word_id])
        else:
            # Save and reset
            if current_entity:
                entities.append({
                    "label": current_entity,
                    "text": " ".join(current_words),
                    "confidence": confidence
                })
            current_entity = None
            current_words = []
    
    # Save final entity
    if current_entity:
        entities.append({
            "label": current_entity,
            "text": " ".join(current_words),
            "confidence": 0.0
        })
    
    return entities


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test inference on a document image")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--model", type=str, default="models/crc_triage/final_model", help="Path to model")
    parser.add_argument("--labels", type=str, default="src.labels_crc_triage", help="Labels module")
    args = parser.parse_args()
    
    # Find a test image
    processed_dir = Path("data/processed")
    images = list(processed_dir.glob("*.png"))
    
    if args.image:
        test_image = args.image
    elif images:
        test_image = str(images[0])
    else:
        print("No images found in data/processed/ - specify with --image")
        import sys
        sys.exit(1)
    
    print(f"\n{'='*50}")
    print("CRC Triage Model - Inference Test")
    print(f"{'='*50}\n")
    
    entities = run_inference(test_image, model_path=args.model, labels_module=args.labels)
    
    print(f"\n{'='*50}")
    print(f"Extracted {len(entities)} entities from {Path(test_image).name}:")
    print(f"{'='*50}")
    
    if entities:
        for e in entities:
            text_preview = e['text'][:60] + "..." if len(e['text']) > 60 else e['text']
            print(f"  [{e['label']:30}] {text_preview} (conf: {e['confidence']:.2f})")
    else:
        print("  No entities extracted")
    
    print()
