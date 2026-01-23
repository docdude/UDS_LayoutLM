"""Add negative (non-mammogram) examples to the mammogram training dataset.

These documents get all O labels, teaching the model that tokens like
"screening", "bilateral", etc. should NOT be labeled as mammogram entities
when they appear in non-mammogram documents.
"""
import json
import os
import sys
from pathlib import Path
from PIL import Image
from datasets import Dataset, DatasetDict, load_from_disk
import random

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.processor import PDFProcessor
from src.labels_mammogram import LABEL2ID, NUM_LABELS


def process_negative_pdfs(pdf_dir: str, output_images_dir: str):
    """Process negative PDFs and create training examples with all O labels."""
    
    pdf_dir = Path(pdf_dir)
    output_images_dir = Path(output_images_dir)
    output_images_dir.mkdir(parents=True, exist_ok=True)
    
    processor = PDFProcessor()
    examples = []
    
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    print(f"Found {len(pdf_files)} negative PDFs to process")
    
    for pdf_path in pdf_files:
        print(f"  Processing: {pdf_path.name}")
        
        try:
            pages = processor.process_pdf(str(pdf_path))
        except Exception as e:
            print(f"    ⚠️ Error processing {pdf_path.name}: {e}")
            continue
        
        for page_idx, page in enumerate(pages):
            # Save page image
            image_filename = f"{pdf_path.stem}_page{page_idx}.png"
            image_path = output_images_dir / image_filename
            
            if page.image is not None:
                page.image.save(image_path)
            else:
                print(f"    ⚠️ No image for page {page_idx}")
                continue
            
            # Get words and boxes from OCR
            words = page.words
            boxes = page.boxes  # Already in 0-1000 scale
            
            if not words:
                print(f"    ⚠️ No words extracted from page {page_idx}")
                continue
            
            # ALL labels are O (outside) for negative examples
            labels = [LABEL2ID["O"]] * len(words)
            
            examples.append({
                "image_path": str(image_path),
                "words": words,
                "boxes": boxes,
                "labels": labels,
                "source": "negative",
                "doc_type": "not_mammogram",
                "original_file": pdf_path.name,
            })
    
    print(f"\nGenerated {len(examples)} negative training examples")
    return examples


def merge_with_existing_dataset(negative_examples: list, existing_dataset_path: str, output_path: str):
    """Merge negative examples with existing mammogram dataset."""
    
    # Load existing dataset
    print(f"\nLoading existing dataset from {existing_dataset_path}...")
    existing = load_from_disk(existing_dataset_path)
    
    print(f"Existing dataset:")
    print(f"  Train: {len(existing['train'])} examples")
    print(f"  Validation: {len(existing['validation'])} examples")
    print(f"  Test: {len(existing['test'])} examples")
    
    # Convert negative examples to match existing format
    def load_image(path):
        return Image.open(path).convert("RGB")
    
    negative_formatted = []
    for ex in negative_examples:
        try:
            img = load_image(ex["image_path"])
            negative_formatted.append({
                "image": img,
                "words": ex["words"],
                "boxes": ex["boxes"],
                "labels": ex["labels"],
            })
        except Exception as e:
            print(f"  ⚠️ Error loading {ex['image_path']}: {e}")
    
    print(f"\nFormatted {len(negative_formatted)} negative examples")
    
    # Split negatives: 70% train, 15% val, 15% test
    random.seed(42)
    random.shuffle(negative_formatted)
    
    n_total = len(negative_formatted)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.15)
    
    neg_train = negative_formatted[:n_train]
    neg_val = negative_formatted[n_train:n_train + n_val]
    neg_test = negative_formatted[n_train + n_val:]
    
    print(f"Negative split: {len(neg_train)} train, {len(neg_val)} val, {len(neg_test)} test")
    
    # Convert existing splits to lists
    def dataset_to_list(ds):
        return [{"image": ds[i]["image"], "words": ds[i]["words"], 
                 "boxes": ds[i]["boxes"], "labels": ds[i]["labels"]} 
                for i in range(len(ds))]
    
    train_list = dataset_to_list(existing['train']) + neg_train
    val_list = dataset_to_list(existing['validation']) + neg_val
    test_list = dataset_to_list(existing['test']) + neg_test
    
    # Shuffle training data
    random.shuffle(train_list)
    
    print(f"\nMerged dataset:")
    print(f"  Train: {len(train_list)} examples")
    print(f"  Validation: {len(val_list)} examples")
    print(f"  Test: {len(test_list)} examples")
    
    # Create new dataset
    def list_to_dataset(examples_list):
        return Dataset.from_dict({
            "image": [ex["image"] for ex in examples_list],
            "words": [ex["words"] for ex in examples_list],
            "boxes": [ex["boxes"] for ex in examples_list],
            "labels": [ex["labels"] for ex in examples_list],
        })
    
    merged = DatasetDict({
        "train": list_to_dataset(train_list),
        "validation": list_to_dataset(val_list),
        "test": list_to_dataset(test_list),
    })
    
    # Save
    print(f"\nSaving merged dataset to {output_path}...")
    merged.save_to_disk(output_path)
    print("✓ Done!")
    
    return merged


def main():
    # Paths
    NEGATIVE_PDF_DIR = "data/raw_pdfs/non_metric_pdf"
    NEGATIVE_IMAGES_DIR = "data/processed/mmg_negative"  # Separate from crc_triage images
    EXISTING_DATASET = "data/dataset_mmg_triage"
    OUTPUT_DATASET = "data/dataset_mmg_triage_with_negatives"
    
    print("=" * 60)
    print("Adding Negative Examples to Mammogram Dataset")
    print("=" * 60)
    
    # Step 1: Process negative PDFs
    print("\n[Step 1] Processing negative PDFs...")
    negative_examples = process_negative_pdfs(NEGATIVE_PDF_DIR, NEGATIVE_IMAGES_DIR)
    
    if not negative_examples:
        print("No negative examples generated. Exiting.")
        return
    
    # Step 2: Merge with existing dataset
    print("\n[Step 2] Merging with existing dataset...")
    merged = merge_with_existing_dataset(
        negative_examples, 
        EXISTING_DATASET, 
        OUTPUT_DATASET
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Negative PDFs processed: {len(list(Path(NEGATIVE_PDF_DIR).glob('*.pdf')))}")
    print(f"Negative examples generated: {len(negative_examples)}")
    print(f"Final dataset saved to: {OUTPUT_DATASET}")
    print(f"  Train: {len(merged['train'])} examples")
    print(f"  Validation: {len(merged['validation'])} examples")
    print(f"  Test: {len(merged['test'])} examples")
    print("=" * 60)
    print("\nTo train with negative examples, update config_mmg_triage.yaml:")
    print(f'  dataset_path: "{OUTPUT_DATASET}"')


if __name__ == "__main__":
    main()
