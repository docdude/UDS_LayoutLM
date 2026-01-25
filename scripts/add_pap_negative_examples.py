"""Add negative (non-PAP/GYN) examples to the PAP triage training dataset.

Reuses already-processed negative images from crc_negative and mmg_negative folders.
Excludes PAP/GYN documents (pap, gyn, cervical, hpv patterns).
"""
import os
import sys
import re
from pathlib import Path
from PIL import Image
from datasets import Dataset, DatasetDict, load_from_disk
import random

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.labels_pap_triage import LABEL2ID, NUM_LABELS


def load_negative_images_with_ocr(image_dirs: list, exclude_patterns: list = None):
    """
    Load already-processed negative images and run OCR to get words/boxes.
    
    Args:
        image_dirs: List of directories containing processed PNG images
        exclude_patterns: Regex patterns for filenames to exclude (PAP/GYN positives)
    """
    from src.processor import PDFProcessor
    import numpy as np
    
    examples = []
    
    # Default exclude patterns for PAP - these are positive examples
    if exclude_patterns is None:
        exclude_patterns = [
            r'pap',
            r'gyn',
            r'cervical',
            r'hpv',
            r'cytology',
            r'thinprep',
            r'bethesda',
        ]
    
    # Compile patterns
    exclude_regex = [re.compile(p, re.IGNORECASE) for p in exclude_patterns]
    
    # Collect all PNG images
    all_images = []
    for img_dir in image_dirs:
        img_dir = Path(img_dir)
        if img_dir.exists():
            imgs = list(img_dir.glob("*.png"))
            all_images.extend(imgs)
            print(f"  Found {len(imgs)} images in {img_dir}")
    
    print(f"Found {len(all_images)} total images")
    
    # Filter out excluded patterns
    filtered_images = []
    excluded_count = 0
    for img_path in all_images:
        fname = img_path.name
        if any(p.search(fname) for p in exclude_regex):
            excluded_count += 1
        else:
            filtered_images.append(img_path)
    
    print(f"Using {len(filtered_images)} images (excluded {excluded_count} PAP/GYN-related)")
    
    # Initialize processor for OCR
    processor = PDFProcessor()
    
    # Process images with OCR
    print(f"\nRunning OCR on {len(filtered_images)} images...")
    for i, img_path in enumerate(filtered_images):
        if i % 10 == 0:
            print(f"  Processing {i+1}/{len(filtered_images)}...")
        
        try:
            img = Image.open(img_path).convert("RGB")
            
            # Run OCR to get words and boxes
            words, boxes, raw_boxes = processor._ocr_with_boxes(img)
            
            if not words:
                continue
            
            # ALL labels are O (outside) for negative examples
            labels = [LABEL2ID["O"]] * len(words)
            
            examples.append({
                "image": img,
                "words": words,
                "boxes": boxes,
                "labels": labels,
                "source_file": img_path.name,
            })
            
        except Exception as e:
            print(f"  ⚠️ Error processing {img_path}: {e}")
            continue
    
    print(f"\nLoaded {len(examples)} negative examples with OCR")
    return examples


def load_negative_from_tasks(task_dirs: list, image_dirs: list, exclude_patterns: list = None):
    """
    Load negative examples from task JSON files with corresponding images.
    
    Args:
        task_dirs: Directories containing *_tasks.json files
        image_dirs: Directories containing PNG images  
        exclude_patterns: Regex patterns for filenames to exclude
    """
    import json
    
    examples = []
    
    # Default exclude patterns for PAP
    if exclude_patterns is None:
        exclude_patterns = [
            r'pap',
            r'gyn',
            r'cervical',
            r'hpv',
            r'cytology',
            r'thinprep',
        ]
    
    exclude_regex = [re.compile(p, re.IGNORECASE) for p in exclude_patterns]
    
    # Build image lookup
    image_lookup = {}
    for img_dir in image_dirs:
        img_dir = Path(img_dir)
        if img_dir.exists():
            for img_path in img_dir.glob("*.png"):
                image_lookup[img_path.name] = img_path
    
    print(f"Image lookup: {len(image_lookup)} images available")
    
    # Load task files
    all_tasks = []
    for task_dir in task_dirs:
        task_dir = Path(task_dir)
        if task_dir.exists():
            for task_file in task_dir.glob("*_tasks.json"):
                # Check exclude patterns on task filename
                if any(p.search(task_file.name) for p in exclude_regex):
                    continue
                    
                with open(task_file) as f:
                    tasks = json.load(f)
                    all_tasks.extend(tasks)
    
    print(f"Loaded {len(all_tasks)} tasks from task files")
    
    for task in all_tasks:
        data = task.get('data', {})
        image_path_str = data.get('image', '')
        ocr_data = data.get('ocr', [])
        
        if not ocr_data:
            continue
        
        # Extract image filename
        image_filename = Path(image_path_str).name
        
        # Check exclude patterns on image filename
        if any(p.search(image_filename) for p in exclude_regex):
            continue
        
        # Find image
        if image_filename not in image_lookup:
            continue
        
        try:
            img = Image.open(image_lookup[image_filename]).convert("RGB")
            words = [w['text'] for w in ocr_data if w.get('text')]
            boxes = [w['bbox'] for w in ocr_data if w.get('text')]
            
            if not words:
                continue
            
            # ALL labels are O for negative examples
            labels = [LABEL2ID["O"]] * len(words)
            
            examples.append({
                "image": img,
                "words": words,
                "boxes": boxes,
                "labels": labels,
                "source_file": image_filename,
            })
            
        except Exception as e:
            continue
    
    print(f"Created {len(examples)} negative examples from tasks")
    return examples


def merge_with_existing_dataset(negative_examples: list, existing_dataset_path: str, output_path: str):
    """Merge negative examples with existing PAP dataset."""
    
    # Load existing dataset
    print(f"\nLoading existing dataset from {existing_dataset_path}...")
    existing = load_from_disk(existing_dataset_path)
    
    print(f"Existing dataset:")
    print(f"  Train: {len(existing['train'])} examples")
    print(f"  Validation: {len(existing['validation'])} examples")
    print(f"  Test: {len(existing['test'])} examples")
    
    # Split negatives: 70% train, 15% val, 15% test
    random.seed(42)
    random.shuffle(negative_examples)
    
    n_total = len(negative_examples)
    n_train = int(n_total * 0.7)
    n_val = int(n_total * 0.15)
    
    neg_train = negative_examples[:n_train]
    neg_val = negative_examples[n_train:n_train + n_val]
    neg_test = negative_examples[n_train + n_val:]
    
    print(f"\nNegative split: {len(neg_train)} train, {len(neg_val)} val, {len(neg_test)} test")
    
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
    print(f"  Train: {len(train_list)} examples ({len(existing['train'])} pos + {len(neg_train)} neg)")
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
    # Paths - reuse existing processed negatives
    IMAGE_DIRS = [
        "data/processed/crc_negative",
        "data/processed/mmg_negative",
    ]
    
    EXISTING_DATASET = "data/dataset_pap_triage"
    OUTPUT_DATASET = "data/dataset_pap_triage_with_negatives"
    
    # Exclude patterns for PAP/GYN documents
    EXCLUDE_PATTERNS = [
        r'pap',
        r'gyn',
        r'cervical',
        r'hpv',
        r'cytology',
        r'thinprep',
        r'colposcopy',
        r'leep',
    ]
    
    print("=" * 60)
    print("Adding Negative Examples to PAP Triage Dataset")
    print("=" * 60)
    
    # Step 1: Load negative examples from existing processed images
    print("\n[Step 1] Loading negative examples with OCR...")
    negative_examples = load_negative_images_with_ocr(
        image_dirs=IMAGE_DIRS,
        exclude_patterns=EXCLUDE_PATTERNS
    )
    
    if not negative_examples:
        print("No negative examples found. Exiting.")
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
    print(f"Negative examples loaded: {len(negative_examples)}")
    print(f"Final dataset saved to: {OUTPUT_DATASET}")
    print(f"  Train: {len(merged['train'])} examples")
    print(f"  Validation: {len(merged['validation'])} examples")
    print(f"  Test: {len(merged['test'])} examples")
    print("=" * 60)
    print("\nTo train with negative examples, update config_pap_triage.yaml:")
    print(f'  dataset: "{OUTPUT_DATASET}"')


if __name__ == "__main__":
    main()
