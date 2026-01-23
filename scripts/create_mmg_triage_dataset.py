"""Create training dataset from Mammogram triage Label Studio export."""
import json
import os
import sys
from pathlib import Path
from PIL import Image
from collections import Counter
from datasets import Dataset, DatasetDict
import random

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Use the mammogram triage labels
from src.labels_mammogram import LABEL2ID, ID2LABEL, NUM_LABELS


def bbox_overlap(box1, box2, threshold=0.3):
    """Check if two boxes overlap by at least threshold percentage."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return False
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    
    if area1 == 0:
        return False
    
    return (intersection / area1) >= threshold


def load_label_studio_export(export_path: str, images_dir: str):
    """Load Label Studio export and convert to training format."""
    
    with open(export_path, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
    
    examples = []
    label_counts = Counter()
    skipped = 0
    
    for task in tasks:
        # Skip unannotated tasks
        annotations = task.get('annotations', [])
        if not annotations:
            continue
        
        # Get the latest annotation
        annotation = annotations[-1]
        results = annotation.get('result', [])
        
        # Get OCR data
        ocr_data = task.get('data', {}).get('ocr', [])
        if not ocr_data:
            continue
        
        # Get image info
        image_url = task.get('data', {}).get('image', '')
        if not image_url:
            continue
        
        # Extract filename from URL (handles Label Studio local-files path)
        # Pattern: /data/local-files/?d=processed_mmg/100311_mmg_page0.png
        if '?d=' in image_url:
            image_filename = image_url.split('?d=')[-1]
            # Remove any prefix path like processed_mmg/
            image_filename = os.path.basename(image_filename)
        else:
            image_filename = image_url.split('/')[-1]
        
        image_path = os.path.join(images_dir, image_filename)
        
        if not os.path.exists(image_path):
            print(f"  Warning: Image not found: {image_path}")
            skipped += 1
            continue
        
        # Load image to get dimensions
        try:
            img = Image.open(image_path).convert("RGB")
            img_width, img_height = img.size
        except Exception as e:
            print(f"  Warning: Could not load image {image_path}: {e}")
            skipped += 1
            continue
        
        # Build word list with labels
        words = []
        boxes = []
        labels = []
        
        for ocr_item in ocr_data:
            word = ocr_item.get('text', '').strip()
            if not word:
                continue
            
            # OCR bbox is in 0-1000 scale (already normalized for LayoutLMv3)
            bbox = ocr_item.get('bbox', [0, 0, 0, 0])
            if len(bbox) != 4:
                continue
            
            # Box is already 0-1000, use directly
            norm_box = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            
            # Find matching label from annotations
            label = "O"
            
            for result in results:
                if result.get('type') != 'rectanglelabels':
                    continue
                
                value = result.get('value', {})
                rect_labels = value.get('rectanglelabels', [])
                if not rect_labels:
                    continue
                
                # Annotation is 0-100 percentage - convert to 0-1000 scale
                ann_x1 = value.get('x', 0) * 10
                ann_y1 = value.get('y', 0) * 10
                ann_x2 = ann_x1 + value.get('width', 0) * 10
                ann_y2 = ann_y1 + value.get('height', 0) * 10
                ann_box = [ann_x1, ann_y1, ann_x2, ann_y2]
                
                # Check overlap (both in 0-1000 space now)
                if bbox_overlap(bbox, ann_box, threshold=0.3):
                    label_name = rect_labels[0]
                    # Use B- prefix (simplification: treat all as beginning)
                    full_label = f"B-{label_name}"
                    if full_label in LABEL2ID:
                        label = full_label
                        break
            
            words.append(word)
            boxes.append(norm_box)
            labels.append(LABEL2ID.get(label, 0))
            
            if label != "O":
                label_counts[label] += 1
        
        if words:
            examples.append({
                "image": img,
                "words": words,
                "boxes": boxes,
                "labels": labels,
                "source_file": image_filename
            })
    
    print(f"\nLabel distribution:")
    for label, count in label_counts.most_common():
        print(f"  {label}: {count}")
    
    if skipped:
        print(f"\nSkipped {skipped} tasks due to missing images")
    
    return examples


def create_dataset_splits(examples, train_ratio=0.7, val_ratio=0.15):
    """Split examples into train/val/test."""
    random.seed(42)
    random.shuffle(examples)
    
    n = len(examples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_examples = examples[:n_train]
    val_examples = examples[n_train:n_train + n_val]
    test_examples = examples[n_train + n_val:]
    
    return train_examples, val_examples, test_examples


def main():
    # Mammogram-specific paths
    export_path = "data/labeled/project-4-at-2026-01-22-05-23-94bcabcf.json"
    images_dir = "data/processed_mmg"
    output_dir = "data/dataset_mmg_triage"
    
    print("=" * 50)
    print("Creating Mammogram Triage Training Dataset")
    print("=" * 50)
    print(f"\nUsing {NUM_LABELS} labels")
    
    # Load and convert
    print(f"\nLoading export from {export_path}...")
    examples = load_label_studio_export(export_path, images_dir)
    print(f"  Loaded {len(examples)} annotated examples")
    
    if len(examples) == 0:
        print("ERROR: No examples loaded! Check image paths.")
        return
    
    # Split
    train, val, test = create_dataset_splits(examples)
    print(f"\nSplit: {len(train)} train, {len(val)} val, {len(test)} test")
    
    # Create HuggingFace datasets
    def to_hf_format(examples_list):
        return {
            "image": [ex["image"] for ex in examples_list],
            "words": [ex["words"] for ex in examples_list],
            "boxes": [ex["boxes"] for ex in examples_list],
            "labels": [ex["labels"] for ex in examples_list],
            "source_file": [ex["source_file"] for ex in examples_list],
        }
    
    dataset = DatasetDict({
        "train": Dataset.from_dict(to_hf_format(train)),
        "validation": Dataset.from_dict(to_hf_format(val)),
        "test": Dataset.from_dict(to_hf_format(test)),
    })
    
    # Save
    print(f"\nSaving to {output_dir}...")
    dataset.save_to_disk(output_dir)
    
    print("\n" + "=" * 50)
    print("Dataset created successfully!")
    print(f"  Train: {len(train)} examples")
    print(f"  Validation: {len(val)} examples")
    print(f"  Test: {len(test)} examples")
    print("=" * 50)


if __name__ == "__main__":
    main()
