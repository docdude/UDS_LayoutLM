#!/usr/bin/env python3
"""
Add negative examples to CRC triage dataset.

Processes PDFs that are NOT CRC-related (mammograms, cardiology, labs, etc.)
and adds them to the training set with all "O" labels.
This teaches the model what CRC entities should NOT look like.
"""

import os
import sys
import re
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_from_disk, Dataset, DatasetDict
from PIL import Image
from tqdm import tqdm

from src.processor import PDFProcessor
from src.labels_crc_triage import LABEL2ID, NUM_LABELS


def process_negative_pdfs(pdf_dirs: list, output_dir: str, exclude_patterns: list = None):
    """
    Process PDFs and create negative examples (all O labels).
    
    Args:
        pdf_dirs: List of directories containing negative PDFs
        output_dir: Directory to save processed images
        exclude_patterns: Regex patterns for filenames to exclude (e.g., colonoscopy, fit)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    processor = PDFProcessor()
    examples = []
    
    # Default exclude patterns for CRC - these are positive examples
    if exclude_patterns is None:
        exclude_patterns = [
            r'colonoscopy',
            r'_fit[_\.]',
            r'fobt',
            r'sigmoidoscopy',
        ]
    
    # Compile patterns
    exclude_regex = [re.compile(p, re.IGNORECASE) for p in exclude_patterns]
    
    # Collect all PDFs
    all_pdfs = []
    for pdf_dir in pdf_dirs:
        pdf_dir = Path(pdf_dir)
        if pdf_dir.exists():
            all_pdfs.extend(list(pdf_dir.glob("*.pdf")))
    
    print(f"Found {len(all_pdfs)} total PDFs")
    
    # Filter out excluded patterns
    filtered_pdfs = []
    excluded_count = 0
    for pdf in all_pdfs:
        fname = pdf.name
        if any(p.search(fname) for p in exclude_regex):
            print(f"  Excluding (CRC positive): {fname}")
            excluded_count += 1
        else:
            filtered_pdfs.append(pdf)
    
    print(f"Using {len(filtered_pdfs)} PDFs (excluded {excluded_count} CRC-related)")
    
    for pdf_path in tqdm(filtered_pdfs, desc="Processing negative PDFs"):
        try:
            pages = processor.process_pdf(str(pdf_path))
            
            for page in pages:
                # Save image
                img_name = f"{pdf_path.stem}_page{page.page_num}.png"
                img_path = output_dir / img_name
                page.image.save(img_path)
                
                # Create example with all O labels
                examples.append({
                    "image": page.image,
                    "words": page.words,
                    "boxes": page.boxes,
                    "labels": [0] * len(page.words),  # All O labels
                    "source_file": str(img_path)
                })
                
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            continue
    
    print(f"\nCreated {len(examples)} negative examples from {len(filtered_pdfs)} PDFs")
    return examples


def merge_with_existing_dataset(
    existing_dataset_path: str,
    negative_examples: list,
    output_path: str,
    neg_train_ratio: float = 0.7,
    neg_val_ratio: float = 0.15,
    seed: int = 42
):
    """
    Merge negative examples with existing dataset.
    """
    import random
    random.seed(seed)
    
    # Load existing dataset
    print(f"\nLoading existing dataset from {existing_dataset_path}...")
    existing = load_from_disk(existing_dataset_path)
    
    print(f"Existing dataset:")
    print(f"  Train: {len(existing['train'])}")
    print(f"  Validation: {len(existing['validation'])}")
    print(f"  Test: {len(existing['test'])}")
    
    # Shuffle negative examples
    random.shuffle(negative_examples)
    
    # Split negatives
    n_total = len(negative_examples)
    n_train = int(n_total * neg_train_ratio)
    n_val = int(n_total * neg_val_ratio)
    
    neg_train = negative_examples[:n_train]
    neg_val = negative_examples[n_train:n_train + n_val]
    neg_test = negative_examples[n_train + n_val:]
    
    print(f"\nNegative examples split:")
    print(f"  Train: {len(neg_train)}")
    print(f"  Validation: {len(neg_val)}")
    print(f"  Test: {len(neg_test)}")
    
    # Convert existing to lists
    train_list = [existing['train'][i] for i in range(len(existing['train']))]
    val_list = [existing['validation'][i] for i in range(len(existing['validation']))]
    test_list = [existing['test'][i] for i in range(len(existing['test']))]
    
    # Merge
    train_list.extend(neg_train)
    val_list.extend(neg_val)
    test_list.extend(neg_test)
    
    # Shuffle merged data
    random.shuffle(train_list)
    random.shuffle(val_list)
    random.shuffle(test_list)
    
    # Create new dataset
    new_dataset = DatasetDict({
        'train': Dataset.from_list(train_list),
        'validation': Dataset.from_list(val_list),
        'test': Dataset.from_list(test_list)
    })
    
    # Save
    print(f"\nSaving merged dataset to {output_path}...")
    new_dataset.save_to_disk(output_path)
    
    print(f"\nFinal dataset:")
    print(f"  Train: {len(new_dataset['train'])} ({len(existing['train'])} pos + {len(neg_train)} neg)")
    print(f"  Validation: {len(new_dataset['validation'])} ({len(existing['validation'])} pos + {len(neg_val)} neg)")
    print(f"  Test: {len(new_dataset['test'])} ({len(existing['test'])} pos + {len(neg_test)} neg)")
    
    return new_dataset


if __name__ == "__main__":
    # Directories containing negative PDFs
    PDF_DIRS = [
        "/opt/UDS_LayoutLM/data/raw_pdfs/non_metric_pdf",  # Local
        "/media/jloyamd/UBUNTU 25_1/negative_pdf",  # USB
    ]
    
    # Output directory for processed images
    OUTPUT_DIR = "/opt/UDS_LayoutLM/data/processed/crc_negative"
    
    # Existing dataset
    EXISTING_DATASET = "/opt/UDS_LayoutLM/data/dataset_crc_triage"
    
    # New merged dataset
    NEW_DATASET = "/opt/UDS_LayoutLM/data/dataset_crc_triage_v2"
    
    # Process negative PDFs (excluding colonoscopy, fit, fobt, sigmoidoscopy)
    negative_examples = process_negative_pdfs(
        pdf_dirs=PDF_DIRS,
        output_dir=OUTPUT_DIR,
        exclude_patterns=[
            r'colonoscopy',
            r'_fit[_\.]',
            r'fobt',
            r'sigmoidoscopy',
            r'ct_colon',
        ]
    )
    
    if negative_examples:
        # Merge with existing dataset
        merge_with_existing_dataset(
            existing_dataset_path=EXISTING_DATASET,
            negative_examples=negative_examples,
            output_path=NEW_DATASET
        )
    else:
        print("No negative examples created!")
