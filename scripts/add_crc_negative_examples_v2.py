#!/usr/bin/env python3
"""
Add negative examples to CRC triage dataset - Memory efficient version.

Processes PDFs in batches and saves incrementally to avoid memory issues.
"""

import os
import sys
import re
import json
import gc
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
from tqdm import tqdm


def process_negative_pdfs_to_disk(pdf_dirs: list, output_dir: str, exclude_patterns: list = None):
    """
    Process PDFs and save images + metadata to disk (no memory accumulation).
    """
    from src.processor import PDFProcessor
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    processor = PDFProcessor()
    
    # Default exclude patterns for CRC
    if exclude_patterns is None:
        exclude_patterns = [r'colonoscopy', r'_fit[_\.]', r'fobt', r'sigmoidoscopy']
    
    exclude_regex = [re.compile(p, re.IGNORECASE) for p in exclude_patterns]
    
    # Collect and filter PDFs
    all_pdfs = []
    for pdf_dir in pdf_dirs:
        pdf_dir = Path(pdf_dir)
        if pdf_dir.exists():
            all_pdfs.extend(list(pdf_dir.glob("*.pdf")))
    
    filtered_pdfs = [p for p in all_pdfs if not any(r.search(p.name) for r in exclude_regex)]
    print(f"Processing {len(filtered_pdfs)} PDFs (excluded {len(all_pdfs) - len(filtered_pdfs)} CRC-related)")
    
    # Metadata file to track processed examples
    metadata_file = output_dir / "negative_examples.json"
    metadata = []
    
    # Load existing metadata if resuming
    if metadata_file.exists():
        with open(metadata_file) as f:
            metadata = json.load(f)
        print(f"Resuming: {len(metadata)} examples already processed")
    
    processed_images = {m["image_path"] for m in metadata}
    
    for pdf_path in tqdm(filtered_pdfs, desc="Processing"):
        try:
            pages = processor.process_pdf(str(pdf_path))
            
            for page in pages:
                img_name = f"{pdf_path.stem}_page{page.page_num}.png"
                img_path = output_dir / img_name
                
                # Skip if already in metadata
                if str(img_path) in processed_images:
                    continue
                
                # Save image
                page.image.save(img_path)
                
                # Save metadata (not image)
                metadata.append({
                    "image_path": str(img_path),
                    "words": page.words,
                    "boxes": page.boxes,
                    "num_tokens": len(page.words)
                })
                processed_images.add(str(img_path))
                
                # Clear image from memory
                page.image.close()
            
            # Save metadata after each PDF (incremental save)
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
            
            # Force garbage collection periodically
            gc.collect()
                
        except Exception as e:
            print(f"Error: {pdf_path.name}: {e}")
            continue
    
    # Save metadata
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f)
    
    print(f"\nSaved {len(metadata)} negative examples to {output_dir}")
    print(f"Metadata: {metadata_file}")
    return metadata_file


def create_merged_dataset(
    existing_dataset_path: str,
    negative_metadata_file: str,
    output_path: str,
    neg_train_ratio: float = 0.7,
    seed: int = 42
):
    """
    Create merged dataset using Arrow format (memory efficient).
    """
    import random
    from datasets import load_from_disk, Dataset, DatasetDict, concatenate_datasets
    
    random.seed(seed)
    
    # Load metadata
    with open(negative_metadata_file) as f:
        neg_metadata = json.load(f)
    
    print(f"Loaded {len(neg_metadata)} negative example metadata")
    
    # Shuffle and split
    random.shuffle(neg_metadata)
    n_total = len(neg_metadata)
    n_train = int(n_total * neg_train_ratio)
    n_val = int(n_total * 0.15)
    
    splits = {
        'train': neg_metadata[:n_train],
        'validation': neg_metadata[n_train:n_train + n_val],
        'test': neg_metadata[n_train + n_val:]
    }
    
    print(f"Negative splits: train={len(splits['train'])}, val={len(splits['validation'])}, test={len(splits['test'])}")
    
    # Load existing dataset
    print(f"\nLoading existing dataset...")
    existing = load_from_disk(existing_dataset_path)
    
    # Process each split
    output_path = Path(output_path)
    
    for split_name in ['train', 'validation', 'test']:
        print(f"\nProcessing {split_name}...")
        
        # Create negative examples for this split (load images one at a time)
        neg_examples = []
        for meta in tqdm(splits[split_name], desc=f"Loading {split_name} negatives"):
            img = Image.open(meta["image_path"]).convert("RGB")
            neg_examples.append({
                "image": img,
                "words": meta["words"],
                "boxes": meta["boxes"],
                "labels": [0] * meta["num_tokens"],  # All O
                "source_file": meta["image_path"]
            })
        
        # Create negative dataset
        if neg_examples:
            neg_ds = Dataset.from_list(neg_examples)
            
            # Concatenate with existing
            merged = concatenate_datasets([existing[split_name], neg_ds])
            print(f"  {split_name}: {len(existing[split_name])} + {len(neg_ds)} = {len(merged)}")
        else:
            merged = existing[split_name]
        
        # Save incrementally
        split_dir = output_path / split_name
        merged.save_to_disk(str(split_dir))
        
        # Clear memory
        del neg_examples
        gc.collect()
    
    # Create dataset_dict.json
    dataset_dict_info = {"splits": ["train", "validation", "test"]}
    with open(output_path / "dataset_dict.json", 'w') as f:
        json.dump(dataset_dict_info, f)
    
    print(f"\nDataset saved to {output_path}")
    
    # Verify
    final = load_from_disk(str(output_path))
    print(f"\nFinal dataset:")
    print(f"  Train: {len(final['train'])}")
    print(f"  Validation: {len(final['validation'])}")
    print(f"  Test: {len(final['test'])}")


if __name__ == "__main__":
    # Step 1: Process PDFs to disk (can be run separately)
    NEGATIVE_PDF_DIRS = [
        "/opt/UDS_LayoutLM/data/raw_pdfs/non_metric_pdf",
        "/media/jloyamd/UBUNTU 25_1/negative_pdf",
    ]
    OUTPUT_DIR = "/opt/UDS_LayoutLM/data/processed/crc_negative"
    
    print("=" * 60)
    print("Step 1: Process negative PDFs to disk")
    print("=" * 60)
    
    metadata_file = process_negative_pdfs_to_disk(NEGATIVE_PDF_DIRS, OUTPUT_DIR)
    
    # Force cleanup
    gc.collect()
    
    print("\n" + "=" * 60)
    print("Step 2: Create merged dataset")
    print("=" * 60)
    
    create_merged_dataset(
        existing_dataset_path="/opt/UDS_LayoutLM/data/dataset_crc_triage",
        negative_metadata_file=str(metadata_file),
        output_path="/opt/UDS_LayoutLM/data/dataset_crc_triage_v2"
    )
