"""Dataset creation and loading utilities."""

import os
from pathlib import Path
from typing import List, Dict, Optional, Any
import json

from datasets import Dataset, DatasetDict, Features, Sequence, Value, Image
from PIL import Image as PILImage
from tqdm import tqdm

from .labels import LABEL2ID, ID2LABEL, UDS_LABELS


def load_label_studio_export(export_path: str) -> List[Dict]:
    """
    Load and parse Label Studio JSON export.
    
    Expected format from Label Studio NER labeling.
    """
    with open(export_path, "r") as f:
        data = json.load(f)
    
    examples = []
    
    for item in tqdm(data, desc="Parsing annotations"):
        # Get image and OCR data
        image_path = item["data"].get("image", "")
        ocr_data = item["data"].get("ocr", [])
        
        if not ocr_data:
            continue
        
        words = [w["text"] for w in ocr_data]
        boxes = [w["bbox"] for w in ocr_data]
        
        # Initialize all labels as "O"
        labels = ["O"] * len(words)
        
        # Get annotations
        annotations = item.get("annotations", [])
        if annotations:
            results = annotations[0].get("result", [])
            
            for result in results:
                if result.get("type") == "labels":
                    value = result.get("value", {})
                    label_names = value.get("labels", [])
                    
                    if not label_names:
                        continue
                    
                    label = label_names[0]
                    start = value.get("start")
                    end = value.get("end")
                    
                    if start is not None and end is not None:
                        # BIO tagging
                        labels[start] = f"B-{label}"
                        for i in range(start + 1, min(end, len(labels))):
                            labels[i] = f"I-{label}"
        
        # Convert labels to IDs (use 0 for unknown labels)
        label_ids = [LABEL2ID.get(l, 0) for l in labels]
        
        # Load image
        try:
            image = PILImage.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Warning: Could not load image {image_path}: {e}")
            continue
        
        examples.append({
            "image": image,
            "words": words,
            "boxes": boxes,
            "labels": label_ids,
            "source_file": image_path
        })
    
    return examples


def create_dataset_from_labeled(
    labeled_dir: str,
    output_dir: str,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42
) -> DatasetDict:
    """
    Create HuggingFace DatasetDict from labeled JSON files.
    """
    labeled_dir = Path(labeled_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all labeled files
    all_examples = []
    for json_file in labeled_dir.glob("*.json"):
        examples = load_label_studio_export(str(json_file))
        all_examples.extend(examples)
    
    if not all_examples:
        raise ValueError(f"No examples found in {labeled_dir}")
    
    print(f"Loaded {len(all_examples)} total examples")
    
    # Create dataset
    dataset = Dataset.from_list(all_examples)
    
    # Split: train / temp
    train_test = dataset.train_test_split(
        test_size=test_size + val_size, 
        seed=seed
    )
    
    # Split temp into val / test
    val_test = train_test["test"].train_test_split(
        test_size=test_size / (test_size + val_size),
        seed=seed
    )
    
    # Create final DatasetDict
    dataset_dict = DatasetDict({
        "train": train_test["train"],
        "validation": val_test["train"],
        "test": val_test["test"]
    })
    
    # Save to disk
    dataset_dict.save_to_disk(str(output_dir))
    
    # Print statistics
    print("\nDataset created:")
    print(f"  Train: {len(dataset_dict['train'])} examples")
    print(f"  Validation: {len(dataset_dict['validation'])} examples")
    print(f"  Test: {len(dataset_dict['test'])} examples")
    print(f"\nSaved to: {output_dir}")
    
    return dataset_dict


def load_dataset(dataset_dir: str) -> DatasetDict:
    """Load a previously created dataset."""
    from datasets import load_from_disk
    return load_from_disk(dataset_dir)


class UDSDataCollator:
    """Custom data collator for LayoutLMv3 token classification."""
    
    def __init__(self, processor, max_length: int = 512):
        self.processor = processor
        self.max_length = max_length
    
    def __call__(self, features: List[Dict]) -> Dict:
        images = [f["image"] for f in features]
        words = [f["words"] for f in features]
        boxes = [f["boxes"] for f in features]
        labels = [f["labels"] for f in features]
        
        # Process with LayoutLMv3Processor
        encoding = self.processor(
            images,
            words,
            boxes=boxes,
            word_labels=labels,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return encoding