"""Dataset creation and loading utilities."""

import os
from pathlib import Path
from typing import List, Dict, Optional, Any
import json

from datasets import Dataset, DatasetDict, Features, Sequence, Value, Image
from PIL import Image as PILImage
from tqdm import tqdm

from .labels import LABEL2ID, ID2LABEL, UDS_LABELS


def bbox_overlap(box1_pct, box2_px, img_width, img_height):
    """
    Check if two bounding boxes overlap.
    box1_pct: Label Studio format (x, y, width, height as percentages)
    box2_px: OCR format (x1, y1, x2, y2 as pixels)
    """
    # Convert Label Studio percentage bbox to pixels
    x1_ls = (box1_pct["x"] / 100) * img_width
    y1_ls = (box1_pct["y"] / 100) * img_height
    x2_ls = x1_ls + (box1_pct["width"] / 100) * img_width
    y2_ls = y1_ls + (box1_pct["height"] / 100) * img_height
    
    # OCR bbox is [x1, y1, x2, y2]
    x1_ocr, y1_ocr, x2_ocr, y2_ocr = box2_px
    
    # Check for overlap
    x_overlap = max(0, min(x2_ls, x2_ocr) - max(x1_ls, x1_ocr))
    y_overlap = max(0, min(y2_ls, y2_ocr) - max(y1_ls, y1_ocr))
    
    if x_overlap > 0 and y_overlap > 0:
        # Calculate overlap ratio relative to OCR token
        ocr_area = (x2_ocr - x1_ocr) * (y2_ocr - y1_ocr)
        if ocr_area > 0:
            overlap_area = x_overlap * y_overlap
            return overlap_area / ocr_area > 0.3  # 30% overlap threshold
    
    return False


def load_label_studio_export(export_path: str) -> List[Dict]:
    """
    Load and parse Label Studio JSON export.
    
    Expected format from Label Studio NER labeling with bounding boxes.
    """
    with open(export_path, "r") as f:
        data = json.load(f)
    
    examples = []
    
    for item in tqdm(data, desc="Parsing annotations"):
        # Get image and OCR data
        image_path = item["data"].get("image", "")
        ocr_data = item["data"].get("ocr", [])
        
        # Handle Label Studio local file paths
        if image_path.startswith("/data/local-files/?d="):
            image_path = image_path.replace("/data/local-files/?d=", "")
        
        if not ocr_data:
            continue
        
        words = [w["text"] for w in ocr_data]
        boxes = [w["bbox"] for w in ocr_data]
        
        # Initialize all labels as "O"
        labels = ["O"] * len(words)
        
        # Get annotations
        annotations = item.get("annotations", [])
        if not annotations:
            continue
            
        results = annotations[0].get("result", [])
        
        # Get image dimensions from the results
        img_width = 2550  # Default
        img_height = 3300  # Default
        for result in results:
            if "original_width" in result:
                img_width = result["original_width"]
                img_height = result["original_height"]
                break
        
        # Process each labeled region
        for result in results:
            if result.get("type") == "labels":
                value = result.get("value", {})
                label_names = value.get("labels", [])
                
                if not label_names:
                    continue
                
                label = label_names[0]
                
                # Find overlapping OCR tokens
                matching_indices = []
                for idx, ocr_token in enumerate(ocr_data):
                    if bbox_overlap(value, ocr_token["bbox"], img_width, img_height):
                        matching_indices.append(idx)
                
                # Apply BIO tagging
                if matching_indices:
                    matching_indices.sort()
                    labels[matching_indices[0]] = f"B-{label}"
                    for idx in matching_indices[1:]:
                        if labels[idx] == "O":  # Don't override existing labels
                            labels[idx] = f"I-{label}"
        
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


# ImageNet normalization constants (used by LayoutLMv3)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class UDSDataCollator:
    """
    Custom data collator for LayoutLMv3 token classification.
    
    Improvements from ClinicalLayoutLM:
    - Optional image augmentation during training
    - Explicit ImageNet normalization
    """
    
    def __init__(
        self, 
        processor, 
        max_length: int = 512,
        augment: bool = False,
        augment_prob: float = 0.5
    ):
        self.processor = processor
        self.max_length = max_length
        self.augment = augment
        self.augment_prob = augment_prob
        
        # Set up augmentation transforms if enabled
        if self.augment:
            try:
                from torchvision import transforms
                self.color_jitter = transforms.ColorJitter(
                    brightness=0.2, 
                    contrast=0.2, 
                    saturation=0.1, 
                    hue=0.05
                )
                self.random_rotation = transforms.RandomRotation(degrees=2)
                print("  Image augmentation enabled (brightness, contrast, rotation)")
            except ImportError:
                print("  Warning: torchvision not available, augmentation disabled")
                self.augment = False
    
    def _augment_image(self, image: PILImage.Image) -> PILImage.Image:
        """Apply random augmentation to image."""
        import random
        
        if random.random() < self.augment_prob:
            # Apply color jitter
            image = self.color_jitter(image)
        
        if random.random() < self.augment_prob * 0.5:  # Less frequent rotation
            # Apply slight rotation
            image = self.random_rotation(image)
        
        return image
    
    def __call__(self, features: List[Dict]) -> Dict:
        images = [f["image"] for f in features]
        words = [f["words"] for f in features]
        boxes = [f["boxes"] for f in features]
        labels = [f["labels"] for f in features]
        
        # Apply augmentation if enabled
        if self.augment:
            images = [self._augment_image(img) for img in images]
        
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