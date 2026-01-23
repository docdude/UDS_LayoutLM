"""Remap mammogram dataset labels after removing INDICATION_SCREENING/DIAGNOSTIC.

Converts old label IDs to new label IDs without needing to re-export from Label Studio.
"""
import sys
from pathlib import Path
from datasets import load_from_disk, Dataset, DatasetDict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.labels_mammogram import LABEL2ID as NEW_LABEL2ID, MAMMOGRAM_TRIAGE_LABELS

# OLD labels (before removing INDICATION)
OLD_LABELS = [
    "O",
    "B-INDICATION_SCREENING", "I-INDICATION_SCREENING",
    "B-INDICATION_DIAGNOSTIC", "I-INDICATION_DIAGNOSTIC",
    "B-EXAM_MAMMOGRAM", "I-EXAM_MAMMOGRAM",
    "B-EXAM_TOMOSYNTHESIS", "I-EXAM_TOMOSYNTHESIS",
    "B-EXAM_ULTRASOUND", "I-EXAM_ULTRASOUND",
    "B-EXAM_DATE", "I-EXAM_DATE",
    "B-REPORT_DATE", "I-REPORT_DATE",
    "B-BIRADS_CATEGORY", "I-BIRADS_CATEGORY",
    "B-BREAST_DENSITY", "I-BREAST_DENSITY",
    "B-LATERALITY_BILATERAL", "I-LATERALITY_BILATERAL",
    "B-LATERALITY_LEFT", "I-LATERALITY_LEFT",
    "B-LATERALITY_RIGHT", "I-LATERALITY_RIGHT",
    "B-FOLLOW_UP_INTERVAL", "I-FOLLOW_UP_INTERVAL",
    "B-FINDING_MASS", "I-FINDING_MASS",
    "B-FINDING_CALCIFICATION", "I-FINDING_CALCIFICATION",
    "B-PROCEDURE_CPT", "I-PROCEDURE_CPT",
]

OLD_ID2LABEL = {idx: label for idx, label in enumerate(OLD_LABELS)}

# Create mapping from old ID to new ID
def create_label_mapping():
    """Map old label IDs to new label IDs."""
    mapping = {}
    
    for old_id, old_label in OLD_ID2LABEL.items():
        if old_label in NEW_LABEL2ID:
            # Label still exists, map to new ID
            mapping[old_id] = NEW_LABEL2ID[old_label]
        else:
            # Label removed (INDICATION_*), map to O
            mapping[old_id] = NEW_LABEL2ID["O"]
    
    return mapping

def remap_labels(example, label_mapping):
    """Remap labels in a single example."""
    new_labels = [label_mapping.get(label_id, 0) for label_id in example["labels"]]
    return {"labels": new_labels}

def remap_dataset(input_path: str, output_path: str):
    """Remap entire dataset."""
    print(f"Loading dataset from {input_path}...")
    ds = load_from_disk(input_path)
    
    label_mapping = create_label_mapping()
    
    print("\nLabel mapping (old_id -> new_id):")
    for old_id, new_id in sorted(label_mapping.items()):
        old_label = OLD_ID2LABEL[old_id]
        new_label = MAMMOGRAM_TRIAGE_LABELS[new_id]
        if old_label != new_label:
            print(f"  {old_id} ({old_label}) -> {new_id} ({new_label})")
    
    print(f"\nRemapping labels...")
    
    remapped = DatasetDict()
    for split_name, split_ds in ds.items():
        print(f"  {split_name}: {len(split_ds)} examples")
        remapped[split_name] = split_ds.map(
            lambda ex: remap_labels(ex, label_mapping),
            desc=f"Remapping {split_name}"
        )
    
    print(f"\nSaving to {output_path}...")
    remapped.save_to_disk(output_path)
    print("✓ Done!")
    
    return remapped

def main():
    # Remap the dataset with negatives
    INPUT = "data/dataset_mmg_triage_with_negatives"
    OUTPUT = "data/dataset_mmg_triage_v2"  # New version without INDICATION labels
    
    print("=" * 60)
    print("Remapping Mammogram Labels")
    print("=" * 60)
    print(f"\nOld labels: {len(OLD_LABELS)}")
    print(f"New labels: {len(MAMMOGRAM_TRIAGE_LABELS)}")
    print(f"\nRemoved labels:")
    for label in OLD_LABELS:
        if label not in NEW_LABEL2ID:
            print(f"  - {label}")
    
    remapped = remap_dataset(INPUT, OUTPUT)
    
    # Verify
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)
    
    # Check that no old INDICATION labels remain
    for split_name, split_ds in remapped.items():
        max_label = max(max(ex["labels"]) for ex in split_ds)
        print(f"{split_name}: max_label_id = {max_label} (should be < {len(MAMMOGRAM_TRIAGE_LABELS)})")
    
    print("\n✓ Dataset ready for training!")
    print(f"  Path: {OUTPUT}")

if __name__ == "__main__":
    main()
