#!/usr/bin/env python3
"""
Remap CRC dataset labels after removing:
- INDICATION_SCREENING, INDICATION_SURVEILLANCE, INDICATION_DIAGNOSTIC
- RESULT_POSITIVE, RESULT_NEGATIVE
- BIOPSY_TAKEN

Maps old label IDs to new label IDs, converting removed labels to O.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_from_disk
from tqdm import tqdm

# OLD labels (45 labels) - before removal
OLD_LABELS = [
    "O",
    "B-DOC_TYPE_COLONOSCOPY", "I-DOC_TYPE_COLONOSCOPY",
    "B-DOC_TYPE_FIT", "I-DOC_TYPE_FIT",
    "B-DOC_TYPE_FOBT", "I-DOC_TYPE_FOBT",
    "B-DOC_TYPE_SIGMOIDOSCOPY", "I-DOC_TYPE_SIGMOIDOSCOPY",
    "B-DOC_TYPE_CT_COLONOGRAPHY", "I-DOC_TYPE_CT_COLONOGRAPHY",
    "B-PROCEDURE_DATE", "I-PROCEDURE_DATE",
    "B-COLLECTION_DATE", "I-COLLECTION_DATE",
    "B-REPORT_DATE", "I-REPORT_DATE",
    "B-POLYP_FINDING", "I-POLYP_FINDING",
    "B-POLYP_LOCATION", "I-POLYP_LOCATION",
    "B-POLYP_SIZE", "I-POLYP_SIZE",
    "B-POLYP_COUNT", "I-POLYP_COUNT",
    "B-BIOPSY_TAKEN", "I-BIOPSY_TAKEN",  # REMOVED
    "B-BIOPSY_RESULT", "I-BIOPSY_RESULT",
    "B-PATHOLOGY_DIAGNOSIS", "I-PATHOLOGY_DIAGNOSIS",
    "B-COMPLICATIONS", "I-COMPLICATIONS",
    "B-RESULT_POSITIVE", "I-RESULT_POSITIVE",  # REMOVED
    "B-RESULT_NEGATIVE", "I-RESULT_NEGATIVE",  # REMOVED
    "B-RESULT_VALUE", "I-RESULT_VALUE",
    "B-PROCEDURE_CPT", "I-PROCEDURE_CPT",
    "B-INDICATION_SCREENING", "I-INDICATION_SCREENING",  # REMOVED
    "B-INDICATION_SURVEILLANCE", "I-INDICATION_SURVEILLANCE",  # REMOVED
    "B-INDICATION_DIAGNOSTIC", "I-INDICATION_DIAGNOSTIC",  # REMOVED
    "B-DIVERTICULA_FINDING", "I-DIVERTICULA_FINDING",
    "B-HEMORRHOIDS_FINDING", "I-HEMORRHOIDS_FINDING",
    "B-COLON_ANATOMY", "I-COLON_ANATOMY",
]

# Labels to remove (map to O)
REMOVED_LABELS = {
    "B-INDICATION_SCREENING", "I-INDICATION_SCREENING",
    "B-INDICATION_SURVEILLANCE", "I-INDICATION_SURVEILLANCE",
    "B-INDICATION_DIAGNOSTIC", "I-INDICATION_DIAGNOSTIC",
    "B-RESULT_POSITIVE", "I-RESULT_POSITIVE",
    "B-RESULT_NEGATIVE", "I-RESULT_NEGATIVE",
    "B-BIOPSY_TAKEN", "I-BIOPSY_TAKEN",
}

# Import new labels
from src.labels_crc_triage import LABEL2ID as NEW_LABEL2ID, CRC_TRIAGE_LABELS as NEW_LABELS

def create_mapping():
    """Create old_id -> new_id mapping."""
    old_id2label = {i: l for i, l in enumerate(OLD_LABELS)}
    
    mapping = {}
    for old_id, label in old_id2label.items():
        if label in REMOVED_LABELS:
            # Map removed labels to O
            mapping[old_id] = NEW_LABEL2ID["O"]
        elif label in NEW_LABEL2ID:
            mapping[old_id] = NEW_LABEL2ID[label]
        else:
            print(f"Warning: Label '{label}' not found in new labels, mapping to O")
            mapping[old_id] = NEW_LABEL2ID["O"]
    
    return mapping


def remap_dataset(input_path: str, output_path: str):
    """Remap dataset labels."""
    print(f"Loading dataset from {input_path}...")
    dataset = load_from_disk(input_path)
    
    mapping = create_mapping()
    
    # Show mapping for removed labels
    print("\nLabel remapping (removed → O):")
    for old_id, label in enumerate(OLD_LABELS):
        if label in REMOVED_LABELS:
            print(f"  {old_id}: {label} → 0 (O)")
    
    def remap_labels(example):
        old_labels = example["labels"]
        new_labels = [mapping.get(l, 0) for l in old_labels]
        return {"labels": new_labels}
    
    print("\nRemapping labels...")
    remapped = dataset.map(remap_labels, desc="Remapping")
    
    print(f"\nSaving to {output_path}...")
    remapped.save_to_disk(output_path)
    
    print(f"\nDone! Remapped dataset saved to {output_path}")
    print(f"  Train: {len(remapped['train'])}")
    print(f"  Validation: {len(remapped['validation'])}")
    print(f"  Test: {len(remapped['test'])}")
    
    return remapped


if __name__ == "__main__":
    print("=" * 60)
    print("CRC Label Remapping")
    print(f"Old labels: {len(OLD_LABELS)}")
    print(f"New labels: {len(NEW_LABELS)}")
    print(f"Removed: {len(REMOVED_LABELS)} labels")
    print("=" * 60)
    
    remap_dataset(
        input_path="data/dataset_crc_triage_v2",  # With negatives
        output_path="data/dataset_crc_triage_v3"  # Remapped labels
    )
