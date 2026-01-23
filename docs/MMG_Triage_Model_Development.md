# Mammogram Triage Model Development

## Overview

This document summarizes the development of a LayoutLMv3-based model for mammogram document triage and entity extraction. The goal is to automatically identify mammogram reports from a mixed document stream and extract relevant UDS (Uniform Data System) metrics.

**Final Model**: `models/mmg_triage_v2/final_model`  
**Test F1**: 0.9557  
**Triage Accuracy**: 100% (72/72 documents)  
**Date**: January 2026

---

## Problem Statement

When processing clinical documents for UDS metrics, we need to:
1. **Triage**: Identify which documents are mammogram reports vs. other document types
2. **Extract**: Pull relevant entities (exam type, BI-RADS, breast density, dates, etc.)

The challenge is that many non-mammogram documents (oncology notes, lab reports, etc.) may mention "mammogram" in the patient history, causing false positives.

---

## Key Findings

### 1. No Image Augmentation

**Finding**: Image augmentation (brightness, contrast, rotation) **hurts performance** for document classification.

| Model | Augmentation | Test F1 | Triage FPs |
|-------|--------------|---------|------------|
| v2    | ❌ Disabled  | 0.9557  | 0          |
| v3    | ✅ Enabled   | 0.9620  | 5          |

Despite v3 having slightly higher token-level F1, it produced bizarre hallucinations:
- "IMMUNOCHEM" → EXAM_MAMMOGRAM (conf=0.82)
- "ENDOSCOPIC" → EXAM_TOMOSYNTHESIS (conf=0.69)
- "MENTAL" → EXAM_MAMMOGRAM (conf=0.93)

**Conclusion**: For document NER with consistent layout/quality, augmentation introduces spurious visual patterns.

### 2. Remove INDICATION Labels

**Finding**: Labels like `INDICATION_SCREENING` and `INDICATION_DIAGNOSTIC` cause cross-contamination.

The word "screening" appears in many document types:
- Colonoscopy reports ("colorectal cancer screening")
- Pap smear reports ("cervical cancer screening")
- General consult notes ("screening mammogram recommended")

**Solution**: Removed all INDICATION labels from the schema (33 → 29 labels).

For UDS numerator calculation, what matters is that the **test was performed**, not the indication for ordering it.

### 3. Negative Training Examples

**Finding**: Training on only positive mammogram examples causes the model to find mammogram entities everywhere.

**Solution**: Added 35 diverse negative PDFs (121 page images) to the training set with all "O" labels:
- Colonoscopy reports
- FIT/FOBT lab results
- Pap smear reports
- General imaging (CXR, DEXA, ultrasound)
- Consult notes (cardiology, neurology, oncology, etc.)

This teaches the model what mammogram entities should NOT look like.

### 4. Multi-Entity Triage Rule

**Finding**: Even with negative training, some documents that mention "mammogram" in text get flagged.

Example: Oncology consult note with "Last mammogram: 2024" in history section.

**Solution**: Post-processing rule requiring ≥2 different entity types:

```python
MAMMOGRAM_TRIAGE_TYPES = {'EXAM_MAMMOGRAM', 'EXAM_TOMOSYNTHESIS', 'BIRADS_CATEGORY', 'BREAST_DENSITY'}

def is_mammogram_report(entities, min_entity_types=2):
    mmg_entity_types = {e.entity_type for e in entities if e.entity_type in MAMMOGRAM_TRIAGE_TYPES}
    return len(mmg_entity_types) >= min_entity_types
```

**Rationale**: A real mammogram report always has multiple entity types (exam + BI-RADS, exam + density, etc.). A document that just mentions "mammogram" will only have EXAM_MAMMOGRAM.

---

## Label Schema

**File**: `src/labels_mammogram.py`

29 labels (14 entity types × 2 for B/I tagging + O):

| Entity Type | Description |
|-------------|-------------|
| EXAM_MAMMOGRAM | "mammogram", "mammography" |
| EXAM_TOMOSYNTHESIS | "tomosynthesis", "DBT", "3D" |
| EXAM_ULTRASOUND | "breast ultrasound" |
| EXAM_DATE | Date of the examination |
| REPORT_DATE | Date report was generated |
| BIRADS_CATEGORY | BI-RADS 0, 1, 2, 3, 4, 5, 6 |
| BREAST_DENSITY | ACR A, B, C, D or descriptive |
| LATERALITY_BILATERAL | "bilateral" |
| LATERALITY_LEFT | "left breast" |
| LATERALITY_RIGHT | "right breast" |
| FOLLOW_UP_INTERVAL | "12 months", "6 months", "immediate" |
| FINDING_MASS | Mass findings (manual annotation only) |
| FINDING_CALCIFICATION | Calcification findings (manual annotation only) |
| PROCEDURE_CPT | CPT procedure codes |

**Removed Labels** (caused cross-contamination):
- INDICATION_SCREENING - "screening" appears in colonoscopy, pap, etc.
- INDICATION_DIAGNOSTIC - Same issue
- RESULT_NEGATIVE/BENIGN/ABNORMAL - Caused FPs (e.g., "no suspicious findings")

---

## Dataset

**Location**: `data/dataset_mmg_triage_v2/`

| Split | Examples | Composition |
|-------|----------|-------------|
| Train | 135 | 52 positive + 83 negative pages |
| Validation | 29 | Mixed positive/negative |
| Test | 31 | Mixed positive/negative |
| **Total** | 195 | |

### Positive Examples
- Source: `data/labeled/mmg_labeled_*.json` (Label Studio exports)
- Images: `data/processed/mmg_triage/` (74 annotated mammogram pages)

### Negative Examples
- Source: `data/raw_pdfs/non_metric_pdf/` (35 diverse PDFs)
- Images: `data/processed/mmg_negative/` (121 page images)
- Labels: All tokens labeled as "O"

---

## Training Configuration

**File**: `config_mmg_triage.yaml`

```yaml
model:
  base_model: "microsoft/layoutlmv3-base"
  max_length: 512

training:
  output_dir: "./models/mmg_triage_v2"
  epochs: 30
  batch_size: 8
  learning_rate: 0.00003
  weight_decay: 0.01
  warmup_ratio: 0.1
  lr_scheduler_type: "cosine"
  early_stopping_patience: 5
  entity_boost: 5.0  # Weight multiplier for entity classes
  o_weight: 1.0
  augment_images: false  # IMPORTANT: Keep disabled

data:
  dataset: "./data/dataset_mmg_triage_v2"

labels:
  module: "src.labels_mammogram"
```

**Key Settings**:
- `augment_images: false` - Augmentation hurts performance
- `entity_boost: 5.0` - Upweight entity classes vs O
- `early_stopping_patience: 5` - Stop if no improvement
- `confidence_threshold: 0.6` - For inference

---

## Workflow

### 1. Prepare Negative Examples
```bash
python scripts/add_negative_examples.py
```
Processes PDFs from `data/raw_pdfs/non_metric_pdf/`, converts to images, creates training examples with O labels, merges with positive dataset.

### 2. Remap Labels (if schema changes)
```bash
python scripts/remap_mmg_labels.py
```
Converts existing dataset to new label schema when labels are added/removed.

### 3. Train Model
```bash
python -m src.train --config config_mmg_triage.yaml
```

### 4. Test Inference
```python
from src.inference import UDSExtractor, is_mammogram_report

extractor = UDSExtractor(
    model_path='models/mmg_triage_v2/final_model',
    device='cpu',
    confidence_threshold=0.6,
    labels_module='src.labels_mammogram'
)

result = extractor.extract_from_pdf('document.pdf')

# Triage: Is this a mammogram report?
if is_mammogram_report(result.entities):
    # Process mammogram entities
    for entity in result.entities:
        print(f"{entity.entity_type}: {entity.text}")
else:
    print("Not a mammogram report")
```

---

## File Structure

```
models/
├── mmg_triage_v2/          # Production model (no augmentation)
│   ├── final_model/
│   └── logs/
└── mmg_triage_v3/          # Experimental (with augmentation - worse)

data/
├── dataset_mmg_triage_v2/  # Training dataset with negatives
├── processed/
│   ├── mmg_triage/         # Positive mammogram images (74)
│   └── mmg_negative/       # Negative example images (121)
├── labeled/
│   └── mmg_labeled_*.json  # Label Studio annotations
└── raw_pdfs/
    └── non_metric_pdf/     # Source negative PDFs (35)

src/
├── labels_mammogram.py     # 29-label schema
├── inference.py            # UDSExtractor + is_mammogram_report()
├── train.py                # Training script
└── dataset.py              # Data loading + collator

scripts/
├── add_negative_examples.py    # Process negative PDFs
├── remap_mmg_labels.py         # Remap labels when schema changes
└── create_mmg_triage_dataset.py # Create dataset from labeled data

config_mmg_triage.yaml      # Training configuration
```

---

## Test Results

### Token-Level NER (Test Set)

| Metric | Value |
|--------|-------|
| F1 | 0.9557 |
| Precision | 0.924 |
| Recall | 0.990 |
| Epochs | 21 (early stopped) |

### Document Triage (72 Real Documents)

| Document Type | Count | Correct | Accuracy |
|---------------|-------|---------|----------|
| Mammogram reports | 19 | 19 TP | 100% |
| Non-mammogram (pdf_01_21/) | 21 | 21 TN | 100% |
| Negative examples (negative_pdf/) | 32 | 32 TN | 100% |
| **Total** | **72** | **72** | **100%** |

---

## Lessons Learned

1. **Negative examples are essential** for document triage models
2. **Avoid labels that appear across document types** (e.g., "screening")
3. **Image augmentation can hurt** document understanding tasks
4. **Post-processing rules** (multi-entity requirement) add robustness
5. **Confidence threshold of 0.6** filters noise without losing recall
6. **Entity extraction ≠ Document classification** - a model can extract entities well but still fail at document-level classification without proper triage logic

---

## Future Improvements

1. **More negative examples**: Add more diverse document types
2. **OCR confidence filtering**: Currently using 30, article recommends 80
3. **Entity linking**: Connect BIRADS to specific breast/laterality
4. **Date parsing**: Normalize extracted dates to standard format
5. **Multi-page aggregation**: Combine entities across pages for final report

---

## References

- LayoutLMv3: [microsoft/layoutlmv3-base](https://huggingface.co/microsoft/layoutlmv3-base)
- ClinicalLayoutLM paper: Improvements for clinical document understanding
- UDS Manual: HRSA Uniform Data System reporting requirements
