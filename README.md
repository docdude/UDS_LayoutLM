# UDS LayoutLM - Clinical Document Understanding for HRSA UDS Metrics

Extract HRSA UDS (Uniform Data System) quality metrics from clinical documents using **LayoutLMv3**, a multimodal transformer that understands both text and document layout.

## 🎯 Purpose

Automate extraction of UDS clinical quality measures from EHR documents (PDFs) to support FQHC (Federally Qualified Health Center) reporting requirements.

### Supported UDS Measures

| UDS Table | Measures Extracted |
|-----------|-------------------|
| **Table 6B** | Colorectal cancer screening (colonoscopy, FIT/FOBT) |
| **Table 6B** | Cervical & breast cancer screening |
| **Table 6B** | Depression screening & follow-up |
| **Table 6B** | Tobacco use screening & cessation |
| **Table 6A** | Hypertension control (BP < 140/90) |
| **Table 6A** | Diabetes control (A1C < 9%) |
| **Table 6B** | HIV screening |
| **Table 5A** | BMI screening |
| **Table 5A** | Immunizations (flu, COVID, pneumonia) |

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/UDS_LayoutLM.git
cd UDS_LayoutLM

# Install dependencies
pip install -r requirements.txt

# Install Poppler (PDF conversion) - Windows
choco install poppler
# Or: conda install -c conda-forge poppler

# Install Tesseract (OCR) - Windows
choco install tesseract
# Or download from: https://github.com/UB-Mannheim/tesseract/wiki
```

### 2. Prepare Documents for Labeling

```bash
# Process PDFs with OCR and export for Label Studio
python scripts/export_for_labeling.py --process-pdfs ./data/raw_pdfs
```

### 3. Label Data with Label Studio

```bash
# Start Label Studio with local files enabled
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT="$(pwd)/data"
label-studio start

# Then in Label Studio UI:
# 1. Create new project
# 2. Settings → Labeling Interface → paste contents of data/label_studio_config.xml
# 3. Settings → Cloud Storage → Add Local Storage → Path: "processed"
# 4. Import → Upload data/label_studio_import.json
# 5. Label 50-100 documents
# 6. Export as JSON to data/labeled/
```

### 4. Validate Annotations

```bash
python scripts/validate_annotations.py
```

### 5. Create Training Dataset

```bash
python scripts/create_training_data.py
```

### 6. Train Model

```bash
python -m src.train
```

### 7. Run Inference

```bash
# Single document
python -m src.inference ./document.pdf --model ./outputs/final_model

# Batch processing
python scripts/batch_inference.py ./new_documents --model ./outputs/final_model
```

## 📊 Architecture

```
Input PDF → OCR (Tesseract) → LayoutLMv3 → Extracted Entities
              ↓                      ↓
         Text + Boxes          Text + Layout + Vision
                                     ↓
                            UDS Metrics by Category
```

**Why LayoutLMv3?**
- Understands document **layout** (where text appears matters)
- Processes **visual features** (tables, forms, formatting)
- Pre-trained on millions of documents
- State-of-the-art for clinical document understanding

## 📁 Project Structure

```
UDS_LayoutLM/
├── config.yaml                 # Training configuration
├── requirements.txt            # Python dependencies
├── .gitignore
├── README.md
├── docs/
│   ├── annotation_guide.md    # Detailed labeling instructions
│   └── quick_reference.md     # Quick labeling cheat sheet
├── src/
│   ├── labels.py              # Entity definitions (60+ labels)
│   ├── processor.py           # PDF processing & OCR
│   ├── dataset.py             # Dataset creation
│   ├── train.py               # Training script
│   └── inference.py           # Production inference
├── scripts/
│   ├── prepare_data.py        # Process PDFs
│   ├── export_for_labeling.py # Export to Label Studio
│   ├── create_training_data.py# Create HuggingFace dataset
│   ├── batch_inference.py     # Batch processing
│   └── validate_annotations.py# Validate before training
└── data/
    ├── raw_pdfs/              # Source PDFs (not committed)
    ├── processed/             # OCR results (not committed)
    ├── labeled/               # Labeled data (not committed)
    └── dataset/               # Training dataset (not committed)
```

## 🏷️ Entity Labels

The model extracts **60+ entity types** organized by UDS measure:

### Colorectal Cancer Screening
- `COLONOSCOPY_DATE`, `COLONOSCOPY_RESULT`, `COLONOSCOPY_INDICATION`
- `POLYP_FINDING`, `POLYP_LOCATION`, `POLYP_SIZE`, `PATHOLOGY_DIAGNOSIS`
- `STOOL_TEST_TYPE`, `STOOL_TEST_RESULT` (FIT/FOBT)

### Other Cancer Screenings
- `CERVICAL_SCREEN`, `CERVICAL_SCREEN_DATE`
- `BREAST_SCREEN`, `BREAST_SCREEN_DATE`

### Chronic Disease Management
- `BLOOD_PRESSURE`, `A1C_VALUE`, `BMI`, `WEIGHT`, `HEIGHT`

### Behavioral Health
- `DEPRESSION_SCREEN`, `DEPRESSION_SCORE`
- `TOBACCO_STATUS`, `TOBACCO_COUNSELING`

### Preventive Care
- `HIV_SCREEN`, `VACCINATION`, `FLU_VACCINE`, `COVID_VACCINE`

[See full list in src/labels.py](src/labels.py)

## 📝 Annotation Guide

Key labeling principles:

1. **Label values only** - Not field names
   - ✅ `120/80` → `BLOOD_PRESSURE`
   - ❌ `BP: 120/80` → Don't include "BP:"

2. **Label complete entities**
   - ✅ `Tubular adenoma with low-grade dysplasia`
   - ❌ `Tubular adenoma` (incomplete)

3. **Codes without descriptions**
   - ✅ `Z12.11` → Just the code
   - ❌ `Z12.11 - CRC Screening` → Don't include description

See [docs/annotation_guide.md](docs/annotation_guide.md) for detailed instructions.

## 📈 Expected Performance

With **50-100 labeled documents**:
- Patient ID extraction: ~95% F1
- Date extraction: ~90% F1
- Clinical codes: ~85% F1
- Clinical findings: ~80% F1

With **200-300 labeled documents**:
- Most entities: ~90%+ F1

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License

## 🙏 Acknowledgments

- **LayoutLMv3** by Microsoft Research
- **HuggingFace Transformers**
- **Label Studio** for annotation tooling
- **HRSA** for UDS measure definitions

---

**Built for FQHC clinical quality reporting** 🏥
