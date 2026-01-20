# CRC Audit Application

## Overview

This directory contains the Streamlit-based audit dashboard for monthly UDS CRC screening metric audits.

## Components

### 1. Audit Dashboard (`app.py`)
Interactive web interface for chart auditors to:
- Load monthly patient list (100 patients with MRNs)
- View model predictions for each patient
- Record agree/disagree decisions
- Export audit results

### 2. API Integration (`../api/nextgen_client.py`)
NextGen Enterprise USCDI API integration for:
- Pre-filtering structured data (procedures, lab results)
- Downloading unstructured documents (PDFs) for model triage

**Status:** Placeholder - requires API credentials from NextGen Partner Program

## Running the Dashboard

```bash
# Activate virtual environment
source /opt/UDS_LayoutLM/.venv/bin/activate

# Install Streamlit if needed
pip install streamlit

# Run the dashboard
streamlit run audit_app/app.py
```

The dashboard will open at `http://localhost:8501`

## Workflow

### Monthly Audit Process

1. **Generate Patient List**
   - Quality team provides CSV with 100 randomized patients
   - Format: `mrn, patient_name, dob, pdf_folder`

2. **Stage 1: Structured Data Query** (Future - with API)
   - Query NextGen API for procedures (colonoscopy, sigmoidoscopy)
   - Query NextGen API for lab results (FIT, FOBT, FIT-DNA)
   - If structured evidence found → auto-approve

3. **Stage 2: Unstructured Document Triage**
   - Download relevant PDFs from EHR
   - Run LayoutLMv3 CRC triage model
   - Generate predictions with confidence scores

4. **Human Review**
   - Auditor reviews model predictions
   - Views source PDF documents
   - Records agree/disagree decision with notes

5. **Export Results**
   - Download audit results as CSV
   - Calculate agreement rate
   - Identify model improvement opportunities

## File Structure

```
audit_app/
├── README.md           # This file
├── app.py              # Streamlit dashboard
├── cache/              # Cached model extractions
│   └── {mrn}_extraction.json
└── exports/            # Audit result exports

api/
├── nextgen_client.py   # NextGen API integration
└── README.md           # API documentation
```

## Patient List CSV Format

```csv
mrn,patient_name,dob,pdf_folder
1944227,Demo Patient 1,1971-03-17,/path/to/pdfs
1936610,Demo Patient 2,1965-08-22,/path/to/pdfs
```

## Audit Decision Options

| Decision | Description |
|----------|-------------|
| `agree` | Model prediction is correct |
| `disagree_not_met` | Model said MET, but auditor says NOT MET |
| `disagree_met` | Model said NOT MET, but auditor found evidence |
| `unable_to_determine` | Needs supervisor review |

## Performance Metrics

Track monthly:
- **Agreement Rate**: % of cases where auditor agrees with model
- **False Positive Rate**: Model said MET incorrectly
- **False Negative Rate**: Model missed evidence
- **Average Review Time**: Time per patient

## Dependencies

```
streamlit>=1.28.0
pandas>=2.0.0
```

Add to requirements.txt:
```bash
echo "streamlit>=1.28.0" >> /opt/UDS_LayoutLM/requirements.txt
```
