"""CRC Triage labels - Simplified for document triage."""

# CRC Triage Labels - focused on key extraction for CRC UDS numerator determination
# 
# REMOVED LABELS (cause cross-contamination with non-CRC documents):
# - INDICATION_SCREENING, INDICATION_SURVEILLANCE, INDICATION_DIAGNOSTIC
#   "screening" appears in mammograms, pap smears, any preventive care
# - RESULT_POSITIVE, RESULT_NEGATIVE
#   Appears in any lab test (COVID, strep, pregnancy, etc.)
# - BIOPSY_TAKEN
#   Appears in breast biopsies, skin biopsies, etc.
#
# KEPT: BIOPSY_RESULT, PATHOLOGY_DIAGNOSIS (colon pathology context differs from pap/mmg)

CRC_TRIAGE_LABELS = [
    "O",  # Outside any entity
    
    # === Document Type Indicators (Most Important for Triage) ===
    "B-DOC_TYPE_COLONOSCOPY", "I-DOC_TYPE_COLONOSCOPY",
    "B-DOC_TYPE_FIT", "I-DOC_TYPE_FIT",
    "B-DOC_TYPE_FOBT", "I-DOC_TYPE_FOBT",
    "B-DOC_TYPE_SIGMOIDOSCOPY", "I-DOC_TYPE_SIGMOIDOSCOPY",
    "B-DOC_TYPE_CT_COLONOGRAPHY", "I-DOC_TYPE_CT_COLONOGRAPHY",
    
    # === Critical Dates (For Lookback Calculation) ===
    "B-PROCEDURE_DATE", "I-PROCEDURE_DATE",
    "B-COLLECTION_DATE", "I-COLLECTION_DATE",
    "B-REPORT_DATE", "I-REPORT_DATE",
    
    # === Polyp Findings (CRC-Specific) ===
    "B-POLYP_FINDING", "I-POLYP_FINDING",
    "B-POLYP_LOCATION", "I-POLYP_LOCATION",
    "B-POLYP_SIZE", "I-POLYP_SIZE",
    "B-POLYP_COUNT", "I-POLYP_COUNT",
    "B-BIOPSY_RESULT", "I-BIOPSY_RESULT",
    "B-PATHOLOGY_DIAGNOSIS", "I-PATHOLOGY_DIAGNOSIS",
    "B-COMPLICATIONS", "I-COMPLICATIONS",
    
    # === Test Results (For FIT/FOBT) ===
    "B-RESULT_VALUE", "I-RESULT_VALUE",
    
    # === Procedure Details ===
    "B-PROCEDURE_CPT", "I-PROCEDURE_CPT",
    "B-DIVERTICULA_FINDING", "I-DIVERTICULA_FINDING",
    "B-HEMORRHOIDS_FINDING", "I-HEMORRHOIDS_FINDING",
    
    # === Colon Anatomy (Procedure Completeness - CRC-Specific) ===
    "B-COLON_ANATOMY", "I-COLON_ANATOMY",  # cecum, ascending, transverse, sigmoid, etc.
]

LABEL2ID = {label: idx for idx, label in enumerate(CRC_TRIAGE_LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(CRC_TRIAGE_LABELS)}
NUM_LABELS = len(CRC_TRIAGE_LABELS)

# Document type classification (for the choices field)
DOC_TYPES = [
    "colonoscopy_report",
    "fit_result", 
    "fobt_result",
    "sigmoidoscopy_report",
    "ct_colonography",
    "not_crc_relevant",
    "gi_note",
    "colon_pathology_report",
]

print(f"CRC Triage labels loaded: {NUM_LABELS} labels")
