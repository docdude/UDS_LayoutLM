"""CRC Triage labels - Simplified for document triage."""

# CRC Triage Labels - focused on key extraction for CRC UDS numerator determination
CRC_TRIAGE_LABELS = [
    "O",  # Outside any entity
    
    # === Document Type Indicators (Most Important) ===
    "B-DOC_TYPE_COLONOSCOPY", "I-DOC_TYPE_COLONOSCOPY",
    "B-DOC_TYPE_FIT", "I-DOC_TYPE_FIT",
    "B-DOC_TYPE_FOBT", "I-DOC_TYPE_FOBT",
    "B-DOC_TYPE_SIGMOIDOSCOPY", "I-DOC_TYPE_SIGMOIDOSCOPY",
    "B-DOC_TYPE_CT_COLONOGRAPHY", "I-DOC_TYPE_CT_COLONOGRAPHY",
    
    # === Critical Dates (For Lookback Calculation) ===
    "B-PROCEDURE_DATE", "I-PROCEDURE_DATE",
    "B-COLLECTION_DATE", "I-COLLECTION_DATE",
    "B-REPORT_DATE", "I-REPORT_DATE",
    
    # === Polyp Findings ===
    "B-POLYP_FINDING", "I-POLYP_FINDING",
    "B-POLYP_LOCATION", "I-POLYP_LOCATION",
    "B-POLYP_SIZE", "I-POLYP_SIZE",
    "B-POLYP_COUNT", "I-POLYP_COUNT",
    "B-BIOPSY_TAKEN", "I-BIOPSY_TAKEN",
    "B-BIOPSY_RESULT", "I-BIOPSY_RESULT",
    "B-PATHOLOGY_DIAGNOSIS", "I-PATHOLOGY_DIAGNOSIS",
    "B-COMPLICATIONS", "I-COMPLICATIONS",
    
    # === Test Results (For FIT/FOBT) ===
    "B-RESULT_POSITIVE", "I-RESULT_POSITIVE",
    "B-RESULT_NEGATIVE", "I-RESULT_NEGATIVE",
    "B-RESULT_VALUE", "I-RESULT_VALUE",
    
    # === Procedure Details (Secondary) ===
    "B-PROCEDURE_CPT", "I-PROCEDURE_CPT",
    "B-INDICATION_SCREENING", "I-INDICATION_SCREENING",
    "B-INDICATION_SURVEILLANCE", "I-INDICATION_SURVEILLANCE",
    "B-INDICATION_DIAGNOSTIC", "I-INDICATION_DIAGNOSTIC",
    "B-DIVERTICULA_FINDING", "I-DIVERTICULA_FINDING",
    "B-HEMORRHOIDS_FINDING", "I-HEMORRHOIDS_FINDING",
    
    # === Colon Anatomy (Procedure Completeness & Image Labels) ===
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
