"""CRC Triage labels - Simplified for document triage."""

# CRC Triage Labels - focused on key extraction for report classification
CRC_TRIAGE_LABELS = [
    "O",  # Outside any entity
    
    # === Document Type Indicators ===
    "B-DOC_TYPE_COLONOSCOPY", "I-DOC_TYPE_COLONOSCOPY",
    "B-DOC_TYPE_FIT", "I-DOC_TYPE_FIT",
    "B-DOC_TYPE_FOBT", "I-DOC_TYPE_FOBT",
    
    # === Key Dates ===
    "B-PROCEDURE_DATE", "I-PROCEDURE_DATE",
    "B-REPORT_DATE", "I-REPORT_DATE",
    "B-COLLECTION_DATE", "I-COLLECTION_DATE",
    
    # === Clinical Codes ===
    "B-PROCEDURE_CPT", "I-PROCEDURE_CPT",
    
    # === Indications ===
    "B-INDICATION_SCREENING", "I-INDICATION_SCREENING",
    "B-INDICATION_SURVEILLANCE", "I-INDICATION_SURVEILLANCE",
    "B-INDICATION_DIAGNOSTIC", "I-INDICATION_DIAGNOSTIC",
    
    # === Results ===
    "B-RESULT_NEGATIVE", "I-RESULT_NEGATIVE",
    "B-RESULT_POSITIVE", "I-RESULT_POSITIVE",
]

LABEL2ID = {label: idx for idx, label in enumerate(CRC_TRIAGE_LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(CRC_TRIAGE_LABELS)}
NUM_LABELS = len(CRC_TRIAGE_LABELS)

# Document type classification (for the choices field)
DOC_TYPES = [
    "colonoscopy_report",
    "fit_result", 
    "fobt_result",
    "gi_note",
    "other",
]

print(f"CRC Triage labels loaded: {NUM_LABELS} labels")
