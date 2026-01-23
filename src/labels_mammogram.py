"""Mammogram Triage labels - Focused on breast cancer screening document triage."""

# Mammogram Triage Labels - for UDS breast cancer screening numerator
# Designed for atomic labeling of key concepts for document triage
MAMMOGRAM_TRIAGE_LABELS = [
    "O",  # Outside any entity
    
    # === INDICATION - REMOVED ===
    # "screening" and "diagnostic" appear across many document types (colonoscopy, pap, etc.)
    # causing significant cross-contamination and false positives.
    # For UDS numerator, what matters is the TEST was done, not the indication.
    # REMOVED: B-INDICATION_SCREENING, I-INDICATION_SCREENING
    # REMOVED: B-INDICATION_DIAGNOSTIC, I-INDICATION_DIAGNOSTIC
    
    # === EXAM TYPE ===
    "B-EXAM_MAMMOGRAM", "I-EXAM_MAMMOGRAM",  # "mammogram", "mammography"
    "B-EXAM_TOMOSYNTHESIS", "I-EXAM_TOMOSYNTHESIS",  # "tomosynthesis", "DBT", "3D"
    "B-EXAM_ULTRASOUND", "I-EXAM_ULTRASOUND",  # "breast ultrasound"
    
    # === KEY DATES (Critical for measurement period) ===
    "B-EXAM_DATE", "I-EXAM_DATE",
    "B-REPORT_DATE", "I-REPORT_DATE",
    
    # === BI-RADS ASSESSMENT (Definitive result - use this instead of RESULT_* labels) ===
    "B-BIRADS_CATEGORY", "I-BIRADS_CATEGORY",
    
    # NOTE: RESULT_NEGATIVE, RESULT_BENIGN, RESULT_ABNORMAL removed
    # These caused false positives (e.g., 'suspicious' in 'no suspicious findings')
    # BI-RADS category provides the definitive result interpretation
    
    # === BREAST DENSITY (ACR categories) ===
    "B-BREAST_DENSITY", "I-BREAST_DENSITY",
    
    # === LATERALITY ===
    "B-LATERALITY_BILATERAL", "I-LATERALITY_BILATERAL",  # "bilateral"
    "B-LATERALITY_LEFT", "I-LATERALITY_LEFT",  # "left breast"
    "B-LATERALITY_RIGHT", "I-LATERALITY_RIGHT",  # "right breast"
    
    # === FOLLOW-UP ===
    "B-FOLLOW_UP_INTERVAL", "I-FOLLOW_UP_INTERVAL",  # "12 months", "6 months", "immediate"
    
    # === FINDINGS (Manual annotation only - not auto-labeled) ===
    # NOTE: Removed from auto-labeling - regex matches 'masses' in 'no suspicious masses'
    # Keep labels for manual annotation during review
    "B-FINDING_MASS", "I-FINDING_MASS",
    "B-FINDING_CALCIFICATION", "I-FINDING_CALCIFICATION",
    
    # === CLINICAL CODES ===
    "B-PROCEDURE_CPT", "I-PROCEDURE_CPT",
]

LABEL2ID = {label: idx for idx, label in enumerate(MAMMOGRAM_TRIAGE_LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(MAMMOGRAM_TRIAGE_LABELS)}
NUM_LABELS = len(MAMMOGRAM_TRIAGE_LABELS)

# Document type classification (for the choices field)
DOC_TYPES = [
    "screening_mammogram",
    "diagnostic_mammogram",
    "screening_with_tomosynthesis",
    "breast_ultrasound",
    "breast_mri",
    "not_mammogram",
    "other",
]

# BI-RADS Categories (for UDS: 1-2 typically meet numerator)
BIRADS_CATEGORIES = {
    0: "Incomplete - Need Additional Imaging",
    1: "Negative",
    2: "Benign",
    3: "Probably Benign",
    4: "Suspicious",
    5: "Highly Suggestive of Malignancy",
    6: "Known Biopsy-Proven Malignancy",
}

# Breast Density Categories (ACR)
DENSITY_CATEGORIES = {
    "A": "Almost entirely fatty",
    "B": "Scattered fibroglandular densities",
    "C": "Heterogeneously dense",
    "D": "Extremely dense",
}

# UDS Numerator logic:
# - Must be SCREENING mammogram (not diagnostic)
# - Must be within measurement period (check EXAM_DATE)
# - Any BI-RADS result counts (completed screening)

print(f"Mammogram Triage labels loaded: {NUM_LABELS} labels")
