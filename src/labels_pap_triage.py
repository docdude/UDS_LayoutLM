"""PAP/Cervical Cancer Triage labels - for UDS cervical cancer screening numerator."""

# PAP Triage Labels - designed for extracting key concepts from cervical screening documents
# Labels align with Label Studio config: label_studio_config_pap.xml
#
# DESIGN NOTES:
# - DOC_TYPE_PAP/HPV: Identifies document as cervical screening vs other lab reports
# - COLLECTION_DATE: Critical for UDS measurement period compliance
# - RESULT_*: Bethesda system results (NILM, ASC-US, LSIL, HSIL, etc.)
# - HPV_*: HPV co-testing results
# - SPECIMEN_*: Adequacy for interpretation
#
# REMOVED from auto-labeling (per model_pap.py notes):
# - Generic terms like "satisfactory", "adequate" without cervical context
# - "screening", "routine" which appear in many document types

PAP_TRIAGE_LABELS = [
    "O",  # Outside any entity
    
    # === DOCUMENT TYPE ===
    "B-DOC_TYPE_PAP", "I-DOC_TYPE_PAP",  # pap smear, pap test, cervical cytology, ThinPrep
    "B-DOC_TYPE_HPV", "I-DOC_TYPE_HPV",  # HPV test, HPV aptima, high-risk HPV
    
    # === KEY DATES (Critical for measurement period) ===
    "B-COLLECTION_DATE", "I-COLLECTION_DATE",  # specimen collection date
    
    # === SPECIMEN ADEQUACY ===
    "B-SPECIMEN_ADEQUATE", "I-SPECIMEN_ADEQUATE",  # endocervical component identified
    "B-SPECIMEN_INADEQUATE", "I-SPECIMEN_INADEQUATE",  # unsatisfactory for evaluation
    "B-SPECIMEN_TYPE", "I-SPECIMEN_TYPE",  # ThinPrep, liquid-based, cervical specimen
    
    # === PAP RESULTS - Bethesda System ===
    "B-RESULT_NILM", "I-RESULT_NILM",  # Negative for intraepithelial lesion or malignancy
    "B-RESULT_ASCUS", "I-RESULT_ASCUS",  # Atypical squamous cells of undetermined significance
    "B-RESULT_ASCH", "I-RESULT_ASCH",  # Atypical squamous cells, cannot exclude HSIL
    "B-RESULT_LSIL", "I-RESULT_LSIL",  # Low-grade squamous intraepithelial lesion (CIN 1)
    "B-RESULT_HSIL", "I-RESULT_HSIL",  # High-grade squamous intraepithelial lesion (CIN 2/3)
    "B-RESULT_AGC", "I-RESULT_AGC",  # Atypical glandular cells
    
    # === HPV RESULTS ===
    "B-HPV_POSITIVE", "I-HPV_POSITIVE",  # HPV detected/positive
    "B-HPV_NEGATIVE", "I-HPV_NEGATIVE",  # HPV not detected/negative
    "B-HPV_GENOTYPE", "I-HPV_GENOTYPE",  # HPV 16/18 genotyping
    
    # === PROCEDURE CODES ===
    "B-PROCEDURE_CPT", "I-PROCEDURE_CPT",  # 88141-88175 (Pap), 87624-87625 (HPV)
]

LABEL2ID = {label: idx for idx, label in enumerate(PAP_TRIAGE_LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(PAP_TRIAGE_LABELS)}
NUM_LABELS = len(PAP_TRIAGE_LABELS)

# Document type classification (for the choices field in Label Studio)
DOC_TYPES = [
    "pap_result",
    "hpv_result", 
    "pap_hpv_cotest",
    "cervical_mention_in_note",
    "not_cervical_screening",
]

# Bethesda System Categories (for reference)
BETHESDA_CATEGORIES = {
    "NILM": "Negative for Intraepithelial Lesion or Malignancy",
    "ASC-US": "Atypical Squamous Cells of Undetermined Significance",
    "ASC-H": "Atypical Squamous Cells, cannot exclude HSIL",
    "LSIL": "Low-grade Squamous Intraepithelial Lesion",
    "HSIL": "High-grade Squamous Intraepithelial Lesion",
    "AGC": "Atypical Glandular Cells",
    "AIS": "Adenocarcinoma in situ",
}

# CIN Grading (histologic, correlates with Bethesda)
CIN_GRADES = {
    "CIN 1": "Mild dysplasia (correlates with LSIL)",
    "CIN 2": "Moderate dysplasia (correlates with HSIL)",
    "CIN 3": "Severe dysplasia/Carcinoma in situ (correlates with HSIL)",
}

# UDS Cervical Cancer Screening Notes:
# - Ages 21-64 for Pap alone or Pap+HPV co-testing
# - Ages 30-64 for primary HPV testing
# - Lookback: 3 years for Pap, 5 years for HPV or co-testing
# - Result (NILM, abnormal) doesn't affect numerator compliance

print(f"PAP Triage labels loaded: {NUM_LABELS} labels")
