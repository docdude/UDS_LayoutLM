"""UDS Metrics entity labels - Complete for all UDS measures."""

# Complete UDS Labels
UDS_LABELS = [
    "O",  # Outside any entity
    
    # === Patient Demographics ===
    "B-PATIENT_ID", "I-PATIENT_ID",
    "B-PATIENT_NAME", "I-PATIENT_NAME",
    "B-DATE_OF_BIRTH", "I-DATE_OF_BIRTH",
    "B-GENDER", "I-GENDER",
    "B-DATE_OF_SERVICE", "I-DATE_OF_SERVICE",
    "B-PROVIDER_NAME", "I-PROVIDER_NAME",
    "B-PROVIDER_NPI", "I-PROVIDER_NPI",
    "B-FACILITY_NAME", "I-FACILITY_NAME",
    
    # === Clinical Codes ===
    "B-DIAGNOSIS_ICD10", "I-DIAGNOSIS_ICD10",
    "B-PROCEDURE_CPT", "I-PROCEDURE_CPT",
    
    # === Vitals (UDS Table 6A/6B) ===
    "B-BLOOD_PRESSURE", "I-BLOOD_PRESSURE",
    "B-BLOOD_PRESSURE_SYSTOLIC", "I-BLOOD_PRESSURE_SYSTOLIC",
    "B-BLOOD_PRESSURE_DIASTOLIC", "I-BLOOD_PRESSURE_DIASTOLIC",
    "B-A1C_VALUE", "I-A1C_VALUE",
    "B-BMI", "I-BMI",
    "B-WEIGHT", "I-WEIGHT",
    "B-HEIGHT", "I-HEIGHT",
    
    # === Screenings (UDS Table 6B) ===
    "B-DEPRESSION_SCREEN", "I-DEPRESSION_SCREEN",
    "B-DEPRESSION_SCORE", "I-DEPRESSION_SCORE",
    "B-TOBACCO_STATUS", "I-TOBACCO_STATUS",
    "B-TOBACCO_COUNSELING", "I-TOBACCO_COUNSELING",
    "B-HIV_SCREEN", "I-HIV_SCREEN",
    "B-HIV_RESULT", "I-HIV_RESULT",
    "B-CERVICAL_SCREEN", "I-CERVICAL_SCREEN",
    "B-CERVICAL_SCREEN_DATE", "I-CERVICAL_SCREEN_DATE",
    "B-BREAST_SCREEN", "I-BREAST_SCREEN",
    "B-BREAST_SCREEN_DATE", "I-BREAST_SCREEN_DATE",
    
    # === Colorectal Cancer Screening ===
    "B-COLONOSCOPY_DATE", "I-COLONOSCOPY_DATE",
    "B-COLONOSCOPY_RESULT", "I-COLONOSCOPY_RESULT",
    "B-COLONOSCOPY_INDICATION", "I-COLONOSCOPY_INDICATION",
    "B-BOWEL_PREP_QUALITY", "I-BOWEL_PREP_QUALITY",
    "B-CECUM_REACHED", "I-CECUM_REACHED",
    "B-WITHDRAWAL_TIME", "I-WITHDRAWAL_TIME",
    
    # === Polyp Findings ===
    "B-POLYP_FINDING", "I-POLYP_FINDING",
    "B-POLYP_LOCATION", "I-POLYP_LOCATION",
    "B-POLYP_SIZE", "I-POLYP_SIZE",
    "B-POLYP_COUNT", "I-POLYP_COUNT",
    "B-BIOPSY_TAKEN", "I-BIOPSY_TAKEN",
    "B-BIOPSY_RESULT", "I-BIOPSY_RESULT",
    "B-PATHOLOGY_DIAGNOSIS", "I-PATHOLOGY_DIAGNOSIS",
    "B-COMPLICATIONS", "I-COMPLICATIONS",
    
    # === Stool Tests (FIT/FOBT) ===
    "B-STOOL_TEST_TYPE", "I-STOOL_TEST_TYPE",
    "B-STOOL_TEST_DATE", "I-STOOL_TEST_DATE",
    "B-STOOL_TEST_RESULT", "I-STOOL_TEST_RESULT",
    "B-STOOL_TEST_VALUE", "I-STOOL_TEST_VALUE",
    "B-REFERENCE_RANGE", "I-REFERENCE_RANGE",
    
    # === Immunizations ===
    "B-VACCINATION", "I-VACCINATION",
    "B-VACCINATION_DATE", "I-VACCINATION_DATE",
    "B-VACCINATION_TYPE", "I-VACCINATION_TYPE",
    "B-FLU_VACCINE", "I-FLU_VACCINE",
    "B-COVID_VACCINE", "I-COVID_VACCINE",
    "B-PNEUMONIA_VACCINE", "I-PNEUMONIA_VACCINE",
    
    # === Lab Results ===
    "B-LAB_NAME", "I-LAB_NAME",
    "B-LAB_VALUE", "I-LAB_VALUE",
    "B-LAB_DATE", "I-LAB_DATE",
    "B-LAB_UNIT", "I-LAB_UNIT",
    "B-LAB_RESULT", "I-LAB_RESULT",
    "B-LAB_REFERENCE_RANGE", "I-LAB_REFERENCE_RANGE",
    
    # === Medications ===
    "B-MEDICATION", "I-MEDICATION",
    "B-MEDICATION_DOSE", "I-MEDICATION_DOSE",
    "B-MEDICATION_FREQUENCY", "I-MEDICATION_FREQUENCY",
    "B-STATIN_THERAPY", "I-STATIN_THERAPY",
    "B-ASPIRIN_THERAPY", "I-ASPIRIN_THERAPY",
    
    # === Prenatal Care ===
    "B-PRENATAL_VISIT", "I-PRENATAL_VISIT",
    "B-GESTATIONAL_AGE", "I-GESTATIONAL_AGE",
    "B-EDD", "I-EDD",
    "B-PRENATAL_LABS", "I-PRENATAL_LABS",
    
    # === Follow-up & Recommendations ===
    "B-NEXT_SCREENING_DATE", "I-NEXT_SCREENING_DATE",
    "B-SCREENING_INTERVAL", "I-SCREENING_INTERVAL",
    "B-RECOMMENDATION", "I-RECOMMENDATION",
    "B-RISK_CATEGORY", "I-RISK_CATEGORY",
    "B-FOLLOW_UP_DATE", "I-FOLLOW_UP_DATE",
]

LABEL2ID = {label: idx for idx, label in enumerate(UDS_LABELS)}
ID2LABEL = {idx: label for idx, label in enumerate(UDS_LABELS)}
NUM_LABELS = len(UDS_LABELS)

# UDS Measure Mapping
UDS_MEASURES = {
    # Colorectal Cancer Screening
    "colorectal_cancer_screening": [
        "COLONOSCOPY_DATE", "COLONOSCOPY_RESULT",
        "STOOL_TEST_TYPE", "STOOL_TEST_DATE", "STOOL_TEST_RESULT",
    ],
    "colonoscopy_quality": [
        "BOWEL_PREP_QUALITY", "CECUM_REACHED", "WITHDRAWAL_TIME",
    ],
    "polyp_findings": [
        "POLYP_FINDING", "POLYP_LOCATION", "POLYP_SIZE",
        "BIOPSY_RESULT", "PATHOLOGY_DIAGNOSIS",
    ],
    
    # Other Cancer Screenings
    "cervical_cancer_screening": [
        "CERVICAL_SCREEN", "CERVICAL_SCREEN_DATE",
    ],
    "breast_cancer_screening": [
        "BREAST_SCREEN", "BREAST_SCREEN_DATE",
    ],
    
    # Chronic Disease Management
    "hypertension_control": [
        "BLOOD_PRESSURE", "BLOOD_PRESSURE_SYSTOLIC", "BLOOD_PRESSURE_DIASTOLIC",
    ],
    "diabetes_control": [
        "A1C_VALUE",
    ],
    "statin_therapy": [
        "STATIN_THERAPY",
    ],
    
    # Behavioral Health
    "depression_screening": [
        "DEPRESSION_SCREEN", "DEPRESSION_SCORE",
    ],
    "tobacco_screening": [
        "TOBACCO_STATUS", "TOBACCO_COUNSELING",
    ],
    
    # Preventive Care
    "hiv_screening": [
        "HIV_SCREEN", "HIV_RESULT",
    ],
    "adult_bmi": [
        "BMI", "WEIGHT", "HEIGHT",
    ],
    "immunizations": [
        "VACCINATION", "VACCINATION_DATE", "VACCINATION_TYPE",
        "FLU_VACCINE", "COVID_VACCINE", "PNEUMONIA_VACCINE",
    ],
    
    # Prenatal
    "prenatal_care": [
        "PRENATAL_VISIT", "GESTATIONAL_AGE", "EDD", "PRENATAL_LABS",
    ],
    
    # Follow-up
    "screening_followup": [
        "NEXT_SCREENING_DATE", "SCREENING_INTERVAL",
        "RECOMMENDATION", "FOLLOW_UP_DATE",
    ],
}

# Reference Code Mappings
CPT_CODES = {
    # Colonoscopy
    "45378": "Colonoscopy, diagnostic",
    "45380": "Colonoscopy with biopsy",
    "45385": "Colonoscopy with polypectomy",
    # Stool Tests
    "82270": "FOBT (guaiac)",
    "82274": "FIT",
    "81528": "Cologuard",
    # Other Screenings
    "77067": "Mammography, screening",
    "77066": "Mammography, diagnostic",
    "88175": "Pap smear, liquid-based",
    "87624": "HPV test",
    "86703": "HIV antibody test",
    "87389": "HIV antigen/antibody combo",
    # Labs
    "83036": "HbA1c",
    "80061": "Lipid panel",
    # Immunizations
    "90686": "Flu vaccine (quadrivalent)",
    "90471": "Immunization administration",
    "91300": "COVID-19 vaccine (Pfizer)",
    "91301": "COVID-19 vaccine (Moderna)",
}

ICD10_CODES = {
    # Screening encounters
    "Z12.11": "Encounter for screening for malignant neoplasm of colon",
    "Z12.12": "Encounter for screening for malignant neoplasm of rectum",
    "Z12.31": "Encounter for screening mammogram",
    "Z12.4": "Encounter for screening for cervical cancer",
    "Z11.4": "Encounter for screening for HIV",
    # History
    "Z86.010": "Personal history of colonic polyps",
    "Z85.038": "Personal history of malignant neoplasm of large intestine",
    "Z80.0": "Family history of malignant neoplasm of digestive organs",
    # Chronic conditions
    "E11.9": "Type 2 diabetes mellitus without complications",
    "E11.65": "Type 2 DM with hyperglycemia",
    "I10": "Essential hypertension",
    # Behavioral
    "F32.9": "Major depressive disorder, single episode",
    "F17.210": "Nicotine dependence, cigarettes",
    "Z87.891": "Personal history of nicotine dependence",
}