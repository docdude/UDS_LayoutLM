# CRC Screening Document Annotation Guide

## Overview

This guide provides instructions for annotating colorectal cancer (CRC) screening documents for the UDS Metrics extraction model. The goal is to identify and label key clinical information that determines whether a patient has completed appropriate CRC screening.

---

## Document Types

### 1. Colonoscopy Reports
Full procedure reports from colonoscopy examinations.

### 2. Pathology Reports  
Biopsy results from tissue samples taken during colonoscopy.

### 3. FIT/FOBT Results
Fecal immunochemical test or fecal occult blood test laboratory results.

### 4. Cologuard Results
Stool DNA test results.

---

## Labeling Instructions

### General Rules

1. **Label complete phrases** - Include all words that form a complete entity
2. **Be consistent** - Same information should always get the same label
3. **When in doubt, label it** - Over-labeling is better than under-labeling
4. **Include units** - For measurements, include the unit (mm, cm, etc.)

### Keyboard Shortcuts

| Key | Label |
|-----|-------|
| 1 | PATIENT_ID |
| 2 | DATE_OF_SERVICE |
| 3 | DIAGNOSIS_ICD10 |
| 4 | PROCEDURE_CPT |
| 5 | COLONOSCOPY_DATE |
| 6 | COLONOSCOPY_RESULT |
| 7 | POLYP_FINDING |
| 8 | PATHOLOGY_DIAGNOSIS |
| 9 | STOOL_TEST_TYPE |
| 0 | STOOL_TEST_RESULT |

---

## Entity Definitions & Examples

### Patient Demographics

#### PATIENT_ID
Medical record number or patient identifier.

**Examples:**
- `MRN: 1079665` → Label: "1079665"
- `Patient ID: 12969` → Label: "12969"  
- `Acct#: 456789` → Label: "456789"

#### PATIENT_NAME
Full patient name.

**Examples:**
- `Patient: John Smith` → Label: "John Smith"
- `SMITH, JOHN A` → Label: "SMITH, JOHN A"

#### DATE_OF_BIRTH
Patient's birth date.

**Examples:**
- `DOB: 05/15/1960` → Label: "05/15/1960"
- `Birth Date: May 15, 1960` → Label: "May 15, 1960"

#### DATE_OF_SERVICE
Date the procedure or test was performed.

**Examples:**
- `Date of Service: 03/15/2024` → Label: "03/15/2024"
- `Procedure Date: March 15, 2024` → Label: "March 15, 2024"
- `DOS: 3/15/24` → Label: "3/15/24"

#### PROVIDER_NAME
Name of performing or ordering physician.

**Examples:**
- `Physician: Dr. Jane Doe, MD` → Label: "Dr. Jane Doe, MD"
- `Performed by: J. Smith, DO` → Label: "J. Smith, DO"

---

### Clinical Codes

#### DIAGNOSIS_ICD10
ICD-10 diagnosis codes.

**Examples:**
- `Diagnosis: Z12.11` → Label: "Z12.11"
- `ICD-10: K63.5, Z86.010` → Label each code separately
- `Primary Dx: Z12.11 - Colorectal cancer screening` → Label: "Z12.11"

**Common CRC ICD-10 Codes:**
| Code | Description |
|------|-------------|
| Z12.11 | Screening for colon cancer |
| Z12.12 | Screening for rectal cancer |
| Z86.010 | History of colonic polyps |
| K63.5 | Polyp of colon |
| Z80.0 | Family history of GI malignancy |

#### PROCEDURE_CPT
CPT procedure codes.

**Examples:**
- `CPT: 45378` → Label: "45378"
- `Procedure Code: 45385` → Label: "45385"
- `82274 - FIT` → Label: "82274"

**Common CRC CPT Codes:**
| Code | Description |
|------|-------------|
| 45378 | Diagnostic colonoscopy |
| 45380 | Colonoscopy with biopsy |
| 45385 | Colonoscopy with polypectomy |
| 82274 | FIT test |
| 82270 | FOBT |
| 81528 | Cologuard |

---

### Colonoscopy-Specific Entities

#### COLONOSCOPY_DATE
Specific date of colonoscopy procedure.

**Examples:**
- `Colonoscopy performed on 03/15/2024` → Label: "03/15/2024"
- `Procedure Date: March 15, 2024` → Label: "March 15, 2024"

#### COLONOSCOPY_RESULT
Overall result or impression of colonoscopy.

**Examples:**
- `Impression: Normal colonoscopy` → Label: "Normal colonoscopy"
- `Result: Polyps found and removed` → Label: "Polyps found and removed"
- `Findings: Diverticulosis, no polyps` → Label: "Diverticulosis, no polyps"
- `IMPRESSION: Unremarkable colonoscopy to cecum` → Label: "Unremarkable colonoscopy to cecum"

#### COLONOSCOPY_INDICATION
Reason for the colonoscopy.

**Examples:**
- `Indication: Screening colonoscopy` → Label: "Screening colonoscopy"
- `Reason: Surveillance for history of polyps` → Label: "Surveillance for history of polyps"
- `Indication: Family history of colon cancer` → Label: "Family history of colon cancer"
- `Indication: Positive FIT test` → Label: "Positive FIT test"

#### BOWEL_PREP_QUALITY
Quality of bowel preparation.

**Examples:**
- `Prep Quality: Excellent` → Label: "Excellent"
- `Bowel preparation was good` → Label: "good"
- `Prep: Fair, some residual stool` → Label: "Fair, some residual stool"
- `Boston Bowel Prep Score: 8` → Label: "8"

**Quality Levels:**
- Excellent / Good / Adequate = Successful exam
- Fair / Poor / Inadequate = May need repeat

#### CECUM_REACHED
Whether the colonoscope reached the cecum (complete exam).

**Examples:**
- `Cecum reached: Yes` → Label: "Yes"
- `Intubated to cecum` → Label: "Intubated to cecum"
- `Complete colonoscopy to cecum` → Label: "Complete colonoscopy to cecum"
- `Unable to reach cecum due to obstruction` → Label: "Unable to reach cecum due to obstruction"

#### WITHDRAWAL_TIME
Time spent withdrawing the scope (quality indicator, should be ≥6 min).

**Examples:**
- `Withdrawal time: 8 minutes` → Label: "8 minutes"
- `Scope withdrawal: 10 min` → Label: "10 min"

---

### Polyp Findings

#### POLYP_FINDING
Description of polyp(s) found.

**Examples:**
- `Single sessile polyp found` → Label: "Single sessile polyp found"
- `Multiple polyps identified` → Label: "Multiple polyps identified"
- `No polyps visualized` → Label: "No polyps visualized"
- `3mm pedunculated polyp` → Label: "3mm pedunculated polyp"

#### POLYP_LOCATION
Anatomical location of polyp.

**Examples:**
- `Location: Ascending colon` → Label: "Ascending colon"
- `Polyp in sigmoid colon` → Label: "sigmoid colon"
- `Cecal polyp` → Label: "Cecal"
- `Found at 30cm from anal verge` → Label: "30cm from anal verge"

**Colon Segments (proximal to distal):**
1. Cecum
2. Ascending colon
3. Hepatic flexure
4. Transverse colon
5. Splenic flexure
6. Descending colon
7. Sigmoid colon
8. Rectum

#### POLYP_SIZE
Size of polyp in mm or cm.

**Examples:**
- `Size: 5mm` → Label: "5mm"
- `8 mm polyp` → Label: "8 mm"
- `1.2 cm sessile lesion` → Label: "1.2 cm"

#### POLYP_COUNT
Number of polyps found.

**Examples:**
- `3 polyps found` → Label: "3"
- `Multiple (>10) polyps` → Label: "Multiple (>10)"
- `Single polyp` → Label: "Single"

#### BIOPSY_TAKEN
Whether biopsy was performed.

**Examples:**
- `Biopsy: Yes` → Label: "Yes"
- `Cold biopsy performed` → Label: "Cold biopsy performed"
- `Specimen sent to pathology` → Label: "Specimen sent to pathology"

#### BIOPSY_RESULT
Result of biopsy/polypectomy.

**Examples:**
- `Biopsy result: Tubular adenoma` → Label: "Tubular adenoma"
- `Pathology: Hyperplastic polyp` → Label: "Hyperplastic polyp"
- `Final path: Adenomatous polyp with low-grade dysplasia` → Label: "Adenomatous polyp with low-grade dysplasia"

#### PATHOLOGY_DIAGNOSIS
Final pathological diagnosis.

**Examples:**
- `Diagnosis: Tubular adenoma` → Label: "Tubular adenoma"
- `Tubulovillous adenoma with high-grade dysplasia` → Label: "Tubulovillous adenoma with high-grade dysplasia"
- `Hyperplastic polyp` → Label: "Hyperplastic polyp"
- `Sessile serrated adenoma` → Label: "Sessile serrated adenoma"

**Polyp Types (risk from low to high):**
| Type | Follow-up |
|------|-----------|
| Hyperplastic | 10 years |
| Tubular adenoma (<10mm) | 5-10 years |
| Tubular adenoma (≥10mm) | 3 years |
| Tubulovillous adenoma | 3 years |
| Villous adenoma | 3 years |
| Sessile serrated adenoma | 3-5 years |
| Any adenoma with HGD | 3 years |

#### COMPLICATIONS
Any complications during procedure.

**Examples:**
- `Complications: None` → Label: "None"
- `No immediate complications` → Label: "No immediate complications"
- `Complication: Minor bleeding, controlled with clips` → Label: "Minor bleeding, controlled with clips"

---

### FIT/FOBT Specific

#### STOOL_TEST_TYPE
Type of stool test performed.

**Examples:**
- `Test: FIT` → Label: "FIT"
- `Fecal Immunochemical Test` → Label: "Fecal Immunochemical Test"
- `FOBT (guaiac)` → Label: "FOBT (guaiac)"
- `iFOBT` → Label: "iFOBT"
- `Cologuard` → Label: "Cologuard"
- `Stool DNA test` → Label: "Stool DNA test"

#### STOOL_TEST_DATE
Date stool test was collected or resulted.

**Examples:**
- `Collection Date: 03/10/2024` → Label: "03/10/2024"
- `Resulted: 3/12/2024` → Label: "3/12/2024"

#### STOOL_TEST_RESULT
Result of stool test.

**Examples:**
- `Result: Negative` → Label: "Negative"
- `POSITIVE` → Label: "POSITIVE"
- `FIT: Negative for occult blood` → Label: "Negative for occult blood"
- `No blood detected` → Label: "No blood detected"
- `Cologuard: Positive` → Label: "Positive"

#### STOOL_TEST_VALUE
Quantitative value if provided.

**Examples:**
- `Value: <50 ng/mL` → Label: "<50 ng/mL"
- `Hemoglobin: 25 ng/mL` → Label: "25 ng/mL"
- `Result: 0.0 ug/g` → Label: "0.0 ug/g"

#### REFERENCE_RANGE
Reference range for the test.

**Examples:**
- `Reference: <50 ng/mL` → Label: "<50 ng/mL"
- `Normal: Negative` → Label: "Negative"
- `Cutoff: 100 ng/mL` → Label: "100 ng/mL"

---

### Follow-up & Recommendations

#### NEXT_SCREENING_DATE
Recommended date for next screening.

**Examples:**
- `Next colonoscopy: March 2029` → Label: "March 2029"
- `Repeat in 2027` → Label: "2027"
- `F/U colonoscopy: 5 years` → Label: "5 years"

#### SCREENING_INTERVAL
Recommended interval between screenings.

**Examples:**
- `Interval: 10 years` → Label: "10 years"
- `Repeat colonoscopy in 3 years` → Label: "3 years"
- `Annual FIT recommended` → Label: "Annual"
- `Surveillance every 5 years` → Label: "every 5 years"

#### RECOMMENDATION
Clinical recommendation.

**Examples:**
- `Recommend repeat colonoscopy in 5 years` → Label: "Recommend repeat colonoscopy in 5 years"
- `Continue annual FIT screening` → Label: "Continue annual FIT screening"
- `Refer to GI for colonoscopy` → Label: "Refer to GI for colonoscopy"

#### RISK_CATEGORY
Patient risk stratification.

**Examples:**
- `Average risk` → Label: "Average risk"
- `High risk due to family history` → Label: "High risk due to family history"
- `Increased risk - history of adenomas` → Label: "Increased risk - history of adenomas"

---

## Common Scenarios

### Scenario 1: Normal Screening Colonoscopy

**Key entities to label:**
- PATIENT_ID
- DATE_OF_SERVICE / COLONOSCOPY_DATE
- COLONOSCOPY_INDICATION: "Screening"
- COLONOSCOPY_RESULT: "Normal"
- CECUM_REACHED: "Yes"
- PROCEDURE_CPT: "45378"
- DIAGNOSIS_ICD10: "Z12.11"
- NEXT_SCREENING_DATE or SCREENING_INTERVAL: "10 years"

### Scenario 2: Colonoscopy with Polyp Removal

**Key entities to label:**
- All demographic entities
- COLONOSCOPY_DATE
- POLYP_FINDING: Description
- POLYP_LOCATION: Where found
- POLYP_SIZE: Size in mm
- BIOPSY_RESULT or PATHOLOGY_DIAGNOSIS: Polyp type
- PROCEDURE_CPT: "45385" (polypectomy)
- SCREENING_INTERVAL: Based on findings (typically 3-5 years)

### Scenario 3: Negative FIT Result

**Key entities to label:**
- PATIENT_ID
- STOOL_TEST_TYPE: "FIT"
- STOOL_TEST_DATE
- STOOL_TEST_RESULT: "Negative"
- STOOL_TEST_VALUE: If quantitative
- PROCEDURE_CPT: "82274"

### Scenario 4: Positive FIT Result

**Key entities to label:**
- All FIT entities above with STOOL_TEST_RESULT: "Positive"
- RECOMMENDATION: Usually "Refer for colonoscopy"

---

## Quality Checklist

Before submitting each annotation:

- [ ] Patient ID labeled
- [ ] Date of service/procedure labeled
- [ ] Primary procedure or test type labeled
- [ ] Result/finding labeled
- [ ] CPT code labeled (if visible)
- [ ] ICD-10 code labeled (if visible)
- [ ] Follow-up recommendation labeled (if present)
- [ ] Document type selected

---

## Tips for Efficiency

1. **Use hotkeys** - Much faster than clicking labels
2. **Zoom in** - For small text or unclear areas
3. **Skip unclear text** - Don't guess, move on
4. **Label codes separately** - Each ICD-10 and CPT as individual entities
5. **Include context** - "5mm" is better labeled with context like "5mm polyp"

---

## Troubleshooting

### Overlapping Labels
If text belongs to multiple categories, prioritize:
1. Most specific label (COLONOSCOPY_DATE over DATE_OF_SERVICE)
2. Clinical codes (always label CPT and ICD-10)

### Unclear Text
- Skip if OCR is poor
- Add note in the notes field
- Don't guess at numbers or codes

### Missing Information
- Only label what's visible
- Don't infer information not explicitly stated