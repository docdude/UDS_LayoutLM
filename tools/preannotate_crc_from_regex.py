"""
Enhanced preannotation for CRC/UDS documents using regex patterns.

This script extracts multiple entity types from OCR tokens and creates
Label Studio predictions with properly positioned bounding boxes.

COORDINATE SYSTEM:
- processor.py saves OCR bboxes as NORMALIZED 0-1000 coordinates (LayoutLMv3 format)
- Label Studio expects PERCENTAGE 0-100 coordinates
- Conversion: percentage = normalized / 10
"""

import json
import re
import uuid
from pathlib import Path
from PIL import Image
from typing import List, Dict, Tuple, Optional

# ============================================================================
# REGEX PATTERNS
# ============================================================================

# Date patterns
DATE_PAT = re.compile(r"^(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4})$")
DATE_LOOSE = re.compile(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}")

# Patient identifiers
MRN_PAT = re.compile(r"^(\d{6,10})$")  # 6-10 digit MRN
ACCOUNT_PAT = re.compile(r"^(\d{9,12})$")  # Account numbers

# Result patterns
RESULT_POSITIVE = re.compile(r"^(positive|pos|detected|abnormal)$", re.I)
RESULT_NEGATIVE = re.compile(r"^(negative|neg|not\s*detected|normal|none|no)$", re.I)

# Stool test keywords
STOOL_TEST_TYPES = {"fit", "fobt", "ifobt", "guaiac", "colofit", "ocolit"}

# Colonoscopy findings
POLYP_LOCATIONS = {"cecum", "ascending", "transverse", "descending", "sigmoid", "rectum", "hepatic", "splenic"}
BOWEL_PREP_TERMS = {"excellent", "good", "fair", "poor", "adequate", "inadequate"}

# ============================================================================
# CONTEXT PATTERNS (for window-based matching)
# ============================================================================

# Date context patterns - used to classify dates (more relaxed for better recall)
CTX_PROCEDURE_DATE = re.compile(
    r"(procedure\s*date|date\s*of\s*procedure|date\s*of\s*service|\bdos\b|"
    r"date\s*of\s*operation|operation\s*date|performed\s*(on|at)|"
    r"exam\s*date|date\s*of\s*exam|study\s*date|service\s*date|"
    r"visit\s*date|date\s*performed|date\s*:)", re.I
)
CTX_COLLECTION_DATE = re.compile(
    r"(collection\s*date|date\s*of\s*collection|\bcollected\b|"
    r"specimen\s*date|test\s*date|order\s*date|received)", re.I
)
CTX_DOB = re.compile(
    r"(date\s*of\s*birth|\bdob\b|birth\s*date|\bborn\b|\bage\b)", re.I
)
CTX_RESULT_DATE = re.compile(
    r"(report\s*date|reported|result\s*date|finalized|signed)", re.I
)

# Entity context patterns
CTX_MRN = re.compile(r"(mrn|medical\s*record|med\s*rec|chart)", re.I)
CTX_ACCOUNT = re.compile(r"(account|acct|encounter|visit)", re.I)
CTX_PATIENT_NAME = re.compile(r"(patient\s*name|name\s*:|pt\s*name)", re.I)
CTX_PROVIDER = re.compile(r"(attending|provider|physician|doctor|md|performed\s*by)", re.I)
CTX_INDICATION = re.compile(r"(indication|reason\s*for|referred\s*for)", re.I)
CTX_FINDINGS = re.compile(r"(finding|impression|conclusion|result)", re.I)
CTX_BOWEL_PREP = re.compile(r"(bowel\s*prep|preparation|prep\s*quality)", re.I)
CTX_POLYP = re.compile(r"(polyp|adenoma|lesion|mass)", re.I)
CTX_RECOMMENDATION = re.compile(r"(recommend|follow\s*up|repeat|next|return)", re.I)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def looks_like_colonoscopy(text: str) -> bool:
    t = (text or "").lower()
    return "colonoscopy" in t

def looks_like_stool_test(text: str) -> bool:
    t = (text or "").lower()
    return (("occult blood" in t and "fecal" in t) or 
            "ifobt" in t or "fobt" in t or "guaiac" in t or 
            "colofit" in t or ("fit" in t and "fecal" in t))

def looks_like_nonclinical_line(ctx: str) -> bool:
    """Filter out non-clinical date contexts (from notebook logic)."""
    l = (ctx or "").lower()
    if "electronically signed" in l or "signed by" in l:
        return True
    if "dob" in l or "date of birth" in l or "birth" in l:
        return True
    if "data:text" in l or "base64" in l:
        return True
    if "from: fax" in l or ("from:" in l and "to:" in l and "page:" in l):
        return True
    # Filter historical/future references
    if "last colonoscopy" in l or "previous colonoscopy" in l or "prior colonoscopy" in l:
        return True
    if "recommend" in l and ("colonoscopy" in l or "repeat" in l):
        return True
    if "next colonoscopy" in l or "follow-up colonoscopy" in l:
        return True
    return False

def classify_date_line(line: str, near_procedure_indicator: bool = False) -> str:
    """
    Classify a line containing a date based on context keywords.
    Returns: 'procedure', 'collection', 'received', 'result', 'dob', or 'other'
    
    This mirrors the notebook's extract_labeled_dates() logic.
    Priority: procedure > collection > received > result > dob > other
    """
    ll = line.lower()
    
    # PRIORITY 1: Strong procedure date patterns (check first!)
    if re.search(r'\bprocedure\b.*\bdate\b', ll) or re.search(r'procedure\s*date|date\s*of\s*procedure|date\s*of\s*service|\bdos\b', ll):
        return "procedure"
    if re.search(r'\bdate\s*of\s*operation\b|\boperation\s*date\b', ll):
        return "procedure"
    if re.search(r'\bperformed\b.*(on|at)', ll) and not looks_like_nonclinical_line(line):
        return "procedure"
    if re.search(r'\bmrn\b.*\bdate\s*:', ll):
        return "procedure"
    if re.search(r'\bexam\s*date\b|\bstudy\s*date\b|\bservice\s*date\b', ll):
        return "procedure"
    
    # If we're near a procedure indicator line, trust that
    if near_procedure_indicator:
        # But still filter out obvious DOB lines
        if re.search(r'\bdob\s*:', ll) or re.search(r'date\s*of\s*birth\s*:', ll):
            return "dob"
        return "procedure"
    
    # Check for non-clinical lines early (after procedure check)
    if looks_like_nonclinical_line(line):
        return "dob"  # Treat non-clinical dates as DOB (to filter out)
    
    # PRIORITY 2: Collection date patterns (stool tests)
    if re.search(r'\bcollec[\s\x00]*[ti]?on\b.*\bdate\b', ll) or re.search(r'\bcollect\b.*\bdate\b', ll):
        return "collection"
    if re.search(r'collection\s*date|date\s*of\s*collection', ll):
        return "collection"
    if re.search(r'\btest\s*date\b', ll):
        return "collection"
    if re.search(r'\breturned\b.*(as|on)', ll) or re.search(r'\bresult.*\b(as|on)\b', ll):
        return "collection"
    if re.search(r'\border\b.*\bdate\b', ll):
        return "collection"
    if re.search(r'\b(collected|obtained)\b.*(on|at|:)', ll):
        return "collection"
    if re.search(r'\bfinal\s*,\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', ll):
        return "collection"
    
    # PRIORITY 3: Received date
    if re.search(r'\breceived\b', ll):
        return "received"
    
    # PRIORITY 4: Result/report date
    if re.search(r'\breport(ed)?\s*:', ll):
        return "result"
    if re.search(r'\bresult\b|\breported\b', ll):
        return "result"
    
    # PRIORITY 5: DOB patterns (check AFTER clinical patterns)
    if re.search(r'\bdob\s*:', ll) or re.search(r'\bdate\s*of\s*birth\s*:', ll) or re.search(r'\bbirth\s*date\s*:', ll):
        return "dob"
    # Only mark as DOB if DOB keyword is BEFORE the date (not just anywhere in context)
    if re.search(r'\bdob\b.*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', ll):
        return "dob"
    if re.search(r'date\s*of\s*birth.*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', ll):
        return "dob"
    
    return "other"
    # Filter historical/future references
    if "last colonoscopy" in l or "previous colonoscopy" in l or "prior colonoscopy" in l:
        return True
    if "recommend" in l and ("colonoscopy" in l or "repeat" in l):
        return True
    if "next colonoscopy" in l or "follow-up colonoscopy" in l:
        return True
    return False

def is_reasonable_procedure_date(date_str: str) -> bool:
    """Check if date is in reasonable range for a procedure (not DOB)."""
    try:
        m = DATE_LOOSE.search(date_str)
        if not m:
            return True
        parts = re.split(r'[/-]', m.group())
        year = int(parts[2])
        if year < 100:
            year = 2000 + year if year < 50 else 1900 + year
        # Procedures should be between 1995-2030
        return 1995 <= year <= 2030
    except:
        return True

def window_text(tokens: List[Dict], idx: int, k: int = 12) -> str:
    """Get text window around token index."""
    lo = max(0, idx - k)
    hi = min(len(tokens), idx + k + 1)
    return " ".join(t.get("text", "") for t in tokens[lo:hi])

def window_before(tokens: List[Dict], idx: int, k: int = 8) -> str:
    """Get text window before token index."""
    lo = max(0, idx - k)
    return " ".join(t.get("text", "") for t in tokens[lo:idx])

def window_after(tokens: List[Dict], idx: int, k: int = 8) -> str:
    """Get text window after token index."""
    hi = min(len(tokens), idx + k + 1)
    return " ".join(t.get("text", "") for t in tokens[idx+1:hi])

def bbox_union(bboxes: List[List[int]]) -> List[int]:
    """Union of multiple bboxes."""
    x0 = min(b[0] for b in bboxes)
    y0 = min(b[1] for b in bboxes)
    x1 = max(b[2] for b in bboxes)
    y1 = max(b[3] for b in bboxes)
    return [x0, y0, x1, y1]

def normalized_to_ls_percent(bbox_normalized: List[int]) -> Tuple[float, float, float, float]:
    """Convert normalized 0-1000 bbox to Label Studio percentage 0-100."""
    x0, y0, x1, y1 = bbox_normalized
    x = x0 / 10.0
    y = y0 / 10.0
    w = (x1 - x0) / 10.0
    h = (y1 - y0) / 10.0
    return x, y, w, h

def add_region(results: List[Dict], img_w: int, img_h: int, 
               bbox_normalized: List[int], label_value: str):
    """Add a labeled region to results."""
    rid = str(uuid.uuid4())[:8]
    x, y, w, h = normalized_to_ls_percent(bbox_normalized)

    results.append({
        "id": rid,
        "from_name": "bbox",
        "to_name": "image",
        "type": "rectangle",
        "value": {"x": x, "y": y, "width": w, "height": h, "rotation": 0},
        "original_width": img_w,
        "original_height": img_h,
        "image_rotation": 0
    })

    results.append({
        "id": rid,
        "from_name": "label",
        "to_name": "image",
        "type": "labels",
        "value": {"labels": [label_value]}
    })

# ============================================================================
# ENTITY EXTRACTION (Two-pass approach from notebook)
# ============================================================================

def find_procedure_indicator_tokens(ocr_tokens: List[Dict]) -> List[int]:
    """
    First pass: Find token indices near strong procedure date indicators.
    This mirrors the notebook's two-pass approach.
    """
    indicator_indices = []
    for i, tok in enumerate(ocr_tokens):
        ctx = window_text(ocr_tokens, i, k=8)
        ll = ctx.lower()
        # Strong indicators that this section contains the procedure date
        if re.search(r'\bprocedure\b.*\bdate\b', ll) or re.search(r'\bdate\s*of\s*operation\b', ll):
            indicator_indices.append(i)
        elif re.search(r'\bexam\s*date\b|\bstudy\s*date\b|\bservice\s*date\b', ll):
            indicator_indices.append(i)
    return indicator_indices

def extract_entities(ocr_tokens: List[Dict], doc_type: str, full_text: str = "") -> List[Tuple[int, str, str]]:
    """
    Extract entities from OCR tokens with their labels.
    Uses two-pass approach for better date detection.
    
    Returns:
        List of (token_index, label, reason) tuples
    """
    entities = []
    is_colon = doc_type == "COLONOSCOPY"
    is_stool = doc_type in ["FIT", "FOBT"]
    
    # First pass: find procedure date indicator tokens
    procedure_indicator_tokens = find_procedure_indicator_tokens(ocr_tokens)
    
    # Build a set of date tokens we've already classified
    classified_dates = set()
    
    for i, tok in enumerate(ocr_tokens):
        text = (tok.get("text") or "").strip()
        if not text:
            continue
            
        ctx_before = window_before(ocr_tokens, i, k=10)
        ctx_after = window_after(ocr_tokens, i, k=8)
        ctx_full = window_text(ocr_tokens, i, k=15)  # Larger window for better context
        ctx_lower = ctx_full.lower()
        
        bb = tok.get("bbox")
        if not bb or len(bb) != 4:
            continue
        
        # === DATES (Two-pass approach) ===
        if DATE_PAT.match(text):
            # Check if near a procedure indicator (within ~5 tokens)
            near_procedure_indicator = any(abs(i - pi) <= 5 for pi in procedure_indicator_tokens)
            
            # Classify using the full context line approach from notebook
            date_class = classify_date_line(ctx_full, near_procedure_indicator)
            
            # Skip DOB
            if date_class == "dob":
                entities.append((i, "DATE_OF_BIRTH", "DOB context"))
                continue
            
            # Skip unreasonable years (likely DOB)
            if not is_reasonable_procedure_date(text):
                entities.append((i, "DATE_OF_BIRTH", "old date"))
                continue
            
            # Classify date by document type and context
            if is_colon:
                if date_class == "procedure" or near_procedure_indicator:
                    entities.append((i, "COLONOSCOPY_DATE", f"procedure date ({date_class})"))
                elif date_class == "other":
                    # For colonoscopy docs, generic dates are often procedure dates
                    # Check additional patterns from notebook
                    if re.search(r'\bdate\s*:', ctx_lower) and not CTX_RESULT_DATE.search(ctx_full):
                        entities.append((i, "COLONOSCOPY_DATE", "date: pattern"))
                    elif re.search(r'\bperformed\b', ctx_lower):
                        entities.append((i, "COLONOSCOPY_DATE", "performed pattern"))
                # Don't label result/received dates as colonoscopy dates
            
            elif is_stool:
                if date_class == "collection":
                    entities.append((i, "STOOL_TEST_DATE", "collection date context"))
                elif date_class == "result" or date_class == "received":
                    entities.append((i, "STOOL_TEST_DATE", "result/received date"))
                elif date_class == "other":
                    # For stool tests, check for common patterns
                    if re.search(r'\bfinal\b', ctx_lower) or re.search(r'\btest\b', ctx_lower):
                        entities.append((i, "STOOL_TEST_DATE", "stool test date"))
                    elif "date" in ctx_lower and not looks_like_nonclinical_line(ctx_full):
                        entities.append((i, "STOOL_TEST_DATE", "generic date in stool test"))
        
        # === PATIENT IDENTIFIERS ===
        elif MRN_PAT.match(text):
            if CTX_MRN.search(ctx_before) or CTX_MRN.search(ctx_full):
                entities.append((i, "PATIENT_ID", "MRN context"))
            elif CTX_ACCOUNT.search(ctx_before):
                entities.append((i, "PATIENT_ID", "account context"))
        
        # === STOOL TEST TYPE ===
        elif text.lower() in STOOL_TEST_TYPES:
            if is_stool:
                entities.append((i, "STOOL_TEST_TYPE", "stool test keyword"))
        
        # === STOOL TEST RESULT ===
        elif is_stool:
            if RESULT_NEGATIVE.match(text):
                # Check if near "occult blood" or result context
                if "occult" in ctx_lower or "result" in ctx_lower or "fecal" in ctx_lower:
                    entities.append((i, "STOOL_TEST_RESULT", "negative result"))
            elif RESULT_POSITIVE.match(text):
                if "occult" in ctx_lower or "result" in ctx_lower or "fecal" in ctx_lower:
                    entities.append((i, "STOOL_TEST_RESULT", "positive result"))
        
        # === COLONOSCOPY SPECIFIC ===
        elif is_colon:
            text_lower = text.lower()
            
            # Bowel prep quality
            if text_lower in BOWEL_PREP_TERMS:
                if CTX_BOWEL_PREP.search(ctx_full) or "prep" in ctx_lower:
                    entities.append((i, "BOWEL_PREP_QUALITY", "prep quality term"))
            
            # Polyp location
            elif text_lower in POLYP_LOCATIONS:
                if CTX_POLYP.search(ctx_full) or "polyp" in ctx_lower or "mm" in ctx_lower:
                    entities.append((i, "POLYP_LOCATION", "polyp location"))
            
            # Colonoscopy result indicators
            elif text_lower == "normal":
                if CTX_FINDINGS.search(ctx_before) or "colon" in ctx_lower:
                    entities.append((i, "COLONOSCOPY_RESULT", "normal finding"))
            
            # Indication patterns
            elif "screening" in text_lower:
                if CTX_INDICATION.search(ctx_before) or "indication" in ctx_lower:
                    entities.append((i, "COLONOSCOPY_INDICATION", "screening indication"))
    
    return entities

# ============================================================================
# MAIN PROCESSING
# ============================================================================

def main(import_json_path: str, out_path: str, data_root: str = None):
    tasks = json.loads(Path(import_json_path).read_text(encoding="utf-8"))
    
    if data_root is None:
        data_root = Path(__file__).parent.parent / "data"
    else:
        data_root = Path(data_root)

    stats = {"total": 0, "with_predictions": 0, "entities": {}}

    for task in tasks:
        stats["total"] += 1
        data = task.get("data", {})
        text = data.get("text", "") or ""
        ocr_tokens = data.get("ocr", []) or []
        image_uri = data.get("image", "")

        # Resolve image path
        img_path = image_uri
        if "local-files" in img_path and "?d=" in img_path:
            img_path = img_path.split("?d=", 1)[1]
        
        img_path = Path(img_path)
        if not img_path.is_absolute():
            img_path = data_root / img_path
        img_path = Path(str(img_path).replace('\\', '/'))

        try:
            im = Image.open(img_path).convert("RGB")
            img_w, img_h = im.size
        except Exception as e:
            print(f"Could not open {img_path}: {e}")
            continue

        # Determine document type
        if looks_like_colonoscopy(text):
            doc_type = "COLONOSCOPY"
        elif looks_like_stool_test(text):
            doc_type = "FIT"
        else:
            doc_type = "UNKNOWN"
        
        # Extract entities
        entities = extract_entities(ocr_tokens, doc_type)
        
        results = []
        for tok_idx, label, reason in entities:
            bb = ocr_tokens[tok_idx].get("bbox")
            if bb and len(bb) == 4:
                # Skip DATE_OF_BIRTH - we identify it but don't want to prelabel
                if label == "DATE_OF_BIRTH":
                    continue
                add_region(results, img_w, img_h, bb, label)
                stats["entities"][label] = stats["entities"].get(label, 0) + 1

        # Add document classification (doc_type) based on detected document type
        if doc_type == "COLONOSCOPY":
            # Check if it's a pathology report
            text_lower = text.lower()
            if "pathology" in text_lower or "biopsy" in text_lower or "adenoma" in text_lower:
                results.append({
                    "from_name": "doc_type",
                    "to_name": "image",
                    "type": "choices",
                    "value": {"choices": ["pathology_report"]}
                })
            else:
                results.append({
                    "from_name": "doc_type",
                    "to_name": "image",
                    "type": "choices",
                    "value": {"choices": ["colonoscopy_report"]}
                })
            # Add UDS measure category
            results.append({
                "from_name": "uds_measure",
                "to_name": "image",
                "type": "choices",
                "value": {"choices": ["colorectal_screening"]}
            })
        elif doc_type == "FIT":
            results.append({
                "from_name": "doc_type",
                "to_name": "image",
                "type": "choices",
                "value": {"choices": ["fit_fobt_result"]}
            })
            results.append({
                "from_name": "uds_measure",
                "to_name": "image",
                "type": "choices",
                "value": {"choices": ["colorectal_screening"]}
            })

        if results:
            task["predictions"] = [{
                "model_version": "regex-prelabel-v2",
                "score": 0.7,
                "result": results
            }]
            stats["with_predictions"] += 1
            print(f"  {Path(img_path).name}: {len([r for r in results if r.get('type') != 'choices'])//2} entities, doc_type={doc_type}")

    Path(out_path).write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total tasks: {stats['total']}")
    print(f"Tasks with predictions: {stats['with_predictions']}")
    print(f"Entities by type:")
    for label, count in sorted(stats["entities"].items(), key=lambda x: -x[1]):
        print(f"  {label}: {count}")
    print(f"\nWrote: {out_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python preannotate_crc_from_regex.py <input.json> <output.json> [data_root]")
        sys.exit(1)
    data_root = sys.argv[3] if len(sys.argv) > 3 else None
    main(sys.argv[1], sys.argv[2], data_root)
