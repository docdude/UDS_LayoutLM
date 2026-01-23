"""Test inference for Mammogram Triage model - Real-world triage test.

Tests the model on a MIX of document types (colonoscopy, mmg, pap, fit)
to see if it correctly identifies mammogram documents vs non-mammogram.
"""
import os
import sys
import glob

# Add project root to path
sys.path.insert(0, '/opt/UDS_LayoutLM')

from src.inference import UDSExtractor

# Configuration
MODEL_PATH = "models/mmg_triage/final_model"
DEVICE = "cpu"  # Use CPU (GPU sm_61 not compatible)
CONFIDENCE_THRESHOLD = 0.5

# REAL-WORLD TEST: Run on ALL PDFs (not just mammogram)
USB_PDF_DIR = "/media/jloyamd/UBUNTU 25_1/pdf_01_21"
ALL_PDFS = sorted(glob.glob(os.path.join(USB_PDF_DIR, "*.pdf")))  # ALL documents

# Track document types for accuracy
doc_types = {"mmg": [], "colonoscopy": [], "pap": [], "fit": [], "other": []}
for pdf in ALL_PDFS:
    fname = os.path.basename(pdf).lower()
    if "_mmg" in fname:
        doc_types["mmg"].append(pdf)
    elif "_colonoscopy" in fname or "colonscopy" in fname:
        doc_types["colonoscopy"].append(pdf)
    elif "_pap" in fname:
        doc_types["pap"].append(pdf)
    elif "_fit" in fname or "_fobt" in fname or "_ifobt" in fname:
        doc_types["fit"].append(pdf)
    else:
        doc_types["other"].append(pdf)

print(f"Real-World Mammogram Triage Test")
print("=" * 70)
print(f"Model: {MODEL_PATH}")
print(f"Device: {DEVICE}")
print("=" * 70)
print(f"\n📂 Document Mix in {os.path.basename(USB_PDF_DIR)}:")
print(f"   Mammogram (mmg): {len(doc_types['mmg'])}")
print(f"   Colonoscopy:     {len(doc_types['colonoscopy'])}")
print(f"   Pap smear:       {len(doc_types['pap'])}")
print(f"   FIT/FOBT:        {len(doc_types['fit'])}")
print(f"   Other:           {len(doc_types['other'])}")
print(f"   TOTAL:           {len(ALL_PDFS)}")
print("=" * 70)

# Initialize extractor
print("\nLoading model...")
extractor = UDSExtractor(
    model_path=MODEL_PATH,
    device=DEVICE,
    confidence_threshold=CONFIDENCE_THRESHOLD,
    labels_module="src.labels_mammogram"
)
print("✓ Model loaded\n")

# Triage thresholds
MMG_ENTITY_THRESHOLD = 0.8  # Confidence threshold for mammogram-specific entities
MIN_MMG_ENTITIES = 3  # Minimum mammogram entities to classify as mammogram

def is_mammogram_document(entities):
    """Determine if document is a mammogram based on extracted entities."""
    # MAMMOGRAM-SPECIFIC entity types (not just screening/diagnostic which appear in other docs)
    mmg_specific_types = {
        "EXAM_MAMMOGRAM", "EXAM_TOMOSYNTHESIS",  # These should have mammogram-related text
        "BIRADS_CATEGORY", "BREAST_DENSITY",
    }
    
    # Check for EXAM_MAMMOGRAM with mammogram-related text
    mammogram_keywords = {"mammogram", "mammography", "mammo", "breast", "tomo", "tomosynthesis", "dbt"}
    
    has_mammogram_exam = False
    for e in entities:
        if e.entity_type in ["EXAM_MAMMOGRAM", "EXAM_TOMOSYNTHESIS"]:
            text_lower = e.text.lower()
            # Only count if the text actually mentions mammogram/breast terms
            if any(kw in text_lower for kw in mammogram_keywords) and e.confidence >= MMG_ENTITY_THRESHOLD:
                has_mammogram_exam = True
                break
    
    # Check for BI-RADS or breast density (very mammogram-specific)
    has_birads = any(e.entity_type == "BIRADS_CATEGORY" and e.confidence >= MMG_ENTITY_THRESHOLD 
                     for e in entities)
    has_density = any(e.entity_type == "BREAST_DENSITY" and e.confidence >= MMG_ENTITY_THRESHOLD 
                      for e in entities)
    
    # Count high-confidence mammogram-specific entities
    mmg_count = sum(1 for e in entities 
                    if e.entity_type in mmg_specific_types and e.confidence >= MMG_ENTITY_THRESHOLD)
    
    # Classification logic - require mammogram exam OR birads/density
    if has_mammogram_exam:
        return True, mmg_count, "Has MAMMOGRAM/TOMO exam entity with breast/mammogram text"
    elif has_birads or has_density:
        return True, mmg_count, "Has BI-RADS or breast density"
    else:
        return False, mmg_count, "No mammogram-specific indicators"

# Track results
results = {
    "true_positive": [],   # mmg correctly identified as mmg
    "true_negative": [],   # non-mmg correctly rejected
    "false_positive": [],  # non-mmg incorrectly identified as mmg
    "false_negative": [],  # mmg incorrectly rejected
}

# Test each PDF
print("\n" + "=" * 70)
print("PROCESSING ALL DOCUMENTS")
print("=" * 70)

for pdf_path in ALL_PDFS:
    print(f"\n{'='*60}")
    print(f"📄 {os.path.basename(pdf_path)}")
    print("=" * 60)
    
    try:
        result = extractor.extract_from_pdf(pdf_path)
        entities = result.entities
        
        # Determine if mammogram
        is_mmg, mmg_count, reason = is_mammogram_document(entities)
        
        # Get ground truth from filename
        fname = os.path.basename(pdf_path).lower()
        actual_mmg = "_mmg" in fname
        
        # Classify result
        if actual_mmg and is_mmg:
            results["true_positive"].append(pdf_path)
            status = "✅ TP"
        elif not actual_mmg and not is_mmg:
            results["true_negative"].append(pdf_path)
            status = "✅ TN"
        elif not actual_mmg and is_mmg:
            results["false_positive"].append(pdf_path)
            status = "❌ FP"
        else:  # actual_mmg and not is_mmg
            results["false_negative"].append(pdf_path)
            status = "❌ FN"
        
        # Determine document type for display
        if "_mmg" in fname:
            doc_type = "MMG"
        elif "_colonoscopy" in fname or "colonscopy" in fname:
            doc_type = "COLO"
        elif "_pap" in fname:
            doc_type = "PAP"
        elif "_fit" in fname or "_fobt" in fname:
            doc_type = "FIT"
        else:
            doc_type = "OTHER"
        
        # Print result
        pred = "MMG" if is_mmg else "---"
        print(f"{status} | {doc_type:5} | Pred: {pred:3} | Entities: {len(entities):2} | MMG-ents: {mmg_count:2} | {os.path.basename(pdf_path)}")
        
    except Exception as e:
        print(f"⚠️  ERROR | {os.path.basename(pdf_path)}: {e}")

# Calculate metrics
print("\n" + "=" * 70)
print("TRIAGE RESULTS")
print("=" * 70)

tp = len(results["true_positive"])
tn = len(results["true_negative"])
fp = len(results["false_positive"])
fn = len(results["false_negative"])

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

print(f"\n📊 Classification Metrics:")
print(f"   True Positives (MMG→MMG):    {tp}")
print(f"   True Negatives (non-MMG→--): {tn}")
print(f"   False Positives (non-MMG→MMG): {fp}")
print(f"   False Negatives (MMG→--):    {fn}")
print(f"\n   Precision: {precision:.1%}")
print(f"   Recall:    {recall:.1%}")
print(f"   F1 Score:  {f1:.1%}")
print(f"   Accuracy:  {accuracy:.1%}")

if results["false_positive"]:
    print(f"\n⚠️  False Positives (non-MMG identified as MMG):")
    for fp_path in results["false_positive"]:
        print(f"   - {os.path.basename(fp_path)}")

# Detailed analysis of false positives
if results["false_positive"]:
    print(f"\n{'='*70}")
    print("DETAILED FALSE POSITIVE ANALYSIS")
    print("=" * 70)
    for fp_path in results["false_positive"]:
        print(f"\n📄 {os.path.basename(fp_path)}")
        print("-" * 50)
        result = extractor.extract_from_pdf(fp_path)
        entities = result.entities
        
        # Group by entity type
        by_type = {}
        for ent in entities:
            if ent.entity_type not in by_type:
                by_type[ent.entity_type] = []
            by_type[ent.entity_type].append(ent)
        
        # Show all entities with confidence
        for etype in sorted(by_type.keys()):
            ents = by_type[etype]
            high_conf = [e for e in ents if e.confidence >= 0.8]
            print(f"   {etype}: {len(high_conf)}/{len(ents)} high-conf")
            for e in ents[:3]:
                marker = "⚠️" if e.confidence >= 0.8 else "  "
                print(f"      {marker} \"{e.text}\" ({e.confidence:.0%})")

if results["false_negative"]:
    print(f"\n⚠️  False Negatives (MMG not identified):")
    for fn_path in results["false_negative"]:
        print(f"   - {os.path.basename(fn_path)}")

print(f"\n{'='*70}")
print("✓ Triage test complete")
print("=" * 70)
