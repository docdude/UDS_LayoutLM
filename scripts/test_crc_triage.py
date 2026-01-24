#!/usr/bin/env python3
"""
Test CRC Triage Accuracy

Tests the CRC model's ability to distinguish CRC-related documents 
(colonoscopy, FIT, FOBT, sigmoidoscopy) from other document types.

Uses entity detection to determine if a document is CRC-related.
"""

import os
import sys
import json
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import UDSExtractor


# CRC-specific entity types for triage
# Split into strong indicators (document types) and supporting evidence
CRC_DOC_TYPES = {
    'DOC_TYPE_COLONOSCOPY',
    'DOC_TYPE_FIT', 
    'DOC_TYPE_FOBT',
    'DOC_TYPE_SIGMOIDOSCOPY',
    'DOC_TYPE_CT_COLONOGRAPHY',
}

CRC_STRONG_EVIDENCE = {
    # Colon-specific anatomy/findings (only appear in colonoscopy reports)
    'COLON_ANATOMY',
    'POLYP_FINDING',
    'POLYP_SIZE',
    'POLYP_LOCATION',
    'POLYP_MORPHOLOGY',
    'POLYP_COUNT',
    'ADENOMA_FINDING',
    'PREP_QUALITY',
    'CECAL_INTUBATION',
    'WITHDRAWAL_TIME',
    'BOWEL_PREP',
    'SCOPE_INSERTION',
    
    # FIT/FOBT specific
    'STOOL_TEST_RESULT',
    'HEMOGLOBIN_LEVEL',
}

CRC_WEAK_EVIDENCE = {
    # These appear in many document types
    'PROCEDURE_DATE',
    'BIOPSY_RESULT',
    'PATHOLOGY_DIAGNOSIS',
    'SEDATION',
}

# All CRC triage entities (like MAMMOGRAM_TRIAGE_TYPES)
CRC_TRIAGE_ENTITIES = CRC_DOC_TYPES | CRC_STRONG_EVIDENCE


def is_crc_report(entities: list, min_entity_types: int = 2) -> tuple[bool, list]:
    """
    Determine if document is CRC-related based on detected entities.
    
    Uses multi-entity rule (like MMG): Require 2+ different CRC entity types.
    This prevents false positives from documents that just mention "colonoscopy".
    
    Args:
        entities: List of (text, entity_type, confidence) tuples from inference
        min_entity_types: Minimum different entity types required (default: 2)
        
    Returns:
        (is_crc, detected_crc_entities)
    """
    detected = []
    for item in entities:
        if len(item) == 3:
            text, entity_type, confidence = item
        else:
            text, entity_type = item
            confidence = None
            
        # Strip B-/I- prefix
        etype = entity_type[2:] if entity_type.startswith(('B-', 'I-')) else entity_type
        
        if etype in CRC_TRIAGE_ENTITIES:
            detected.append((text, etype, confidence))
    
    unique_types = {ent[1] for ent in detected}
    return len(unique_types) >= min_entity_types, detected


def test_folder(extractor: UDSExtractor, folder_path: str, expected_crc: bool) -> dict:
    """Test all PDFs in a folder."""
    folder = Path(folder_path)
    if not folder.exists():
        print(f"  Folder not found: {folder}")
        return {'total': 0, 'correct': 0, 'files': []}
    
    results = {
        'total': 0,
        'correct': 0,
        'files': []
    }
    
    pdf_files = list(folder.glob("*.pdf"))
    print(f"  Found {len(pdf_files)} PDFs")
    
    for pdf_path in sorted(pdf_files):
        try:
            # Run inference
            result = extractor.extract_from_pdf(str(pdf_path))
            
            # Convert ExtractedEntity objects to (text, entity_type, confidence) tuples
            entities = [(ent.text, ent.entity_type, ent.confidence) for ent in result.entities]
            
            # Check if CRC-related
            is_crc, detected = is_crc_report(entities)
            
            # Determine if correct
            correct = is_crc == expected_crc
            
            file_result = {
                'file': pdf_path.name,
                'predicted_crc': is_crc,
                'expected_crc': expected_crc,
                'correct': correct,
                'crc_entities': [(t, e, c) for t, e, c in detected[:5]],  # First 5
                'entity_count': len(entities),
                'crc_entity_types': list(set(e[1] for e in detected))
            }
            
            results['total'] += 1
            if correct:
                results['correct'] += 1
            else:
                # Print misclassification with confidence
                status = "FALSE POSITIVE" if is_crc else "FALSE NEGATIVE"
                print(f"    {status}: {pdf_path.name}")
                if detected:
                    # Show entity types and their confidences
                    for text, etype, conf in detected[:5]:
                        conf_str = f"{conf:.3f}" if conf else "N/A"
                        print(f"      {etype}: '{text[:30]}' (conf: {conf_str})")
                else:
                    print(f"      No CRC entities detected (total entities: {len(entities)})")
            
            results['files'].append(file_result)
            
        except Exception as e:
            print(f"    ERROR: {pdf_path.name}: {e}")
            results['files'].append({
                'file': pdf_path.name,
                'error': str(e)
            })
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test CRC triage accuracy')
    parser.add_argument('--model', type=str, 
                       default='./models/crc_triage_v2_L4/final_model',
                       help='Path to model')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu or cuda)')
    parser.add_argument('--positive-folders', type=str, nargs='+',
                       help='Folders with CRC documents (colonoscopy, FIT, FOBT)')
    parser.add_argument('--negative-folder', type=str,
                       help='Folder with non-CRC documents')
    parser.add_argument('--output', type=str, default='crc_triage_results.json',
                       help='Output JSON file')
    args = parser.parse_args()
    
    print("=" * 60)
    print("CRC Triage Accuracy Test")
    print("=" * 60)
    print(f"\nModel: {args.model}")
    
    # Load model
    print("\nLoading model...")
    extractor = UDSExtractor(
        model_path=args.model,
        device=args.device,
        labels_module="src.labels_crc_triage"
    )
    
    results = {
        'model': args.model,
        'triage_entities': list(CRC_TRIAGE_ENTITIES),
        'positive_results': {},
        'negative_results': {},
        'summary': {}
    }
    
    total_correct = 0
    total_files = 0
    
    # Test positive folders (CRC documents)
    if args.positive_folders:
        print("\n" + "-" * 60)
        print("Testing CRC POSITIVE documents (expected: CRC detected)")
        print("-" * 60)
        
        for folder in args.positive_folders:
            print(f"\n{folder}:")
            folder_results = test_folder(extractor, folder, expected_crc=True)
            results['positive_results'][folder] = folder_results
            
            if folder_results['total'] > 0:
                acc = folder_results['correct'] / folder_results['total'] * 100
                print(f"  Accuracy: {folder_results['correct']}/{folder_results['total']} ({acc:.1f}%)")
                total_correct += folder_results['correct']
                total_files += folder_results['total']
    
    # Test negative folder (non-CRC documents)
    if args.negative_folder:
        print("\n" + "-" * 60)
        print("Testing CRC NEGATIVE documents (expected: CRC NOT detected)")
        print("-" * 60)
        
        print(f"\n{args.negative_folder}:")
        folder_results = test_folder(extractor, args.negative_folder, expected_crc=False)
        results['negative_results'][args.negative_folder] = folder_results
        
        if folder_results['total'] > 0:
            acc = folder_results['correct'] / folder_results['total'] * 100
            print(f"  Accuracy: {folder_results['correct']}/{folder_results['total']} ({acc:.1f}%)")
            total_correct += folder_results['correct']
            total_files += folder_results['total']
    
    # Summary
    print("\n" + "=" * 60)
    print("OVERALL TRIAGE ACCURACY")
    print("=" * 60)
    
    if total_files > 0:
        overall_acc = total_correct / total_files * 100
        print(f"\nTotal: {total_correct}/{total_files} ({overall_acc:.1f}%)")
        
        results['summary'] = {
            'total_files': total_files,
            'correct': total_correct,
            'accuracy': overall_acc
        }
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")
    
    return results


if __name__ == "__main__":
    main()
