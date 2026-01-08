"""Batch inference on multiple documents."""

import argparse
from pathlib import Path
import sys
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.inference import UDSExtractor


def main():
    parser = argparse.ArgumentParser(description="Run batch inference on documents")
    parser.add_argument("input", help="Input directory containing PDFs")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--output", default="./data/extractions", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.7, help="Confidence threshold")
    parser.add_argument("--extensions", nargs="+", default=[".pdf"], help="File extensions to process")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("UDS Metrics Extraction - Batch Inference")
    print("=" * 60)
    print(f"Input directory: {input_path}")
    print(f"Output directory: {output_path}")
    print(f"Model: {args.model}")
    print(f"Confidence threshold: {args.threshold}")
    print("-" * 60)
    
    # Initialize extractor
    extractor = UDSExtractor(
        model_path=args.model,
        device=args.device,
        confidence_threshold=args.threshold
    )
    
    # Run batch extraction
    results = extractor.batch_extract(
        input_dir=str(input_path),
        output_dir=str(output_path),
        extensions=args.extensions
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    
    total_entities = 0
    uds_summary = {}
    
    for result in results:
        total_entities += len(result.entities)
        for measure, entities in result.uds_metrics.items():
            if measure not in uds_summary:
                uds_summary[measure] = 0
            uds_summary[measure] += len(entities)
    
    print(f"Documents processed: {len(results)}")
    print(f"Total entities extracted: {total_entities}")
    print("\nUDS Metrics Found:")
    for measure, count in sorted(uds_summary.items()):
        print(f"  {measure}: {count}")
    
    # Save detailed summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": args.model,
            "threshold": args.threshold,
            "input_dir": str(input_path),
            "output_dir": str(output_path)
        },
        "results": {
            "documents_processed": len(results),
            "total_entities": total_entities,
            "uds_metrics_summary": uds_summary
        },
        "documents": [
            {
                "file": r.source_file,
                "pages": r.num_pages,
                "entities": len(r.entities),
                "uds_metrics": {k: len(v) for k, v in r.uds_metrics.items()}
            }
            for r in results
        ]
    }
    
    summary_file = output_path / "batch_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nDetailed summary saved to: {summary_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()