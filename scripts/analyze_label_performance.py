"""Analyze per-label performance on the test set."""
import torch
import json
import sys
from pathlib import Path
from collections import defaultdict
from datasets import load_from_disk
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
from seqeval.metrics import classification_report

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.labels_crc_triage import ID2LABEL, LABEL2ID
from src.dataset import UDSDataCollator


def analyze_per_label_performance():
    """Evaluate model on test set and show per-label metrics."""
    
    model_path = "models/crc_triage/final_model"
    dataset_path = "data/dataset_crc_triage"
    
    print("=" * 60)
    print("Per-Label Performance Analysis")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
    processor = LayoutLMv3Processor.from_pretrained(model_path, apply_ocr=False)
    model.eval()
    
    # Load test dataset
    print(f"Loading test set from {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    test_set = dataset["test"]
    print(f"  Test examples: {len(test_set)}")
    
    # Data collator
    data_collator = UDSDataCollator(processor=processor, max_length=512)
    
    all_true_labels = []
    all_pred_labels = []
    
    # Per-label stats
    label_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    
    print("\nRunning predictions...")
    
    for i, example in enumerate(test_set):
        # Prepare batch
        batch = data_collator([example])
        
        # Get ground truth
        labels = batch["labels"][0].tolist()
        
        # Run inference
        with torch.no_grad():
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                bbox=batch["bbox"],
                pixel_values=batch["pixel_values"]
            )
        
        predictions = torch.argmax(outputs.logits, dim=-1)[0].tolist()
        
        # Convert to label strings (for seqeval)
        true_seq = []
        pred_seq = []
        
        for true_id, pred_id in zip(labels, predictions):
            if true_id == -100:  # Skip padding
                continue
            
            true_label = ID2LABEL[true_id]
            pred_label = ID2LABEL[pred_id]
            
            true_seq.append(true_label)
            pred_seq.append(pred_label)
            
            # Count per-label stats (for entity labels only)
            if true_label != "O" or pred_label != "O":
                if true_label == pred_label:
                    label_stats[true_label]["tp"] += 1
                else:
                    if true_label != "O":
                        label_stats[true_label]["fn"] += 1
                    if pred_label != "O":
                        label_stats[pred_label]["fp"] += 1
        
        all_true_labels.append(true_seq)
        all_pred_labels.append(pred_seq)
    
    # Print seqeval classification report
    print("\n" + "=" * 60)
    print("SEQEVAL Classification Report:")
    print("=" * 60)
    print(classification_report(all_true_labels, all_pred_labels, digits=3))
    
    # Print per-label breakdown
    print("\n" + "=" * 60)
    print("Per-Label Breakdown:")
    print("=" * 60)
    print(f"{'Label':<35} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    print("-" * 85)
    
    for label in sorted(label_stats.keys()):
        if label == "O":
            continue
        
        stats = label_stats[label]
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{label:<35} {tp:>6} {fp:>6} {fn:>6} {precision:>8.3f} {recall:>8.3f} {f1:>8.3f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary by Category:")
    print("=" * 60)
    
    categories = {
        "DOC_TYPE": ["B-DOC_TYPE_COLONOSCOPY", "B-DOC_TYPE_FIT", "B-DOC_TYPE_FOBT"],
        "DATES": ["B-PROCEDURE_DATE", "B-REPORT_DATE", "B-COLLECTION_DATE"],
        "INDICATIONS": ["B-INDICATION_SCREENING", "B-INDICATION_SURVEILLANCE", "B-INDICATION_DIAGNOSTIC"],
        "RESULTS": ["B-RESULT_NEGATIVE", "B-RESULT_POSITIVE", "B-RESULT_ADENOMA", "B-RESULT_POLYP"]
    }
    
    for cat_name, cat_labels in categories.items():
        cat_tp = sum(label_stats[l]["tp"] for l in cat_labels)
        cat_fp = sum(label_stats[l]["fp"] for l in cat_labels)
        cat_fn = sum(label_stats[l]["fn"] for l in cat_labels)
        
        precision = cat_tp / (cat_tp + cat_fp) if (cat_tp + cat_fp) > 0 else 0
        recall = cat_tp / (cat_tp + cat_fn) if (cat_tp + cat_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"  {cat_name}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f} (TP={cat_tp}, FP={cat_fp}, FN={cat_fn})")


if __name__ == "__main__":
    analyze_per_label_performance()
