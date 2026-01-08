"""Validate Label Studio annotations before training."""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass, field
from collections import Counter
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.labels import UDS_LABELS, LABEL2ID, UDS_MEASURES


@dataclass
class ValidationIssue:
    """A single validation issue."""
    severity: str  # "error", "warning", "info"
    issue_type: str
    message: str
    file: str = ""
    task_id: int = -1
    details: Dict = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Complete validation report."""
    total_files: int = 0
    total_tasks: int = 0
    total_annotations: int = 0
    issues: List[ValidationIssue] = field(default_factory=list)
    label_counts: Dict[str, int] = field(default_factory=dict)
    doc_type_counts: Dict[str, int] = field(default_factory=dict)
    
    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]
    
    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]
    
    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


def get_valid_labels() -> Set[str]:
    """Get set of valid label names (without B-/I- prefix)."""
    valid = set()
    for label in UDS_LABELS:
        if label.startswith("B-") or label.startswith("I-"):
            valid.add(label[2:])
    return valid


def validate_label_name(label: str, valid_labels: Set[str]) -> Tuple[bool, str]:
    """Check if a label name is valid."""
    if label in valid_labels:
        return True, ""
    
    # Check for common typos
    label_upper = label.upper()
    for valid in valid_labels:
        if label_upper == valid.upper():
            return False, f"Case mismatch: '{label}' should be '{valid}'"
    
    # Check for partial matches
    for valid in valid_labels:
        if label_upper in valid.upper() or valid.upper() in label_upper:
            return False, f"Similar label exists: '{valid}'"
    
    return False, f"Unknown label: '{label}'"


def validate_annotation_result(
    result: Dict, 
    valid_labels: Set[str],
    task_id: int,
    file_path: str
) -> List[ValidationIssue]:
    """Validate a single annotation result."""
    issues = []
    
    # Check result type
    result_type = result.get("type", "")
    
    if result_type == "labels":
        # Check label value
        labels = result.get("value", {}).get("labels", [])
        for label in labels:
            is_valid, message = validate_label_name(label, valid_labels)
            if not is_valid:
                issues.append(ValidationIssue(
                    severity="error",
                    issue_type="invalid_label",
                    message=message,
                    file=file_path,
                    task_id=task_id,
                    details={"label": label}
                ))
        
        # Check bounding box
        value = result.get("value", {})
        required_bbox_fields = ["x", "y", "width", "height"]
        for field in required_bbox_fields:
            if field not in value:
                issues.append(ValidationIssue(
                    severity="error",
                    issue_type="missing_bbox",
                    message=f"Missing bounding box field: {field}",
                    file=file_path,
                    task_id=task_id
                ))
        
        # Check for zero-size bounding box
        if value.get("width", 0) == 0 or value.get("height", 0) == 0:
            issues.append(ValidationIssue(
                severity="warning",
                issue_type="zero_size_bbox",
                message="Bounding box has zero width or height",
                file=file_path,
                task_id=task_id
            ))
    
    return issues


def validate_task(
    task: Dict,
    valid_labels: Set[str],
    task_idx: int,
    file_path: str
) -> Tuple[List[ValidationIssue], Dict[str, int]]:
    """Validate a single task."""
    issues = []
    label_counts = Counter()
    
    task_id = task.get("id", task_idx)
    
    # Check for required data fields
    data = task.get("data", {})
    if "image" not in data:
        issues.append(ValidationIssue(
            severity="error",
            issue_type="missing_image",
            message="Task missing 'image' field in data",
            file=file_path,
            task_id=task_id
        ))
    
    # Check image path exists
    image_path = data.get("image", "")
    if image_path:
        # Handle Label Studio path formats
        if image_path.startswith("/data/local-files/?d="):
            image_path = image_path.replace("/data/local-files/?d=", "")
        
        if not Path(image_path).exists():
            issues.append(ValidationIssue(
                severity="warning",
                issue_type="image_not_found",
                message=f"Image file not found: {image_path}",
                file=file_path,
                task_id=task_id
            ))
    
    # Check annotations
    annotations = task.get("annotations", [])
    
    if not annotations:
        issues.append(ValidationIssue(
            severity="warning",
            issue_type="no_annotations",
            message="Task has no annotations",
            file=file_path,
            task_id=task_id
        ))
    else:
        for annotation in annotations:
            results = annotation.get("result", [])
            
            if not results:
                issues.append(ValidationIssue(
                    severity="warning",
                    issue_type="empty_annotation",
                    message="Annotation has no results",
                    file=file_path,
                    task_id=task_id
                ))
            
            for result in results:
                # Validate each result
                result_issues = validate_annotation_result(
                    result, valid_labels, task_id, file_path
                )
                issues.extend(result_issues)
                
                # Count labels
                if result.get("type") == "labels":
                    for label in result.get("value", {}).get("labels", []):
                        label_counts[label] += 1
    
    return issues, dict(label_counts)


def validate_file(file_path: str, valid_labels: Set[str]) -> Tuple[List[ValidationIssue], Dict]:
    """Validate a single Label Studio export file."""
    issues = []
    stats = {
        "tasks": 0,
        "annotations": 0,
        "label_counts": Counter(),
        "doc_types": Counter()
    }
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        issues.append(ValidationIssue(
            severity="error",
            issue_type="invalid_json",
            message=f"Invalid JSON: {e}",
            file=file_path
        ))
        return issues, stats
    except Exception as e:
        issues.append(ValidationIssue(
            severity="error",
            issue_type="file_read_error",
            message=f"Could not read file: {e}",
            file=file_path
        ))
        return issues, stats
    
    # Handle both list and single task formats
    if isinstance(data, dict):
        data = [data]
    
    stats["tasks"] = len(data)
    
    for idx, task in enumerate(data):
        task_issues, label_counts = validate_task(
            task, valid_labels, idx, file_path
        )
        issues.extend(task_issues)
        stats["label_counts"].update(label_counts)
        
        # Count annotations
        annotations = task.get("annotations", [])
        stats["annotations"] += len(annotations)
        
        # Count document types
        for annotation in annotations:
            for result in annotation.get("result", []):
                if result.get("from_name") == "doc_type":
                    choices = result.get("value", {}).get("choices", [])
                    for choice in choices:
                        stats["doc_types"][choice] += 1
    
    return issues, stats


def check_label_distribution(label_counts: Dict[str, int]) -> List[ValidationIssue]:
    """Check for potential issues in label distribution."""
    issues = []
    
    if not label_counts:
        issues.append(ValidationIssue(
            severity="error",
            issue_type="no_labels",
            message="No labels found in any annotations"
        ))
        return issues
    
    total_labels = sum(label_counts.values())
    
    # Check for severely imbalanced labels
    for label, count in label_counts.items():
        ratio = count / total_labels
        
        if count < 5:
            issues.append(ValidationIssue(
                severity="warning",
                issue_type="low_label_count",
                message=f"Label '{label}' has very few examples ({count}). "
                        f"Consider adding more or removing if not needed.",
                details={"label": label, "count": count}
            ))
        
        if ratio > 0.5:
            issues.append(ValidationIssue(
                severity="info",
                issue_type="dominant_label",
                message=f"Label '{label}' represents {ratio:.1%} of all labels. "
                        f"This is expected for common entities.",
                details={"label": label, "ratio": ratio}
            ))
    
    # Check for missing critical labels
    critical_labels = ["PATIENT_ID", "DATE_OF_SERVICE"]
    for label in critical_labels:
        if label not in label_counts:
            issues.append(ValidationIssue(
                severity="warning",
                issue_type="missing_critical_label",
                message=f"Critical label '{label}' not found in annotations",
                details={"label": label}
            ))
    
    return issues


def check_uds_measure_coverage(label_counts: Dict[str, int]) -> List[ValidationIssue]:
    """Check coverage of UDS measures."""
    issues = []
    
    for measure, required_labels in UDS_MEASURES.items():
        found_labels = [l for l in required_labels if l in label_counts]
        
        if not found_labels:
            issues.append(ValidationIssue(
                severity="info",
                issue_type="uds_measure_not_covered",
                message=f"UDS measure '{measure}' has no labeled examples",
                details={"measure": measure, "required_labels": required_labels}
            ))
        elif len(found_labels) < len(required_labels):
            missing = set(required_labels) - set(found_labels)
            issues.append(ValidationIssue(
                severity="info",
                issue_type="uds_measure_partial",
                message=f"UDS measure '{measure}' partially covered. "
                        f"Missing: {missing}",
                details={"measure": measure, "missing": list(missing)}
            ))
    
    return issues


def validate_annotations(
    labeled_dir: str,
    verbose: bool = True
) -> ValidationReport:
    """
    Validate all Label Studio export files in a directory.
    
    Args:
        labeled_dir: Directory containing Label Studio JSON exports
        verbose: Print progress information
    
    Returns:
        ValidationReport with all issues and statistics
    """
    report = ValidationReport()
    valid_labels = get_valid_labels()
    
    labeled_path = Path(labeled_dir)
    
    if not labeled_path.exists():
        report.issues.append(ValidationIssue(
            severity="error",
            issue_type="directory_not_found",
            message=f"Labeled directory not found: {labeled_dir}"
        ))
        return report
    
    json_files = list(labeled_path.glob("*.json"))
    
    if not json_files:
        report.issues.append(ValidationIssue(
            severity="error",
            issue_type="no_files",
            message=f"No JSON files found in {labeled_dir}"
        ))
        return report
    
    report.total_files = len(json_files)
    
    if verbose:
        print(f"Validating {len(json_files)} files in {labeled_dir}")
        print("-" * 60)
    
    all_label_counts = Counter()
    all_doc_types = Counter()
    
    for json_file in json_files:
        if verbose:
            print(f"  Checking: {json_file.name}...", end=" ")
        
        issues, stats = validate_file(str(json_file), valid_labels)
        report.issues.extend(issues)
        report.total_tasks += stats["tasks"]
        report.total_annotations += stats["annotations"]
        all_label_counts.update(stats["label_counts"])
        all_doc_types.update(stats["doc_types"])
        
        if verbose:
            error_count = len([i for i in issues if i.severity == "error"])
            warn_count = len([i for i in issues if i.severity == "warning"])
            if error_count:
                print(f"❌ {error_count} errors, {warn_count} warnings")
            elif warn_count:
                print(f"⚠️  {warn_count} warnings")
            else:
                print(f"✅ OK ({stats['tasks']} tasks)")
    
    report.label_counts = dict(all_label_counts)
    report.doc_type_counts = dict(all_doc_types)
    
    # Distribution checks
    dist_issues = check_label_distribution(dict(all_label_counts))
    report.issues.extend(dist_issues)
    
    # UDS coverage checks
    coverage_issues = check_uds_measure_coverage(dict(all_label_counts))
    report.issues.extend(coverage_issues)
    
    return report


def print_report(report: ValidationReport):
    """Print a formatted validation report."""
    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)
    
    # Summary
    print(f"\n📊 SUMMARY")
    print(f"   Files: {report.total_files}")
    print(f"   Tasks: {report.total_tasks}")
    print(f"   Annotations: {report.total_annotations}")
    print(f"   Unique labels: {len(report.label_counts)}")
    
    # Status
    if report.is_valid:
        print(f"\n✅ VALIDATION PASSED")
    else:
        print(f"\n❌ VALIDATION FAILED")
    
    # Errors
    if report.errors:
        print(f"\n🔴 ERRORS ({len(report.errors)})")
        for issue in report.errors:
            print(f"   • [{issue.issue_type}] {issue.message}")
            if issue.file:
                print(f"     File: {issue.file}, Task: {issue.task_id}")
    
    # Warnings
    if report.warnings:
        print(f"\n🟡 WARNINGS ({len(report.warnings)})")
        for issue in report.warnings[:10]:  # Show first 10
            print(f"   • [{issue.issue_type}] {issue.message}")
        if len(report.warnings) > 10:
            print(f"   ... and {len(report.warnings) - 10} more warnings")
    
    # Label distribution
    print(f"\n📈 LABEL DISTRIBUTION")
    if report.label_counts:
        sorted_labels = sorted(
            report.label_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        for label, count in sorted_labels[:15]:
            bar = "█" * min(count // 5, 30)
            print(f"   {label:30} {count:5} {bar}")
        if len(sorted_labels) > 15:
            print(f"   ... and {len(sorted_labels) - 15} more labels")
    else:
        print("   No labels found!")
    
    # Document types
    if report.doc_type_counts:
        print(f"\n📄 DOCUMENT TYPES")
        for doc_type, count in sorted(report.doc_type_counts.items(), key=lambda x: -x[1]):
            print(f"   {doc_type:30} {count:5}")
    
    # UDS Coverage
    print(f"\n🎯 UDS MEASURE COVERAGE")
    for measure, labels in UDS_MEASURES.items():
        found = sum(1 for l in labels if l in report.label_counts)
        total = len(labels)
        status = "✅" if found == total else "⚠️ " if found > 0 else "❌"
        print(f"   {status} {measure:35} {found}/{total} labels")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Validate Label Studio annotations before training"
    )
    parser.add_argument(
        "--labeled",
        default="./data/labeled",
        help="Directory containing Label Studio JSON exports"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save report to JSON file"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors"
    )
    
    args = parser.parse_args()
    
    # Run validation
    report = validate_annotations(args.labeled, verbose=not args.quiet)
    
    # Print report
    print_report(report)
    
    # Save report if requested
    if args.output:
        output_data = {
            "summary": {
                "total_files": report.total_files,
                "total_tasks": report.total_tasks,
                "total_annotations": report.total_annotations,
                "is_valid": report.is_valid,
                "error_count": len(report.errors),
                "warning_count": len(report.warnings),
            },
            "label_counts": report.label_counts,
            "doc_type_counts": report.doc_type_counts,
            "issues": [
                {
                    "severity": i.severity,
                    "type": i.issue_type,
                    "message": i.message,
                    "file": i.file,
                    "task_id": i.task_id,
                    "details": i.details
                }
                for i in report.issues
            ]
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nReport saved to: {args.output}")
    
    # Exit code
    if args.strict and report.warnings:
        sys.exit(1)
    elif not report.is_valid:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()