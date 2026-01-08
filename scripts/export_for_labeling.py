"""Export processed PDFs to Label Studio format for annotation."""

import argparse
from pathlib import Path
import sys
import json
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.processor import PDFProcessor


def create_label_studio_tasks(
    processed_dir: str,
    output_file: str,
    image_base_url: str = "/data/local-files/?d="
) -> List[Dict]:
    """
    Create Label Studio import file from processed documents.
    
    Args:
        processed_dir: Directory containing processed images and JSON files
        output_file: Output JSON file for Label Studio import
        image_base_url: Base URL for images in Label Studio
    
    Returns:
        List of Label Studio tasks
    """
    processed_path = Path(processed_dir)
    tasks = []
    
    # Find all task JSON files
    for task_file in sorted(processed_path.glob("*_tasks.json")):
        with open(task_file, "r") as f:
            file_tasks = json.load(f)
        
        for task in file_tasks:
            # Update image path for Label Studio
            original_path = task["data"]["image"]
            
            # Create Label Studio compatible path
            if image_base_url:
                task["data"]["image"] = f"{image_base_url}{original_path}"
            
            # Add OCR text as a reference field
            if "ocr" in task["data"]:
                task["data"]["text"] = " ".join([w["text"] for w in task["data"]["ocr"]])
            
            tasks.append(task)
    
    # Save combined tasks file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(tasks, f, indent=2)
    
    return tasks


def create_label_studio_config() -> str:
    """Generate Label Studio labeling configuration XML."""
    
    config = """<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  
  <Labels name="label" toName="image">
    <!-- Patient Demographics -->
    <Label value="PATIENT_ID" background="#1f77b4"/>
    <Label value="DATE_OF_SERVICE" background="#2ca02c"/>
    <Label value="PROVIDER_NPI" background="#17becf"/>
    
    <!-- Clinical Codes -->
    <Label value="DIAGNOSIS_ICD10" background="#d62728"/>
    <Label value="PROCEDURE_CPT" background="#ff7f0e"/>
    
    <!-- Vitals (UDS Table 6A/6B) -->
    <Label value="BLOOD_PRESSURE" background="#9467bd"/>
    <Label value="A1C_VALUE" background="#e377c2"/>
    <Label value="BMI" background="#8c564b"/>
    <Label value="WEIGHT" background="#bcbd22"/>
    
    <!-- Screenings (UDS Table 6B) -->
    <Label value="DEPRESSION_SCREEN" background="#7f7f7f"/>
    <Label value="TOBACCO_STATUS" background="#aec7e8"/>
    <Label value="CERVICAL_SCREEN" background="#ffbb78"/>
    <Label value="BREAST_SCREEN" background="#98df8a"/>
    <Label value="COLORECTAL_SCREEN" background="#ff9896"/>
    <Label value="HIV_SCREEN" background="#c5b0d5"/>
    
    <!-- Lab Results -->
    <Label value="LAB_RESULT" background="#c49c94"/>
    <Label value="LAB_NAME" background="#f7b6d2"/>
    <Label value="LAB_VALUE" background="#c7c7c7"/>
    <Label value="LAB_DATE" background="#dbdb8d"/>
    
    <!-- Medications & Vaccinations -->
    <Label value="MEDICATION" background="#9edae5"/>
    <Label value="VACCINATION" background="#393b79"/>
    <Label value="VACCINATION_DATE" background="#5254a3"/>
  </Labels>
  
  <Rectangle name="bbox" toName="image" strokeWidth="2"/>
  
  <TextArea name="notes" toName="image" 
            placeholder="Add any notes about this document..."
            rows="3" maxSubmissions="1"/>
</View>"""
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Export processed documents for Label Studio annotation"
    )
    parser.add_argument(
        "--processed", 
        default="./data/processed",
        help="Directory containing processed documents"
    )
    parser.add_argument(
        "--output", 
        default="./data/label_studio_import.json",
        help="Output JSON file for Label Studio import"
    )
    parser.add_argument(
        "--config-output",
        default="./data/label_studio_config.xml",
        help="Output file for Label Studio labeling config"
    )
    parser.add_argument(
        "--image-base-url",
        default="/data/local-files/?d=",
        help="Base URL for images in Label Studio"
    )
    parser.add_argument(
        "--process-pdfs",
        default=None,
        help="Optional: Process PDFs from this directory first"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for PDF processing"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Export for Label Studio")
    print("=" * 60)
    
    # Optionally process PDFs first
    if args.process_pdfs:
        print(f"\nProcessing PDFs from: {args.process_pdfs}")
        processor = PDFProcessor(dpi=args.dpi)
        results = processor.batch_process(args.process_pdfs, args.processed)
        success = sum(1 for r in results if r["status"] == "success")
        print(f"Processed {success}/{len(results)} PDFs")
    
    # Create Label Studio tasks
    print(f"\nCreating Label Studio tasks from: {args.processed}")
    tasks = create_label_studio_tasks(
        processed_dir=args.processed,
        output_file=args.output,
        image_base_url=args.image_base_url
    )
    print(f"Created {len(tasks)} tasks")
    print(f"Tasks saved to: {args.output}")
    
    # Save labeling configuration
    config = create_label_studio_config()
    config_path = Path(args.config_output)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        f.write(config)
    print(f"Labeling config saved to: {args.config_output}")
    
    # Print instructions
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("""
1. Start Label Studio:
   label-studio start

2. Create a new project

3. Go to Settings > Labeling Interface
   - Copy contents of: {config_file}
   - Paste into the Code editor

4. Go to Settings > Cloud Storage
   - Add Local Storage pointing to your processed images directory
   
5. Import tasks:
   - Go to project
   - Click "Import"
   - Upload: {tasks_file}

6. Start labeling!

7. When done, export annotations:
   - Click "Export"
   - Choose JSON format
   - Save to: ./data/labeled/
""".format(
        config_file=args.config_output,
        tasks_file=args.output
    ))
    print("=" * 60)


if __name__ == "__main__":
    main()