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
  <Header value="UDS Metrics Document Annotation"/>
  <Image name="image" value="$image" zoom="true" zoomControl="true" rotateControl="true"/>
  
  <View style="padding: 10px; margin-top: 10px; background: #f0f7ff; border-radius: 5px;">
    <Header value="Instructions" size="5"/>
    <Text name="instructions" value="Select a label, then click and drag to highlight text regions. Use keyboard shortcuts for faster labeling."/>
  </View>
  
  <Labels name="label" toName="image">
    
    <!-- ============================================ -->
    <!-- PATIENT DEMOGRAPHICS (Blue shades) -->
    <!-- ============================================ -->
    <Header value="── Patient Demographics ──"/>
    <Label value="PATIENT_ID" background="#1f77b4" hotkey="1"/>
    <Label value="PATIENT_NAME" background="#aec7e8"/>
    <Label value="DATE_OF_BIRTH" background="#17becf"/>
    <Label value="GENDER" background="#9edae5"/>
    <Label value="DATE_OF_SERVICE" background="#2ca02c" hotkey="2"/>
    <Label value="PROVIDER_NAME" background="#98df8a"/>
    <Label value="PROVIDER_NPI" background="#c5b0d5"/>
    <Label value="FACILITY_NAME" background="#c7c7c7"/>
    
    <!-- ============================================ -->
    <!-- CLINICAL CODES (Red/Orange shades) -->
    <!-- ============================================ -->
    <Header value="── Clinical Codes ──"/>
    <Label value="DIAGNOSIS_ICD10" background="#d62728" hotkey="3"/>
    <Label value="PROCEDURE_CPT" background="#ff7f0e" hotkey="4"/>
    
    <!-- ============================================ -->
    <!-- VITALS - UDS Table 6A/6B (Green shades) -->
    <!-- ============================================ -->
    <Header value="── Vitals (UDS 6A/6B) ──"/>
    <Label value="BLOOD_PRESSURE" background="#2ca02c"/>
    <Label value="BLOOD_PRESSURE_SYSTOLIC" background="#98df8a"/>
    <Label value="BLOOD_PRESSURE_DIASTOLIC" background="#d4edda"/>
    <Label value="A1C_VALUE" background="#00ff00"/>
    <Label value="BMI" background="#32cd32"/>
    <Label value="WEIGHT" background="#7cfc00"/>
    <Label value="HEIGHT" background="#adff2f"/>
    
    <!-- ============================================ -->
    <!-- SCREENINGS - UDS Table 6B (Teal shades) -->
    <!-- ============================================ -->
    <Header value="── Screenings (UDS 6B) ──"/>
    <Label value="DEPRESSION_SCREEN" background="#008080"/>
    <Label value="DEPRESSION_SCORE" background="#20b2aa"/>
    <Label value="TOBACCO_STATUS" background="#40e0d0"/>
    <Label value="TOBACCO_COUNSELING" background="#00ced1"/>
    <Label value="HIV_SCREEN" background="#48d1cc"/>
    <Label value="HIV_RESULT" background="#afeeee"/>
    <Label value="CERVICAL_SCREEN" background="#5f9ea0"/>
    <Label value="CERVICAL_SCREEN_DATE" background="#b0e0e6"/>
    <Label value="BREAST_SCREEN" background="#87ceeb"/>
    <Label value="BREAST_SCREEN_DATE" background="#add8e6"/>
    
    <!-- ============================================ -->
    <!-- COLORECTAL CANCER SCREENING (Purple shades) -->
    <!-- ============================================ -->
    <Header value="── Colorectal Screening ──"/>
    <Label value="COLONOSCOPY_DATE" background="#9467bd" hotkey="5"/>
    <Label value="COLONOSCOPY_RESULT" background="#8c564b" hotkey="6"/>
    <Label value="COLONOSCOPY_INDICATION" background="#e377c2"/>
    <Label value="BOWEL_PREP_QUALITY" background="#bcbd22"/>
    <Label value="CECUM_REACHED" background="#dbdb8d"/>
    <Label value="WITHDRAWAL_TIME" background="#f7b6d2"/>
    
    <!-- ============================================ -->
    <!-- POLYP FINDINGS (Pink/Magenta shades) -->
    <!-- ============================================ -->
    <Header value="── Polyp Findings ──"/>
    <Label value="POLYP_FINDING" background="#ff69b4" hotkey="7"/>
    <Label value="POLYP_LOCATION" background="#da70d6"/>
    <Label value="POLYP_SIZE" background="#ba55d3"/>
    <Label value="POLYP_COUNT" background="#9932cc"/>
    <Label value="BIOPSY_TAKEN" background="#8b008b"/>
    <Label value="BIOPSY_RESULT" background="#ff1493"/>
    <Label value="PATHOLOGY_DIAGNOSIS" background="#c71585" hotkey="8"/>
    <Label value="COMPLICATIONS" background="#dc143c"/>
    
    <!-- ============================================ -->
    <!-- STOOL TESTS - FIT/FOBT (Lime shades) -->
    <!-- ============================================ -->
    <Header value="── Stool Tests (FIT/FOBT) ──"/>
    <Label value="STOOL_TEST_TYPE" background="#228b22" hotkey="9"/>
    <Label value="STOOL_TEST_DATE" background="#32cd32"/>
    <Label value="STOOL_TEST_RESULT" background="#00ff7f" hotkey="0"/>
    <Label value="STOOL_TEST_VALUE" background="#7cfc00"/>
    <Label value="REFERENCE_RANGE" background="#adff2f"/>
    
    <!-- ============================================ -->
    <!-- IMMUNIZATIONS (Yellow shades) -->
    <!-- ============================================ -->
    <Header value="── Immunizations ──"/>
    <Label value="VACCINATION" background="#ffd700"/>
    <Label value="VACCINATION_DATE" background="#ffec8b"/>
    <Label value="VACCINATION_TYPE" background="#fff68f"/>
    <Label value="FLU_VACCINE" background="#fffacd"/>
    <Label value="COVID_VACCINE" background="#fafad2"/>
    <Label value="PNEUMONIA_VACCINE" background="#eee8aa"/>
    
    <!-- ============================================ -->
    <!-- LAB RESULTS (Gray shades) -->
    <!-- ============================================ -->
    <Header value="── Lab Results ──"/>
    <Label value="LAB_NAME" background="#696969"/>
    <Label value="LAB_VALUE" background="#808080"/>
    <Label value="LAB_DATE" background="#a9a9a9"/>
    <Label value="LAB_UNIT" background="#d3d3d3"/>
    <Label value="LAB_RESULT" background="#c0c0c0"/>
    <Label value="LAB_REFERENCE_RANGE" background="#dcdcdc"/>
    
    <!-- ============================================ -->
    <!-- MEDICATIONS (Coral shades) -->
    <!-- ============================================ -->
    <Header value="── Medications ──"/>
    <Label value="MEDICATION" background="#ff7f50"/>
    <Label value="MEDICATION_DOSE" background="#ffa07a"/>
    <Label value="MEDICATION_FREQUENCY" background="#fa8072"/>
    <Label value="STATIN_THERAPY" background="#e9967a"/>
    <Label value="ASPIRIN_THERAPY" background="#f08080"/>
    
    <!-- ============================================ -->
    <!-- PRENATAL (Light Pink - if applicable) -->
    <!-- ============================================ -->
    <Header value="── Prenatal Care ──"/>
    <Label value="PRENATAL_VISIT" background="#ffb6c1"/>
    <Label value="GESTATIONAL_AGE" background="#ffc0cb"/>
    <Label value="EDD" background="#ffe4e1"/>
    <Label value="PRENATAL_LABS" background="#fff0f5"/>
    
    <!-- ============================================ -->
    <!-- FOLLOW-UP & RECOMMENDATIONS (Cyan shades) -->
    <!-- ============================================ -->
    <Header value="── Follow-up ──"/>
    <Label value="NEXT_SCREENING_DATE" background="#008b8b"/>
    <Label value="SCREENING_INTERVAL" background="#00bfff"/>
    <Label value="RECOMMENDATION" background="#87cefa"/>
    <Label value="RISK_CATEGORY" background="#b0c4de"/>
    <Label value="FOLLOW_UP_DATE" background="#e0ffff"/>
    
  </Labels>
  
  <Rectangle name="bbox" toName="image" strokeWidth="2"/>
  
  <View style="margin-top: 20px; padding: 10px; background: #fff3cd; border-radius: 5px;">
    <Header value="Document Classification" size="5"/>
    <Choices name="doc_type" toName="image" choice="single" showInline="true">
      <Choice value="colonoscopy_report"/>
      <Choice value="pathology_report"/>
      <Choice value="fit_fobt_result"/>
      <Choice value="lab_report"/>
      <Choice value="clinical_note"/>
      <Choice value="screening_form"/>
      <Choice value="immunization_record"/>
      <Choice value="referral"/>
      <Choice value="other"/>
    </Choices>
  </View>
  
  <View style="margin-top: 10px; padding: 10px; background: #f8f9fa; border-radius: 5px;">
    <Header value="UDS Measure Category" size="5"/>
    <Choices name="uds_measure" toName="image" choice="multiple" showInline="true">
      <Choice value="colorectal_screening"/>
      <Choice value="cervical_screening"/>
      <Choice value="breast_screening"/>
      <Choice value="diabetes_control"/>
      <Choice value="hypertension_control"/>
      <Choice value="depression_screening"/>
      <Choice value="tobacco_screening"/>
      <Choice value="hiv_screening"/>
      <Choice value="immunizations"/>
      <Choice value="prenatal_care"/>
      <Choice value="bmi_screening"/>
      <Choice value="statin_therapy"/>
    </Choices>
  </View>
  
  <View style="margin-top: 10px;">
    <Header value="Document Notes" size="5"/>
    <TextArea name="notes" toName="image" 
              placeholder="Add notes about document quality, unclear text, or special observations..."
              rows="2" maxSubmissions="1"/>
  </View>
  
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