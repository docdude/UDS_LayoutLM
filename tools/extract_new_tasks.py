"""
Extract only NEW tasks from label_studio_import.json that weren't in the original import.

This allows you to add new tasks to an existing Label Studio project
without duplicating existing tasks or losing your annotations.
"""

import json
import argparse
from pathlib import Path
from typing import Set


def get_image_filename(task: dict) -> str:
    """Extract the image filename from a task's image path."""
    image_path = task.get("data", {}).get("image", "")
    # Handle various path formats
    if "?d=" in image_path:
        image_path = image_path.split("?d=")[-1]
    return Path(image_path).name


def load_existing_images(existing_file: str) -> Set[str]:
    """Load image filenames from an existing JSON file."""
    with open(existing_file, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
    
    return {get_image_filename(task) for task in tasks}


def load_existing_from_directory(processed_dir: str, cutoff_date: str = None) -> Set[str]:
    """
    Load image filenames from the processed directory.
    Optionally filter by files created before a cutoff date.
    """
    processed_path = Path(processed_dir)
    images = set()
    
    for img_file in processed_path.glob("*.png"):
        images.add(img_file.name)
    
    return images


def extract_new_tasks(
    new_import_file: str,
    existing_file: str = None,
    existing_images: Set[str] = None,
    output_file: str = None
) -> list:
    """
    Extract tasks from new_import_file that don't exist in existing_file.
    
    Args:
        new_import_file: The full import JSON with all tasks
        existing_file: Previous import JSON to compare against
        existing_images: Set of image filenames already in Label Studio
        output_file: Where to save the new tasks
    
    Returns:
        List of new tasks
    """
    # Load the new full import
    with open(new_import_file, 'r', encoding='utf-8') as f:
        all_tasks = json.load(f)
    
    print(f"Total tasks in new import: {len(all_tasks)}")
    
    # Get existing image filenames
    if existing_images is None:
        if existing_file:
            existing_images = load_existing_images(existing_file)
        else:
            existing_images = set()
    
    print(f"Existing images to exclude: {len(existing_images)}")
    
    # Filter to only new tasks
    new_tasks = []
    skipped = []
    
    for task in all_tasks:
        img_name = get_image_filename(task)
        if img_name not in existing_images:
            new_tasks.append(task)
        else:
            skipped.append(img_name)
    
    print(f"New tasks to add: {len(new_tasks)}")
    print(f"Skipped (already exist): {len(skipped)}")
    
    # Save new tasks
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(new_tasks, f, indent=2, ensure_ascii=False)
        print(f"\nSaved new tasks to: {output_file}")
    
    return new_tasks


def main():
    parser = argparse.ArgumentParser(
        description="Extract only new tasks that don't exist in Label Studio yet"
    )
    parser.add_argument(
        "--new-import",
        default="./data/label_studio_import.json",
        help="The new import file with all tasks (default: ./data/label_studio_import.json)"
    )
    parser.add_argument(
        "--existing",
        default=None,
        help="Previous import JSON to compare against (to identify existing tasks)"
    )
    parser.add_argument(
        "--existing-list",
        default=None,
        help="Text file with list of existing image filenames (one per line)"
    )
    parser.add_argument(
        "--output",
        default="./data/label_studio_import_new_only.json",
        help="Output file for new tasks only"
    )
    parser.add_argument(
        "--preannotate",
        action="store_true",
        help="Run preannotation on the new tasks after extraction"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Extract New Tasks for Label Studio")
    print("=" * 60)
    
    existing_images = None
    
    # Load existing images from various sources
    if args.existing:
        print(f"\nLoading existing tasks from: {args.existing}")
        existing_images = load_existing_images(args.existing)
    elif args.existing_list:
        print(f"\nLoading existing image list from: {args.existing_list}")
        with open(args.existing_list, 'r') as f:
            existing_images = {line.strip() for line in f if line.strip()}
    else:
        # Default: use the v3 preannotated file as reference for existing tasks
        default_existing = "./data/label_studio_import_preannotated_v3.json"
        if Path(default_existing).exists():
            print(f"\nUsing default existing file: {default_existing}")
            existing_images = load_existing_images(default_existing)
        else:
            print("\nWARNING: No existing file specified. Will include ALL tasks.")
            print("Use --existing to specify the original import file.")
    
    new_tasks = extract_new_tasks(
        new_import_file=args.new_import,
        existing_images=existing_images,
        output_file=args.output
    )
    
    if args.preannotate and new_tasks:
        print("\n" + "=" * 60)
        print("Running preannotation on new tasks...")
        print("=" * 60)
        
        import subprocess
        preannotate_output = args.output.replace(".json", "_preannotated.json")
        result = subprocess.run([
            "python", "tools/preannotate_crc_from_regex.py",
            args.output,
            preannotate_output
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(result.stdout)
            print(f"\nPreannotated file saved to: {preannotate_output}")
        else:
            print(f"Preannotation failed: {result.stderr}")
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print(f"""
1. In Label Studio, go to your existing project

2. Click "Import" 

3. Upload the new tasks file:
   {args.output}
   
   (or the preannotated version if you used --preannotate)

4. The new tasks will be ADDED to your project
   Your existing annotations are preserved!

5. Continue labeling the new documents
""")
    print("=" * 60)


if __name__ == "__main__":
    main()
