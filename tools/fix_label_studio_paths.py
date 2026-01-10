"""Fix image paths in Label Studio JSON to use relative paths."""

import json
import argparse
from pathlib import Path


def fix_paths(input_file: str, output_file: str, base_path: str = None):
    """
    Fix absolute Windows paths in Label Studio JSON to relative paths.
    
    Args:
        input_file: Input JSON file with absolute paths
        output_file: Output JSON file with fixed paths
        base_path: Base path to make paths relative to (optional)
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
    
    fixed_count = 0
    
    for task in tasks:
        if 'data' in task and 'image' in task['data']:
            image_path = task['data']['image']
            
            # Check if it's a local-files URL with absolute path
            if image_path.startswith('/data/local-files/?d='):
                # Extract the path after the prefix
                path_after_prefix = image_path.replace('/data/local-files/?d=', '')
                
                # If it's an absolute path, convert to relative
                if path_after_prefix.startswith('C:\\') or path_after_prefix.startswith('C:/'):
                    path_obj = Path(path_after_prefix)
                    filename = path_obj.name
                    # Use relative path from document root
                    task['data']['image'] = f"/data/local-files/?d=processed/{filename}"
                    fixed_count += 1
            
            # Also handle raw absolute paths (no prefix)
            elif image_path.startswith('C:\\') or image_path.startswith('C:/'):
                path_obj = Path(image_path)
                filename = path_obj.name
                task['data']['image'] = f"/data/local-files/?d=processed/{filename}"
                fixed_count += 1
        
        # Fix paths in predictions if they exist
        if 'predictions' in task:
            for prediction in task['predictions']:
                if 'result' in prediction:
                    for result in prediction['result']:
                        if 'value' in result and 'image' in result['value']:
                            image_path = result['value']['image']
                            if image_path.startswith('C:\\') or image_path.startswith('C:/'):
                                path_obj = Path(image_path)
                                filename = path_obj.name
                                result['value']['image'] = f"processed/{filename}"
    
    # Write fixed JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)
    
    print(f"Fixed {fixed_count} image paths")
    print(f"Output saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Fix absolute paths in Label Studio JSON to relative paths"
    )
    parser.add_argument(
        "--input",
        default="./data/label_studio_import_preannotated.json",
        help="Input JSON file with absolute paths"
    )
    parser.add_argument(
        "--output",
        default="./data/label_studio_import_preannotated_fixed.json",
        help="Output JSON file with fixed paths"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Fixing Label Studio Image Paths")
    print("=" * 60)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print()
    
    fix_paths(args.input, args.output)
    
    print()
    print("=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. Delete the old project in Label Studio (or clear tasks)")
    print("2. Import the fixed JSON file:")
    print(f"   {args.output}")
    print("3. Images should now load correctly!")
    print("=" * 60)


if __name__ == "__main__":
    main()
