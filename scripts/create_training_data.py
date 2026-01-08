"""Create training dataset from labeled exports."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dataset import create_dataset_from_labeled


def main():
    parser = argparse.ArgumentParser(description="Create training dataset")
    parser.add_argument("--labeled", default="./data/labeled", help="Labeled data directory")
    parser.add_argument("--output", default="./data/dataset", help="Output directory")
    parser.add_argument("--test-size", type=float, default=0.15, help="Test split size")
    parser.add_argument("--val-size", type=float, default=0.15, help="Validation split size")
    
    args = parser.parse_args()
    
    create_dataset_from_labeled(
        labeled_dir=args.labeled,
        output_dir=args.output,
        test_size=args.test_size,
        val_size=args.val_size
    )


if __name__ == "__main__":
    main()