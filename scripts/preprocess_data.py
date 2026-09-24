"""
CLI script to execute the complete Data Preprocessing Pipeline.
Usage: python scripts/preprocess_data.py [--force]
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.preprocessing.pipeline import run_preprocessing_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run RoadSafety_ML Data Preprocessing Pipeline")
    parser.add_argument("--force", action="store_true", help="Force recalculation even if cache exists")
    parser.add_argument("--nrows", type=int, default=100000, help="Row limit for processing")
    args = parser.parse_args()

    print("==================================================")
    print("  RoadSafety_ML — Data Preprocessing Pipeline")
    print("==================================================")
    results = run_preprocessing_pipeline(force_run=args.force, nrows=args.nrows)
    print("\n✓ Preprocessing completed successfully!")
    print(f"  Missing Value Columns Analyzed: {len(results.get('missing_analysis', {}).get('col_info', {}))}")
    print(f"  Imputation Benchmark Strategies: {len(results.get('imputation', []))}")
    print(f"  IQR Outlier Features Capped: {len(results.get('outliers', []))}")
    print(f"  Categorical Nominal Encodings: {results.get('encoding', {}).get('ohe_count', 0)} OHE features")
    print("==================================================")


if __name__ == "__main__":
    main()
