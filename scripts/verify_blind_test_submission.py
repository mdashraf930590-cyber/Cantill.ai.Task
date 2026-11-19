#!/usr/bin/env python3
"""
Quick Blind-Test Pre-Flight Verifier.
Run this script on Thursday before 12:00 noon to verify your submission files:
    python3 scripts/verify_blind_test_submission.py
"""

from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from spend_pipeline.exporter import validate_output_schema, SchemaValidationError

REQUIRED_FILES = [
    "submission/ashraf__A2.csv",
    "submission/ashraf__B2.csv",
    "submission/ashraf__C.csv",
    "configs/company_C.yaml",
]

def main():
    print("=" * 65)
    print("      CANTILL.AI BLIND TEST SUBMISSION VERIFICATION")
    print("=" * 65)

    all_passed = True

    # 1. Check file existence
    print("\n[1/3] Checking Required Files Existence...")
    for f in REQUIRED_FILES:
        p = Path(f)
        if p.is_file():
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  [OK] Found: {f} ({size_mb:.2f} MB)")
        else:
            print(f"  [FAIL] Missing file: {f}")
            all_passed = False

    if not all_passed:
        print("\nERROR: Please generate all missing files before submitting.")
        sys.exit(1)

    # 2. Schema Validation
    print("\n[2/3] Validating Submission CSV Schemas & Integrity...")
    for csv_file in REQUIRED_FILES[:3]:
        df = pd.read_csv(csv_file, low_memory=False)
        try:
            validate_output_schema(df)
            inc_count = (df["include"] == 1).sum()
            exc_count = (df["include"] == 0).sum()
            print(f"  [OK] {csv_file}:")
            print(f"       Total rows: {len(df):,} | Included: {inc_count:,} | Excluded: {exc_count:,}")
        except SchemaValidationError as e:
            print(f"  [FAIL] Schema error in {csv_file}: {e}")
            all_passed = False

    # 3. Git Status Reminder
    print("\n[3/3] Ready for Git Submission:")
    print("  Run these commands before 12:00 noon:")
    print("    git add configs/company_C.yaml submission/ashraf__A2.csv submission/ashraf__B2.csv submission/ashraf__C.csv")
    print("    git commit -m 'Submit blind test: company_C.yaml and ashraf outputs'")
    print("    git push origin main")

    if all_passed:
        print("\n" + "=" * 65)
        print(" SUCCESS: All files are valid and ready for final scoring!")
        print("=" * 65)
    else:
        print("\n" + "=" * 65)
        print(" WARNING: Some verification checks failed. Review errors above.")
        print("=" * 65)
        sys.exit(1)

if __name__ == "__main__":
    main()

