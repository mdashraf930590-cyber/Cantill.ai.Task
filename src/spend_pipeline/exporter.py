"""
Stage 5: Strict schema validator and exporter.
Guarantees every input row is preserved, validates output column types and constraints,
and writes standard submission CSV files.
"""

from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "record_id",
    "include",
    "exclusion_reason",
    "supplier_key",
    "item_key",
    "spend",
    "l1",
    "l2",
    "l3",
    "l4",
    "confidence",
]

VALID_EXCLUSION_REASONS = {"CANCELLED", "NOT_APPROVED", "NON_PROCUREMENT", "DUPLICATE"}


class SchemaValidationError(Exception):
    """Raised when output DataFrame fails schema or constraint validation."""
    pass


def validate_output_schema(df: pd.DataFrame, expected_row_count: Optional[int] = None) -> None:
    """Validate that the DataFrame strictly conforms to the required output schema and business rules."""
    # 1. Column presence and order
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaValidationError(f"Missing required output columns: {missing}")

    # 2. Row count preservation
    if expected_row_count is not None and len(df) != expected_row_count:
        raise SchemaValidationError(f"Row count mismatch! Expected {expected_row_count}, got {len(df)}")

    # 3. record_id uniqueness and non-null
    if df["record_id"].isna().sum() > 0:
        raise SchemaValidationError("Found null values in record_id!")
    if df["record_id"].duplicated().sum() > 0:
        dups = df[df["record_id"].duplicated()]["record_id"].tolist()[:5]
        raise SchemaValidationError(f"Found duplicate record_ids: {dups}")

    # 4. include values must be 0 or 1
    invalid_include = df[~df["include"].isin([0, 1])]
    if len(invalid_include) > 0:
        raise SchemaValidationError(f"Found invalid include values: {invalid_include['include'].unique()}")

    # 5. exclusion_reason validation
    included = df[df["include"] == 1]
    if included["exclusion_reason"].fillna("").str.strip().ne("").sum() > 0:
        raise SchemaValidationError("Included records must have a blank exclusion_reason!")

    excluded = df[df["include"] == 0]
    invalid_reasons = set(excluded["exclusion_reason"].dropna().unique()) - VALID_EXCLUSION_REASONS
    if invalid_reasons:
        raise SchemaValidationError(f"Found invalid exclusion reasons: {invalid_reasons}")
    if excluded["exclusion_reason"].isna().sum() > 0 or (excluded["exclusion_reason"].str.strip() == "").sum() > 0:
        raise SchemaValidationError("Excluded records must specify an exclusion_reason!")

    # 6. spend constraint
    if (excluded["spend"] != 0.0).sum() > 0:
        raise SchemaValidationError("Excluded records must have spend == 0.0!")

    # 7. confidence bounds
    if (df["confidence"] < 0.0).sum() > 0 or (df["confidence"] > 1.0).sum() > 0:
        raise SchemaValidationError("Confidence scores must be between 0.0 and 1.0!")


def export_submission_csv(
    df: pd.DataFrame,
    output_path: str,
    expected_row_count: Optional[int] = None,
) -> Path:
    """
    Format and export final DataFrame to CSV according to exact output specification.
    """
    out_df = pd.DataFrame()
    out_df["record_id"] = df["record_id"].astype(str)
    out_df["include"] = df["include"].astype(int)
    out_df["exclusion_reason"] = df["exclusion_reason"].fillna("").astype(str)
    out_df["supplier_key"] = df["supplier_key"].fillna("supp_unknown").astype(str)
    out_df["item_key"] = df["item_key"].fillna("item_unknown").astype(str)
    out_df["spend"] = df["spend_reporting"].round(2)

    # Ensure excluded rows have spend=0 and blank l1-l4
    mask_ex = out_df["include"] == 0
    out_df.loc[mask_ex, "spend"] = 0.0

    for i in range(1, 5):
        col = f"l{i}"
        if col in df.columns:
            out_df[col] = df[col].fillna("").astype(str)
        else:
            out_df[col] = ""
        out_df.loc[mask_ex, col] = ""

    out_df["confidence"] = df["confidence"].round(2)

    # Validate schema
    validate_output_schema(out_df, expected_row_count=expected_row_count)

    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out_df[REQUIRED_COLUMNS].to_csv(dest, index=False)
    return dest
