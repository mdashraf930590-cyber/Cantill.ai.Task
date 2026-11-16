"""
Unit tests for Stage 5: Strict Schema Validator and Exporter.
"""

import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spend_pipeline.exporter import SchemaValidationError, export_submission_csv, validate_output_schema


def test_validate_output_schema_valid():
    df = pd.DataFrame(
        {
            "record_id": ["ERP:1", "TOOL:2"],
            "include": [1, 0],
            "exclusion_reason": ["", "DUPLICATE"],
            "supplier_key": ["supp_1", "supp_1"],
            "item_key": ["item_1", "item_1"],
            "spend": [123.45, 0.0],
            "l1": ["MRO", ""],
            "l2": ["Mechanical", ""],
            "l3": ["Valves", ""],
            "l4": ["", ""],
            "confidence": [0.85, 0.95],
        }
    )
    # Should not raise
    validate_output_schema(df, expected_row_count=2)


def test_validate_output_schema_failures():
    # 1. Missing columns
    with pytest.raises(SchemaValidationError, match="Missing required output columns"):
        validate_output_schema(pd.DataFrame({"record_id": ["1"]}))

    # 2. Row count mismatch
    df_valid = pd.DataFrame(
        {
            "record_id": ["ERP:1"],
            "include": [1],
            "exclusion_reason": [""],
            "supplier_key": ["supp_1"],
            "item_key": ["item_1"],
            "spend": [100.0],
            "l1": ["A"],
            "l2": ["B"],
            "l3": ["C"],
            "l4": [""],
            "confidence": [0.9],
        }
    )
    with pytest.raises(SchemaValidationError, match="Row count mismatch"):
        validate_output_schema(df_valid, expected_row_count=10)

    # 3. Excluded record with spend > 0
    df_bad_spend = df_valid.copy()
    df_bad_spend.loc[0, "include"] = 0
    df_bad_spend.loc[0, "exclusion_reason"] = "CANCELLED"
    df_bad_spend.loc[0, "spend"] = 50.0
    with pytest.raises(SchemaValidationError, match="Excluded records must have spend == 0.0"):
        validate_output_schema(df_bad_spend)

    # 4. Included record with non-blank exclusion_reason
    df_bad_inc = df_valid.copy()
    df_bad_inc.loc[0, "exclusion_reason"] = "DUPLICATE"
    with pytest.raises(SchemaValidationError, match="Included records must have a blank exclusion_reason"):
        validate_output_schema(df_bad_inc)


def test_export_submission_csv(tmp_path):
    df = pd.DataFrame(
        {
            "record_id": ["A1", "A2"],
            "include": [1, 0],
            "exclusion_reason": ["", "NON_PROCUREMENT"],
            "supplier_key": ["s1", "s2"],
            "item_key": ["i1", "i2"],
            "spend_reporting": [100.0, 50.0],  # export should force A2 spend to 0.0
            "l1": ["Cat1", "Cat1"],
            "l2": ["Cat2", "Cat2"],
            "l3": ["Cat3", "Cat3"],
            "confidence": [0.88, 0.95],
        }
    )
    out_file = tmp_path / "output.csv"
    res_path = export_submission_csv(df, str(out_file), expected_row_count=2)

    assert res_path.is_file()
    read_back = pd.read_csv(res_path)
    assert len(read_back) == 2
    assert read_back.loc[0, "spend"] == 100.0
    assert read_back.loc[1, "spend"] == 0.0
    assert read_back.loc[1, "l1"] == "" or pd.isna(read_back.loc[1, "l1"])
