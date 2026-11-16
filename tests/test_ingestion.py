"""
Unit tests for Stage 1: Ingestion Layer.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spend_pipeline.ingestion import FXLookupManager, parse_clean_amount, parse_date_with_flag, parse_vat_rate


def test_parse_clean_amount():
    assert parse_clean_amount(1234.56) == 1234.56
    assert parse_clean_amount("1,234.56") == 1234.56
    assert parse_clean_amount("1.234,56") == 1234.56
    assert parse_clean_amount("5,266.63") == 5266.63
    assert parse_clean_amount("(1,234.56)") == -1234.56
    assert parse_clean_amount("-500.00") == -500.00
    assert parse_clean_amount("R 45,000.50") == 45000.50
    assert parse_clean_amount("$ 120.00") == 120.00
    assert parse_clean_amount("") == 0.0
    assert parse_clean_amount(None) == 0.0
    assert parse_clean_amount(np.nan) == 0.0


def test_parse_vat_rate():
    assert parse_vat_rate("15%") == 0.15
    assert parse_vat_rate("15") == 0.15
    assert parse_vat_rate(0.15) == 0.15
    assert parse_vat_rate("0") == 0.0
    assert parse_vat_rate(0.0) == 0.0
    assert parse_vat_rate(np.nan, default_rate=0.15) == 0.15
    assert parse_vat_rate("", default_rate=0.15) == 0.15
    assert parse_vat_rate("invalid", default_rate=0.15) == 0.15


def test_parse_date_with_flag():
    dates = pd.Series(["2024-03-29", "27-Mar-24", "04/06/2023", "invalid-date", np.nan])
    parsed, is_unparseable = parse_date_with_flag(dates)

    assert parsed.iloc[0].year == 2024
    assert parsed.iloc[0].month == 3
    assert parsed.iloc[1].year == 2024
    assert parsed.iloc[1].month == 3
    assert not is_unparseable.iloc[0]
    assert not is_unparseable.iloc[1]
    assert is_unparseable.iloc[3]  # "invalid-date" flagged
    assert not is_unparseable.iloc[4]  # NaN is not unparseable, just missing


def test_fx_lookup_manager(tmp_path):
    fx_file = tmp_path / "fx.csv"
    fx_file.write_text("currency,month,units_per_usd\nEUR,2024-01,0.92\nEUR,2024-03,0.91\nCAD,2023-01,1.35\n")

    mgr = FXLookupManager(str(fx_file))
    # Exact match
    assert mgr.get_rate("EUR", pd.to_datetime("2024-01-15")) == 0.92
    assert mgr.get_rate("EUR", pd.to_datetime("2024-03-01")) == 0.91

    # Nearest month fallback
    assert mgr.get_rate("EUR", pd.to_datetime("2024-02-15")) in (0.91, 0.92)

    # Currency mean fallback
    assert mgr.get_rate("CAD", pd.to_datetime("2024-05-01")) == 1.35

    # Unknown currency default 1.0
    assert mgr.get_rate("XYZ", pd.to_datetime("2024-01-01")) == 1.0
