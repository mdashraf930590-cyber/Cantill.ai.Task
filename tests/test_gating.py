"""
Unit tests for Stage 2: Deterministic and Heuristic Gating Engine.
"""

import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spend_pipeline.config import CompanyConfig, DeduplicationConfig, GatingConfig
from spend_pipeline.gating import apply_gating_and_deduplication, clean_supplier_for_matching, extract_normalized_po


def test_extract_normalized_po():
    assert extract_normalized_po("4510085877/10.0") == "4510085877"
    assert extract_normalized_po("PO 4510109330") == "4510109330"
    assert extract_normalized_po("PO#4510059209") == "4510059209"
    assert extract_normalized_po("4510119046") == "4510119046"
    assert extract_normalized_po("PO-30000001") == "30000001"
    assert extract_normalized_po("") == ""


def test_clean_supplier_for_matching():
    assert clean_supplier_for_matching("ACME SUPPLIES (PTY) LTD") == "acme"
    assert clean_supplier_for_matching("DUNTON (PTY) LTD") == "dunton"
    assert clean_supplier_for_matching("CALDALE ELECTRICAL CC") == "caldale electrical"
    assert clean_supplier_for_matching("YARAX TECHNOLOGIES (PTY) LTD") == "yarax"


def test_gating_precedence():
    """Verify strict precedence: CANCELLED -> NOT_APPROVED -> NON_PROCUREMENT -> DUPLICATE."""
    data = [
        # 1. CANCELLED (even if also IC or Tax)
        {
            "record_id": "R1", "status_raw": "Cancelled", "gl_account_desc_raw": "IC Payables",
            "item_description_raw": "Sales Tax", "is_primary": True, "spend_reporting": 100.0,
            "invoice_raw": "INV1", "supplier_name_raw": "Acme", "po_raw": "PO1", "date_parsed": pd.to_datetime("2024-01-01"),
            "gl_account_raw": "GL1", "document_type_raw": "PV",
        },
        # 2. NOT_APPROVED (rejected, not cancelled)
        {
            "record_id": "R2", "status_raw": "Rejected", "gl_account_desc_raw": "Normal Spend",
            "item_description_raw": "Valve parts", "is_primary": True, "spend_reporting": 200.0,
            "invoice_raw": "INV2", "supplier_name_raw": "Acme", "po_raw": "PO2", "date_parsed": pd.to_datetime("2024-01-01"),
            "gl_account_raw": "GL2", "document_type_raw": "PV",
        },
        # 3. NON_PROCUREMENT (approved, but tax)
        {
            "record_id": "R3", "status_raw": "Paid", "gl_account_desc_raw": "Operating Expense",
            "item_description_raw": "State & Local Taxes - Expense", "is_primary": True, "spend_reporting": 300.0,
            "invoice_raw": "INV3", "supplier_name_raw": "Acme", "po_raw": "PO3", "date_parsed": pd.to_datetime("2024-01-01"),
            "gl_account_raw": "GL3", "document_type_raw": "PV",
        },
        # 4. Valid Spend Primary
        {
            "record_id": "ERP:R4", "status_raw": "Paid", "gl_account_desc_raw": "Direct Spend",
            "item_description_raw": "Ball Valve 2in", "is_primary": True, "spend_reporting": 400.0,
            "invoice_raw": "INV4", "supplier_name_raw": "Acme Ltd", "po_raw": "4510000001", "date_parsed": pd.to_datetime("2024-01-01"),
            "gl_account_raw": "GL4", "document_type_raw": "PV",
        },
        # 5. DUPLICATE Cross-System Tool record matching ERP:R4
        {
            "record_id": "TOOL:R5", "status_raw": "Paid", "gl_account_desc_raw": "Direct Spend",
            "item_description_raw": "Ball Valve 2in", "is_primary": False, "spend_reporting": 400.0,
            "invoice_raw": "INV4", "supplier_name_raw": "Acme", "po_raw": "4510000001", "date_parsed": pd.to_datetime("2024-01-02"),
            "gl_account_raw": "GL4", "document_type_raw": "PV",
        },
    ]

    df = pd.DataFrame(data)
    config = CompanyConfig(
        company_id="test",
        reporting_currency="USD",
        taxonomy_file="dummy",
        sources=[],
        deduplication=DeduplicationConfig(cross_system_linkage=True),
    )

    result = apply_gating_and_deduplication(df, config)

    # R1: CANCELLED
    assert result.loc[result["record_id"] == "R1", "include"].values[0] == 0
    assert result.loc[result["record_id"] == "R1", "exclusion_reason"].values[0] == "CANCELLED"

    # R2: NOT_APPROVED
    assert result.loc[result["record_id"] == "R2", "include"].values[0] == 0
    assert result.loc[result["record_id"] == "R2", "exclusion_reason"].values[0] == "NOT_APPROVED"

    # R3: NON_PROCUREMENT
    assert result.loc[result["record_id"] == "R3", "include"].values[0] == 0
    assert result.loc[result["record_id"] == "R3", "exclusion_reason"].values[0] == "NON_PROCUREMENT"

    # ERP:R4: INCLUDED
    assert result.loc[result["record_id"] == "ERP:R4", "include"].values[0] == 1
    assert result.loc[result["record_id"] == "ERP:R4", "exclusion_reason"].values[0] == ""
    assert result.loc[result["record_id"] == "ERP:R4", "spend_reporting"].values[0] == 400.0

    # TOOL:R5: DUPLICATE
    assert result.loc[result["record_id"] == "TOOL:R5", "include"].values[0] == 0
    assert result.loc[result["record_id"] == "TOOL:R5", "exclusion_reason"].values[0] == "DUPLICATE"
    assert result.loc[result["record_id"] == "TOOL:R5", "spend_reporting"].values[0] == 0.0
