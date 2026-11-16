"""
Unit tests for Stage 3: Entity Resolution Module.
"""

import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spend_pipeline.config import CompanyConfig
from spend_pipeline.entity_resolution import cluster_strings_tfidf, normalize_entity_text, resolve_entities


def test_normalize_entity_text():
    assert normalize_entity_text("ACME EQUIPMENT (PTY) LTD") == "acme equipment"
    assert normalize_entity_text("PELFIELD STEEL (PTY) LTD", strip_boilerplate=True) == "pelfield steel"
    assert normalize_entity_text("KINORA  INDUSTRIAL SUPPLIES ENGINEERING", strip_boilerplate=True) == "kinora industrial"
    assert normalize_entity_text("München GmbH") == "munchen"


def test_cluster_strings_tfidf():
    names = [
        "kinora industrial supplies",
        "kinora  industrial supplies",
        "kinora industrial",
        "dunton equipment",
        "dunton",
    ]
    mapping = cluster_strings_tfidf(names, similarity_threshold=0.75)
    # The variations of kinora should map to the same cluster key
    assert mapping["kinora industrial supplies"] == mapping["kinora  industrial supplies"]
    # dunton variations should map to the same key, but different from kinora
    assert mapping["dunton equipment"] == mapping["dunton"]
    assert mapping["kinora industrial supplies"] != mapping["dunton"]


def test_resolve_entities_integration():
    df = pd.DataFrame(
        {
            "supplier_name_raw": ["DUNVALE MINING", "DUNVALE MINING (PTY) LTD", "PELMERE CC"],
            "vendor_id_raw": ["V1", "V2", "V3"],
            "item_description_raw": ["BOLT M10X50 STL", "BOLT M10X50 STL", "O-RING 20MM"],
            "material_number_raw": ["M100", "M100", ""],
        }
    )
    config = CompanyConfig(
        company_id="test",
        reporting_currency="USD",
        taxonomy_file="dummy",
        sources=[],
    )
    res = resolve_entities(df, config)

    # Suppliers for Dunvale should cluster together
    assert res.loc[0, "supplier_key"] == res.loc[1, "supplier_key"]
    assert res.loc[0, "supplier_key"] != res.loc[2, "supplier_key"]

    # Items with same material_number should match
    assert res.loc[0, "item_key"] == res.loc[1, "item_key"]
    assert res.loc[0, "item_key"] == "item_m100"
    assert res.loc[2, "item_key"].startswith("item_")
