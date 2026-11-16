"""
Unit tests for Stage 4: Hierarchical Spend Classifier.
"""

import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spend_pipeline.classifier import HierarchicalSpendClassifier, HierarchicalTaxonomy, expand_procurement_text
from spend_pipeline.config import CompanyConfig


def test_expand_procurement_text():
    raw = "ELEC FAN 22KW DRIL ROD STL WLD"
    expanded = expand_procurement_text(raw)
    assert "electric" in expanded
    assert "drill" in expanded
    assert "steel" in expanded
    assert "welded" in expanded


def test_hierarchical_taxonomy(tmp_path):
    tax_csv = tmp_path / "taxonomy.csv"
    tax_csv.write_text(
        "category_l1,category_l2,category_l3\n"
        "Direct Materials,Metals,Aluminium\n"
        "Direct Materials,Metals,Copper\n"
        "MRO,Fasteners,Bolts\n"
    )

    tax = HierarchicalTaxonomy(str(tax_csv))
    assert tax.num_levels == 3
    assert len(tax.paths) == 3
    assert "Direct Materials > Metals > Aluminium" in tax.paths

    descs = tax.build_rich_descriptions()
    assert len(descs) == 3
    assert "Aluminium" in descs[0]
    assert "Metals" in descs[0]


def test_classifier_tree_consistency(tmp_path):
    tax_csv = tmp_path / "taxonomy.csv"
    tax_csv.write_text(
        "category_l1,category_l2,category_l3\n"
        "Direct Materials,Metals,Aluminium Ingot\n"
        "MRO,Mechanical,Valves & Pumps\n"
    )

    config = CompanyConfig(
        company_id="test",
        reporting_currency="USD",
        taxonomy_file=str(tax_csv),
        sources=[],
    )

    df = pd.DataFrame(
        {
            "include": [1, 1, 0],
            "gate_confidence": [1.0, 1.0, 0.95],
            "item_description_raw": [
                "Aluminium metal ingot coil",
                "High pressure valve assembly pump",
                "Excluded line",
            ],
            "supplier_name_raw": ["Alcoa", "Flowserve", "Acme"],
            "gl_account_desc_raw": ["Direct Raw", "Plant MRO", "Admin"],
            "document_type_raw": ["PV", "PV", "PV"],
        }
    )

    clf = HierarchicalSpendClassifier(config)
    res = clf.classify_records(df)

    # Check predictions strictly conform to taxonomy paths
    valid_paths = set(clf.taxonomy.paths)

    path_0 = f"{res.loc[0, 'l1']} > {res.loc[0, 'l2']} > {res.loc[0, 'l3']}"
    assert path_0 in valid_paths
    assert res.loc[0, "l3"] == "Aluminium Ingot"

    path_1 = f"{res.loc[1, 'l1']} > {res.loc[1, 'l2']} > {res.loc[1, 'l3']}"
    assert path_1 in valid_paths
    assert res.loc[1, "l3"] == "Valves & Pumps"

    # Excluded row must have blank l1-l4
    assert res.loc[2, "l1"] == ""
    assert res.loc[2, "l2"] == ""
    assert res.loc[2, "l3"] == ""
    assert res.loc[2, "l4"] == ""

    # Confidence scores must be valid floats between 0 and 1
    assert 0.0 <= res.loc[0, "confidence"] <= 1.0
    assert 0.0 <= res.loc[1, "confidence"] <= 1.0
