"""
End-to-end pipeline integration test.
"""

import sys
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spend_pipeline.pipeline import run_pipeline_for_company


def test_pipeline_mini_e2e(tmp_path):
    # Create mini source CSVs
    erp_file = tmp_path / "erp.csv"
    erp_file.write_text(
        "line_id,po_num,amt,curr,supp,mat_num,desc,post_date\n"
        "A1,4510000001,1000.0,ZAR,Acme Ltd,M01,Flanged Ball Valve,2024-01-10\n"
        "A2,4510000002,2500.0,ZAR,Apex Mining,M02,Electric Motor 15KW,2024-01-15\n"
    )

    tool_file = tmp_path / "tool.csv"
    tool_file.write_text(
        "rec_id,po_ref,gross,vat,curr,supp,text,inv_date,status\n"
        "T1,PO 4510000001,1150.0,15%,ZAR,Acme,Ball Valve,2024-01-12,Paid\n"
        "T2,PO 4510000099,575.0,15%,ZAR,Acme,Small gaskets,2024-01-20,Paid\n"
        "T3,PO 4510000100,100.0,15%,ZAR,Acme,Cancelled item,2024-01-22,Cancelled\n"
    )

    tax_file = tmp_path / "tax.csv"
    tax_file.write_text(
        "category_l1,category_l2,category_l3\n"
        "MRO,Valves,Ball Valves\n"
        "MRO,Motors,Electric Motors\n"
        "MRO,Hardware,Gaskets\n"
    )

    config_yaml = tmp_path / "config.yaml"
    config_yaml.write_text(f"""
company_id: test_mini
reporting_currency: ZAR
taxonomy_file: "{tax_file}"

sources:
  - name: erp
    file_path: "{erp_file}"
    id_column: line_id
    record_id_prefix: "ERP:"
    is_primary: true
    amount_type: net
    column_mapping:
      amount: amt
      date: post_date
      currency: curr
      po_number: po_num
      supplier_name: supp
      material_number: mat_num
      item_description: desc

  - name: tool
    file_path: "{tool_file}"
    id_column: rec_id
    record_id_prefix: "TOOL:"
    is_primary: false
    amount_type: gross
    default_vat_rate: 0.15
    column_mapping:
      amount: gross
      vat_rate: vat
      date: inv_date
      currency: curr
      po_reference: po_ref
      supplier_name: supp
      item_description: text
      status: status

gating:
  approved_statuses: ["paid"]
  cancelled_statuses: ["cancelled"]

deduplication:
  enabled: true
  cross_system_linkage: true
  po_match_tolerance_abs: 1.0

classifier:
  model_name: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
  batch_size: 16
  use_few_shot: false
""")

    out_csv = tmp_path / "out.csv"
    df, metrics = run_pipeline_for_company(str(config_yaml), output_path=str(out_csv))

    assert len(df) == 5  # 2 ERP + 3 Tool records preserved
    assert out_csv.is_file()

    out_df = pd.read_csv(out_csv)
    assert len(out_df) == 5

    # T1 must be DUPLICATE (gross 1150 / 1.15 = 1000 net, matches ERP:A1 on PO 4510000001)
    t1_row = out_df[out_df["record_id"] == "TOOL:T1"].iloc[0]
    assert t1_row["include"] == 0
    assert t1_row["exclusion_reason"] == "DUPLICATE"
    assert t1_row["spend"] == 0.0

    # T3 must be CANCELLED
    t3_row = out_df[out_df["record_id"] == "TOOL:T3"].iloc[0]
    assert t3_row["include"] == 0
    assert t3_row["exclusion_reason"] == "CANCELLED"

    # ERP:A1 must be included spend
    a1_row = out_df[out_df["record_id"] == "ERP:A1"].iloc[0]
    assert a1_row["include"] == 1
    assert a1_row["spend"] == 1000.0
    assert a1_row["l1"] == "MRO"
