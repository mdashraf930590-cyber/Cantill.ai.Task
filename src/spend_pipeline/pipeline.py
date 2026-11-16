"""
End-to-end Pipeline Orchestrator.
Glues together the 6 stages:
1. Ingestion
2. Gating & Deduplication
3. Entity Resolution
4. Hierarchical Classification
5. Export & Schema Validation
6. Offline Evaluation Harness
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import pandas as pd

from .classifier import HierarchicalSpendClassifier
from .config import CompanyConfig, load_config
from .entity_resolution import resolve_entities
from .evaluation import evaluate_entity_clustering, evaluate_gating_exclusions, evaluate_spend_classification
from .exporter import export_submission_csv
from .gating import apply_gating_and_deduplication
from .ingestion import ingest_company_data


def run_pipeline_for_company(
    config_path: str,
    output_path: Optional[str] = None,
    evaluate: bool = False,
    labels_path: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run end-to-end spend analytics & zero-shot classification pipeline for a single company config.
    """
    start_time = time.time()
    config = load_config(config_path, data_dir=data_dir)
    metrics: Dict[str, Any] = {"company_id": config.company_id}

    print(f"\n[{config.company_id}] Stage 1: Ingesting data & normalizing schemas...")
    t0 = time.time()
    df = ingest_company_data(config)
    t_ingest = time.time() - t0
    total_raw_rows = len(df)
    print(f"[{config.company_id}] Ingested {total_raw_rows:,} records across {len(config.sources)} sources in {t_ingest:.2f}s.")

    print(f"[{config.company_id}] Stage 2: Gating non-spend rows & executing multi-pass linkage...")
    t0 = time.time()
    df = apply_gating_and_deduplication(df, config)
    t_gate = time.time() - t0
    included_count = (df["include"] == 1).sum()
    excluded_count = (df["include"] == 0).sum()
    print(f"[{config.company_id}] Gating complete in {t_gate:.2f}s: {included_count:,} included, {excluded_count:,} excluded.")
    print(f"[{config.company_id}] Exclusion breakdown: {dict(df[df['include'] == 0]['exclusion_reason'].value_counts())}")

    print(f"[{config.company_id}] Stage 3: Resolving supplier and item canonical entities...")
    t0 = time.time()
    df = resolve_entities(df, config)
    t_er = time.time() - t0
    print(f"[{config.company_id}] Entity resolution complete in {t_er:.2f}s ({df['supplier_key'].nunique():,} suppliers, {df['item_key'].nunique():,} items).")

    print(f"[{config.company_id}] Stage 4: Hierarchical classification with multilingual bi-encoder...")
    t0 = time.time()
    classifier = HierarchicalSpendClassifier(config)
    df = classifier.classify_records(df)
    t_clf = time.time() - t0
    print(f"[{config.company_id}] Classification complete in {t_clf:.2f}s.")

    # Stage 5: Exporting & Strict Schema Validation
    if output_path:
        print(f"[{config.company_id}] Stage 5: Validating schema and exporting output to {output_path}...")
        t0 = time.time()
        export_file = export_submission_csv(df, output_path, expected_row_count=total_raw_rows)
        t_exp = time.time() - t0
        print(f"[{config.company_id}] Schema validated! Preserved all {total_raw_rows:,} rows in {export_file} in {t_exp:.2f}s.")

    total_time = time.time() - start_time
    metrics["runtime_seconds"] = round(total_time, 2)
    metrics["total_records"] = total_raw_rows
    metrics["included_records"] = int(included_count)
    metrics["excluded_records"] = int(excluded_count)
    metrics["total_spend"] = float(round(df[df["include"] == 1]["spend_reporting"].sum(), 2))

    # Stage 6: Offline Evaluation Harness (if labels provided)
    if evaluate:
        lbl_path = labels_path or config.seed_labels_file
        if lbl_path and Path(lbl_path).is_file():
            print(f"[{config.company_id}] Stage 6: Running offline evaluation harness against {lbl_path}...")
            labels_df = pd.read_csv(lbl_path, low_memory=False)
            lbl_id = "record_id" if "record_id" in labels_df.columns else "line_id"
            eval_clf = evaluate_spend_classification(df, labels_df, id_col_labels=lbl_id)
            eval_er = evaluate_entity_clustering(df, labels_df, id_col_labels=lbl_id)
            eval_gate = evaluate_gating_exclusions(df, labels_df, id_col_labels=lbl_id)
            metrics["classification_evaluation"] = eval_clf
            metrics["entity_clustering_evaluation"] = eval_er
            metrics["gating_evaluation"] = eval_gate
            print(f"[{config.company_id}] Evaluation metrics computed successfully.")

    print(f"[{config.company_id}] Pipeline completed in {total_time:.2f} seconds ({total_raw_rows / total_time:.1f} rows/sec).\n")
    return df, metrics
