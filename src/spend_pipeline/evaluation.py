"""
Stage 6: Offline Evaluation Harness.
Benchmarks pipeline metrics against hidden ground-truth labels:
- Macro & Weighted F1 across hierarchy levels (l1-l4)
- Spend-Weighted Classification Accuracy
- Entity Clustering Purity (Adjusted Rand Index, V-Measure, Homogeneity, Completeness)
- Gating and Exclusion Precision, Recall, and F1
- Total Spend Reconciliation
"""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    classification_report,
    f1_score,
    homogeneity_completeness_v_measure,
    precision_recall_fscore_support,
)


def evaluate_spend_classification(
    pred_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    id_col_pred: str = "record_id",
    id_col_labels: str = "record_id",
) -> Dict[str, Any]:
    """
    Evaluate classification performance on ground truth labels.
    Calculates Macro F1, Weighted F1, Unweighted Accuracy, and Spend-Weighted Accuracy.
    """
    merged = pred_df.merge(labels_df, left_on=id_col_pred, right_on=id_col_labels, suffixes=("_pred", "_true"))
    results = {"total_evaluated_records": len(merged)}

    # Filter to included records for category evaluation
    inc_col = "include" if "include" in merged.columns else ("include_pred" if "include_pred" in merged.columns else "include_true")
    active = merged[merged[inc_col] == 1].copy()
    results["active_procurement_records"] = len(active)

    spend_col = "spend" if "spend" in active.columns else "spend_reporting"
    spends = active[spend_col].values
    total_spend = np.sum(spends)

    for i in range(1, 5):
        pred_col = f"l{i}" if f"l{i}" in active.columns else f"category_l{i}_pred"
        true_col = f"category_l{i}" if f"category_l{i}" in active.columns else f"category_l{i}_true"

        if pred_col in active.columns and true_col in active.columns:
            y_pred = active[pred_col].astype(str)
            y_true = active[true_col].astype(str)

            # Ignore empty true labels if tree has fewer than 4 levels
            valid_mask = y_true.str.strip().ne("") & y_true.str.lower().ne("nan")
            if valid_mask.sum() > 0:
                y_p = y_pred[valid_mask]
                y_t = y_true[valid_mask]
                s_v = spends[valid_mask]

                acc = accuracy_score(y_t, y_p)
                macro_f1 = f1_score(y_t, y_p, average="macro", zero_division=0)
                weighted_f1 = f1_score(y_t, y_p, average="weighted", zero_division=0)

                # Spend-weighted accuracy
                correct_mask = (y_p == y_t).values
                spend_weighted_acc = float(np.sum(s_v[correct_mask]) / (np.sum(s_v) + 1e-9))

                results[f"l{i}"] = {
                    "accuracy": float(acc),
                    "spend_weighted_accuracy": spend_weighted_acc,
                    "macro_f1": float(macro_f1),
                    "weighted_f1": float(weighted_f1),
                    "num_classes": int(y_t.nunique()),
                }

    return results


def evaluate_entity_clustering(
    pred_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    id_col_pred: str = "record_id",
    id_col_labels: str = "record_id",
) -> Dict[str, Any]:
    """
    Evaluate entity resolution clustering purity against golden keys.
    Computes Adjusted Rand Index (ARI), V-Measure, Homogeneity, and Completeness.
    """
    merged = pred_df.merge(labels_df, left_on=id_col_pred, right_on=id_col_labels, suffixes=("_pred", "_true"))
    results = {}

    # 1. Supplier Clustering
    if "supplier_key_true" in merged.columns or "supplier_key" in labels_df.columns:
        s_true_col = "supplier_key_true" if "supplier_key_true" in merged.columns else "supplier_key"
        s_pred_col = "supplier_key" if "supplier_key" in merged.columns else "supplier_key_pred"

        valid = merged[merged[s_true_col].notna() & merged[s_true_col].astype(str).str.strip().ne("")]
        if len(valid) > 0:
            y_true = valid[s_true_col].astype(str)
            y_pred = valid[s_pred_col].astype(str)

            ari = adjusted_rand_score(y_true, y_pred)
            h, c, v = homogeneity_completeness_v_measure(y_true, y_pred)
            results["supplier_clustering"] = {
                "adjusted_rand_index": float(ari),
                "v_measure": float(v),
                "homogeneity": float(h),
                "completeness": float(c),
                "golden_clusters": int(y_true.nunique()),
                "predicted_clusters": int(y_pred.nunique()),
            }

    # 2. Item Clustering
    if "item_key_true" in merged.columns or "item_key" in labels_df.columns:
        i_true_col = "item_key_true" if "item_key_true" in merged.columns else "item_key"
        i_pred_col = "item_key" if "item_key" in merged.columns else "item_key_pred"

        valid_i = merged[merged[i_true_col].notna() & merged[i_true_col].astype(str).str.strip().ne("") & merged[i_true_col].astype(str).str.lower().ne("nan")]
        if len(valid_i) > 0:
            y_true_i = valid_i[i_true_col].astype(str)
            y_pred_i = valid_i[i_pred_col].astype(str)

            ari_i = adjusted_rand_score(y_true_i, y_pred_i)
            h_i, c_i, v_i = homogeneity_completeness_v_measure(y_true_i, y_pred_i)
            results["item_clustering"] = {
                "adjusted_rand_index": float(ari_i),
                "v_measure": float(v_i),
                "homogeneity": float(h_i),
                "completeness": float(c_i),
                "golden_clusters": int(y_true_i.nunique()),
                "predicted_clusters": int(y_pred_i.nunique()),
            }

    return results


def evaluate_gating_exclusions(
    pred_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    id_col_pred: str = "record_id",
    id_col_labels: str = "record_id",
) -> Dict[str, Any]:
    """
    Evaluate gating and exclusion classification against seed or ground-truth labels.
    """
    merged = pred_df.merge(labels_df, left_on=id_col_pred, right_on=id_col_labels, suffixes=("_pred", "_true"))
    results = {}

    if "include_true" in merged.columns or "include" in labels_df.columns:
        inc_true_col = "include_true" if "include_true" in merged.columns else "include"
        inc_pred_col = "include" if "include" in merged.columns else "include_pred"

        y_true = merged[inc_true_col].astype(int)
        y_pred = merged[inc_pred_col].astype(int)

        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        results["include_binary"] = {
            "accuracy": float(acc),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
        }

    return results
