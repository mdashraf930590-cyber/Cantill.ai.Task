"""
Stage 2: Deterministic and heuristic gating engine.
Executes non-spend exclusion checks in strict precedence order:
1. CANCELLED
2. NOT_APPROVED
3. NON_PROCUREMENT
4. DUPLICATE (Intra-system deduplication + Multi-pass cross-system record linkage)
"""

from datetime import timedelta
import re
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd

from .config import CompanyConfig


def extract_normalized_po(val: str) -> str:
    """Extract standard 10-digit PO or cleaned alphanumeric PO identifier."""
    if pd.isna(val) or not str(val).strip():
        return ""
    s = str(val).strip()
    # Check for 10-digit SAP / standard PO format
    m10 = re.search(r"(\d{10})", s)
    if m10:
        return m10.group(1)
    # Strip common PO prefixes
    s_clean = re.sub(r"^(po[-_#:\s]*)+", "", s, flags=re.IGNORECASE).strip().upper()
    s_clean = re.sub(r"[^\w]", "", s_clean)
    return s_clean


def clean_supplier_for_matching(val: str) -> str:
    """Normalize supplier name for record linkage blocking."""
    if pd.isna(val) or not str(val).strip():
        return ""
    s = str(val).lower()
    # Strip legal entity suffixes and corporate abbreviations
    legal = r"\b(pty\s+ltd|proprietary\s+limited|pty|ltd|limited|cc|corp|corporation|inc|incorporated|gmbh|ag|sa|bv|co|company|holdings|services|supplies|equipment|solutions|engineering|technologies|technology|group|enterprises)\b"
    s = re.sub(legal, " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def apply_gating_and_deduplication(df: pd.DataFrame, config: CompanyConfig) -> pd.DataFrame:
    """
    Apply gating exclusions in strict precedence order:
    CANCELLED -> NOT_APPROVED -> NON_PROCUREMENT -> DUPLICATE.
    """
    df = df.copy()

    # Default include=1, exclusion_reason=""
    df["include"] = 1
    df["exclusion_reason"] = ""
    df["gate_confidence"] = 1.0

    g_cfg = config.gating
    d_cfg = config.deduplication

    # ---------------------------------------------------------
    # Precedence Step 1: CANCELLED
    # ---------------------------------------------------------
    status_lower = df["status_raw"].astype(str).str.strip().str.lower()
    is_cancelled = status_lower.isin(g_cfg.cancelled_statuses)
    mask_c = is_cancelled
    df.loc[mask_c, "include"] = 0
    df.loc[mask_c, "exclusion_reason"] = "CANCELLED"
    df.loc[mask_c, "gate_confidence"] = 0.99

    # ---------------------------------------------------------
    # Precedence Step 2: NOT_APPROVED
    # (Evaluated only on remaining active rows)
    # ---------------------------------------------------------
    active_mask = df["include"] == 1
    has_status = status_lower.str.len() > 0
    is_not_approved = has_status & (~status_lower.isin(g_cfg.approved_statuses)) & (~is_cancelled)

    mask_na = active_mask & is_not_approved
    df.loc[mask_na, "include"] = 0
    df.loc[mask_na, "exclusion_reason"] = "NOT_APPROVED"
    df.loc[mask_na, "gate_confidence"] = 0.98

    # ---------------------------------------------------------
    # Precedence Step 3: NON_PROCUREMENT
    # (GL account patterns, tax/intercompany/royalty/financing keywords, reversals)
    # ---------------------------------------------------------
    active_mask = df["include"] == 1

    gl_desc_lower = df["gl_account_desc_raw"].astype(str).str.strip().str.lower()
    line_desc_lower = df["item_description_raw"].astype(str).str.strip().str.lower()
    gl_code = df["gl_account_raw"].astype(str).str.strip().str.upper()

    # GL prefixes (e.g. IC*)
    is_ic_gl = pd.Series(False, index=df.index)
    for pfx in g_cfg.non_procurement_gl_prefixes:
        is_ic_gl = is_ic_gl | gl_desc_lower.str.startswith(pfx.lower())

    # GL keywords (e.g. IC Payables, Intercompany, Tax)
    is_nonproc_gl_kw = pd.Series(False, index=df.index)
    for kw in g_cfg.non_procurement_gl_keywords:
        pattern = rf"\b{re.escape(kw)}\b"
        is_nonproc_gl_kw = is_nonproc_gl_kw | gl_desc_lower.str.contains(pattern, regex=True)

    # Line description keywords (e.g. Sales Tax, GST Input Tax, Royalty, Dividend)
    is_nonproc_desc = pd.Series(False, index=df.index)
    for kw in g_cfg.non_procurement_desc_keywords:
        pattern = rf"\b{re.escape(kw)}\b"
        is_nonproc_desc = is_nonproc_desc | line_desc_lower.str.contains(pattern, regex=True)

    # Reversals / Offsetting zero pairs check
    is_offset_pair = pd.Series(False, index=df.index)
    if g_cfg.filter_zero_negative_offsets:
        # Check matching invoice_number + abs(spend) with net sum 0
        has_inv = df["invoice_raw"].str.len() > 0
        inv_amt = df[has_inv & active_mask].copy()
        if len(inv_amt) > 0:
            inv_amt["abs_amt"] = inv_amt["spend_reporting"].abs().round(2)
            grouped = inv_amt.groupby(["invoice_raw", "abs_amt"])["spend_reporting"].agg(["count", "sum"])
            # Pairs where count > 1 and sum is approximately 0
            cancelled_pairs = grouped[(grouped["count"] >= 2) & (grouped["sum"].abs() < 0.01)].index
            if len(cancelled_pairs) > 0:
                pair_keys = set(cancelled_pairs)
                is_offset_pair = df.apply(
                    lambda r: (r["invoice_raw"], round(abs(r["spend_reporting"]), 2)) in pair_keys if r["invoice_raw"] else False,
                    axis=1,
                )

    is_non_procurement = is_ic_gl | is_nonproc_gl_kw | is_nonproc_desc | is_offset_pair
    mask_np = active_mask & is_non_procurement
    df.loc[mask_np, "include"] = 0
    df.loc[mask_np, "exclusion_reason"] = "NON_PROCUREMENT"
    df.loc[mask_np, "gate_confidence"] = 0.95

    # ---------------------------------------------------------
    # Precedence Step 4: DUPLICATE
    # Intra-system deduplication & Multi-pass cross-system linkage
    # ---------------------------------------------------------
    if d_cfg.enabled:
        active_mask = df["include"] == 1

        # A. Intra-system deduplication
        if d_cfg.intra_system_dedup:
            # Check secondary tool records with identical invoice_number + supplier + amount
            tool_mask = active_mask & (~df["is_primary"])
            if tool_mask.sum() > 0:
                tool_sub = df[tool_mask].copy()
                dup_mask = tool_sub.duplicated(
                    subset=["invoice_raw", "supplier_name_raw", "spend_reporting"],
                    keep="first",
                )
                tool_dup_ids = set(tool_sub[dup_mask]["record_id"])
                mask_intra_dup = df["record_id"].isin(tool_dup_ids)
                df.loc[mask_intra_dup, "include"] = 0
                df.loc[mask_intra_dup, "exclusion_reason"] = "DUPLICATE"
                df.loc[mask_intra_dup, "gate_confidence"] = 0.95

        # B. Multi-pass Cross-System Record Linkage
        if d_cfg.cross_system_linkage:
            active_mask = df["include"] == 1
            primary_mask = active_mask & (df["is_primary"])
            secondary_mask = active_mask & (~df["is_primary"])

            if primary_mask.sum() > 0 and secondary_mask.sum() > 0:
                erp_df = df[primary_mask].copy()
                tool_df = df[secondary_mask].copy()

                erp_df["po_norm"] = erp_df["po_raw"].apply(extract_normalized_po)
                tool_df["po_norm"] = tool_df["po_raw"].apply(extract_normalized_po)

                erp_df["supp_match"] = erp_df["supplier_name_raw"].apply(clean_supplier_for_matching)
                tool_df["supp_match"] = tool_df["supplier_name_raw"].apply(clean_supplier_for_matching)

                erp_df["amt_round"] = erp_df["spend_reporting"].round(0)
                tool_df["amt_round"] = tool_df["spend_reporting"].round(0)

                matched_tool_ids = set()

                # Pass 1: Exact PO match with net amount tolerance
                tool_has_po = tool_df[tool_df["po_norm"] != ""]
                erp_has_po = erp_df[erp_df["po_norm"] != ""]

                if len(tool_has_po) > 0 and len(erp_has_po) > 0:
                    m1 = tool_has_po.merge(
                        erp_has_po[["po_norm", "spend_reporting", "record_id"]],
                        on="po_norm",
                        suffixes=("_tool", "_erp"),
                    )
                    amt_diff = (m1["spend_reporting_tool"] - m1["spend_reporting_erp"]).abs()
                    pct_diff = amt_diff / (m1["spend_reporting_erp"].abs() + 1e-5)

                    pass1_valid = m1[
                        (amt_diff <= d_cfg.po_match_tolerance_abs) | (pct_diff <= d_cfg.po_match_tolerance_pct)
                    ]
                    matched_tool_ids.update(pass1_valid["record_id_tool"].unique())

                # Pass 2: Composite blocking (supplier + rounded amount + date window)
                unmatched_tool = tool_df[~tool_df["record_id"].isin(matched_tool_ids)]
                if len(unmatched_tool) > 0:
                    m2 = unmatched_tool.merge(
                        erp_df[["supp_match", "amt_round", "spend_reporting", "date_parsed", "record_id"]],
                        on=["supp_match", "amt_round"],
                        suffixes=("_tool", "_erp"),
                    )
                    # Exclude empty suppliers
                    m2 = m2[m2["supp_match"] != ""]
                    dt_diff = (m2["date_parsed_tool"] - m2["date_parsed_erp"]).dt.days.abs()
                    amt_diff_2 = (m2["spend_reporting_tool"] - m2["spend_reporting_erp"]).abs()

                    pass2_valid = m2[
                        (dt_diff <= d_cfg.composite_date_window_days)
                        & (amt_diff_2 <= d_cfg.composite_amount_tolerance_abs)
                    ]
                    matched_tool_ids.update(pass2_valid["record_id_tool"].unique())

                # Apply DUPLICATE to secondary matched records
                mask_cross_dup = df["record_id"].isin(matched_tool_ids)
                df.loc[mask_cross_dup, "include"] = 0
                df.loc[mask_cross_dup, "exclusion_reason"] = "DUPLICATE"
                df.loc[mask_cross_dup, "gate_confidence"] = 0.95

    # Zero spend for all excluded records
    df.loc[df["include"] == 0, "spend_reporting"] = 0.0

    return df
