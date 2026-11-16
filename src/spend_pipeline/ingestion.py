"""
Stage 1: Schema-agnostic ingestion layer.
Normalizes heterogeneous column schemas, parses messy amounts, deducts tax,
converts currencies using line FX or monthly fallback tables, and parses dates.
"""

from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .config import CompanyConfig, SourceConfig


def parse_clean_amount(val) -> float:
    """Robustly parse numeric spend amounts across messy procurement representations."""
    if pd.isna(val):
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)

    s = str(val).strip()
    if not s:
        return 0.0

    is_negative = False
    if s.startswith("(") and s.endswith(")"):
        is_negative = True
        s = s[1:-1].strip()

    # Strip symbols except digits, periods, commas, and plus/minus
    s = re.sub(r"[^\d.,\-+]", "", s)
    if not s:
        return 0.0

    if s.startswith("-"):
        is_negative = True
        s = s[1:]
    elif s.startswith("+"):
        s = s[1:]

    # Resolve European vs American number formats
    if "," in s and "." in s:
        if s.rfind(".") > s.rfind(","):
            # 1,234.56
            s = s.replace(",", "")
        else:
            # 1.234,56
            s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        parts = s.split(",")
        if len(parts) == 2 and len(parts[1]) in (1, 2):
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")

    try:
        num = float(s)
        return -num if is_negative else num
    except ValueError:
        return 0.0


def parse_vat_rate(val, default_rate: float = 0.15) -> float:
    """Parse VAT/tax rates handling percentages ('15%'), decimals (0.15), integers ('15'), and blanks."""
    if pd.isna(val) or val is None:
        return float(default_rate)

    s = str(val).strip().rstrip("%")
    if not s:
        return float(default_rate)

    try:
        v = float(s)
        if v > 1.0:
            return v / 100.0
        return v
    except ValueError:
        return float(default_rate)


def parse_date_with_flag(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Parse dates with multiple format fallback, returning parsed datetime and boolean unparseable flag."""
    parsed = pd.to_datetime(series, format="mixed", errors="coerce")
    is_unparseable = series.notna() & parsed.isna()
    return parsed, is_unparseable


class FXLookupManager:
    """Manages monthly FX rates lookup table with nearest-month fallback."""

    def __init__(self, fx_file_path: Optional[str] = None):
        self.fx_table = {}
        self.currency_means = {}
        if fx_file_path and Path(fx_file_path).is_file():
            df = pd.read_csv(fx_file_path)
            # Expects columns: currency, month, units_per_usd
            curr_col = [c for c in df.columns if "curr" in c.lower()][0]
            month_col = [c for c in df.columns if "month" in c.lower() or "date" in c.lower()][0]
            rate_col = [c for c in df.columns if "rate" in c.lower() or "units" in c.lower()][0]

            for _, r in df.iterrows():
                c = str(r[curr_col]).strip().upper()
                m = str(r[month_col]).strip()
                v = float(r[rate_col])
                self.fx_table[(c, m)] = v

            means = df.groupby(curr_col)[rate_col].mean()
            self.currency_means = {str(k).strip().upper(): float(v) for k, v in means.items()}

    def get_rate(self, currency: str, date_val: Optional[pd.Timestamp]) -> float:
        """Retrieve FX rate (units per target currency) with monthly table lookup and nearest month fallback."""
        curr = str(currency).strip().upper() if pd.notna(currency) else "USD"
        if curr in ("USD", "ZAR", ""):
            # When line currency is already USD or matching
            pass

        month_str = ""
        if pd.notna(date_val):
            try:
                month_str = date_val.strftime("%Y-%m")
            except Exception:
                month_str = ""

        # 1. Exact match on (currency, YYYY-MM)
        if (curr, month_str) in self.fx_table:
            return self.fx_table[(curr, month_str)]

        # 2. Match nearest month in table for this currency
        available_months = [m for (c, m) in self.fx_table.keys() if c == curr]
        if available_months:
            if month_str:
                nearest = min(available_months, key=lambda m: abs(pd.to_datetime(m) - pd.to_datetime(month_str)))
                return self.fx_table[(curr, nearest)]
            return self.fx_table[(curr, available_months[0])]

        # 3. Currency mean rate
        if curr in self.currency_means:
            return self.currency_means[curr]

        # 4. Default 1.0
        return 1.0


def ingest_company_data(config: CompanyConfig) -> pd.DataFrame:
    """
    Ingest all configured sources for a company, normalize schemas, parse amounts,
    deduct VAT to net, convert currencies to target reporting currency, and return a unified DataFrame.
    """
    fx_mgr = FXLookupManager(config.fx_rates_file)
    unified_frames = []

    for source in config.sources:
        df_raw = pd.read_csv(source.file_path, low_memory=False)
        mapping = source.column_mapping

        # 1. Canonical Record ID
        id_col = source.id_column
        raw_ids = df_raw[id_col].astype(str)
        if source.record_id_prefix:
            record_ids = source.record_id_prefix + raw_ids
        else:
            record_ids = raw_ids

        # 2. Amounts and Tax Deductions
        amt_col = mapping.get("amount")
        if amt_col and amt_col in df_raw.columns:
            raw_amounts = df_raw[amt_col].apply(parse_clean_amount)
        else:
            raw_amounts = pd.Series(0.0, index=df_raw.index)

        # VAT / Gross to Net
        if source.amount_type == "gross":
            vat_col = mapping.get("vat_rate")
            if vat_col and vat_col in df_raw.columns:
                vat_rates = df_raw[vat_col].apply(lambda v: parse_vat_rate(v, source.default_vat_rate))
            else:
                vat_rates = pd.Series(source.default_vat_rate, index=df_raw.index)
            net_amounts = raw_amounts / (1.0 + vat_rates)
        else:
            net_amounts = raw_amounts
            vat_rates = pd.Series(0.0, index=df_raw.index)

        # 3. Date Parsing & Unparseable Flagging
        date_col = mapping.get("date")
        if date_col and date_col in df_raw.columns:
            parsed_dates, is_unparseable = parse_date_with_flag(df_raw[date_col])
        else:
            parsed_dates = pd.Series(pd.NaT, index=df_raw.index)
            is_unparseable = pd.Series(False, index=df_raw.index)

        # 4. Currency Conversion to Reporting Currency
        curr_col = mapping.get("currency")
        line_curr = df_raw[curr_col].astype(str).str.strip().str.upper() if curr_col and curr_col in df_raw.columns else pd.Series(config.reporting_currency, index=df_raw.index)

        rate_col = mapping.get("currency_rate")
        has_rate_col = rate_col and rate_col in df_raw.columns
        line_rates = pd.to_numeric(df_raw[rate_col], errors="coerce") if has_rate_col else pd.Series(np.nan, index=df_raw.index)

        spend_reporting = []
        for i in range(len(df_raw)):
            amt = net_amounts.iloc[i]
            curr = line_curr.iloc[i]
            dt = parsed_dates.iloc[i]
            rate = line_rates.iloc[i]

            if curr == config.reporting_currency:
                spend_val = amt
            elif pd.notna(rate) and rate > 0:
                spend_val = amt / rate
            else:
                # Lookup table fallback
                lookup_rate = fx_mgr.get_rate(curr, dt)
                spend_val = amt / lookup_rate if lookup_rate > 0 else amt

            spend_reporting.append(spend_val)

        # 5. Extract Canonical Text Fields
        supp_col = mapping.get("supplier_name")
        supp_series = df_raw[supp_col].astype(str) if supp_col and supp_col in df_raw.columns else pd.Series("", index=df_raw.index)

        vend_col = mapping.get("vendor_id")
        vend_series = df_raw[vend_col].astype(str) if vend_col and vend_col in df_raw.columns else pd.Series("", index=df_raw.index)

        # Item description (can be single column or list of columns)
        desc_spec = mapping.get("item_description")
        if isinstance(desc_spec, list):
            desc_parts = [df_raw[c].fillna("").astype(str) for c in desc_spec if c in df_raw.columns]
            item_desc_series = desc_parts[0] if desc_parts else pd.Series("", index=df_raw.index)
            for p in desc_parts[1:]:
                item_desc_series = item_desc_series.str.cat(p, sep=" ").str.strip()
        elif desc_spec and desc_spec in df_raw.columns:
            item_desc_series = df_raw[desc_spec].fillna("").astype(str)
        else:
            item_desc_series = pd.Series("", index=df_raw.index)

        mat_col = mapping.get("material_number")
        mat_series = df_raw[mat_col].fillna("").astype(str) if mat_col and mat_col in df_raw.columns else pd.Series("", index=df_raw.index)

        po_col = mapping.get("po_number") or mapping.get("po_reference")
        po_series = df_raw[po_col].fillna("").astype(str) if po_col and po_col in df_raw.columns else pd.Series("", index=df_raw.index)

        inv_col = mapping.get("invoice_number")
        inv_series = df_raw[inv_col].fillna("").astype(str) if inv_col and inv_col in df_raw.columns else pd.Series("", index=df_raw.index)

        status_col = mapping.get("status")
        status_series = df_raw[status_col].fillna("").astype(str) if status_col and status_col in df_raw.columns else pd.Series("", index=df_raw.index)

        gl_col = mapping.get("gl_account")
        gl_series = df_raw[gl_col].fillna("").astype(str) if gl_col and gl_col in df_raw.columns else pd.Series("", index=df_raw.index)

        gl_desc_col = mapping.get("gl_account_description")
        gl_desc_series = df_raw[gl_desc_col].fillna("").astype(str) if gl_desc_col and gl_desc_col in df_raw.columns else pd.Series("", index=df_raw.index)

        doc_type_col = mapping.get("document_type")
        doc_type_series = df_raw[doc_type_col].fillna("").astype(str) if doc_type_col and doc_type_col in df_raw.columns else pd.Series("", index=df_raw.index)

        # Assemble unified DataFrame
        source_df = pd.DataFrame(
            {
                "record_id": record_ids,
                "raw_id": raw_ids,
                "source_name": source.name,
                "is_primary": source.is_primary,
                "supplier_name_raw": supp_series,
                "vendor_id_raw": vend_series,
                "item_description_raw": item_desc_series,
                "material_number_raw": mat_series,
                "po_raw": po_series,
                "invoice_raw": inv_series,
                "date_parsed": parsed_dates,
                "is_date_unparseable": is_unparseable,
                "status_raw": status_series,
                "gl_account_raw": gl_series,
                "gl_account_desc_raw": gl_desc_series,
                "document_type_raw": doc_type_series,
                "gross_amount": raw_amounts,
                "net_amount": net_amounts,
                "vat_rate": vat_rates,
                "currency_raw": line_curr,
                "spend_reporting": pd.Series(spend_reporting, dtype=float),
            }
        )

        unified_frames.append(source_df)

    combined_df = pd.concat(unified_frames, ignore_index=True)
    return combined_df
