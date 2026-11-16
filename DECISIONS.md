# Engineering & Domain Decisions Log (DECISIONS.md)

This document details the anomalies, schema ambiguities, and business logic discovered across the datasets, along with the architectural decisions and mathematical formulations implemented in the pipeline.

---

## 1. Schema Ingestion, Amount Normalization & Tax Handling

### Data Findings
1. **Heterogeneous Gross vs Net Amounts**:
   - Company A's SAP ERP (`erp_goods_receipts.csv`) records `value_zar` as net spend.
   - Company A's AP invoice tool (`invoice_tool_export.csv`) captures `gross_amount` including VAT.
   - Company B's invoice lines (`invoice_lines.csv`) capture `line_amount` as net accounts payable.
2. **Messy VAT Representation**:
   - In Company A's invoice tool export, `vat_rate` appears across 5 distinct representations: string percentage `"15%"`, float decimal `0.15`, integer string `"15"`, zero `"0"`, and blank/`NaN`.
   - The brief specifies: *"If the rate is blank, use 15%"*.
3. **Dirty Number Formats**:
   - Amount columns contained thousands separators (e.g., `"5,266.63"`, `"27,764.58"`), currency symbols, whitespace, and accounting parentheses for negative values (e.g., `"(1,234.56)"`).

### Architectural Decisions
- **Unified Amount Parser (`parse_clean_amount`)**:
  - Handles American (`1,234.56`) vs European (`1.234,56`) number formats, accounting parentheses `(x) -> -x`, and strips non-numeric characters.
- **Robust Tax Deductor (`parse_vat_rate`)**:
  - Automatically parses `"15%"`, `"15"`, `0.15` to `0.15`, maps `"0"` to `0.0`, and defaults missing/blank values to `0.15`.
  - Normalizes gross amounts to net spend: $\text{net\_amount} = \frac{\text{gross\_amount}}{1.0 + \text{vat\_rate}}$.

---

## 2. Multi-Currency Conversion & FX Fallback Hierarchy

### Data Findings
1. **Company B Currency Fragmentation**:
   - Contains 25 distinct currencies across 8 regional ERPs.
   - Lines with `currency == "USD"` already represent USD spend.
   - For non-USD lines, `currency_rate` represents units of the line currency per 1 USD (e.g., MXN ~17.9, THB ~35.5).
   - 3,343 non-USD rows have null `currency_rate`.
2. **Monthly FX Table (`fx_rates_usd.csv`)**:
   - Contains monthly rates (`units_per_usd`).
   - 3,338 of the 3,343 missing rate rows match `(currency, YYYY-MM)` in `fx_rates_usd.csv`.
   - 5 rows represent historical CAD transactions in 2022-05, 2022-06, and 2022-07, which are prior to the table's CAD start date (2022-12).

### Architectural Decisions
- **Conversion Formula**:
  $$\text{spend\_usd} = \frac{\text{line\_amount}}{\text{rate}}$$
- **Three-Tier Fallback Hierarchy (`FXLookupManager`)**:
  1. *Tier 1 (Exact Match)*: Match `(currency, invoice_month)` in `fx_rates_usd.csv`.
  2. *Tier 2 (Nearest Month / Currency Mean)*: For unlisted months (e.g., CAD mid-2022), fall back to nearest available month or currency mean (CAD is constant at 1.35).
  3. *Tier 3 (Default)*: If currency is completely unlisted, fall back to 1.0 with a logged warning.

---

## 3. Date Parsing & Unparseable Date Flagging

### Data Findings
- Multiple date formats coexist: ISO 8601 (`YYYY-MM-DD`), alphanumeric mixed (`27-Mar-24`), slash-delimited (`04/06/2023`, `01/01/2024`).
- Potential for corrupted or missing timestamps in raw exports.

### Architectural Decisions
- **Two-Pass Date Parser (`parse_date_with_flag`)**:
  - Evaluates mixed format parsing via `pd.to_datetime(series, format="mixed", errors="coerce")`.
  - Emits an explicit boolean column `is_date_unparseable` (`True` when raw date string is present but invalid).
  - Preserves data auditability without dropping records.

---

## 4. Gating Precedence & Non-Procurement Filtration

### Data Findings
- Section 5 of the brief defines four exclusion codes: `CANCELLED`, `NOT_APPROVED`, `NON_PROCUREMENT`, `DUPLICATE`.
- In Company B's 3,000 seed labels, 450 rows are marked `include = 0` with exclusion reason `NON_PROCUREMENT`.
- Analysis showed that 100% of these 450 rows belong to two patterns:
  1. Intercompany transactions: `gl_account_description` matching `^IC\d+` (`IC06`, `IC05`, `IC67`), `"IC Payables"`, or `"I/C Payable"`.
  2. Tax charges: `line_description` or `gl_account_description` containing `"Sales Tax Expense - "`, `"State & Local Taxes - "`, `"GST Input Tax - "`.

### Architectural Decisions
- **Strict Precedence Enforcement**:
  1. **CANCELLED**: Status in `['cancelled', 'canceled', 'void', 'reversed']` $\rightarrow$ `include = 0, exclusion_reason = 'CANCELLED'`.
  2. **NOT_APPROVED**: Status is present and not in `['paid', 'approved', 'posted', 'cleared']` (e.g., `rejected`, `on hold`, `submitted`, `disputed`) $\rightarrow$ `include = 0, exclusion_reason = 'NOT_APPROVED'`.
  3. **NON_PROCUREMENT**:
     - GL account descriptions matching intercompany patterns: `(?i)(^ic\d+|\bi/c\b|\bintercompany\b|\bic payables?\b)`.
     - Text descriptions matching tax patterns: `(?i)\b(tax|taxes|sales tax|vat|gst|input tax)\b`.
     - Other non-procurement keywords: `(?i)\b(royalt(y|ies)|dividend|financing fee|interest expense|bank charges|payroll|salary|wages|bad debt|depreciation)\b`.
     - Reversals / Zero-sum offset checks: Exact matching invoice pairs with positive and negative amounts cancelling to 0.0.
  4. **DUPLICATE**: Secondary records identified through intra-system deduplication or cross-system record linkage.
- **Spend Zeroing**: All excluded records strictly have `spend = 0.0` and blank category columns `l1`–`l4`.

---

## 5. Intra-System Deduplication & Multi-Pass Cross-System Record Linkage

### Data Findings
- Company A operates two systems: SAP ERP (37,029 lines) and AP Invoice Tool (28,520 lines).
- In the invoice tool, `po_reference` is reformatted with prefixes (`"PO 4510109330"`, `"PO#4510059209"`), item line suffixes (`"4510085877/10.0"`), or missing entirely.
- Exact PO string matching identified only 4,464 matches.
- Extracting the 10-digit PO regex revealed 17,620 matching records.
- 265 intra-system duplicate invoice lines exist within the AP invoice tool export itself.

### Architectural Decisions
- **Intra-System Deduplication**:
  - Secondary invoice-tool exports with identical `(invoice_number, supplier, spend)` retain the first instance; subsequent instances are marked `DUPLICATE`.
- **Two-Pass Cross-System Record Linkage**:
  - **Pass 1 (Exact Normalized PO Matching)**:
    - Normalizes PO numbers via regex extraction of the 10-digit core identifier.
    - Matches secondary tool records against primary ERP records with net-amount tolerance $\le 1.0$ ZAR or $\le 2\%$.
    - Matched tool records are marked `DUPLICATE` (ERP primary retained).
    - Result: 16,743 cross-system duplicates identified.
  - **Pass 2 (Composite Blocking on Residuals)**:
    - For tool lines with missing or reformatted POs, block on `(normalized_supplier, rounded_net_amount)`.
    - Apply constraints: absolute date difference $|t_{\text{invoice}} - t_{\text{posting}}| \le 60\text{ days}$ and net amount difference $\le 1.0$ ZAR.
    - Matched tool records marked `DUPLICATE`.
    - Result: 1,737 additional duplicates identified.
  - **Total Cross-System Duplicates**: 18,480 tool records successfully linked and deduplicated.

---

## 6. Entity Resolution (Supplier & Item Canonical Keys)

### Data Findings
- In Company A, 193 raw supplier name strings map to 70 client golden keys.
- Legal suffixes (`(PTY) LTD`, `CC`, `SA`, `GmbH`) and punctuation/spacing differences create artificial duplicates (e.g., `"YARFIELD"` vs `"YARF IELD"`, `"KINORA INDUSTRIAL SUPPLIES ENGINEERING"` vs `"KINORA  INDUSTRIAL SUPPLIES ENGINEERING"`).
- In Company B, vendor IDs (`vendor_id`) are present for 100% of rows, but supplier names are omitted.

### Architectural Decisions
- **Supplier Canonical Key Generation**:
  - Text normalization: Unicode NFKD folding, lowercasing, stripping punctuation, legal entity suffixes, and generic boilerplate (`engineering`, `supplies`, `solutions`, `technologies`).
  - Prefix blocking (first 4 characters) followed by TF-IDF character n-gram (3-4) cosine similarity matrix.
  - Graph edge augmentation via RapidFuzz token set ratio ($\ge 90$) to handle word order permutations and substring containment.
  - Connected-components graph partitioning generates canonical cluster hashes (`supp_<hash>`).
  - Result: Achieved **0.9999 Homogeneity** and **0.687 V-Measure** on Company A golden supplier keys.
- **Item Canonical Key Generation**:
  - When `material_number` is present: Generates canonical key `item_<material_number_lower>`.
  - When absent: Normalizes item description and computes deterministic MD5 hash (`item_<hash>`).
  - Result: Achieved **1.000 Homogeneity**, **0.937 Completeness**, and **0.968 V-Measure** on Company A golden items.

---

## 7. Multilingual Zero-Shot & Few-Shot Hierarchical Classification

### Data Findings
- Zero hardcoding rule: Category trees are provided as runtime CSV files with arbitrary depths (up to 4 levels).
- Descriptions are abbreviated (`"ELEC FAN"`, `"DRIL ROD"`, `"WLD MESH"`, `"FLAME RET CBL"`).
- In Company B, 3,000 seed labels are available, whereas Company A and blind-test Company C require zero-shot classification from the tree alone.

### Architectural Decisions
- **Hierarchical Path Ingestion**:
  - Ingests `taxonomy.csv` dynamically, extracting paths from root to leaf. Parent-child consistency is mathematically guaranteed by construction because predictions are selected from valid taxonomy paths.
- **Path-Aware Description Synthesis**:
  - Generates rich natural language descriptions for each leaf path:
    `"{leaf}. Category path: {ancestors} -> {leaf}. Procurement keywords: {deduplicated_path_tokens}"`
- **Procurement Abbreviation Expansion**:
  - Expands domain acronyms (`elec` $\rightarrow$ `electric electrical`, `dril` $\rightarrow$ `drill drilling`, `stl` $\rightarrow$ `steel`, `ss` $\rightarrow$ `stainless steel`, `hyd` $\rightarrow$ `hydraulic`, `vlv` $\rightarrow$ `valve`).
- **Multilingual Bi-Encoder (`paraphrase-multilingual-MiniLM-L12-v2`)**:
  - Pre-cached locally to ensure 100% offline, reproducible execution without internet dependency.
  - Encodes line descriptions enriched with supplier and GL account context.
- **Dual-Mode Inference Engine**:
  - **Few-Shot Prototype Mode** (when seed labels exist, as in Company B):
    - Computes category prototype embeddings and 1-NN similarity.
    - Yields **98.56% L1 Accuracy**, **98.52% L2 Accuracy**, **97.39% L3 Accuracy**, and **99.82% Spend-Weighted Accuracy**.
  - **Zero-Shot Hybrid Mode** (when labels are hidden or unavailable, as in Company A & C):
    - Ensembles dense bi-encoder cosine similarity with sparse TF-IDF n-gram matching ($\alpha = 0.35$):
      $$S_{\text{hybrid}} = \alpha \cdot S_{\text{dense}} + (1 - \alpha) \cdot S_{\text{sparse}}$$
    - Delivers **45.72% L1 Accuracy** and **44.71% Spend-Weighted Accuracy** on Company A with labels 100% hidden.
- **Calibrated Softmax Margin Confidence**:
  - Calculates temperature-scaled softmax probabilities over candidate paths ($T = 0.15$).
  - Blends top-1 probability with margin over second candidate:
    $$\text{Confidence} = \text{clip}(0.7 \cdot p_{(1)} + 0.3 \cdot (p_{(1)} - p_{(2)}), 0.1, 0.98)$$

---

## 8. Preserving Every Input Row & Submission Integrity

### Data Findings
- Section 6 mandates preserving every input row:
  - Company A: 37,029 ERP lines + 28,520 Invoice Tool lines = **65,549 rows**.
  - Company B: **41,028 rows**.

### Architectural Decisions
- Stage 5 strictly validates schema, non-null `record_id`, valid exclusion reasons, zero spend on excluded lines, and exact row count preservation before writing to disk.
- Runtime performance:
  - Company A (65,549 rows): **27.41 seconds** (2,391 rows/sec).
  - Company B (41,028 rows): **28.14 seconds** (1,458 rows/sec).
  - Total end-to-end execution: **55.55 seconds**, exceeding the 60-minute threshold by over 65x.

