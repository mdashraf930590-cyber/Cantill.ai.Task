# Production Spend Analytics & Zero-Shot Classification Pipeline

A production-grade, single-command, YAML-config-driven Python pipeline designed to process 25,000+ messy procurement rows in under 60 seconds without hardcoded taxonomy names or manual intervention.

---

## Key Features

1. **Schema-Agnostic Ingestion**: Maps arbitrary column aliases, parses heterogeneous date formats with explicit unparseable-date flagging, normalizes gross-to-net spend using line-specific or default VAT rates, and converts multi-currency values to reporting currency via line FX or monthly fallback lookup tables.
2. **Deterministic & Heuristic Gating**: Enforces strict precedence order (`CANCELLED` $\rightarrow$ `NOT_APPROVED` $\rightarrow$ `NON_PROCUREMENT` $\rightarrow$ `DUPLICATE`) before executing intra-system deduplication and two-pass cross-system record linkage (exact PO matching + composite blocking with date window & amount tolerance).
3. **Entity Resolution**: Generates canonical `supplier_key` and `item_key` identifiers through text normalization (Unicode NFKD folding, legal suffix stripping, boilerplate removal), prefix blocking, TF-IDF character n-grams, RapidFuzz token matching, and connected-components graph clustering.
4. **Multilingual Zero-Shot & Few-Shot Classification**: Dynamically ingests category trees (levels 1–4) at runtime without hardcoding, synthesizes rich path-aware descriptions, and classifies lines using a fast multilingual bi-encoder (`paraphrase-multilingual-MiniLM-L12-v2`). Enforces strict parent-child tree consistency and outputs calibrated 0–1 confidence scores from softmax margins.
5. **Strict Schema Exporter**: Preserves 100% of input rows and validates schema adherence (`record_id`, `include`, `exclusion_reason`, `supplier_key`, `item_key`, `spend`, `l1`–`l4`, `confidence`).
6. **Offline Evaluation Harness**: Hides labels during inference and computes Macro/Weighted F1, spend-weighted accuracy, entity clustering purity (V-Measure, Adjusted Rand Index), and gating classification metrics.

---

## Quick Start (Single Command)

### 1. Installation
Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```
*(Dependencies: `pandas`, `numpy`, `scipy`, `scikit-learn`, `rapidfuzz`, `sentence-transformers`, `torch`, `pyyaml`, `pytest`)*

### 2. Run All Companies (Single Command)
Run the complete pipeline end-to-end on both Company A and Company B, write output submission files, and display evaluation metrics:
```bash
python3 run_pipeline.py --all
```

Outputs are automatically validated and written to:
- `submission/__A.csv`
- `submission/__B.csv`

### 3. Run a Single Company
```bash
# Company A
python3 run_pipeline.py --config configs/company_A.yaml --output submission/__A.csv --evaluate

# Company B
python3 run_pipeline.py --config configs/company_B.yaml --output submission/__B.csv --evaluate
```

---

## Running Blind-Test Company C

To process an unseen Company C with an arbitrary tree and layout at the blind test:
1. Copy the configuration template:
   ```bash
   cp configs/company_template.yaml configs/company_C.yaml
   ```
2. Update `configs/company_C.yaml` with Company C's file path, column aliases, reporting currency, and taxonomy path.
3. Run the pipeline:
   ```bash
   python3 run_pipeline.py --config configs/company_C.yaml --output submission/ashraf__C.csv
   ```
*No code modifications required!*

---

## Repository Structure

```
.
├── configs/
│   ├── company_A.yaml          # Pipeline config for Company A
│   ├── company_B.yaml          # Pipeline config for Company B
│   └── company_template.yaml   # Template for unseen Company C
├── src/
│   └── spend_pipeline/
│       ├── __init__.py
│       ├── config.py           # YAML config parser & data contracts
│       ├── ingestion.py        # Stage 1: Schema ingestion, VAT & FX conversion
│       ├── gating.py           # Stage 2: Gating precedence & multi-pass deduplication
│       ├── entity_resolution.py# Stage 3: Supplier/item clustering & key generation
│       ├── classifier.py       # Stage 4: Zero-shot/few-shot multilingual classification
│       ├── exporter.py         # Stage 5: Strict schema validator & CSV exporter
│       ├── evaluation.py       # Stage 6: Offline benchmarking harness
│       ├── pipeline.py         # End-to-end orchestrator
│       └── cli.py              # CLI interface
├── tests/
│   ├── test_ingestion.py       # Unit tests for amounts, VAT, dates, FX lookup
│   ├── test_gating.py          # Unit tests for gating precedence & deduplication
│   ├── test_entity_resolution.py# Unit tests for clustering & key resolution
│   ├── test_classifier.py      # Unit tests for taxonomy tree consistency & confidence
│   ├── test_exporter.py        # Unit tests for schema validation & export constraints
│   └── test_e2e.py             # Mini end-to-end integration test
├── submission/
│   ├── ashraf__A.csv           # Formatted output for Company A (65,549 rows)
│   ├── ashraf__B.csv           # Formatted output for Company B (41,028 rows)
│   ├── __A.csv                 # Alias for grading compatibility
│   └── __B.csv                 # Alias for grading compatibility
├── APPROACH.md                 # One-page executive summary & architecture overview
├── DECISIONS.md                # Detailed engineering decisions and data findings log
├── run_pipeline.py             # Root execution entrypoint
└── requirements.txt            # Dependency manifest
```

---

## Running Unit Tests

Execute the comprehensive test suite with `pytest`:
```bash
pytest -v
```
All 17 tests validate ingestion parsing, tax deduction, FX fallbacks, gating precedence, deduplication linkage, entity clustering, taxonomy consistency, and schema preservation.

---

## Benchmark Performance Summary

| Company | Dataset Size | Total Execution Time | Throughput | L1 Accuracy | L2 Accuracy | L3 Accuracy | Spend-Weighted Accuracy (L1) | Gating F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Company A** | 65,549 rows | **27.41 s** | **2,391 rows/sec** | **45.72%** | **20.87%** | **12.55%** | **44.71%** | 100% |
| **Company B** | 41,028 rows | **28.14 s** | **1,458 rows/sec** | **98.56%** | **98.52%** | **97.39%** | **99.82%** | **98.89%** |
| **Combined** | **106,577 rows** | **55.55 s** | **1,918 rows/sec** | — | — | — | — | — |

*The pipeline comfortably exceeds the 60-minute threshold for 25,000 rows by completing over 106,000 rows in under 1 minute.*

