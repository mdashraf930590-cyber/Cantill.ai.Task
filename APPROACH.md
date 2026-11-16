# Spend Analytics & Classification Pipeline Architecture (APPROACH.md)

## Executive Summary
This architecture delivers an end-to-end, single-command, YAML-config-driven spend analytics and zero-shot hierarchical classification system. The pipeline processes 25,000+ procurement lines in under 30 seconds—over 100x faster than the 60-minute requirement—without hardcoded category names or manual intervention.

---

## 1. Pipeline Architecture (Six Modular Stages)
```
[Raw Ingestion & YAML Config]
       │
       ▼
[Stage 1: Schema Ingestion] ──► Column aliasing, Gross-to-Net VAT deduction, Multi-currency FX table fallback
       │
       ▼
[Stage 2: Gating Engine]    ──► Strict Precedence: CANCELLED -> NOT_APPROVED -> NON_PROCUREMENT -> DUPLICATE
       │
       ▼
[Stage 3: Entity Resolution]──► Unicode/legal normalization, char n-gram TF-IDF + RapidFuzz connected components
       │
       ▼
[Stage 4: Hierarchical Clf] ──► Multilingual Bi-Encoder (MiniLM-L12-v2), path synthesis, hybrid dense+sparse scoring
       │
       ▼
[Stage 5: Schema Exporter]  ──► 100% row preservation, strict schema validation, standard CSV output
       │
       ▼
[Stage 6: Offline Evaluator]──► Spend-weighted accuracy, Macro F1, clustering purity benchmarks
```

---

## 2. Matching and De-Duplication Method
Cross-system deduplication between disparate procurement systems (e.g., SAP ERP goods receipts and AP invoice tool exports) employs a deterministic two-pass linkage strategy:
1. **Intra-System Deduplication**: Filters secondary exports sharing identical `(invoice_number, supplier, spend)` tuples.
2. **Pass 1 (Normalized PO Key Linkage)**: Extracts core 10-digit PO numbers via regex from noisy references (e.g., `"PO 4510109330"` $\rightarrow$ `4510109330`). Matches secondary invoice records against primary ERP records with net-amount tolerance $\le 1.0$ currency unit or $\le 2\%$. Identified 16,743 duplicates in Company A.
3. **Pass 2 (Composite Blocking on Residuals)**: Blocks unlinked lines on `(normalized_supplier, rounded_net_amount)` with constraints: $|t_{\text{invoice}} - t_{\text{posting}}| \le 60\text{ days}$ and $|net_{\text{tool}} - net_{\text{erp}}| \le 1.0\text{ ZAR}$. Identified 1,737 additional duplicates.
*Outcome*: Preserves the primary ERP record as spend and flags matched secondary tool records as `DUPLICATE` (`spend = 0.0`).

---

## 3. How We Classify into an Unseen Tree
To classify into an arbitrary, unseen category hierarchy (levels 1–4) without labeled examples or code changes:
1. **Dynamic Taxonomy Ingestion**: Recursively discovers all valid paths from root to leaf in `taxonomy.csv` at runtime.
2. **Path-Aware Description Synthesis**: Synthesizes rich contextual descriptions for each candidate leaf path by combining leaf names, ancestor hierarchies, and deduplicated keyword tokens:
   $$\text{Desc} = \text{"\{leaf\}. Belongs to \{ancestor\}, within \{domain\}. Keywords: \{tokens\}"}$$
3. **Domain Text Normalization**: Line descriptions are expanded using domain acronym mappings (`elec` $\rightarrow$ `electrical`, `dril` $\rightarrow$ `drill`, `stl` $\rightarrow$ `steel`, `ss` $\rightarrow$ `stainless steel`, `vlv` $\rightarrow$ `valve`) and enriched with supplier and GL account context.
4. **Hybrid Dense + Sparse Scoring**: Dense embeddings from `paraphrase-multilingual-MiniLM-L12-v2` capture multilingual semantics, while sparse character n-gram TF-IDF captures exact part codes and technical terms ($\alpha = 0.35$):
   $$S_{\text{final}} = \alpha \cdot S_{\text{dense}} + (1 - \alpha) \cdot S_{\text{sparse}}$$
5. **Parent-Child Tree Consistency**: Guaranteed by scoring directly over valid taxonomy paths.
6. **Confidence Calibration**: Computed via temperature-scaled softmax margin over candidate paths:
   $$\text{Confidence} = \text{clip}(0.7 \cdot p_{(1)} + 0.3 \cdot (p_{(1)} - p_{(2)}), 0.1, 0.98)$$

---

## 4. Benchmark Performance & Accuracy

| Company | Mode | Records | Runtime | Level 1 Acc | Level 2 Acc | Level 3 Acc | Spend-Weighted Acc (L1) | Gating F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Company A** | Zero-Shot (Labels Hidden) | 65,549 | **27.41s** | **45.72%** | **20.87%** | **12.55%** | **44.71%** | 100% |
| **Company B** | Few-Shot (Seed Prototypes) | 41,028 | **28.14s** | **98.56%** | **98.52%** | **97.39%** | **99.82%** | **98.89%** |

- **Entity Resolution Clustering Purity (Company A)**:
  - Supplier Clustering: **Homogeneity 0.9999**, **V-Measure 0.6872**.
  - Item Clustering: **Homogeneity 1.0000**, **Completeness 0.9373**, **V-Measure 0.9677**, **ARI 0.6373**.

---

## 5. Runtime and LLM Cost Analysis
- **Pipeline Throughput**:
  - Company A: 65,549 rows processed in 27.41 seconds (**2,391 rows/sec**).
  - Company B: 41,028 rows processed in 28.14 seconds (**1,458 rows/sec**).
  - Both companies (106,577 rows total): **55.55 seconds total execution time**.
- **LLM API Cost Analysis**:
  - *Default Local Inference*: **$0.00 / 1,000 lines**. Runs 100% locally and offline on CPU/MPS without external API dependencies.
  - *Optional Batched LLM Fallback (for ambiguous lines with margin < 0.05)*:
    Using Gemini 1.5 Flash or GPT-4o-mini in batched JSON mode (~80 tokens in, ~20 tokens out per procurement line):
    $$\text{Cost per 1,000 lines} \approx 1,000 \times \left( \frac{80}{10^6} \times \$0.075 + \frac{20}{10^6} \times \$0.30 \right) \approx \mathbf{\$0.012\text{ per 1,000 lines}}$$
    Even routing 100% of rows to an external LLM costs only ~$0.012 per 1,000 lines ($0.30 for 25,000 rows).

---

## 6. AI Tools & Unattended Execution Statement
- **AI Tools Used**:
  - Local Multilingual Bi-Encoder: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (pre-cached, 100% offline, zero-network).
  - Lexical Matching & String Similarity: `scikit-learn` TF-IDF character/word n-grams and `rapidfuzz` token-set ratio.
  - Graph Partitioning: `scipy.sparse.csgraph.connected_components`.
- **Unattended Execution**:
  - The pipeline runs 100% unattended on new input files via a single CLI command (`python3 run_pipeline.py --config <file> --output <dest>`).
  - No interactive prompts, no manual label input, and no runtime API key requirements.
  - Free & Local Fallback: Runs entirely offline on standard CPU without external keys or network calls. To plug in an external API key if desired, set `OPENAI_API_KEY` or `GEMINI_API_KEY` in environment variables.

