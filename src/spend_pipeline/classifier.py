"""
Stage 4: Multilingual Zero-Shot & Few-Shot Hierarchical Classifier.
Dynamically ingests runtime category trees (l1-l4), expands procurement abbreviations,
constructs rich path-aware taxonomy representations, encodes lines with multilingual bi-encoder
(paraphrase-multilingual-MiniLM-L12-v2), supports few-shot prototype transfer when seed labels exist,
enforces parent-child tree consistency, and outputs calibrated 0-1 confidence scores.
"""

import os
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure offline execution for sentence-transformers and huggingface hub
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer
from .config import CompanyConfig

PROCUREMENT_ABBREVIATIONS = {
    r"\belec\b": "electric electrical",
    r"\bdril\b": "drill drilling",
    r"\bwld\b": "welded welding",
    r"\bstl\b": "steel",
    r"\bss\b": "stainless steel",
    r"\bflw\b": "flow airflow",
    r"\bflng\b": "flange",
    r"\bwd\b": "width",
    r"\blg\b": "length",
    r"\bln\b": "length",
    r"\bdia\b": "diameter",
    r"\bhyd\b": "hydraulic hydraulics",
    r"\beng\b": "engine diesel",
    r"\bmtr\b": "motor electric motor",
    r"\bvlv\b": "valve",
    r"\bbrg\b": "bearing",
    r"\bgskt\b": "gasket seal",
    r"\bpmp\b": "pump",
    r"\bassy\b": "assembly",
    r"\bwod\b": "wood wooden timber",
    r"\bl/dty\b": "light duty",
    r"\bh/dty\b": "heavy duty",
    r"\bflame ret\b": "flame retardant flameproof",
    r"\bcbl\b": "cable electrical wire",
    r"\baugr\b": "auger drilling",
    r"\bmro\b": "maintenance repair operations",
    r"\bpp\b": "polypropylene",
    r"\balu\b": "aluminium metal",
}


def expand_procurement_text(text: str) -> str:
    """Expand standard procurement acronyms and clean tokens."""
    if pd.isna(text) or not str(text).strip():
        return ""
    s = str(text).lower()
    for pattern, rep in PROCUREMENT_ABBREVIATIONS.items():
        s = re.sub(pattern, rep, s)
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


class HierarchicalTaxonomy:
    """Manages dynamic taxonomy ingestion, valid path enumeration, and rich path descriptions."""

    def __init__(self, taxonomy_file_path: str):
        self.path = taxonomy_file_path
        self.df = pd.read_csv(taxonomy_file_path)
        # Determine category levels dynamically (up to l4)
        cols = self.df.columns.tolist()
        self.level_cols = []
        for i in range(1, 5):
            for c in cols:
                if f"l{i}" in c.lower() or f"level_{i}" in c.lower() or f"category_{i}" in c.lower():
                    self.level_cols.append(c)
                    break
        if not self.level_cols:
            self.level_cols = cols  # fallback

        self.num_levels = len(self.level_cols)
        self.paths = []
        self.path_tuples = []

        for _, row in self.df.iterrows():
            parts = [str(row[c]).strip() for c in self.level_cols if pd.notna(row[c]) and str(row[c]).strip() != ""]
            if parts:
                self.paths.append(" > ".join(parts))
                self.path_tuples.append(parts)

        # Unique paths
        seen = set()
        unique_paths = []
        unique_tuples = []
        for p, t in zip(self.paths, self.path_tuples):
            if p not in seen:
                seen.add(p)
                unique_paths.append(p)
                unique_tuples.append(t)
        self.paths = unique_paths
        self.path_tuples = unique_tuples

    def build_rich_descriptions(self) -> List[str]:
        """Construct rich path-aware taxonomy descriptions combining leaf and hierarchical ancestors."""
        rich_descs = []
        for parts in self.path_tuples:
            leaf = parts[-1]
            ancestors = parts[:-1]
            # Deduplicate repeating words
            words = []
            for part in parts:
                for w in re.split(r"[\s,/]+", part):
                    w_clean = w.lower()
                    if w_clean not in [x.lower() for x in words]:
                        words.append(w)
            dedup_keywords = " ".join(words)
            ancestor_str = ", ".join(ancestors) if ancestors else "Root"
            desc = f"{leaf}. Category path: {ancestor_str} -> {leaf}. Procurement keywords: {dedup_keywords}"
            rich_descs.append(desc)
        return rich_descs


class HierarchicalSpendClassifier:
    """Production-grade bi-encoder and hybrid few-shot spend classifier."""

    def __init__(self, config: CompanyConfig):
        self.config = config
        self.clf_cfg = config.classifier
        self.taxonomy = HierarchicalTaxonomy(config.taxonomy_file)
        self.model: Optional[SentenceTransformer] = None
        self.tax_embeddings: Optional[np.ndarray] = None
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tax_tfidf: Optional[np.ndarray] = None

        # Few-shot prototypes
        self.has_few_shot = False
        self.prototype_embs: Optional[np.ndarray] = None
        self.prototype_paths: List[str] = []

    def load_model(self):
        """Load sentence transformer model offline."""
        if self.model is None:
            self.model = SentenceTransformer(self.clf_cfg.model_name)
            rich_descs = self.taxonomy.build_rich_descriptions()
            self.tax_embeddings = self.model.encode(
                rich_descs,
                batch_size=self.clf_cfg.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            # TF-IDF on taxonomy paths
            self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
            self.tax_tfidf = self.tfidf_vectorizer.fit_transform(rich_descs)

            # Check if seed labels exist for few-shot learning
            if self.clf_cfg.use_few_shot and self.config.seed_labels_file and Path(self.config.seed_labels_file).is_file():
                self._build_few_shot_prototypes()

    def _build_few_shot_prototypes(self):
        """Construct category prototype embeddings from seed labels."""
        seed_df = pd.read_csv(self.config.seed_labels_file, low_memory=False)
        include_col = [c for c in seed_df.columns if "include" in c.lower()]
        if include_col:
            seed_df = seed_df[seed_df[include_col[0]] == 1]

        # Find category columns in seed_labels
        l_cols = []
        for i in range(1, 5):
            for c in seed_df.columns:
                if f"category_l{i}" in c.lower() or f"l{i}" in c.lower():
                    l_cols.append(c)
                    break

        if not l_cols:
            return

        # Merge with invoice lines or source data to get text descriptions
        if len(self.config.sources) > 0:
            primary_src = self.config.sources[0].file_path
            src_df = pd.read_csv(primary_src, low_memory=False)
            id_col = self.config.sources[0].id_column
            seed_id_col = [c for c in seed_df.columns if "id" in c.lower()][0]
            merged_seed = seed_df.merge(src_df, left_on=seed_id_col, right_on=id_col, how="inner")

            # Extract text
            mapping = self.config.sources[0].column_mapping
            desc_spec = mapping.get("item_description")
            gl_desc = mapping.get("gl_account_description")

            texts = []
            paths = []
            for _, r in merged_seed.iterrows():
                path_parts = [str(r[c]).strip() for c in l_cols if pd.notna(r[c]) and str(r[c]).strip() != ""]
                if not path_parts:
                    continue
                path_str = " > ".join(path_parts)

                desc_str = ""
                if isinstance(desc_spec, list):
                    desc_str = " ".join([str(r[c]) for c in desc_spec if c in r and pd.notna(r[c])])
                elif desc_spec and desc_spec in r and pd.notna(r[desc_spec]):
                    desc_str = str(r[desc_spec])

                gl_str = str(r[gl_desc]) if gl_desc and gl_desc in r and pd.notna(r[gl_desc]) else ""
                combined_txt = f"{expand_procurement_text(desc_str)} {expand_procurement_text(gl_str)}".strip()
                if combined_txt:
                    texts.append(combined_txt)
                    paths.append(path_str)

            if texts:
                seed_embs = self.model.encode(texts, batch_size=self.clf_cfg.batch_size, show_progress_bar=False, normalize_embeddings=True)
                self.prototype_embs = seed_embs
                self.prototype_paths = paths
                self.has_few_shot = True

    def classify_records(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify active spend rows into the hierarchical taxonomy with tree consistency and calibrated confidence.
        """
        df = df.copy()
        for i in range(1, 5):
            df[f"l{i}"] = ""
        df["confidence"] = df["gate_confidence"]

        active_indices = df[df["include"] == 1].index
        if len(active_indices) == 0:
            return df

        self.load_model()

        # Construct enriched line representations
        active_texts = []
        for idx in active_indices:
            r = df.loc[idx]
            desc = expand_procurement_text(r["item_description_raw"])
            supp = expand_procurement_text(r["supplier_name_raw"])
            gl_desc = expand_procurement_text(r["gl_account_desc_raw"])
            doc_type = str(r["document_type_raw"]).strip()

            tokens = []
            if desc:
                tokens.append(desc)
            if supp:
                tokens.append(f"supplier {supp}")
            if gl_desc:
                tokens.append(f"account {gl_desc}")
            if doc_type and doc_type != "nan":
                tokens.append(f"type {doc_type}")

            full_text = " | ".join(tokens) if tokens else "unknown procurement item"
            active_texts.append(full_text)

        # Encode unique texts to optimize speed and deduplicate computation
        unique_texts, inverse_indices = np.unique(active_texts, return_inverse=True)

        unique_dense_embs = self.model.encode(
            unique_texts.tolist(),
            batch_size=self.clf_cfg.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        dense_embs = unique_dense_embs[inverse_indices]
        dense_embs = np.nan_to_num(dense_embs, nan=0.0, posinf=1.0, neginf=-1.0)

        # Compute predictions
        if self.has_few_shot and self.prototype_embs is not None:
            # Few-shot 1-NN similarity
            few_sim = dense_embs @ self.prototype_embs.T
            top_nn = np.argmax(few_sim, axis=1)
            nn_sims = np.max(few_sim, axis=1)

            # Map predicted paths
            pred_paths = [self.prototype_paths[i] for i in top_nn]
            confidences = np.clip(nn_sims, 0.5, 0.99)
        else:
            # Pure Zero-shot Hybrid: Dense bi-encoder + Sparse TF-IDF
            dense_sim = dense_embs @ self.tax_embeddings.T

            # Sparse lexical matching
            text_tfidf = self.tfidf_vectorizer.transform(active_texts)
            sparse_sim = (text_tfidf @ self.tax_tfidf.T).toarray()

            alpha = self.clf_cfg.hybrid_alpha
            hybrid_sim = alpha * dense_sim + (1.0 - alpha) * sparse_sim

            top1_idx = np.argmax(hybrid_sim, axis=1)
            pred_paths = [self.taxonomy.paths[i] for i in top1_idx]

            # Calibrate confidence using temperature-scaled softmax margin
            temp = self.clf_cfg.temperature
            probs = softmax(hybrid_sim / temp, axis=1)
            top1_prob = np.max(probs, axis=1)
            sorted_probs = np.sort(probs, axis=1)
            margin = sorted_probs[:, -1] - sorted_probs[:, -2]
            confidences = np.clip(top1_prob * 0.7 + margin * 0.3, 0.1, 0.98)

        # Unpack hierarchical levels into l1, l2, l3, l4 columns
        for idx, path_str, conf in zip(active_indices, pred_paths, confidences):
            levels = [p.strip() for p in path_str.split(" > ")]
            for lvl_num, lvl_name in enumerate(levels, start=1):
                if lvl_num <= 4:
                    df.loc[idx, f"l{lvl_num}"] = lvl_name
            df.loc[idx, "confidence"] = round(float(conf), 3)

        return df
