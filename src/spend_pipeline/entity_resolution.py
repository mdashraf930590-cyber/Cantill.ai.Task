"""
Stage 3: Entity Resolution Module.
Generates canonical supplier_key and item_key identifiers through text normalization
(Unicode accent folding, legal suffix stripping, boilerplate removal), blocking,
TF-IDF cosine similarity, RapidFuzz token matching, and connected-components graph clustering.
"""

import hashlib
import re
import unicodedata
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import rapidfuzz
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import CompanyConfig, EntityResolutionConfig

LEGAL_SUFFIXES_REGEX = re.compile(
    r"\b(pty\s+ltd|proprietary\s+limited|pty|ltd|limited|cc|corp|corporation|inc|incorporated|"
    r"gmbh|ag|sa|bv|nv|srl|llc|plc|co|company|holdings|holding|group|enterprises|industries)\b",
    re.IGNORECASE,
)

BOILERPLATE_TOKENS_REGEX = re.compile(
    r"\b(services|supplies|solutions|engineering|equipment|technologies|technology|"
    r"international|national|trading|logistics|commercial)\b",
    re.IGNORECASE,
)


def normalize_entity_text(val: str, strip_boilerplate: bool = False) -> str:
    """Normalize text: lowercasing, Unicode/accent folding, stripping legal entity suffixes & boilerplate."""
    if pd.isna(val) or not str(val).strip():
        return ""

    # Unicode / accent folding
    s = unicodedata.normalize("NFKD", str(val)).encode("ASCII", "ignore").decode("utf-8").lower()
    # Strip legal suffixes
    s = LEGAL_SUFFIXES_REGEX.sub(" ", s)
    if strip_boilerplate:
        s = BOILERPLATE_TOKENS_REGEX.sub(" ", s)
    # Strip special characters
    s = re.sub(r"[^\w\s]", " ", s)
    tokens = s.split()
    return " ".join(tokens)


def cluster_strings_tfidf(
    unique_strings: List[str],
    similarity_threshold: float = 0.82,
    prefix_block_len: int = 3,
) -> Dict[str, str]:
    """
    Cluster unique strings using prefix blocking, TF-IDF character n-grams,
    RapidFuzz token set ratio, and connected-components graph partitioning.
    Returns mapping from raw string -> canonical cluster key.
    """
    if not unique_strings:
        return {}

    mapping = {}
    valid_strings = [s for s in unique_strings if s.strip()]

    # If small number of strings, cluster globally
    if len(valid_strings) <= 3000:
        blocks = {"all": valid_strings}
    else:
        # Prefix blocking
        blocks = {}
        for s in valid_strings:
            pfx = s[:prefix_block_len].lower()
            blocks.setdefault(pfx, []).append(s)

    cluster_id_counter = 0

    for pfx, block_items in blocks.items():
        n = len(block_items)
        if n == 1:
            item = block_items[0]
            cid = f"supp_{hashlib.md5(item.encode()).hexdigest()[:8]}"
            mapping[item] = cid
            continue

        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=1)
        try:
            X = vec.fit_transform(block_items)
            sim_matrix = (X * X.T).toarray()
            adj = (sim_matrix >= similarity_threshold).astype(int)

            # Augment with RapidFuzz token matching for containment and word reordering
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i, j] == 0:
                        s1 = block_items[i]
                        s2 = block_items[j]
                        # Token set ratio handles substring containment e.g. 'dunton' and 'dunton equipment'
                        score = rapidfuzz.fuzz.token_set_ratio(s1, s2)
                        if score >= 90 and min(len(s1), len(s2)) >= 4:
                            adj[i, j] = 1
                            adj[j, i] = 1

            n_comp, comp_labels = connected_components(adj)

            for i, item in enumerate(block_items):
                label = comp_labels[i] + cluster_id_counter
                cid = f"supp_{hashlib.md5(f'{pfx}_{label}'.encode()).hexdigest()[:8]}"
                mapping[item] = cid

            cluster_id_counter += n_comp
        except Exception:
            for item in block_items:
                cid = f"supp_{hashlib.md5(item.encode()).hexdigest()[:8]}"
                mapping[item] = cid

    return mapping


def resolve_entities(df: pd.DataFrame, config: CompanyConfig) -> pd.DataFrame:
    """
    Generate canonical supplier_key and item_key identifiers for all rows.
    """
    df = df.copy()
    er_cfg = config.entity_resolution

    # ---------------------------------------------------------
    # 1. Supplier Entity Resolution
    # ---------------------------------------------------------
    has_supp_name = df["supplier_name_raw"].astype(str).str.strip().str.len() > 0
    df["supp_source_text"] = np.where(
        has_supp_name,
        df["supplier_name_raw"].astype(str),
        df["vendor_id_raw"].astype(str),
    )

    df["supp_norm"] = df["supp_source_text"].apply(lambda s: normalize_entity_text(s, strip_boilerplate=True))

    unique_supps = df[df["supp_norm"] != ""]["supp_norm"].unique().tolist()
    cluster_map = cluster_strings_tfidf(
        unique_supps,
        similarity_threshold=er_cfg.supplier_similarity_threshold,
        prefix_block_len=er_cfg.blocking_prefix_len,
    )

    def assign_supplier_key(row) -> str:
        s_norm = row["supp_norm"]
        if s_norm and s_norm in cluster_map:
            return cluster_map[s_norm]
        v_id = str(row["vendor_id_raw"]).strip()
        if v_id and v_id.lower() != "nan":
            return f"supp_{hashlib.md5(v_id.encode()).hexdigest()[:8]}"
        return "supp_unknown"

    df["supplier_key"] = df.apply(assign_supplier_key, axis=1)

    # ---------------------------------------------------------
    # 2. Item Entity Resolution
    # ---------------------------------------------------------
    df["item_norm"] = df["item_description_raw"].apply(lambda s: normalize_entity_text(s, strip_boilerplate=False))

    def assign_item_key(row) -> str:
        mat = str(row["material_number_raw"]).strip()
        if mat and mat.lower() != "nan":
            return f"item_{mat.lower()}"
        item_text = row["item_norm"]
        if item_text:
            slug = hashlib.md5(item_text.encode()).hexdigest()[:8]
            return f"item_{slug}"
        return "item_unknown"

    df["item_key"] = df.apply(assign_item_key, axis=1)

    return df
