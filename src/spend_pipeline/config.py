"""
Configuration parser and validator for YAML-driven spend pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class SourceConfig:
    name: str
    file_path: str
    id_column: str
    record_id_prefix: str = ""
    is_primary: bool = True
    amount_type: str = "net"  # 'net' or 'gross'
    default_vat_rate: float = 0.0
    column_mapping: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GatingConfig:
    approved_statuses: List[str] = field(default_factory=lambda: ["paid", "approved", "posted", "cleared"])
    cancelled_statuses: List[str] = field(default_factory=lambda: ["cancelled", "canceled", "void", "reversed"])
    not_approved_statuses: List[str] = field(
        default_factory=lambda: ["rejected", "on hold", "submitted", "disputed", "pending", "draft"]
    )
    non_procurement_gl_prefixes: List[str] = field(default_factory=lambda: ["IC", "I/C"])
    non_procurement_gl_keywords: List[str] = field(
        default_factory=lambda: ["ic payables", "i/c payable", "intercompany", "tax", "taxes", "gst", "vat"]
    )
    non_procurement_desc_keywords: List[str] = field(
        default_factory=lambda: [
            "sales tax", "state & local taxes", "gst input tax", "tax expense", "vat",
            "intercompany", "royalty", "royalties", "dividend", "financing fee",
            "interest expense", "bank charges", "payroll", "salary", "wages",
            "bad debt", "depreciation", "amortization"
        ]
    )
    filter_zero_negative_offsets: bool = True


@dataclass
class DeduplicationConfig:
    enabled: bool = True
    intra_system_dedup: bool = True
    cross_system_linkage: bool = False
    po_match_tolerance_abs: float = 1.0
    po_match_tolerance_pct: float = 0.02
    composite_date_window_days: int = 60
    composite_amount_tolerance_abs: float = 1.0


@dataclass
class EntityResolutionConfig:
    supplier_similarity_threshold: float = 0.82
    item_similarity_threshold: float = 0.85
    blocking_prefix_len: int = 4
    min_ngram: int = 3
    max_ngram: int = 4


@dataclass
class ClassifierConfig:
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    batch_size: int = 256
    hybrid_alpha: float = 0.4  # Weight for dense bi-encoder vs sparse TF-IDF (0.0 to 1.0)
    temperature: float = 0.15   # Softmax calibration temperature
    use_few_shot: bool = True  # Utilize seed labels as prototypes when available
    few_shot_knn_k: int = 1


@dataclass
class CompanyConfig:
    company_id: str
    reporting_currency: str
    taxonomy_file: str
    sources: List[SourceConfig]
    seed_labels_file: Optional[str] = None
    fx_rates_file: Optional[str] = None
    gating: GatingConfig = field(default_factory=GatingConfig)
    deduplication: DeduplicationConfig = field(default_factory=DeduplicationConfig)
    entity_resolution: EntityResolutionConfig = field(default_factory=EntityResolutionConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    raw_config: Dict[str, Any] = field(default_factory=dict)


def load_config(config_path: str) -> CompanyConfig:
    """Load and validate a company YAML configuration file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    sources_data = data.get("sources", [])
    sources = []
    for s in sources_data:
        sources.append(
            SourceConfig(
                name=s.get("name", "source"),
                file_path=s.get("file_path", ""),
                id_column=s.get("id_column", "line_id"),
                record_id_prefix=s.get("record_id_prefix", ""),
                is_primary=s.get("is_primary", True),
                amount_type=s.get("amount_type", "net").lower(),
                default_vat_rate=float(s.get("default_vat_rate", 0.0)),
                column_mapping=s.get("column_mapping", {}),
            )
        )

    gating_dict = data.get("gating", {})
    gating = GatingConfig(
        approved_statuses=[s.lower() for s in gating_dict.get("approved_statuses", GatingConfig().approved_statuses)],
        cancelled_statuses=[s.lower() for s in gating_dict.get("cancelled_statuses", GatingConfig().cancelled_statuses)],
        not_approved_statuses=[s.lower() for s in gating_dict.get("not_approved_statuses", GatingConfig().not_approved_statuses)],
        non_procurement_gl_prefixes=gating_dict.get("non_procurement_gl_prefixes", GatingConfig().non_procurement_gl_prefixes),
        non_procurement_gl_keywords=[s.lower() for s in gating_dict.get("non_procurement_gl_keywords", GatingConfig().non_procurement_gl_keywords)],
        non_procurement_desc_keywords=[s.lower() for s in gating_dict.get("non_procurement_desc_keywords", GatingConfig().non_procurement_desc_keywords)],
        filter_zero_negative_offsets=gating_dict.get("filter_zero_negative_offsets", True),
    )

    dedup_dict = data.get("deduplication", {})
    dedup = DeduplicationConfig(
        enabled=dedup_dict.get("enabled", True),
        intra_system_dedup=dedup_dict.get("intra_system_dedup", True),
        cross_system_linkage=dedup_dict.get("cross_system_linkage", False),
        po_match_tolerance_abs=float(dedup_dict.get("po_match_tolerance_abs", 1.0)),
        po_match_tolerance_pct=float(dedup_dict.get("po_match_tolerance_pct", 0.02)),
        composite_date_window_days=int(dedup_dict.get("composite_date_window_days", 60)),
        composite_amount_tolerance_abs=float(dedup_dict.get("composite_amount_tolerance_abs", 1.0)),
    )

    er_dict = data.get("entity_resolution", {})
    er = EntityResolutionConfig(
        supplier_similarity_threshold=float(er_dict.get("supplier_similarity_threshold", 0.82)),
        item_similarity_threshold=float(er_dict.get("item_similarity_threshold", 0.85)),
        blocking_prefix_len=int(er_dict.get("blocking_prefix_len", 4)),
        min_ngram=int(er_dict.get("min_ngram", 3)),
        max_ngram=int(er_dict.get("max_ngram", 4)),
    )

    clf_dict = data.get("classifier", {})
    clf = ClassifierConfig(
        model_name=clf_dict.get("model_name", ClassifierConfig().model_name),
        batch_size=int(clf_dict.get("batch_size", 256)),
        hybrid_alpha=float(clf_dict.get("hybrid_alpha", 0.4)),
        temperature=float(clf_dict.get("temperature", 0.15)),
        use_few_shot=bool(clf_dict.get("use_few_shot", True)),
        few_shot_knn_k=int(clf_dict.get("few_shot_knn_k", 1)),
    )

    return CompanyConfig(
        company_id=data.get("company_id", path.stem),
        reporting_currency=data.get("reporting_currency", "USD").upper(),
        taxonomy_file=data.get("taxonomy_file", ""),
        sources=sources,
        seed_labels_file=data.get("seed_labels_file"),
        fx_rates_file=data.get("fx_rates_file"),
        gating=gating,
        deduplication=dedup,
        entity_resolution=er,
        classifier=clf,
        raw_config=data,
    )
