"""Dataset-aware configuration objects for memory, sleep, retrieval, and ablations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class AblationConfig:
    """Feature switches used for ablation studies."""

    no_sleep: bool = False
    episodic_only: bool = False
    summarization_only: bool = False
    disable_schema: bool = False
    disable_replay_selection: bool = False
    disable_conflict_handling: bool = False
    disable_evidence_priority: bool = False


@dataclass
class DatasetMemoryConfig:
    """Dataset-specific memory policy while preserving unified agent interface."""

    dataset_name: str = "personamem"
    replay_top_k: int = 8
    replay_top_k_min: int = 4
    replay_top_k_max: int = 14
    salience_weights: Dict[str, float] = field(default_factory=dict)
    retrieval_weights: Dict[str, float] = field(default_factory=dict)
    schema_abstraction_strength: float = 0.55
    decay_rate: float = 1.0
    evidence_required: bool = False
    factuality_threshold: float = 0.55
    ablations: AblationConfig = field(default_factory=AblationConfig)


DEFAULT_POLICY_MAP: Dict[str, DatasetMemoryConfig] = {
    "personachat": DatasetMemoryConfig(
        dataset_name="personachat",
        replay_top_k=6,
        replay_top_k_min=3,
        replay_top_k_max=10,
        salience_weights={
            "salience": 0.20,
            "novelty": 0.15,
            "uncertainty": 0.10,
            "repetition": 0.20,
            "recency": 0.20,
            "persona_relevance": 0.15,
        },
        retrieval_weights={
            "semantic": 0.26,
            "lexical": 0.22,
            "recency": 0.16,
            "salience": 0.16,
            "persona": 0.14,
            "evidence": 0.02,
            "contradiction_penalty": 0.25,
        },
        schema_abstraction_strength=0.45,
        decay_rate=1.2,
        evidence_required=False,
        factuality_threshold=0.45,
    ),
    "personamem": DatasetMemoryConfig(
        dataset_name="personamem",
        replay_top_k=10,
        replay_top_k_min=5,
        replay_top_k_max=16,
        salience_weights={
            "salience": 0.20,
            "novelty": 0.10,
            "uncertainty": 0.12,
            "repetition": 0.26,
            "recency": 0.14,
            "persona_relevance": 0.18,
        },
        retrieval_weights={
            "semantic": 0.24,
            "lexical": 0.20,
            "recency": 0.12,
            "salience": 0.18,
            "persona": 0.20,
            "evidence": 0.03,
            "contradiction_penalty": 0.30,
        },
        schema_abstraction_strength=0.65,
        decay_rate=0.85,
        evidence_required=False,
        factuality_threshold=0.58,
    ),
    "locomo": DatasetMemoryConfig(
        dataset_name="locomo",
        replay_top_k=12,
        replay_top_k_min=6,
        replay_top_k_max=18,
        salience_weights={
            "salience": 0.18,
            "novelty": 0.10,
            "uncertainty": 0.16,
            "repetition": 0.12,
            "recency": 0.14,
            "persona_relevance": 0.10,
            "evidence_strength": 0.20,
        },
        retrieval_weights={
            "semantic": 0.20,
            "lexical": 0.18,
            "recency": 0.10,
            "salience": 0.16,
            "persona": 0.10,
            "evidence": 0.26,
            "contradiction_penalty": 0.35,
        },
        schema_abstraction_strength=0.35,
        decay_rate=0.55,
        evidence_required=True,
        factuality_threshold=0.72,
    ),
    "okvqa": DatasetMemoryConfig(
        dataset_name="okvqa",
        replay_top_k=10,
        replay_top_k_min=4,
        replay_top_k_max=14,
        salience_weights={
            "salience": 0.18,
            "novelty": 0.12,
            "uncertainty": 0.18,
            "repetition": 0.12,
            "recency": 0.14,
            "persona_relevance": 0.08,
            "evidence_strength": 0.18,
        },
        retrieval_weights={
            "semantic": 0.24,
            "lexical": 0.18,
            "recency": 0.10,
            "salience": 0.16,
            "persona": 0.07,
            "evidence": 0.25,
            "contradiction_penalty": 0.32,
        },
        schema_abstraction_strength=0.30,
        decay_rate=0.65,
        evidence_required=True,
        factuality_threshold=0.70,
    ),
}


def resolve_dataset_config(
    dataset_name: str,
    overrides: Dict[str, Any] | None = None,
    ablations: AblationConfig | None = None,
) -> DatasetMemoryConfig:
    """Create dataset policy with optional runtime overrides."""

    key = (dataset_name or "personamem").strip().lower()
    base = DEFAULT_POLICY_MAP.get(key, DEFAULT_POLICY_MAP["personamem"])

    cfg = DatasetMemoryConfig(
        dataset_name=base.dataset_name,
        replay_top_k=base.replay_top_k,
        replay_top_k_min=base.replay_top_k_min,
        replay_top_k_max=base.replay_top_k_max,
        salience_weights=dict(base.salience_weights),
        retrieval_weights=dict(base.retrieval_weights),
        schema_abstraction_strength=base.schema_abstraction_strength,
        decay_rate=base.decay_rate,
        evidence_required=base.evidence_required,
        factuality_threshold=base.factuality_threshold,
        ablations=ablations or AblationConfig(),
    )

    if overrides:
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    return cfg
