# Research-Grade Upgrade Notes

These upgrades make the sleep-inspired memory system more structured, dataset-aware, and paper-ready.

- Added dataset-specific memory policies for PersonaChat, PersonaMem, and LOCOMO in [memory/config.py](memory/config.py).
- Added ablation switches (no sleep, episodic only, summarization only, no schema, no replay selection, no conflict handling, no evidence priority).

- Extended episodic memories with richer encoding metadata:
  `salience_score`, `novelty_score`, `persona_relevance`, `factuality_risk`, `temporal_recency`, `evidence_strength`, `episode_type`, `confidence`, `uncertainty`, `repetition_count`, `memory_consistency_score`.
- Added episodic candidate retrieval using mixed lexical + metadata scoring.

- Upgraded replay selection to weighted, adaptive top-k replay with dataset-aware priorities.
- Added replay modes: `verbatim`, `lossy`, `contrastive`, `schema`, and `evidence`.

- Upgraded consolidation to produce structured memory records, not only plain summaries.
- Consolidated memories now track: `core_fact`, `supporting_context`, `confidence`, `time_span`, `persona_link`, `evidence_link`, `contradiction_flags`, `schema_label`, and stability/consistency fields.

- Added conflict detection and conflict-aware memory consistency updates in consolidated storage.
- Added hybrid retrieval scoring with interpretable retrieval bundles and contradiction suppression.

- Upgraded schema store with lifecycle controls:
  `status` (`emergent`/`stable`/`conflicted`), `version`, `schema_type`, and parent linking.
- Schema formation now marks conflict-sensitive schemas and updates confidence/version over time.

- Reworked agent prompting to structured sections:
  task instruction, user profile, retrieved memory bundles, conflict handling, and current query.
- Added dataset-aware response constraints (grounding for LOCOMO, preference stability for PersonaMem, continuity for PersonaChat).

- Added stronger automatic metrics in evaluator:
  answer utility, retrieval success (`Recall@k`, `MRR`, `nDCG`, evidence hit rate), memory fidelity, unit-normalized hallucination, and efficiency metrics.
- Kept LLM-judge metrics for secondary validation.

- Updated all three runners to pass dataset-aware configs and output richer paper-facing metadata.
- Added ablation CLI flag support in runners.

- Expanded CSV postprocessing outputs to include new Table 1 research metrics.

These changes keep the unified `interact()` interface while enabling dataset-specific behavior and stronger scientific evaluation.
