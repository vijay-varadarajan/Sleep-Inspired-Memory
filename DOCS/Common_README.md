# Common Workflow (After Preprocessing, Before Postprocessing)

This document summarizes the shared pipeline used by **PersonaChat**, **PersonaMem**, and **LOCOMO** once each dataset has already been converted into the common benchmark sample format.

## 1) Inputs to the shared runtime

All runners load two preprocessed artifacts:
- `*_processed.json`: flat list of benchmark samples
- `*_persona_sessions.json`: the same samples grouped by `persona_id`

Each sample provides the same core fields used by the runtime:
- `query`
- `correct_answer`
- `incorrect_answers`
- `persona`
- `related_conversation_snippet`
- `topic_query`

## 2) Agent layer (method selection)

The runner creates one method-specific agent via `create_agent()` from [evaluation/baselines.py](evaluation/baselines.py):
- `vanilla`
- `rag`
- `episodic`
- `summarization`
- `sleep`

Each method uses the same interaction API (`agent.interact(...)`) so evaluation remains comparable across datasets.

## 3) Memory + sleep methodology

During multi-turn evaluation per persona:
- First turn usually runs with limited/no memory usage.
- Later turns run with memory enabled (`use_memory=True`) to test carry-over.

For the sleep-enabled method:
- `agent.sleep()` is called to trigger consolidation-style processing.
- This is central to pre-vs-post memory probe comparisons in Table 2.

Conceptually, the shared backend combines:
- episodic memory traces,
- consolidated memory,
- schema-style abstractions,
- optional retrieval augmentation (RAG baseline),
- and sleep-inspired consolidation steps.

## 4) Evaluation methodology (shared across datasets)

All runners use `PersonaMemEvaluator` from [evaluation/personamem_benchmark.py](evaluation/personamem_benchmark.py).

### Table 1: Task-based memory behavior
Per generated response, the runner computes:
- **Long-Horizon QA**: whether response matches the gold answer better than negatives.
- **Multi-Session Continuity**: consistency with prior relevant context.
- **Hallucination Rate**: unsupported content normalized by response length.

### Table 2: Cognitive-style probes (sleep-focused)
Using the sleep method, each probe is measured before and after consolidation:
- Delayed Recall
- Cue-Based Recall
- Cross-Episode Integration
- Schema Utilization

Reported values include pre-score, post-score, and delta improvement.

## 5) Runtime control and robustness

Across runners, the methodology includes:
- grouping by persona to preserve session structure,
- bounded samples per persona for tractable runtime,
- exception-safe interaction/evaluation blocks,
- and retry/timeout-aware evaluator behavior.

This keeps comparisons stable across all three datasets while using the same backend protocol.

## 6) Output handoff to postprocessing

Before postprocessing, each runner writes a raw results JSON containing:
- run metadata (timestamp, dataset, split, methods, sample counts),
- `table1` method-level metric rows,
- `table2` probe-level sleep comparison results.

Postprocessing scripts then convert this common raw structure into final CSV tables.

## 7) Function-level implementation notes + dataset limitations

- `create_agent(method)` ([evaluation/baselines.py](evaluation/baselines.py)) instantiates one of five backends with a unified `interact()` API, so runners can switch methods without changing evaluation code.  
	Limitation: all datasets are forced through the same interface, so LOCOMO evidence-grounded QA and PersonaChat dialog flow are both treated as generic text turns.

- `MemoryAgent.interact(...)` ([agent/agent.py](agent/agent.py)) builds a system prompt with optional persona and retrieved memories, sends it to Gemini, then stores `User+Agent` text as an episode.  
	Limitation: this is single-shot prompt assembly without dataset-specific decoding rules, so strict factual answering in LOCOMO can be diluted by conversational behavior tuned for PersonaChat/PersonaMem.

- `MemoryAgent._retrieve_relevant_memories(...)` ([agent/agent.py](agent/agent.py)) extracts concepts from the query, retrieves overlapping consolidated memories and schemas, and injects short bullet context into the prompt.  
	Limitation: concept-overlap retrieval is lexical/LLM-extracted rather than evidence-indexed, so exact `dia_id` grounding in LOCOMO is weaker than direct evidence lookup.

- `EpisodicMemoryStore.add_episode()/get_recent()` ([memory/episodic.py](memory/episodic.py)) stores raw interactions with metadata and returns newest-first episodes for short-horizon recall.  
	Limitation: recency-first retrieval can overfit recent PersonaChat turns while under-serving long-range PersonaMem preference changes and distant LOCOMO sessions.

- `ConsolidatedMemoryStore.add_memory()/search_by_concepts()` ([memory/consolidated.py](memory/consolidated.py)) keeps compressed summaries with concept lists and returns memories ranked by concept overlap count.  
	Limitation: overlap scoring ignores temporal and speaker constraints, which matters more in LOCOMO and multi-session PersonaMem than in short PersonaChat contexts.

- `SchemaStore.find_by_concepts(...)` ([memory/schema.py](memory/schema.py)) retrieves abstract schemas by concept intersection and supports generalized persona-conditioned responses.  
	Limitation: abstraction helps PersonaMem/PersonaChat personalization but can introduce over-generalized answers for LOCOMO questions needing precise date/event details.

- `SleepCycle.run_sleep_cycle()` ([sleep/consolidation.py](sleep/consolidation.py)) executes replay selection, LLM compression, schema formation, and episodic decay in a fixed multi-phase pipeline.  
	Limitation: the same sleep schedule is applied across datasets, so consolidation granularity is not tuned to PersonaChat brevity vs LOCOMO long-session density.

- `MemoryCompressor.compress_single_episode()/compress_episode_batch()` ([sleep/compression.py](sleep/compression.py)) uses Gemini prompts to summarize episodes, extract concepts/themes, and enrich consolidated memories with batch-level shared concepts.  
	Limitation: LLM summarization is lossy, so fine-grained facts (especially LOCOMO evidence-linked facts) can be dropped even when high-level PersonaMem/PersonaChat themes are preserved.

- `MemoryCompressor.extract_concepts_from_text()` ([sleep/compression.py](sleep/compression.py)) requests a comma-separated concept list from Gemini and feeds these concepts into retrieval, novelty, and schema matching.  
	Limitation: concept extraction quality is model-dependent and can vary with style differences across the three datasets (chatty PersonaChat, preference-heavy PersonaMem, event-heavy LOCOMO).

- `evaluate_long_horizon_qa(...)` ([evaluation/personamem_benchmark.py](evaluation/personamem_benchmark.py)) uses an LLM judge with a compact `MATCH: YES/NO` prompt against gold answers (with timeout/retry safeguards).  
	Limitation: judge-based semantic matching is less strict than exact-match scoring and may score conversationally-correct PersonaChat answers more generously than precise LOCOMO fact answers.

- `evaluate_multi_session_continuity(...)` ([evaluation/personamem_benchmark.py](evaluation/personamem_benchmark.py)) asks an LLM to score whether the response correctly uses prior snippet context and returns a 0–1 continuity score.  
	Limitation: snippet quality differs by dataset preprocessing, so LOCOMO continuity depends on evidence text mapping while PersonaChat continuity depends on shorter history windows.

- `evaluate_hallucination_rate(...)` ([evaluation/personamem_benchmark.py](evaluation/personamem_benchmark.py)) prompts an LLM to count unsupported claims and normalizes by response length (per 100 words).  
	Limitation: verbose outputs are penalized more and short factual outputs are penalized less, which can skew cross-dataset comparisons when average response lengths differ.

- `evaluate_delayed_recall(...)`, `evaluate_cue_based_recall(...)`, `evaluate_cross_episode_integration(...)`, and `evaluate_schema_utilization(...)` ([evaluation/personamem_benchmark.py](evaluation/personamem_benchmark.py)) are all LLM-judged Table 2 probes applied pre/post sleep.  
	Limitation: these probes are methodologically consistent across datasets but not dataset-calibrated, so absolute scores are not directly comparable as intrinsic dataset difficulty measures.

