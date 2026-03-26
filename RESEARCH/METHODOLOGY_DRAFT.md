## 1. Methodology

### 1.1 High-Level Overview of the Framework

The proposed framework implements a three-tier memory architecture inspired by the complementary learning systems (CLS) theory of hippocampal-neocortical consolidation. In biological memory, the hippocampus rapidly encodes episodic experiences while the neocortex slowly integrates them into stable semantic knowledge during sleep. Our framework mirrors this with three analogous stores: an **EpisodicMemoryStore** (hippocampal rapid encoding), a **ConsolidatedMemoryStore** (neocortical compressed representations), and a **SchemaStore** (semantic long-term knowledge).

This tiered design was chosen deliberately over simpler approaches such as flat retrieval-augmented generation (RAG) because it separates fast, lossy episodic storage from stable, generalised semantic knowledge. A flat vector store conflates these roles, leading to retrieval noise from redundant or contradictory episodes. By contrast, the offline sleep cycle—triggered periodically—compresses and abstracts episodic content without disrupting online inference.

Memory stability in the consolidated tier is modelled analogously to the synaptic consolidation curve:

$$S(t) = S_0 \cdot e^{-\lambda t} + S_{\infty}$$

where $S_0$ is initial memory strength, $\lambda$ is the decay rate, $t$ is elapsed time, and $S_{\infty}$ is asymptotic long-term stability. The CLS principle further motivates the dual-system design: rapid hippocampal binding coexists with slow cortical integration to prevent catastrophic interference.

---

### 1.2 Conceptual Pipeline

The end-to-end pipeline proceeds through six stages:

- **(1) Episode Encoding:** Each user–agent interaction turn is encoded as an `Episode` object with metadata fields including `importance`, `novelty_score`, `salience_score`, `persona_relevance`, `factuality_risk`, and `temporal_recency`. Rich metadata is stored immediately so that downstream retrieval and replay can be priority-weighted without re-reading raw text.

- **(2) Hybrid Retrieval:** At query time, `_retrieve_memory_bundles()` scores all memory objects using a weighted combination of signals. The retrieval score is:
$$r(q, m) = w_s \cdot \text{sem}(q,m) + w_l \cdot \text{lex}(q,m) + w_r \cdot \text{rec}(m) + w_e \cdot \text{evid}(m)$$
where $w_s, w_l, w_r, w_e$ are dataset-specific weights for semantic, lexical, recency, and evidence signals respectively.

- **(3) Response Generation:** Retrieved memory bundles are injected into a structured system prompt via `_build_system_prompt()`, which layers task instruction, user persona, memory context, and conflict guidance. The last five conversation turns are appended as short-term context.

- **(4) Sleep Trigger:** The agent tracks `episodes_since_sleep`. Consolidation is triggered automatically when this counter reaches threshold $\tau = 4$, ensuring the episodic store never grows unboundedly between sleep cycles.

- **(5) Sleep Consolidation Cycle:** The `SleepCycle` executes five sequential phases: replay selection, LLM compression, consolidation, schema formation, and decay (detailed in §1.6).

- **(6) Schema Formation and Decay:** Abstract schemas are induced from clusters of consolidated memories. Low-salience episodes are pruned via exponential decay, freeing capacity for new experiences.

---

### 1.3 Dataset Descriptions

Four datasets were selected to evaluate distinct memory capabilities across varied conversational modalities.

**PersonaChat** is a large-scale social dialogue dataset in which two speakers exchange persona-grounded conversation turns. It was chosen to evaluate preference retention and multi-turn social coherence. We use 50 samples from the validation split. Configuration: schema abstraction strength $\alpha = 0.45$, replay top-$k = 6$, decay rate $\delta = 0.8$.

**PersonaMem** is a persona-based question-answering benchmark that tests fine-grained persona recall across sessions. It was chosen because it explicitly separates persona facts from conversational content, stressing the consolidated memory tier. We use 50 samples from the benchmark split with $\alpha = 0.65$, replay top-$k = 10$, factuality threshold $\theta_f = 0.58$.

**LOCOMO** is an evidence-grounded long-form conversational QA dataset with timestamped, multi-speaker dialogues. It was chosen to stress evidence retrieval and temporal reasoning. Evidence weight $w_e = 0.26$, $\theta_f = 0.72$, replay top-$k = 8$.

**OK-VQA** is an open-knowledge visual QA dataset pairing natural images with knowledge-intensive questions. It was chosen to evaluate multimodal episodic memory via text-encoded image metadata (see §1.7). We use 50 samples. Configuration: schema abstraction strength $\alpha = 0.30$, replay top-$k = 10$, decay rate $\delta = 0.65$, evidence weight $w_e = 0.25$, factuality threshold $\theta_f = 0.70$.

Dataset size per split follows a uniform sample distribution: $|D_i| = 50 \;\forall\; i \in \{\text{PersonaChat, PersonaMem, LOCOMO, OK-VQA}\}$, ensuring comparability across benchmarks.

---

### 1.4 Preprocessing Pipelines

Each dataset undergoes a dedicated preprocessing pipeline that transforms raw data into structured episodic inputs consumable by the memory agent.

**PersonaMem** (`personamem_preprocessing.py`) loads the dataset and groups samples by `persona_id`. For each persona, conversations are flattened into sequential turns, QA pairs are extracted with their ground-truth answers, and supporting evidence snippets are collected. Grouping by persona is essential so that cross-session persona facts are co-located in episodic memory from the outset.

**PersonaChat** (`personachat_preprocessing.py`) parses the dialogue format to extract per-speaker persona statement lists and conversation turn sequences. Persona statements are treated as high-importance seed episodes injected before the first conversation turn. This design ensures persona facts receive elevated salience scores during replay selection.

**LOCOMO** (`locomo_preprocessing.py`) processes structured conversation files with speaker labels and timestamps. Evidence passages linked to QA pairs are extracted and stored as separate episodic fields so that the evidence retrieval weight $w_e$ can be applied at query time. Temporal ordering is preserved to support recency-weighted retrieval.

**OK-VQA** (`okvqa_preprocessing.py`) reads image metadata—object labels and human-generated captions—and converts them to natural language descriptions that substitute for raw visual input. Question–answer pairs are paired with these text descriptions as episodic context.

Across all pipelines, a shared normalization step applies:

$$\hat{x} = \frac{x - \mu_{\text{field}}}{\sigma_{\text{field}} + \epsilon}$$

to numeric importance and salience fields, preventing scale imbalance during retrieval scoring. Concept extraction uses a TF-IDF-style term weighting:

$$w_{t,d} = \text{tf}(t,d) \cdot \log\!\left(\frac{N}{1 + \text{df}(t)}\right)$$

where $\text{tf}(t,d)$ is the term frequency in document $d$, $\text{df}(t)$ is the document frequency, and $N$ is the total number of episodes. High-weight terms are stored as `key_concepts` in consolidated memories, enabling fast lexical retrieval without re-parsing full episode text.

---

### 1.5 Memory Store

The three-tier memory store mirrors the neuroscientific distinction between short-term episodic, long-term declarative, and abstract semantic memory.

**EpisodicMemoryStore** (`episodic.py`) stores raw interaction episodes with 20 metadata fields, including `importance`, `novelty_score`, `salience_score`, `persona_relevance`, `factuality_risk`, `temporal_recency`, `evidence_strength`, `contradiction_flags`, `confidence`, `uncertainty`, and `memory_consistency_score`. The richness of this metadata is intentional: it allows every downstream phase—retrieval, replay selection, and decay—to operate on pre-computed signals rather than re-analysing text at inference time.

**ConsolidatedMemoryStore** (`consolidated.py`) stores LLM-generated summaries of episode batches. Each `ConsolidatedMemory` record includes `core_fact`, `supporting_context`, `key_concepts`, `schema_label`, `stability_score`, and `contradiction_flags`. Conflict detection is built into the consolidation step: if a new consolidated memory contradicts an existing one, both receive updated `contradiction_flags` and the agent's system prompt is notified. This prevents silent hallucination propagation.

**SchemaStore** (`schema.py`) stores abstract knowledge patterns extracted from clusters of consolidated memories. Schemas have a lifecycle (`emergent` → `stable` → `conflicted`) and support versioning and parent–child relationships for hierarchical knowledge.

The hybrid retrieval score combining all three tiers is:

$$R(q) = \arg\max_{m \in \mathcal{M}} \bigl[ w_s \cdot \cos(\mathbf{q}, \mathbf{m}) + w_l \cdot \text{BM25}(q,m) + w_r \cdot e^{-\lambda_r \Delta t} + w_e \cdot \text{evid}(m) \bigr]$$

Conflict detection fires when the cosine distance between two consolidated memories falls below a threshold $\theta_c$:

$$\text{conflict}(m_i, m_j) = \mathbf{1}\!\left[\cos(\mathbf{m}_i, \mathbf{m}_j) < \theta_c \;\land\; \text{topic}(m_i) = \text{topic}(m_j)\right]$$

---

### 1.6 Sleep Cycle

The `SleepCycle` class in `consolidation.py` implements a five-phase offline consolidation process that runs between agent interactions. Each phase has a direct biological analogy to slow-wave and REM sleep processes.

**Phase 1 — Replay Selection:** Episodes are ranked by `calculate_replay_priority()`, a weighted combination of recency, importance, novelty, and access frequency. Recency uses exponential decay with a half-life of 7 days:

$$P(e) = w_1 \cdot e^{-\ln 2 \cdot \Delta t / 7} + w_2 \cdot \text{importance}(e) + w_3 \cdot \text{novelty}(e) + w_4 \cdot \log(1 + \text{access\_count}(e))$$

This mirrors hippocampal sharp-wave ripples during slow-wave sleep, which preferentially reactivate salient recent experiences. The top-$k$ episodes (dataset-specific, $k \in [6,10]$) are selected for compression.

**Phase 2 — Compression:** The Gemini LLM summarises batches of selected episodes into semantic units via `compress_episode_batch()`. LLM-based compression was chosen because rule-based summarisation cannot capture the inferential leaps needed to bridge disparate episodic fragments into coherent facts.

**Phase 3 — Consolidation:** Compressed summaries are merged into `ConsolidatedMemory` objects. Conflict-sensitive merging checks for contradictions before inserting, analogous to memory reconsolidation during REM sleep.

**Phase 4 — Schema Formation:** Clusters of consolidated memories with shared `key_concepts` are abstracted into new or updated `Schema` objects via `merge_schemas()`, which deduplicates overlapping schemas. This phase mirrors neocortical slow integration of repeated patterns into generalised knowledge.

**Phase 5 — Decay:** Low-priority episodic memories are pruned according to:

$$\text{importance}_{t+1}(e) = \text{importance}_t(e) \cdot (1 - \delta)$$

where $\delta \in [0.55, 1.2]$ is the dataset-specific decay rate. Episodes whose importance falls below a minimum threshold are deleted, implementing biologically motivated forgetting to prevent memory saturation.

---

### 1.7 Image Processing

OK-VQA images are not processed visually in the current implementation. Instead, each image is represented by a structured text description composed of human-annotated object labels and image captions. These text descriptions are stored as episodic content in the `EpisodicMemoryStore`, allowing the same retrieval pipeline used for text-only datasets to operate without modification.

This design was chosen because it decouples multimodal grounding from memory architecture, ensuring reproducibility and reducing inference latency. Gemini's native multimodal capabilities are invoked at the response generation stage when the raw image is available, but the memory layer operates entirely on text. For future work, visual embeddings can be integrated using CLIP-style cosine similarity for cross-modal retrieval:

$$\text{sim}_{\text{CLIP}}(v, t) = \frac{\mathbf{f}_v \cdot \mathbf{f}_t}{\|\mathbf{f}_v\| \, \|\mathbf{f}_t\|}$$

where $\mathbf{f}_v$ and $\mathbf{f}_t$ are the visual and textual embedding vectors respectively.
