# Design Decisions & Rationale

This document explains the key architectural decisions and their biological/computational justifications.

## 1. Memory Systems Architecture

### Three-Level Memory Hierarchy

**Decision**: Separate episodic, consolidated, and schema stores

**Rationale**: 
- **Biological**: Mirrors hippocampus (episodic) → neocortex (semantic) → abstract schemas
- **Computational**: Different representations serve different purposes:
  - Episodic: Fast encoding, detailed recall, context-specific
  - Consolidated: Efficient storage, semantic retrieval, cross-context
  - Schemas: Generalization, transfer learning, pattern recognition

**Alternative Considered**: Single unified memory store with metadata
**Why Rejected**: Harder to implement differential consolidation, decay, and retrieval strategies

---

## 2. Prioritized Replay Mechanism

### Multi-Factor Priority Score

**Decision**: Priority = f(recency, importance, novelty, access_count)

**Rationale**:
- **Recency**: Exponential decay matches biological memory curves (Ebbinghaus)
- **Importance**: Emotionally salient events are preferentially consolidated (Cahill & McGaugh, 1998)
- **Novelty**: Novel information gets priority (hippocampal pattern separation)
- **Access**: Repeated retrieval strengthens memories (testing effect)

**Weights**: Default (0.3, 0.4, 0.3, 0.1)
- Importance weighted highest (most reliable predictor of consolidation)
- Recency and novelty balanced
- Access bonus prevents neglect of useful memories

**Alternative Considered**: Random sampling or purely recency-based
**Why Rejected**: Doesn't capture biological principles; misses important/novel information

---

## 3. LLM-Based Generative Compression

### Why LLM for Compression?

**Decision**: Use Gemini for episode compression rather than extractive methods

**Rationale**:
- **Abstraction**: LLMs can identify high-level themes, not just keywords
- **Integration**: Can combine multiple episodes semantically
- **Concept Extraction**: Better at identifying entities and relationships
- **Flexibility**: Natural language summaries are human-readable

**Trade-offs**:
- ✅ Higher quality, semantic compression
- ✅ Captures nuance and relationships
- ❌ Requires API calls (cost)
- ❌ Slower than rule-based methods
- ❌ Non-deterministic (mitigated by low temperature)

**Alternative Considered**: TF-IDF, TextRank, or other extractive summarization
**Why Rejected**: Poor at abstraction and semantic integration

---

## 4. Schema Formation Strategy

### Concept-Based Grouping

**Decision**: Form schemas by grouping memories with shared concepts

**Rationale**:
- **Simple**: Easy to implement and understand
- **Effective**: Captures basic patterns across memories
- **Scalable**: Works with growing memory stores
- **Biological Parallel**: Similar to how cortex extracts statistical regularities

**Current Implementation**: Pairwise concept overlap → group formation

**Limitations**:
- Doesn't capture hierarchical relationships
- May miss abstract patterns
- Limited to explicit concept overlap

**Future Enhancement**: Could use LLM-based pattern extraction or embedding clustering

---

## 5. Forgetting/Decay Mechanism

### Age + Importance-Based Decay

**Decision**: Forget unconsolidated, low-importance episodes after 30 days

**Rationale**:
- **Synaptic Homeostasis**: Prevents memory accumulation from overwhelming system
- **Value-Based**: Preserves important information (high importance or consolidated)
- **Resource Management**: Bounded memory usage in long-running systems

**Parameters**:
- Threshold: 30 days (configurable)
- Protection: Consolidated episodes never decay
- Importance cutoff: <0.3 importance more likely to decay

**Alternative Considered**: Random decay or FIFO
**Why Rejected**: Would lose valuable information indiscriminately

---

## 6. Sleep Cycle Phases

### Four-Phase Consolidation

**Decision**: Replay → Compression → Schema Formation → Decay

**Rationale**:
- **Phase 1 (Replay)**: Select candidates (mirrors hippocampal replay during SWS)
- **Phase 2 (Compression)**: Transform to semantic form (neocortical integration)
- **Phase 3 (Schema Formation)**: Extract patterns (abstraction and generalization)
- **Phase 4 (Decay)**: Prune low-value memories (synaptic homeostasis)

**Sequential vs Parallel**: Currently sequential for clarity
**Could Parallelize**: Phases 2 and 4 could run concurrently in future

---

## 7. Episode Metadata

### Importance, Novelty, Access Tracking

**Decision**: Store importance, novelty, access_count, tags with each episode

**Rationale**:
- **Importance**: User-specified or auto-estimated (query length, markers)
- **Novelty**: Computed from concept overlap with existing memories
- **Access Count**: Tracks retrieval frequency (strengthening effect)
- **Tags**: Enable categorical organization and retrieval

**Auto-Estimation Heuristics**:
- Question marks → +0.1 importance
- Long inputs (>100 chars) → +0.1 importance
- Novel concepts → high novelty score

**Could Improve**: Use LLM to estimate importance/novelty more accurately

---

## 8. Memory Retrieval

### Concept-Based Search

**Decision**: Retrieve memories by extracting concepts from query and matching

**Rationale**:
- **Semantic**: Matches meaning, not just keywords
- **Fast**: Simple set intersection (O(n) with n memories)
- **LLM Integration**: Uses same LLM for concept extraction

**Current Approach**:
1. Extract concepts from query (LLM)
2. Search consolidated memories by concept overlap
3. Search schemas by concept overlap
4. Return top-k matches

**Limitations**:
- No vector embeddings (could add for semantic similarity)
- No attention to recency in retrieval (could weight recent memories)
- No explicit relevance ranking beyond concept overlap

**Future Enhancement**: Add embedding-based semantic search (e.g., using sentence-transformers)

---

## 9. Consolidation Batching

### Single-Episode Consolidation (Default)

**Decision**: Compress episodes individually by default

**Rationale**:
- **Simple**: Each episode → one consolidated memory
- **Transparent**: Easy to trace source episodes
- **Works Well**: Gemini can compress single episodes effectively

**Alternative**: Batch-compress related episodes together
**Trade-off**: Better integration but harder to implement grouping logic

**Hybrid Approach**: Current system supports both (see `compress_episode_batch`)

---

## 10. Persistence Strategy

### JSON Serialization

**Decision**: Save/load memory stores as JSON files

**Rationale**:
- **Human-Readable**: Can inspect saved memories
- **Simple**: No database dependency
- **Portable**: Works across systems

**Trade-offs**:
- ✅ Easy debugging and inspection
- ✅ No infrastructure required
- ❌ Not suitable for very large memory stores
- ❌ No concurrent access support

**Alternative**: SQLite, PostgreSQL, or vector database
**When to Switch**: If memory stores exceed ~10K episodes or need concurrent access

---

## 11. LLM Temperature Settings

### Different Temperatures for Different Tasks

**Decision**: 
- Compression: 0.3 (deterministic)
- Agent responses: 0.7 (creative)
- Concept extraction: 0.0 (most deterministic)

**Rationale**:
- **Compression**: Want consistent, reliable summaries
- **Conversation**: Want natural, varied responses
- **Extraction**: Want precise, reproducible concept lists

---

## 12. Auto-Sleep Threshold

### Default: 10 Interactions

**Decision**: Trigger consolidation after 10 new episodes

**Rationale**:
- **Balance**: Not too frequent (cost) or rare (forget before consolidation)
- **Batch Size**: Gives enough episodes for pattern detection
- **User Experience**: Noticeable consolidation effect

**Recommendations by Use Case**:
- Real-time chatbot: 5-7 interactions
- Batch processing: 20-50 interactions
- Resource-constrained: 3-5 interactions

---

## Summary of Key Principles

1. **Biological Inspiration First**: Design follows neuroscience principles
2. **Modular Architecture**: Each component has clear responsibility
3. **Configurable**: Parameters exposed for different use cases
4. **Research-Oriented**: Clear assumptions, documented trade-offs
5. **Extensible**: Easy to add multimodal, graph-based, or other extensions
6. **Practical**: Works with current LLM APIs, manageable costs

## References

- **Squire & Alvarez (1995)**: Retrograde amnesia and memory consolidation
- **Born & Wilhelm (2012)**: System consolidation during sleep
- **Tononi & Cirelli (2014)**: Synaptic homeostasis hypothesis
- **Cahill & McGaugh (1998)**: Emotional memory enhancement
- **Mnih et al. (2015)**: Experience replay in DQN
