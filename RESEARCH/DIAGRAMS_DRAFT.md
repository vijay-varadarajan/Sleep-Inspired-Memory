# Workflow Diagrams Draft

## 1) Perspective A — System Architecture (Implementation-Oriented)

```mermaid
flowchart LR
    U[User Query] --> A[MemoryAgent\nagent/agent.py]

    subgraph Retrieval[Online Inference Path]
      A --> R[_retrieve_memory_bundles]
      R --> E[EpisodicMemoryStore\nmemory/episodic.py]
      R --> C[ConsolidatedMemoryStore\nmemory/consolidated.py]
      R --> S[SchemaStore\nmemory/schema.py]
      E --> HS[Hybrid Scoring\nsemantic + lexical + recency + evidence]
      C --> HS
      S --> HS
      HS --> P[_build_system_prompt]
      P --> LLM[Gemini / LLM Response Generation]
    end

    LLM --> O[Assistant Response]
    O --> EN[Episode Encoding\nimportance, novelty, salience, persona_relevance, factuality_risk]
    EN --> E

    subgraph Offline[Periodic Sleep Consolidation]
      T[episodes_since_sleep >= tau=4] --> SC[SleepCycle\nsleep/consolidation.py]
      E --> SC
      SC --> RS[Replay Selection\ncalculate_replay_priority]
      RS --> CP[Compression\ncompress_episode_batch]
      CP --> CM[Consolidation\nconflict-aware merge]
      CM --> C
      CM --> SF[Schema Formation\nmerge_schemas]
      SF --> S
      SC --> DY[Decay & Pruning]
      DY --> E
    end
```

---

## 2) Perspective B — Temporal Workflow (Per Turn + Sleep Trigger)

```mermaid
sequenceDiagram
    autonumber
    participant User
    participant Agent as MemoryAgent
    participant Episodic as EpisodicStore
    participant Consolidated as ConsolidatedStore
    participant Schema as SchemaStore
    participant Sleep as SleepCycle
    participant LLM

    User->>Agent: Send prompt / question
    Agent->>Agent: Parse intent and current context

    Agent->>Episodic: Retrieve top episodic candidates
    Agent->>Consolidated: Retrieve consolidated memories
    Agent->>Schema: Retrieve relevant schemas
    Agent->>Agent: Hybrid rank (semantic+lexical+recency+evidence)

    Agent->>LLM: Build prompt + inject memory bundles
    LLM-->>Agent: Generated response
    Agent-->>User: Final answer

    Agent->>Episodic: Store new Episode(turn)
    Agent->>Agent: Increment episodes_since_sleep

    alt episodes_since_sleep >= 4
        Agent->>Sleep: Trigger sleep consolidation
        Sleep->>Episodic: Phase 1: replay selection (top-k)
        Sleep->>LLM: Phase 2: compression of episode batch
        Sleep->>Consolidated: Phase 3: merge compressed memories + contradiction checks
        Sleep->>Schema: Phase 4: schema abstraction / merge
        Sleep->>Episodic: Phase 5: decay + prune low-salience episodes
        Sleep-->>Agent: Updated memory stores
    else threshold not reached
        Agent->>Agent: Continue online turns only
    end
```

---

## 3) Perspective C — Simple Backend Architecture (Vertical)

```mermaid
flowchart TB
  API[Request / Runner Layer\n*_runner.py] --> AG[MemoryAgent\nagent/agent.py]

  AG --> RET[Online Retrieval + Prompt Build]
  RET --> E[(EpisodicMemoryStore\nmemory/episodic.py)]
  RET --> C[(ConsolidatedMemoryStore\nmemory/consolidated.py)]
  RET --> S[(SchemaStore\nmemory/schema.py)]
  RET --> LLM[LLM Inference]
  LLM --> RESP[Response]
  RESP --> ENC[Encode New Episode]
  ENC --> E

  E --> TR{episodes_since_sleep >= 4?}
  TR -- No --> AG
  TR -- Yes --> SC[SleepCycle\nsleep/consolidation.py]

  SC --> RS[1. Replay Selection]
  RS --> CP[2. Compression]
  CP --> CM[3. Consolidation]
  CM --> C
  CM --> SF[4. Schema Formation]
  SF --> S
  SC --> DY[5. Decay / Prune]
  DY --> E
  SC --> AG
```
