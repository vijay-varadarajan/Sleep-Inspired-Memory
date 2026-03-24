# PersonaMem Workflow

## Dataset Format

Source: `PERSONAMEM/benchmark_text/`, `PERSONAMEM/train_text/`, `PERSONAMEM/val_text/` (HuggingFace `load_from_disk`).

Each row stores several JSON-string fields alongside plain-text fields:

```
user_query            → JSON string: {"role": "user", "content": "<question text>"}
short_persona         → JSON string: {"persona": "<persona description>"}
expanded_persona      → JSON string: {"traits": [...]}
related_conversation_snippet → JSON string: [list of prior turns]
correct_answer        → plain string
incorrect_answers     → JSON string / list
persona_id            → int
topic_query           → string
preference            → string
topic_preference      → string
conversation_scenario → string
pref_type             → string (e.g. "anti_stereotypical_pref")
who                   → string
updated               → string
prev_pref             → string
```

A single raw sample looks like:

```json
{
  "persona_id": 521,
  "user_query": "{\"role\": \"user\", \"content\": \"What are some calming morning routines...\"}",
  "short_persona": "{\"persona\": \"A liberal Democrat from Kansas...\"}",
  "correct_answer": "Since you already practice yoga and meditation...",
  "incorrect_answers": "[\"Since you enjoy experimenting with new vegetarian recipes...\", ...]",
  "topic_query": "Health",
  "conversation_scenario": "personal_email",
  "pref_type": "anti_stereotypical_pref"
}
```

---

## Preprocessing

```bash
python personamem_preprocessing.py
```

Processes `benchmark_text`, `train_text`, and `val_text` splits.

**Mapping logic:**
- All JSON-string fields (`user_query`, `short_persona`, `expanded_persona`, `related_conversation_snippet`) are parsed via `json.loads` / `ast.literal_eval`
- `query` ← `user_query["content"]` (or the string directly if already plain)
- `persona` ← `short_persona["persona"]`
- `incorrect_answers` ← parsed list from the JSON string
- All remaining metadata fields are preserved as-is

**Processed sample format:**

```json
{
  "id": "sample_0",
  "persona_id": 521,
  "query": "What are some calming morning routines that can help set a positive tone for the rest of the day?",
  "correct_answer": "Since you already practice yoga and meditation daily...",
  "incorrect_answers": ["Since you enjoy experimenting with new vegetarian recipes..."],
  "persona": "A liberal Democrat from Kansas, slightly skeptical about Republican candidates...",
  "expanded_persona": {"traits": ["..."]},
  "related_conversation_snippet": [...],
  "topic_query": "Health",
  "preference": "...",
  "topic_preference": "...",
  "conversation_scenario": "personal_email",
  "pref_type": "anti_stereotypical_pref",
  "who": "...",
  "updated": "...",
  "prev_pref": "...",
  "distance_from_related_snippet_to_query_32k": 0,
  "num_persona_relevant_tokens_32k": 0,
  "num_persona_irrelevant_tokens_32k": 0
}
```

**Output files** (`PERSONAMEM/preprocessed/`):

| File | Description |
|---|---|
| `benchmark_processed.json` | Flat list of samples from `benchmark_text` |
| `benchmark_persona_sessions.json` | Samples grouped by `persona_id` |
| `train_processed.json` | Same for `train_text` |
| `train_persona_sessions.json` | Same for `train_text` |
| `val_processed.json` | Same for `val_text` |
| `val_persona_sessions.json` | Same for `val_text` |

---

## Running the Benchmark

```bash
python benchmark_runner.py --split benchmark --num_samples 100 --methods all
```

**Arguments:**

| Argument | Default | Options |
|---|---|---|
| `--split` | `benchmark` | `benchmark`, `val`, `train` |
| `--num_samples` | `100` | Any int |
| `--methods` | all | `vanilla rag episodic summarization sleep` or `all` |
| `--output_dir` | `results` | Any path |

**Table 1** — runs all specified methods. Caps at 10 personas; 2 interactions per persona:
1. `agent.interact(query, use_memory=(i > 0))`
2. Evaluates: Long-Horizon QA (every interaction), Multi-Session Continuity (when `i > 0` and context exists), Hallucination Rate (every other interaction, `i % 2 == 0`)
3. 0.5 s delay between each evaluator call
4. Sleep agent calls `agent.sleep()` after each persona

**Table 2** — skipped by default (`skip_table2=True`). When enabled, runs sleep method only, with the same 4 cognitive probes as PersonaChat Table 2, using up to 3 personas × 2 samples.

---

## Result Format

**Raw JSON** (`results/benchmark_results_<timestamp>.json`):

```json
{
  "timestamp": "20260131_111250",
  "split": "benchmark",
  "num_samples": 100,
  "methods": ["vanilla", "rag", "episodic", "summarization", "sleep"],
  "table1": [
    {
      "method": "vanilla",
      "long_horizon_qa": 45.0,
      "multi_session_continuity": 38.2,
      "hallucination_rate": 0.12,
      "num_qa_samples": 20,
      "num_continuity_samples": 10,
      "num_hallucination_samples": 10
    }
  ],
  "table2": [
    {"method": "sleep", "applicable": false, "skipped": true}
  ]
}
```

**Table 1 CSV** (`results/table1_<timestamp>.csv`):

```
Method,Long-Horizon QA,Multi-Session Continuity,Hallucination Rate
vanilla,45.0000,38.2000,0.120000
rag,51.0000,42.5000,0.095000
...
```

**Table 2 CSV** (`results/table2_<timestamp>.csv`) — only written when Table 2 is enabled and applicable:

```
Probe,Pre-Consolidation,Post-Consolidation,Delta Improvement
Delayed Recall Accuracy,52.0000,68.0000,16.0000
Cue-Based Recall,55.0000,70.0000,15.0000
Cross-Episode Integration,40.0000,58.0000,18.0000
Schema Utilization Rate,48.0000,62.0000,14.0000
```
