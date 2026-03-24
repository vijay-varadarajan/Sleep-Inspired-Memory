# PersonaChat Workflow

## Dataset Format

Source: `PERSONACHAT/train/` and `PERSONACHAT/validation/` (HuggingFace `load_from_disk`).

Each row has two fields:

```json
{
  "personality": [
    "i read twenty books a year .",
    "i'm a stunt double as my second job .",
    "i only eat kosher .",
    "i was raised in a single parent household ."
  ],
  "utterances": [
    {
      "history": ["hello what are doing today ?"],
      "candidates": [
        "maybe he can skydive to see a better view",
        "yes , my favorite is broccoli and tofu ...",
        "i am good , i just got off work and tired , i have two jobs ."
      ]
    }
  ]
}
```

Each row has multiple utterances (turns). `candidates` lists distractor responses; **the last candidate is the correct response**.

---

## Preprocessing

```bash
python personachat_preprocessing.py
```

Processes both `train` and `validation` splits. For each row, iterates over all utterances and emits one sample per turn.

**Mapping logic:**
- `query` ← last item in `history`
- `correct_answer` ← last item in `candidates`
- `incorrect_answers` ← remaining candidates (up to 5)
- `persona` ← `personality` lines joined as `"- trait"` bullets
- `related_conversation_snippet` ← `history[:-1]` joined (prior turns before the query)
- `topic_query` ← first 80 chars of `query`
- `persona_id` ← row index in the dataset

**Processed sample format:**

```json
{
  "id": "validation_p0_t0",
  "persona_id": 0,
  "query": "hello what are doing today ?",
  "correct_answer": "i am good , i just got off work and tired , i have two jobs .",
  "incorrect_answers": ["maybe he can skydive ...", "yes , my favorite ..."],
  "persona": "- i read twenty books a year .\n- i'm a stunt double as my second job .",
  "expanded_persona": ["i read twenty books a year .", "i'm a stunt double ..."],
  "related_conversation_snippet": "",
  "topic_query": "hello what are doing today ?"
}
```

**Output files** (`PERSONACHAT/preprocessed/`):

| File | Description |
|---|---|
| `train_processed.json` | Flat list of all turn-level samples from train |
| `train_persona_sessions.json` | Samples grouped by `persona_id` |
| `validation_processed.json` | Same for validation |
| `validation_persona_sessions.json` | Same for validation |
| `benchmark_processed.json` | Alias of `validation_processed.json` |
| `benchmark_persona_sessions.json` | Alias of `validation_persona_sessions.json` |
| `preprocessing_summary.json` | Sample counts per split |

---

## Running the Benchmark

```bash
python personachat_runner.py --split validation --num_samples 200 --methods all
```

**Arguments:**

| Argument | Default | Options |
|---|---|---|
| `--split` | `validation` | `train`, `validation`, `benchmark` |
| `--num_samples` | `200` | Any int |
| `--methods` | `all` | `vanilla rag episodic summarization sleep` or `all` |
| `--output_dir` | `personachat_results` | Any path |

**Table 1** — runs all specified methods. Per persona (up to 3 samples per persona):
1. `agent.interact(query, use_memory=(idx > 0))`
2. Evaluates: Long-Horizon QA, Multi-Session Continuity (if prior context exists), Hallucination Rate
3. Sleep agent calls `agent.sleep()` after each persona

**Table 2** — sleep method only. Per persona (2 samples):
1. Ingest both samples without memory
2. Run 4 probes **pre-consolidation**: Delayed Recall, Cue-Based Recall, Cross-Episode Integration, Schema Utilization
3. Call `agent.sleep()`
4. Repeat all 4 probes **post-consolidation**

---

## Result Format

**Raw JSON** (`personachat_results/personachat_results_<timestamp>.json`):

```json
{
  "timestamp": "20260131_111250",
  "dataset": "PERSONACHAT",
  "split": "validation",
  "num_samples": 200,
  "methods": ["vanilla", "rag", "episodic", "summarization", "sleep"],
  "table1": [
    {
      "method": "vanilla",
      "long_horizon_qa": 45.0,
      "multi_session_continuity": 38.2,
      "hallucination_rate": 0.12,
      "num_qa_samples": 60,
      "num_continuity_samples": 40,
      "num_hallucination_samples": 60
    }
  ],
  "table2": [
    {
      "method": "sleep",
      "applicable": true,
      "delayed_recall_pre": 52.0,
      "delayed_recall_post": 68.0,
      "delayed_recall_improvement": 16.0,
      "cue_based_pre": 55.0,
      "cue_based_post": 70.0,
      "cue_based_improvement": 15.0,
      "integration_pre": 40.0,
      "integration_post": 58.0,
      "integration_improvement": 18.0,
      "schema_util_pre": 48.0,
      "schema_util_post": 62.0,
      "schema_util_improvement": 14.0
    }
  ]
}
```

**Table 1 CSV** (`personachat_results/results_table_1.csv`):

```
Method,Long-Horizon QA,Multi-Session Continuity,Hallucination Rate
vanilla,45.0000,38.2000,0.120000
rag,51.0000,42.5000,0.095000
...
```

**Table 2 CSV** (`personachat_results/results_table_2.csv`):

```
Probe,Pre-Consolidation,Post-Consolidation,Delta Improvement
Delayed Recall Accuracy,52.0000,68.0000,16.0000
Cue-Based Recall,55.0000,70.0000,15.0000
Cross-Episode Integration,40.0000,58.0000,18.0000
Schema Utilization Rate,48.0000,62.0000,14.0000
```

To regenerate CSVs from an existing JSON without re-running the benchmark:

```bash
python personachat_postprocessing.py --output_dir personachat_results
# or point to a specific file:
python personachat_postprocessing.py --results_json personachat_results/personachat_results_<timestamp>.json
```
