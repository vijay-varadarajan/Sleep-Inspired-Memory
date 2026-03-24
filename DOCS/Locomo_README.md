# LOCOMO Workflow

## Dataset Format

Source: `LOCOMO/locomo10.json` — a list of **10 conversation records**.

Each record has the following top-level structure:

```json
{
  "sample_id": "conv-26",
  "qa": [...],
  "conversation": {...}
}
```

**`qa`** — list of question/answer items (≈60–200 per record):

```json
{
  "question": "When did Caroline go to the LGBTQ support group?",
  "answer": "7 May 2023",
  "evidence": ["D1:3"],
  "category": 2,
  "adversarial_answer": "14 May 2023"   // optional
}
```

**`conversation`** — multi-session dialog:

```json
{
  "speaker_a": "Caroline",
  "speaker_b": "Melanie",
  "session_1": [
    {"speaker": "Caroline", "dia_id": "D1:1", "text": "Hey Mel! Good to see you!"},
    {"speaker": "Melanie",  "dia_id": "D1:2", "text": "Great to see you too!"}
  ],
  "session_1_date_time": "...",
  "session_summary": {
    "session_1": "Caroline and Melanie met at a coffee shop..."
  }
}
```

`dia_id` format: `"D<session>:<turn>"` (e.g. `D1:3` = session 1, turn 3).

---

## Preprocessing

```bash
python locomo_preprocessing.py
```

Input: `LOCOMO/locomo10.json`. Each of the 10 records becomes one `persona_id`.

**Mapping logic:**
- A `dia_id → text` index is built from every `session_N` list in the record's `conversation`
- `related_conversation_snippet` ← evidence `dia_id`s resolved to `"D1:3: <text>"` lines, joined with newlines
- `incorrect_answers` ← `adversarial_answer` first (if present), then answers from other QA items in the same record (up to 5 total)
- `persona` ← `"Primary speakers: A and B."` + first 3 `session_summary` values as bullet points
- `expanded_persona` ← `[speaker_a, speaker_b]`
- `category` and `evidence` fields are preserved as-is from the QA item

**Processed sample format:**

```json
{
  "id": "conv-26_q0",
  "persona_id": 0,
  "query": "When did Caroline go to the LGBTQ support group?",
  "correct_answer": "7 May 2023",
  "incorrect_answers": ["14 May 2023", "12 June 2023"],
  "persona": "Primary speakers: Caroline and Melanie.\nContext summaries:\n- Caroline and Melanie met at a coffee shop...",
  "expanded_persona": ["Caroline", "Melanie"],
  "related_conversation_snippet": "D1:3: Caroline mentioned she went to the LGBTQ support group.",
  "topic_query": "When did Caroline go to the LGBTQ support group?",
  "category": 2,
  "evidence": ["D1:3"],
  "source_sample_id": "conv-26"
}
```

**Output files** (`LOCOMO/preprocessed/`):

| File | Description |
|---|---|
| `benchmark_processed.json` | Flat list of all QA samples across all 10 records |
| `benchmark_persona_sessions.json` | Samples grouped by `persona_id` (0–9) |
| `preprocessing_summary.json` | Record/sample/persona counts |

---

## Running the Benchmark

```bash
python locomo_runner.py --split benchmark --num_samples 200 --methods all
```

**Arguments:**

| Argument | Default | Options |
|---|---|---|
| `--split` | `benchmark` | `benchmark` (only split produced by preprocessing) |
| `--num_samples` | `200` | Any int; LOCOMO has ~600 total samples across 10 personas |
| `--methods` | `all` | `vanilla rag episodic summarization sleep` or `all` |
| `--output_dir` | `locomo_results` | Any path |

**Table 1** — same structure as PersonaChat runner: up to 3 samples per persona, `use_memory=(idx > 0)`, evaluates QA / Continuity / Hallucination. Sleep agent calls `agent.sleep()` after each persona.

**Table 2** — sleep method only, 2 samples per persona. Pre/post consolidation probes: Delayed Recall, Cue-Based Recall, Cross-Episode Integration, Schema Utilization.

---

## Result Format

**Raw JSON** (`locomo_results/locomo_results_<timestamp>.json`):

```json
{
  "timestamp": "20260131_111250",
  "dataset": "LOCOMO",
  "split": "benchmark",
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

**Table 1 CSV** (`locomo_results/results_table_1.csv`):

```
Method,Long-Horizon QA,Multi-Session Continuity,Hallucination Rate
vanilla,45.0000,38.2000,0.120000
rag,51.0000,42.5000,0.095000
...
```

**Table 2 CSV** (`locomo_results/results_table_2.csv`):

```
Probe,Pre-Consolidation,Post-Consolidation,Delta Improvement
Delayed Recall Accuracy,52.0000,68.0000,16.0000
Cue-Based Recall,55.0000,70.0000,15.0000
Cross-Episode Integration,40.0000,58.0000,18.0000
Schema Utilization Rate,48.0000,62.0000,14.0000
```

To regenerate CSVs from an existing JSON:

```bash
python locomo_postprocessing.py --output_dir locomo_results
# or point to a specific file:
python locomo_postprocessing.py --results_json locomo_results/locomo_results_<timestamp>.json
```
