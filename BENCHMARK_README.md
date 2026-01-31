# PersonaMem Benchmark for Sleep-Inspired Memory Consolidation

This repository contains a comprehensive benchmarking system for evaluating sleep-inspired memory consolidation against baseline methods using the PersonaMem-v2 dataset.

## Overview

The system evaluates memory consolidation approaches across two main dimensions:

### Table 1: Task-Based Memory Performance
Compares different memory methods on practical tasks:
- **Long-Horizon QA**: Accuracy on multi-turn question answering
- **Multi-Session Continuity**: Ability to reference prior conversation sessions
- **Hallucination Rate**: Unsupported claims per 100 responses

**Methods Compared:**
1. Vanilla LLM (no memory)
2. RAG (Vector Database retrieval)
3. Episodic Only (episodic memory without consolidation)
4. Episodic + Summarization (basic summarization)
5. Ours: Sleep-Consolidated (full biologically-inspired consolidation)

### Table 2: Cognitive-Style Probes (Before vs After Sleep)
Validates biological inspiration by measuring changes after consolidation:
- **Delayed Recall Accuracy**: Recall after time delay
- **Cue-Based Recall**: Improvement when given memory cues
- **Cross-Episode Integration**: Connecting information across sessions
- **Schema Utilization Rate**: Use of learned persona schemas

## Setup

### Prerequisites
```bash
# Python 3.10+ required
python --version

# Install dependencies
pip install -r requirements.txt
```

### Environment Configuration
Create a `.env` file with your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```

## Dataset Preprocessing

Before running benchmarks, preprocess the PersonaMem dataset:

```bash
python personamem_preprocessing.py
```

This will:
- Load benchmark_text, train_text, and val_text splits
- Parse and clean the data
- Group samples by persona_id for multi-session testing
- Create evaluation subsets
- Save preprocessed data to `PERSONAMEM/preprocessed/`

**Output:**
- `benchmark_processed.json` - Processed benchmark samples
- `benchmark_persona_sessions.json` - Samples grouped by persona
- `benchmark_evaluation_subsets.json` - Specialized subsets for different tests
- Similar files for train and val splits

## Running Benchmarks

### Quick Test
Test that everything works:
```bash
python test_benchmark.py
```

### Run Small Test
Test with a small sample (faster):
```bash
python benchmark_runner.py --split benchmark --num_samples 20 --methods sleep
```

### Run Full Benchmark
Evaluate all methods on the benchmark split:
```bash
python benchmark_runner.py --split benchmark --num_samples 100 --methods all
```

### Run Specific Method
Test only the sleep-consolidated approach:
```bash
python benchmark_runner.py --split benchmark --num_samples 50 --methods sleep
```

### Run Multiple Specific Methods
```bash
python benchmark_runner.py --split benchmark --num_samples 100 --methods vanilla rag sleep
```

### Command-Line Arguments
- `--split`: Dataset split to use (`benchmark`, `val`, `train`)
- `--num_samples`: Number of samples to evaluate (default: 100)
- `--methods`: Methods to evaluate (options: `vanilla`, `rag`, `episodic`, `summarization`, `sleep`, `all`)
- `--output_dir`: Directory to save results (default: `results`)

## Output

Results are saved to the `results/` directory with timestamps:

```
results/
├── benchmark_results_20240131_143022.json
└── .csv
```

### Sample Output Format

**Table 1:**
```
======================================================================
TABLE 1: Task-Based Memory Performance
======================================================================
Method                    Long-Horizon QA ↑     Multi-Session Continuity ↑     Hallucination Rate ↓     
----------------------------------------------------------------------
vanilla                    45.23%                32.15%                         8.42
rag                        52.67%                41.33%                         7.18
episodic                   58.91%                48.22%                         6.85
summarization              63.45%                53.78%                         5.92
sleep                      71.28%                64.51%                         4.33
======================================================================
```

**Table 2:**
```
======================================================================
TABLE 2: Cognitive-Style Probes (Before vs After Sleep)
======================================================================
Probe                          Pre-Consolidation     Post-Consolidation    Δ Improvement      
----------------------------------------------------------------------
Delayed Recall Accuracy        52.34%                67.82%                +15.48%
Cue-Based Recall               61.45%                73.29%                +11.84%
Cross-Episode Integration      43.67%                58.91%                +15.24%
Schema Utilization Rate        48.23%                62.45%                +14.22%
======================================================================
```

## Architecture

### Key Components

1. **`personamem_preprocessing.py`**: Dataset preprocessing pipeline
   - Loads and cleans PersonaMem data
   - Groups by persona for multi-session testing
   - Creates specialized evaluation subsets

2. **`evaluation/baselines.py`**: Baseline method implementations
   - `VanillaLLM`: No memory system
   - `RAGBaseline`: Vector database retrieval
   - `EpisodicOnlyAgent`: Episodic memory only
   - `EpisodicSummarizationAgent`: Basic summarization
   - `SleepConsolidatedAgent`: Full sleep consolidation

3. **`evaluation/personamem_benchmark.py`**: Evaluation metrics
   - `PersonaMemEvaluator`: Implements all evaluation functions
   - Long-Horizon QA evaluation
   - Multi-Session Continuity scoring
   - Hallucination detection
   - Cognitive probe measurements

4. **`benchmark_runner.py`**: Main orchestration
   - Runs experiments across all methods
   - Manages persona-based multi-session interactions
   - Aggregates results
   - Generates formatted tables

5. **Existing Memory System**: Core consolidation logic
   - `agent/agent.py`: Memory agent with sleep cycles
   - `memory/episodic.py`: Episodic memory store
   - `memory/consolidated.py`: Consolidated memory store
   - `memory/schema.py`: Schema induction
   - `sleep/consolidation.py`: Sleep-based consolidation
   - `sleep/compression.py`: LLM-based memory compression

## Workflow

```
PersonaMem Dataset
       ↓
personamem_preprocessing.py
       ↓
Preprocessed Data (PERSONAMEM/preprocessed/)
       ↓
benchmark_runner.py
       ↓
   ┌─────────────────────────────────┐
   │  For each method & persona:     │
   │  1. Load persona samples        │
   │  2. Simulate multi-session      │
   │  3. Run sleep (if applicable)   │
   │  4. Evaluate metrics            │
   └─────────────────────────────────┘
       ↓
Results (JSON + Formatted Tables)
```

## Extending the System

### Adding New Datasets
1. Create a new preprocessing file (e.g., `dataset_name_preprocessing.py`)
2. Follow the same output format as `personamem_preprocessing.py`
3. Ensure preprocessed data has the required fields:
   - `query`, `correct_answer`, `persona`, `related_conversation_snippet`
4. Update `benchmark_runner.py` to load from the new dataset

### Adding New Metrics
1. Add evaluation method to `evaluation/personamem_benchmark.py`
2. Update `PersonaMemEvaluator` class with new metric
3. Integrate into `benchmark_runner.py` evaluation loops

### Adding New Baselines
1. Create new agent class in `evaluation/baselines.py`
2. Implement `interact()` and `get_memory_summary()` methods
3. Add to `create_agent()` factory function
4. Include in `benchmark_runner.py` methods list

## Performance Considerations

### API Costs
- Each evaluation makes multiple LLM calls
- Reduce `--num_samples` for testing
- Consider using `gemini-flash` instead of `gemini-pro`

### Runtime
- Full benchmark (100 samples, all methods) takes ~30-60 minutes
- Use `--methods sleep` to test only the main approach
- Consider running on smaller val split first

### Memory
- FAISS vector store (RAG baseline) scales with interactions
- Consider limiting episodes per persona for large-scale tests

## Troubleshooting

### "GOOGLE_API_KEY not found"
- Create `.env` file with your API key
- Or export: `export GOOGLE_API_KEY=your_key`

### "No module named 'langchain_community'"
```bash
pip install langchain-community --upgrade
```

### "Dataset not found"
- Run `personamem_preprocessing.py` first
- Ensure `PERSONAMEM/benchmark_text` directory exists

### Out of Memory
- Reduce `--num_samples`
- Use `--methods` to test one method at a time

## Citation

If you use this benchmarking system, please cite:

```bibtex
@misc{personamem-sleep-benchmark,
  title={PersonaMem Benchmark for Sleep-Inspired Memory Consolidation},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/sleep-inspired-memory}
}
```

## Cost Estimation

### Per Sample (approximate)
- Preprocessing: Free (one-time)
- Vanilla LLM: 1-2 API calls
- RAG: 2-3 API calls (interaction + embedding)
- Episodic: 2-3 API calls
- Summarization: 3-5 API calls (includes summarization)
- Sleep: 5-10 API calls (includes consolidation)
- Evaluation: 3-5 API calls per metric

### Full Benchmark (100 samples, all methods)
- Estimated: 2,000-3,000 API calls
- Cost: ~$0.50-$1.50 (using Gemini Flash)
- Time: 30-60 minutes

## License

MIT License - See LICENSE file for details

## Acknowledgments

- PersonaMem-v2 dataset by Bowen et al.
- Inspired by biological sleep-dependent memory consolidation
- Built with LangChain and Google Gemini API
