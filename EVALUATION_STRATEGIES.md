# Evaluation Strategies

## Benchmarking Framework

**Table 1: Task-Based Performance** (5 methods × 3 metrics)
- **Long-Horizon QA**: Semantic accuracy via LLM-as-judge
- **Multi-Session Continuity**: Correct reference to prior sessions
- **Hallucination Rate**: Unsupported claims per 100 responses

**Table 2: Cognitive Probes** (sleep method before/after)
- **Delayed Recall**: Memory retention after time delay
- **Cue-Based Recall**: Improvement with conversation cues
- **Cross-Episode Integration**: Connect info across sessions
- **Schema Utilization**: Use of persona patterns

## Evaluation Setup
- **Dataset**: PersonaMem-v2 (5,000 samples, 200 personas)
- **Protocol**: Multi-session interactions (5 per persona, first without memory, rest with memory)
- **Scoring**: LLM-as-judge (Gemini, temp=0.0, deterministic)
- **Aggregation**: Average across samples

## Expected Results
- Table 1: Sleep method highest multi-session continuity
- Table 2: Sleep method shows positive delta on cue-based recall & schema utilization

## Run Commands
```bash
# Quick test (20 samples)
python benchmark_runner.py --split benchmark --num_samples 20 --methods all

# Full benchmark (100 samples)
python benchmark_runner.py --split benchmark --num_samples 100 --methods all
```
