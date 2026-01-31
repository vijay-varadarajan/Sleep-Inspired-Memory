# Getting Started Guide

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Key

Create a `.env` file in the project root:

```bash
GOOGLE_API_KEY=your_gemini_api_key_here
```

Get your API key from: https://makersuite.google.com/app/apikey

## Running the System

### Option 1: Run the Full Demo

```bash
python main.py
```

This will demonstrate:
- Basic interactions and memory storage
- A complete sleep cycle
- Memory recall after consolidation
- Memory persistence (save/load)

### Option 2: Interactive Usage

```python
from agent.agent import MemoryAgent

# Initialize agent
agent = MemoryAgent(
    api_key="your_key",  # Or set GOOGLE_API_KEY env var
    auto_sleep_threshold=10  # Auto-sleep after 10 interactions
)

# Have a conversation
response1 = agent.interact(
    "Teach me about neural networks",
    importance=0.8,  # High importance
    tags=["ml", "neural-networks"],
    use_memory=False  # Don't use memory yet (first interaction)
)
print(response1)

response2 = agent.interact(
    "What is backpropagation?",
    importance=0.7,
    tags=["ml", "neural-networks"],
    use_memory=True  # Now use any existing memories
)
print(response2)

# Manually trigger sleep consolidation
print("\n[Entering sleep mode...]")
stats = agent.sleep(verbose=True)
print(f"Consolidated {stats['memories_consolidated']} memories")

# Test memory recall
response3 = agent.interact(
    "What have we discussed about machine learning?",
    importance=0.6,
    tags=["ml"],
    use_memory=True
)
print(response3)

# Check memory statistics
summary = agent.get_memory_summary()
print(f"\nMemory Summary:")
print(f"  Total interactions: {summary['interactions']}")
print(f"  Episodic memories: {summary['episodic']['total_episodes']}")
print(f"  Consolidated memories: {summary['consolidated']['total_memories']}")
print(f"  Schemas: {summary['schemas']['total_schemas']}")
```

### Option 3: Run Tests

```bash
python -m evaluation.tests
```

## Understanding Memory States

### Before Sleep
- **Episodic memories**: Raw interactions stored as-is
- **Consolidated memories**: 0 (nothing consolidated yet)
- **Schemas**: 0 (no patterns identified)

### After Sleep
- **Episodic memories**: Marked as consolidated (preserved)
- **Consolidated memories**: Compressed summaries created
- **Schemas**: Patterns identified across related memories

## Key Parameters

### Episode Importance (0-1)
- **0.1-0.3**: Low importance (casual chat, small talk)
- **0.4-0.6**: Medium importance (general queries)
- **0.7-0.9**: High importance (key information, learning)
- **0.9-1.0**: Critical importance (must remember)

### Auto-Sleep Threshold
- Default: 10 interactions
- Higher values: Less frequent consolidation (may miss patterns)
- Lower values: More frequent consolidation (higher API costs)

### Replay Weights
```python
from sleep.replay import select_episodes_for_replay

selected = select_episodes_for_replay(
    episodes=episodes,
    recency_weight=0.3,    # Weight for recent memories
    importance_weight=0.4,  # Weight for important memories
    novelty_weight=0.3,     # Weight for novel information
    access_bonus=0.1        # Bonus for accessed memories
)
```

## Saving and Loading Memories

```python
# Save current memory state
agent.save_memories("my_memories")

# Later, in a new session
agent2 = MemoryAgent()
agent2.load_memories("my_memories")
# All memories restored!
```

## Advanced: Custom Sleep Cycles

```python
from sleep.consolidation import SleepCycle
from memory.episodic import EpisodicMemoryStore
from memory.consolidated import ConsolidatedMemoryStore
from memory.schema import SchemaStore
from sleep.compression import MemoryCompressor

# Create memory stores
episodic = EpisodicMemoryStore()
consolidated = ConsolidatedMemoryStore()
schemas = SchemaStore()

# Create compressor
compressor = MemoryCompressor(api_key="your_key")

# Create custom sleep cycle
sleep_cycle = SleepCycle(
    episodic_store=episodic,
    consolidated_store=consolidated,
    schema_store=schemas,
    compressor=compressor,
    replay_batch_size=15,        # Replay top 15 episodes
    consolidation_batch_size=3,  # Consolidate in batches of 3
    schema_min_memories=4        # Need 4 memories to form schema
)

# Add some episodes manually
episodic.add_episode("Content 1", importance=0.8)
episodic.add_episode("Content 2", importance=0.6)

# Run consolidation
stats = sleep_cycle.run_sleep_cycle(verbose=True)
```

## Troubleshooting

### API Key Issues
```
Error: Google API key required
```
**Solution**: Set `GOOGLE_API_KEY` environment variable or pass `api_key` to constructor

### Import Errors
```
ModuleNotFoundError: No module named 'langchain'
```
**Solution**: Run `pip install -r requirements.txt`

### Low Quality Consolidation
**Issue**: Memories not being compressed well
**Solution**: 
- Increase `importance` scores for valuable interactions
- Add descriptive `tags` to episodes
- Ensure episodes have substantial content

### Too Many API Calls
**Issue**: High costs from frequent LLM calls
**Solution**:
- Increase `auto_sleep_threshold`
- Reduce `replay_batch_size` in sleep cycle
- Use manual sleep triggering instead of auto-sleep

## Performance Tips

1. **Batch interactions before sleep**: Gather 10-20 interactions before consolidating
2. **Use tags strategically**: Help group related memories
3. **Set importance accurately**: Focus consolidation on valuable memories
4. **Monitor memory stats**: Use `agent.get_memory_summary()` regularly
5. **Save periodically**: Use `agent.save_memories()` to persist state

## Next Steps

- Read the full [README.md](README.md) for architecture details
- Explore [memory/episodic.py](memory/episodic.py) for data structures
- Check [sleep/consolidation.py](sleep/consolidation.py) for consolidation logic
- Review [evaluation/tests.py](evaluation/tests.py) for usage examples
- Extend the system for your specific use case!
