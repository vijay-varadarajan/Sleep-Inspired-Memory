"""
Quick Test Script for PersonaMem Benchmark

Tests the benchmark runner with a small sample to ensure everything works.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check API key
if not os.getenv("GOOGLE_API_KEY"):
    print("ERROR: GOOGLE_API_KEY not found in environment")
    print("Please set it in your .env file")
    exit(1)

print("Testing PersonaMem Benchmark System")
print("="*70)

# Test 1: Check preprocessed data
print("\n1. Checking preprocessed data...")
data_dir = Path("PERSONAMEM/preprocessed")
required_files = [
    "benchmark_processed.json",
    "benchmark_persona_sessions.json",
    "benchmark_evaluation_subsets.json"
]

for file in required_files:
    file_path = data_dir / file
    if file_path.exists():
        print(f"   ✓ {file} exists")
    else:
        print(f"   ✗ {file} missing!")

# Test 2: Import modules
print("\n2. Testing imports...")
try:
    from evaluation.baselines import create_agent
    print("   ✓ baselines module imported")
except Exception as e:
    print(f"   ✗ Error importing baselines: {e}")

try:
    from evaluation.personamem_benchmark import PersonaMemEvaluator
    print("   ✓ personamem_benchmark module imported")
except Exception as e:
    print(f"   ✗ Error importing personamem_benchmark: {e}")

try:
    from benchmark_runner import BenchmarkRunner
    print("   ✓ benchmark_runner module imported")
except Exception as e:
    print(f"   ✗ Error importing benchmark_runner: {e}")

# Test 3: Create agents
print("\n3. Testing agent creation...")
methods = ['vanilla', 'rag', 'episodic', 'summarization', 'sleep']
for method in methods:
    try:
        agent = create_agent(method)
        print(f"   ✓ {method} agent created")
    except Exception as e:
        print(f"   ✗ Error creating {method} agent: {e}")

# Test 4: Quick interaction test
print("\n4. Testing agent interaction...")
try:
    agent = create_agent('vanilla')
    response = agent.interact("Hello, how are you?")
    print(f"   ✓ Agent interaction successful")
    print(f"   Response preview: {response[:100]}...")
except Exception as e:
    print(f"   ✗ Error in agent interaction: {e}")

# Test 5: Evaluator test
print("\n5. Testing evaluator...")
try:
    evaluator = PersonaMemEvaluator()
    print(f"   ✓ Evaluator created")
except Exception as e:
    print(f"   ✗ Error creating evaluator: {e}")

print("\n" + "="*70)
print("Test complete!")
print("\nTo run a small test benchmark:")
print("  python benchmark_runner.py --split benchmark --num_samples 20 --methods sleep")
print("\nTo run the full benchmark:")
print("  python benchmark_runner.py --split benchmark --num_samples 100 --methods all")
