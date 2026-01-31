#!/bin/bash

# Quick Start Guide for PersonaMem Benchmark
# This script provides commands to run the benchmark system

echo "=========================================="
echo "PersonaMem Benchmark - Quick Start Guide"
echo "=========================================="
echo ""

echo "1. SETUP"
echo "   source .venv/bin/activate"
echo ""

echo "2. PREPROCESS DATA (run once)"
echo "   python personamem_preprocessing.py"
echo ""

echo "3. TEST INSTALLATION"
echo "   python test_benchmark.py"
echo ""

echo "4. RUN BENCHMARKS"
echo ""
echo "   a) Small test (20 samples, ~5-10 minutes):"
echo "      python benchmark_runner.py --split benchmark --num_samples 20 --methods sleep"
echo ""
echo "   b) All methods, 100 samples (~30-60 minutes):"
echo "      python benchmark_runner.py --split benchmark --num_samples 100 --methods all"
echo ""
echo "   c) Just sleep method, 50 samples (~10-15 minutes):"
echo "      python benchmark_runner.py --split benchmark --num_samples 50 --methods sleep"
echo ""
echo "   d) Specific methods:"
echo "      python benchmark_runner.py --split benchmark --num_samples 100 --methods vanilla sleep"
echo ""

echo "5. RESULTS"
echo "   Results are saved to: results/benchmark_results_TIMESTAMP.json"
echo "   Formatted tables are printed to console"
echo ""

echo "=========================================="
echo "For more details, see BENCHMARK_README.md"
echo "=========================================="
