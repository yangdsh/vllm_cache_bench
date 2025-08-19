#!/bin/bash

# VLLM Cache Benchmark Runner Script
# Usage: ./scripts/run_experiments.sh [config_file] [gpus]

set -e

# Default values
CONFIG_FILE="configs/models/qwen-8b.yaml"
GPUS=8

# Parse command line arguments
if [ $# -ge 1 ]; then
    CONFIG_FILE="$1"
fi

if [ $# -ge 2 ]; then
    GPUS="$2"
fi

echo "Running experiments with config: $CONFIG_FILE"
echo "Using GPUs: $GPUS"

# Run the benchmark
python src/entrypoint/benchmark_runner.py --yaml "$CONFIG_FILE" --gpus "$GPUS"

echo "Experiments completed!"
