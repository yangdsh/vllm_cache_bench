# vLLM Cache Benchmark Runner

A comprehensive benchmarking system for evaluating vLLM and LMCache cache performance with different eviction strategies and configurations.

## Project Structure

```
v1/
├── src/                          # Main source code
│   ├── entrypoint/               # Entry points of benchmark
│   ├── simulation/               # Cache simulation and analysis
│   ├── client/                   # Core of benchmark
│   ├── runner/                   # Experiment execution
│   ├── utils/                    # Shared utilities of benchmark and sim
│   ├── config/                   # Configuration management
│   └── environment/              # Environment-specific settings
├── configs/                      # Configuration files
├── data/                         # Data files and datasets
├── outputs/                      # Generated outputs
│   ├── logs/                     # Runtime logs
│   ├── simulation/               # Simulation results and replays
│   └── plots/                    # Generated plots
├── scripts/                      # Utility scripts
└── docs/                         # Documentation
```

## Quick Start

### Running Real Experiments

```bash
# Run demo with test data
python src/entrypoint/benchmark_runner.py
```

```bash
# Run experiments from YAML configuration
python src/entrypoint/benchmark_runner.py --yaml configs/qwen-8b.yaml --gpus 8
```

### Running Cache Simulation

```bash
# Run cache simulation analysis
python src/simulation/cache_simulator.py --input-file outputs/logs/experiments/qwen-8b_2025-08-12/server_64.0gb_0.9rps_29_5am_6am.log --eviction-policy lru > outputs/simulation/sim_lru.log 2>&1
```

```bash
# Run cache emulator
python src/simulation/cache_replay.py --mode server --input-file outputs/logs/experiments/qwen-8b_2025-08-12/server_64.0gb_0.9rps_29_5am_6am.log --eviction-policy lru > outputs/simulation/emu_lru.log 2>&1
```

## Features

- **Model Support**: Qwen, Llama, and other models
- **Flexible Configuration**: YAML-based experiment configuration
- **Cache Analysis**: Comprehensive cache hit/miss analysis
- **Conversation-aware Eviction**: Advanced eviction strategies
- **Resource Management**: GPU resource allocation and management
- **Result Visualization**: Automated plotting and analysis

## Configuration

Experiments are configured using YAML files in `configs/models/`. Each file defines:
- Model specifications
- Cache sizes to test
- Request rates
- Dataset paths
- Eviction strategies

## Outputs

All generated files are organized in the `outputs/` directory:
- `logs/`: Runtime logs from experiments
- `simulation/`: Simulation results and replay files
- `plots/`: Generated visualizations
