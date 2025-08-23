# InferenceLab

A comprehensive benchmarking system for evaluating vLLM and LMCache performance with different eviction strategies and configurations.

## Project Structure

```
InferenceLab/
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

### Running Benchmark

```bash
# Run demo with test data
python src/entrypoint/benchmark_runner.py
```

```bash
# Run experiments from YAML configuration
python src/entrypoint/benchmark_runner.py --yaml configs/qwen-8b.yaml --gpus 8
```

### Running Prefix Cache Simulation

```bash
# Run cache simulation analysis (lru)
python src/simulation/cache_simulator.py --input-file outputs/logs/experiments/qwen-8b_2025-08-12/server_64.0gb_0.9rps_29_5am_6am.log --eviction-policy lru > outputs/simulation/sim_lru.log 2>&1
```

```bash
# Run cache simulation analysis (rule based)
python src/simulation/cache_simulator.py --input-file outputs/logs/experiments/qwen-8b_2025-08-12/server_64.0gb_0.9rps_29_5am_6am.log --eviction-policy conversation_aware > outputs/simulation/sim_rule.log 2>&1
```

```bash
# Run cache emulator (deprecated)
python src/simulation/cache_replay.py --mode server --input-file outputs/logs/experiments/qwen-8b_2025-08-12/server_64.0gb_0.9rps_29_5am_6am.log --eviction-policy lru > outputs/simulation/emu_lru.log 2>&1
```

Use synthetic run to generate simulator input.

```bash
# generate simulator input with simulation run
python src/simulation/csv_to_logs_converter.py data/test_data.csv qwen-8b --tokens-per-second 100 --output-dir outputs/logs/
```

Run simulator with different eviction policies 

```bash
# Run cache simulation analysis (lru)
python src/simulation/cache_simulator.py --input-file outputs/logs/server_qwen-8b.log --eviction-policy lru > outputs/simulation/sim_sim_lru.log 2>&1
```

```bash
# Run cache simulation analysis (lightgbm)
python src/simulation/cache_simulator.py --input-file outputs/logs/server_qwen-8b.log --eviction-policy lightgbm > outputs/simulation/sim_sim_lightgbm.log 2>&1
```

```bash
# Run cache simulation analysis (rule based)
python src/simulation/cache_simulator.py --input-file outputs/logs/server_qwen-8b.log --eviction-policy conversation_aware > outputs/simulation/sim_sim_rule.log 2>&1
```

Search for eviction rule hypeerparameters

```bash
# Search rules
python src/simulation/scorer_sweep.py  --input-file outputs/logs/server_qwen-8b.log
```

Run simulation in batches

```bash
# run simulation in batches
python src/simulation/simulator_runner.py --yaml configs/simulation_aws_offpeak.yaml > outputs/simulation/sim_sim_batched.log 2>&1

# plotting is called in the above script but can also run individually
python src/utils/simulation_plotter.py --summary-file outputs/simulation/summary_cw_logs_5_29_5am_6am.json
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
