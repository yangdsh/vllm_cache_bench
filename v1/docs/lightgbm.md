# LightGBM-based Cache Eviction

This document explains how to use the new LightGBM-based cache eviction policy that was added to the cache simulator.

## Overview

The LightGBM eviction policy uses machine learning to predict which cache entries should be evicted based on conversation features. It automatically trains a model from log data and uses it to make more intelligent caching decisions compared to simple LRU or rule-based approaches.

## Usage

### Basic Usage

```bash
python src/simulation/cache_simulator.py \
    --input-file outputs/logs/experiments/qwen-8b_2025-08-12/server_64.0gb_0.9rps_29_5am_6am.log \
    --eviction-policy lightgbm \
    --cache-size 64.0
```

### Advanced Options

```bash
python src/simulation/cache_simulator.py \
    --input-file your_log_file.log \
    --eviction-policy lightgbm \
    --cache-size 64.0 \
    --model-dir models/ \
    --train-max-events 10000 \
    --force-retrain \
    --max-events 50000
```

## Command Line Arguments

- `--eviction-policy lightgbm`: Use the LightGBM-based eviction policy
- `--model-dir`: Directory to store/load trained models (default: `models/`)
- `--train-max-events`: Maximum number of events to use for training (default: 10000)
- `--force-retrain`: Force retraining even if a model already exists
- `--max-events`: Limit simulation to N events

## How It Works

1. **Model Path Generation**: A unique model file is generated based on the input log file name
2. **Training Data Preparation**: Events are processed to extract features from the selected ConversationFeature:
   - `turn_number`: Turn number within the conversation  
   - `previous_interval`: Time since last access to this conversation
   - `CustomerQueryLength`: Length of the user query
   - `GenerativeModelResponseLength`: Length of the model response
   - `cumulative_time`: Total time spent in conversation
   - `avg_interval_so_far`: Average reuse interval up to this point
   - `avg_query_length_so_far`: Average query length up to this point
   - `avg_response_length_so_far`: Average response length up to this point

3. **Ranking Labels**: Time-to-next-reuse labels where:
   - Lower values = higher priority to keep in cache (more likely to be reused soon)
   - Higher values = lower priority (less likely to be reused soon)
   - Events with no future reuse get maximum time value (24 hours)

4. **Model Training**: LightGBM ranking model (LambdaRank) is trained to predict time-to-next-reuse

5. **Eviction Scoring**: During simulation, the model predicts time-to-reuse scores, and entries with highest predicted times (least likely to be reused soon) are evicted first

## Model Persistence

- Models are automatically saved after training with filenames based on the input log file
- Existing models are reused unless `--force-retrain` is specified
- Models are stored as pickle files containing both the LightGBM model and metadata

## Comparison with Other Policies

You can compare the LightGBM policy against other eviction strategies:

```bash
# LRU policy
python src/simulation/cache_simulator.py --eviction-policy lru --input-file your_log.log

# Rule-based conversation-aware policy  
python src/simulation/cache_simulator.py --eviction-policy conversation_aware --input-file your_log.log

# LightGBM policy
python src/simulation/cache_simulator.py --eviction-policy lightgbm --input-file your_log.log
```

## Requirements

- LightGBM: `pip install lightgbm`
- NumPy: Usually already installed
- Pickle: Part of Python standard library

## Performance Notes

- Initial training may take a few minutes depending on the size of your log file
- Consider using `--train-max-events` to limit training data size for faster training
- Trained models are reused across runs for the same input file
- The model training uses only a subset of events to avoid memory issues with large log files
