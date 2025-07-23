
# VLLM Cache Benchmark Runner

This document provides instructions on how to use the VLLM Cache Benchmark Runner to run experiments and analyze the performance of different caching strategies.

## Overview

The benchmark runner is a Python-based tool for running batch experiments to evaluate the performance of VLLM with different cache configurations. It is designed to be highly configurable, allowing you to specify the model, dataset, and various experiment parameters through a YAML configuration file.

## Running Experiments

The main entry point for running experiments is the `benchmark_runner.py` script. You can run it from the command line, and the primary way to configure the experiments is by providing a YAML file.

### Basic Usage

To run a set of experiments defined in a YAML file, use the following command:

```bash
python3 benchmark_runner.py --yaml <path_to_your_config>.yaml
```

For example, to run the experiments defined in `qwen8b.yaml`, you would use:

```bash
python3 benchmark_runner.py --yaml qwen8b.yaml
```

You can also specify the number of GPUs to use for the experiments with the `--gpus` flag:

```bash
python3 benchmark_runner.py --yaml llama70b.yaml --gpus 4
```

If you run the script without any arguments, it will run a single demo experiment with default settings.

## Configuration

The benchmark runner is configured using a YAML file. This file contains all the necessary parameters to define the model, the experiments, and the environment.

### Model Configuration

The model to be benchmarked is specified using the `model_name` key at the top level of the YAML file. The model name should correspond to one of the models registered in the `ModelRegistry`.

**Example:**

```yaml
model_name: "qwen-8b"
```

### Experiment Configuration

The `experiments` section of the YAML file is where you define the experiments to be run. You can specify a list of datasets and a set of parameters to be swept for each dataset.

The following parameters can be configured for each experiment:

- `path`: The path to the dataset file.
- `cache_sizes`: A list of cache sizes (in GB) to be tested.
- `request_rates`: A list of request rates (in requests per second) to be tested.
- `num_prompts`: The maximum number of prompts to process from the dataset.
- `time_limit`: The maximum time (in seconds) to run the experiment.
- `use_conversation_evictions`: A list of booleans to enable or disable conversation-aware eviction.
- `mock_decoding`: A boolean to enable or disable mock decoding.

**Example:**

```yaml
experiments:
    datasets:
      - path: "../data/cw_logs_5_29_5am_6am.csv"
        cache_sizes: [64, 128, 256]
        request_rates: [1.0]
      - path: "../data/cw_logs_5_28_19pm_20pm.csv"
        cache_sizes: [64, 128, 256]
        request_rates: [0.7]
```

You can also set default values for these parameters in the `defaults` section.

### Environment Configuration

The environment-specific settings are configured in the `environment/environment_provider.py` file. This file contains a dictionary of `EnvironmentSettings` for different environments. The appropriate environment is detected automatically based on the username.

The following settings can be configured for each environment:

- `shell_command_prefix`: A command to be prepended to the shell command used to run the experiment.
- `server_ready_pattern`: A regex pattern to detect when the vLLM server is ready.

You can add new environments to the `_ENVIRONMENT_SETTINGS` dictionary in `environment/environment_provider.py` to support new setups.

#### LMCACHE Environment Variables

For the `LMCACHE` environment, you can set the following environment variables in the `shell_command_prefix`:

- `LMCACHE_CHUNK_SIZE`: The chunk size for the cache.
- `LMCACHE_LOCAL_CPU`: A boolean to enable or disable local CPU caching.

**Example:**

```python
Environment.LMCACHE: EnvironmentSettings(
    environment_type=Environment.LMCACHE,
    shell_command_prefix=(
        "LMCACHE_CHUNK_SIZE=256 "
        "LMCACHE_LOCAL_CPU=True "
    ),
    ...
)
```

### Registering a New Model

To benchmark a new model, you need to register it in the `config/model_registry.py` file. This involves adding a new `ModelSpec` to the `ModelRegistry`.

The `ModelSpec` includes the following parameters:

- `name`: A unique name for the model.
- `huggingface_path`: The path to the model on Hugging Face Hub.
- `tensor_parallel_size`: The tensor parallel size for the model.
- `pipeline_parallel_size`: The pipeline parallel size for the model.
- `max_model_length`: The maximum model length.

**Example:**

To register a new model, add the following to the `initialize_default_models` method in `config/model_registry.py`:

```python
cls.register_model(ModelSpec(
    name="my-new-model",
    huggingface_path="my-org/my-new-model",
    tensor_parallel_size=2,
    max_model_length=8192
))
```

If your model has environment-specific paths, you can register them using the `register_environment_path` method.

## Results

The results of the experiments are stored in the directory specified by the `log_base_dir` key in the YAML configuration file. For each experiment, a directory is created with a name based on the `tag` and the current date.

Inside the experiment directory, you will find the following:

- **Server and client logs**: Log files for the vLLM server and the benchmark client.
- **Configuration summary**: A JSON file (`summary.json`) containing a summary of the experiment configurations and results.

The benchmark runner also prints a summary of the results to the console at the end of the run, including the total number of experiments, the number of successful and failed experiments, and the average execution time. 