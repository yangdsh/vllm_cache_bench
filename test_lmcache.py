#!/usr/bin/env python3
"""
Debug script to test preemption hypothesis:
Different GPU memory utilization -> more preemptions -> more cache queries
"""
import subprocess
import os
import re
import time
import json
import argparse
import asyncio
from dataclasses import dataclass
from typing import Optional, Dict, Any
from utils import kill_server

# Import environment configuration from constants.py
from constants import (
    ENV, 
    SERVER_COMMAND_PREFIX, 
    HOME, 
    DATA_HOME, 
    SERVER_COMMAND_SUFFIX, 
    MODEL, 
    DIR, 
    SERVER_READY_PATTERN,
    VLLM_SERVER_CMD_TEMPLATE,
    CLIENT_CMD_TEMPLATE
)

class ExperimentConfig:
    """Structured configuration for a single experiment"""
    memory_utilization: float
    request_rate: float
    dataset_name: str = "conversational_csv"
    dataset_path: str = f'set by argument'
    num_prompts: int = 30000
    time_limit: int = 1200
    gpu_device: int = 0
    server_port: Optional[int] = None
    log_prefix: Optional[str] = None
    log_suffix: Optional[str] = None
    use_conversation_eviction: bool = False

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        if self.log_suffix is None:
            conv_eviction_suffix = "_conv_evict" if self.use_conversation_eviction else ""
            object.__setattr__(self, 'log_suffix', f"_{self.memory_utilization}_{self.request_rate}"
                               f"_{conv_eviction_suffix}")
        if self.log_prefix is None:
            object.__setattr__(self, 'log_prefix', "logs")
    
    def get_client_args(self) -> Dict[str, Any]:
        """Get client arguments as a dictionary"""
        return {
            'result-dir': self.log_prefix,
            'model': MODEL,
            'endpoint': '/v1/chat/completions',
            'dataset-name': self.dataset_name,
            'host': 'localhost',
            'port': self.server_port,
            'result-filename': f'vllm{self.log_suffix}.log',
            'num-prompts': self.num_prompts,
            'use-oracle': 0,
            'request-rate': self.request_rate,  
            'session-rate': -1, 
            'max-active-conversations': -1,
            'checkpoint': 'None',
            'dataset-path': self.dataset_path,
            'time-limit': self.time_limit,
            'save-result': None,
        }
    
    def get_server_config(self) -> Dict[str, Any]:
        """Get server configuration as a dictionary"""
        return {
            'host': 'localhost',
            'cuda_devices': f'CUDA_VISIBLE_DEVICES={self.gpu_device}',
            'eviction_algorithm': 'ml',
            'port': self.server_port,
            'size': 4000,
            'scale': 1,
        }
    
    def get_key(self) -> str:
        """Get a string key for this configuration"""
        conv_eviction_suffix = "_conv_evict" if self.use_conversation_eviction else ""
        return f"{self.memory_utilization}_{self.request_rate}_{self.dataset_name}{conv_eviction_suffix}"

def create_experiment_configs(
    memory_utilizations: list[float],
    request_rates: list[float],
    dataset_paths: list[str],
    use_conversation_evictions: list[bool] = [True]
) -> list[ExperimentConfig]:
    """
    Create a list of experiment configurations from parameter lists.
    
    Args:
        memory_utilizations: List of GPU memory utilization values
        request_rates: List of request rate values  
        dataset_paths: List of dataset file paths
        use_conversation_evictions: Whether to enable conversation-aware eviction
    
    Returns:
        List of ExperimentConfig objects
    """
    configs = []
    for memory_util in memory_utilizations:
        for request_rate in request_rates:
            for use_conversation_eviction in use_conversation_evictions:
                for dataset_path in dataset_paths:
                    config = ExperimentConfig(
                        memory_utilization=memory_util,
                        request_rate=request_rate,
                        dataset_path=dataset_path,
                        use_conversation_eviction=use_conversation_eviction
                    )
                    configs.append(config)
    
    return configs

async def run_experiment(config: ExperimentConfig):
    """Run a single experiment with the given configuration"""
    config_id = config.get_key()
    print(f"\n{'='*50}")
    print(f"RUNNING EXPERIMENT: {config_id}")
    print(f"{'='*50}")

    # --- Server config ---
    server_config = config.get_server_config()
    eac = {"enable_online_learning": 1}
    eviction_algorithm_config_str = json.dumps(eac)
    
    if ENV == 'ec2':
        kv_config = {"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}
        server_args = (
            f'--port {server_config["port"]} '
            f'--max-num-batched-tokens 16384 '
            f'--gpu-memory-utilization {0.95} '
            f'--kv-transfer-config \'{json.dumps(kv_config)}\''
        )
        server_prefix = SERVER_COMMAND_PREFIX + f"LMCACHE_MAX_LOCAL_CPU_SIZE={config.memory_utilization} "
        if config.use_conversation_eviction:
            # Enable conversation-aware eviction through LMCache extra_config
            # This sets the use_conversation_eviction flag in the LMCache configuration
            extra_config = {"use_conversation_eviction": True}
            server_prefix += f"LMCACHE_EXTRA_CONFIG='{json.dumps(extra_config)}' "
    else:
        server_args = (
            f'--port {server_config["port"]} '
            f'--eviction_algorithm {server_config["eviction_algorithm"]} '
            f'--max-num-batched-tokens 2048 '
            f'--gpu-memory-utilization {config.memory_utilization} '
            f'--num-gpu-blocks-override {server_config["size"]} '
            f"--eviction_algorithm_config '{eviction_algorithm_config_str}'"
        )
        server_prefix = SERVER_COMMAND_PREFIX
        if config.use_conversation_eviction:
            extra_config = {"use_conversation_eviction": True}
            server_prefix += f"LMCACHE_EXTRA_CONFIG='{json.dumps(extra_config)}' "
    
    server_cmd = VLLM_SERVER_CMD_TEMPLATE.format(args=server_args)
    ssh_command = f"{server_prefix} {server_config['cuda_devices']} {server_cmd} {SERVER_COMMAND_SUFFIX}"

    # --- Client args config ---
    all_args = config.get_client_args()

    # Build client args using loop
    client_args_list = []
    for arg_name, arg_value in all_args.items():
        if arg_value is None:
            client_args_list.append(f"--{arg_name}")
        else:
            client_args_list.append(f"--{arg_name} {arg_value}")

    client_args = " ".join(client_args_list)
    client_cmd = CLIENT_CMD_TEMPLATE.format(args=client_args)
    client_cmd = f"{server_config['cuda_devices']} {client_cmd}"

    # --- Start server ---
    server_log_file = f"{config.log_prefix}/server{config.log_suffix}.log"
    server_log = open(server_log_file, "w")

    print(f"Starting server for config {config_id} on port {config.server_port}")
    print(f"Server logs: {server_log_file}")
    # print(ssh_command)
    server_proc = await asyncio.create_subprocess_shell(
        ssh_command, 
        stdout=server_log, 
        stderr=server_log
    )

    # --- Wait for server to be ready ---
    ready = False
    for i in range(300):  # Reduced timeout for faster testing
        if os.path.exists(server_log_file):
            with open(server_log_file) as f:
                if re.search(SERVER_READY_PATTERN, f.read()):
                    ready = True
                    print(f"Server ready for config {config_id} after {i} seconds\n")
                    break
        await asyncio.sleep(1)
    
    if not ready:
        print(f"Server for config {config_id} not ready, terminating.")
        server_proc.terminate()
        await server_proc.wait()
        server_log.close()
        return False

    # --- Start client ---
    client_log_file = f"{config.log_prefix}/client{config.log_suffix}.log"
    print(f"Starting client for config {config_id}; log file: {client_log_file}\n")
    # print(client_cmd)
    client_proc = await asyncio.create_subprocess_shell(
        client_cmd, 
        stdout=asyncio.subprocess.PIPE, 
        stderr=asyncio.subprocess.PIPE
    )
    
    stdout, stderr = await client_proc.communicate()
    client_log = open(client_log_file, "w")
    client_log.write(stdout.decode())
    client_log.write(stderr.decode())
    client_log.close()
    
    # --- Clean up ---
    print(f"Cleaning up server for config {config_id}\n")
    server_proc.terminate()
    await server_proc.wait()
    server_log.close()
    
    success = client_proc.returncode == 0
    print(f"Experiment {config_id}: {'SUCCESS' if success else 'FAILED'}")
    return success

def analyze_logs(configs: list[ExperimentConfig]):
    """Analyze server and client logs for preemption, cache query patterns, and conversation analytics"""
    print(f"\n{'='*80}")
    print("ANALYZING SERVER AND CLIENT LOGS FOR COMPREHENSIVE ANALYTICS")
    print("="*80)
    
    for config in configs:
        config_id = config.get_key()
        print(f"\n--- Config: {config_id} ---")
        server_log_file = f"{config.log_prefix}/server{config.log_suffix}.log"
        client_log_file = f"{config.log_prefix}/client{config.log_suffix}.log"
        
        # Initialize stats for this config
        stats = {
            'cache_stats': {},
            'conversation_features': {}
        }

        # Parse client log file
        if os.path.exists(client_log_file):
            print(f"  📄 Parsing client log: {client_log_file}")
            parsing_client_stats = False
            parsing_benchmark_results = False
            benchmark_result_lines = []
            
            with open(client_log_file, 'r') as f:
                for line in f:
                    line_stripped = line.strip()
                    
                    # Check for benchmark result section
                    if "============ Serving Benchmark Result ============" in line_stripped:
                        parsing_benchmark_results = True
                        benchmark_result_lines = [line_stripped]
                        continue
                    elif parsing_benchmark_results:
                        benchmark_result_lines.append(line_stripped)
                        if "==================================================" in line_stripped:
                            parsing_benchmark_results = False
                            # Print the complete benchmark result section
                            print(f"  📊 BENCHMARK RESULTS:")
                            for result_line in benchmark_result_lines:
                                print(f"    {result_line}")
                            print()  # Add blank line after benchmark results
                            benchmark_result_lines = []
                        continue
                    
                    # Parse client statistics (existing functionality)
                    if "CLIENT_STATISTICS_BEGIN" in line_stripped:
                        parsing_client_stats = True
                    elif "CLIENT_STATISTICS_END" in line_stripped:
                        parsing_client_stats = False
                    elif parsing_client_stats and ":" in line_stripped:
                        try:
                            key, value = line_stripped.split(":", 1)
                            key = key.strip()
                            value = value.strip()
                            
                            # Parse different types of values
                            if key.startswith('conversation_'):
                                stats['conversation_features'][key] = value
                            elif key.endswith('_rate'):
                                stats['cache_stats'][key] = float(value)
                            else:
                                try:
                                    stats['cache_stats'][key] = int(value)
                                except ValueError:
                                    stats['cache_stats'][key] = value
                        except ValueError:
                            pass
        else:
            print(f"  ⚠️  Client log file {client_log_file} not found")

        print(f"  CLIENT CACHE ANALYTICS:")
        if stats['cache_stats']:
            # Separate vLLM and LMCache stats for cleaner output
            vllm_stats = {}
            lmcache_stats = {}
            
            for key, value in stats['cache_stats'].items():
                if key.startswith('vllm_'):
                    vllm_stats[key] = value
                elif key.startswith('lmcache_'):
                    lmcache_stats[key] = value
            
            if vllm_stats:
                print(f"    vLLM Local Prefix Cache:")
                for key, value in vllm_stats.items():
                    clean_key = key.replace('vllm_', '')
                    if clean_key.endswith('_rate'):
                        print(f"      {clean_key}: {float(value)*100:.1f}%")
                    else:
                        print(f"      {clean_key}: {value}")
            
            if lmcache_stats:
                print(f"    LMCache External Cache:")
                for key, value in lmcache_stats.items():
                    clean_key = key.replace('lmcache_', '')
                    if clean_key.endswith('_rate'):
                        print(f"      {clean_key}: {float(value)*100:.1f}%")
                    else:
                        print(f"      {clean_key}: {value}")
        else:
            print(f"    No client cache statistics found")

        print(f"  CONVERSATION ANALYTICS:")
        # Client conversation analytics
        if stats['conversation_features']:
            for key, value in stats['conversation_features'].items():
                print(f"      {key}: {value}")

async def run_all_experiments_concurrently(configs: list[ExperimentConfig]):
    """Run all experiments concurrently using asyncio"""
    print(f"Running {len(configs)} experiments concurrently...")
    print(f"Configs: {[config.get_key() for config in configs]}")
    
    # Create tasks for all experiments with different GPU devices
    tasks = []
    for i, config in enumerate(configs):
        gpu_device = i % 8  # Support up to 8 GPUs, cycle if more experiments
        config.gpu_device = gpu_device
        config.server_port = 8000 + gpu_device

        task = asyncio.create_task(run_experiment(config))
        tasks.append((config, task))
        print(f"  Experiment {config.get_key()} assigned to GPU {gpu_device}")
    
    # Wait for all experiments to complete
    results = {}
    for config, task in tasks:
        try:
            success = await task
            results[config.get_key()] = success
        except Exception as e:
            print(f"Experiment {config.get_key()} failed with exception: {e}")
            results[config.get_key()] = False
    
    return results

async def main():
    parser = argparse.ArgumentParser(description="Debug preemption and cache query correlation")
    parser.add_argument("--memory-utilizations", nargs="+", type=float, 
                        default=[32, 64], 
                        help="GPU memory utilization values to test")
    parser.add_argument("--request-rates", nargs="+", type=float, 
                        default=[1], 
                        help="Request rate values to test")
    parser.add_argument("--dataset-paths", nargs="+", type=str,
                        default=[f'{HOME}/PrefixCacheInternProject/Qdata/cw_logs_5_29_5am_6am.csv'],
                        help="Dataset paths to test (default: Qdata/cw_logs_5_29_5am_6am.csv)")
    parser.add_argument("--use-conversation-evictions", nargs="+", type=int, 
                        default=[0, 1], 
                        help="Enable conversation-aware eviction policy")
    parser.add_argument("--analyze-only", action="store_true", 
                        help="Only analyze existing logs")
    
    args = parser.parse_args()

    if not args.analyze_only:
        # Ensure no conflicting server processes
        kill_server('')

        # Create experiment configurations
        configs = create_experiment_configs(
            memory_utilizations=args.memory_utilizations,
            request_rates=args.request_rates,
            dataset_paths=args.dataset_paths,
            use_conversation_evictions=args.use_conversation_evictions
        )
        
        results = await run_all_experiments_concurrently(configs)
        
        print(f"\n{'='*50}")
        print("EXPERIMENT RESULTS")
        print("="*50)
        for config_key, success in results.items():
            print(f"{config_key}: {'SUCCESS' if success else 'FAILED'}")
    else:
        # For analyze-only mode, create configs from existing log files
        # This is a simplified approach - you might want to enhance this
        configs = create_experiment_configs(
            memory_utilizations=args.memory_utilizations,
            request_rates=args.request_rates,
            dataset_paths=args.dataset_paths,
            use_conversation_evictions=args.use_conversation_evictions
        )
    
    # Analyze the logs
    successful_configs = [config for config in configs if results.get(config.get_key())]
    analyze_logs(successful_configs)

if __name__ == "__main__":
    asyncio.run(main()) 