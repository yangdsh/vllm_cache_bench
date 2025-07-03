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

async def run_experiment(memory_utilization, log_suffix="", gpu_device=0):
    """Run a single experiment with specified GPU memory utilization"""
    print(f"\n{'='*50}")
    print(f"RUNNING EXPERIMENT WITH GPU_MEMORY_UTILIZATION={memory_utilization} ON GPU {gpu_device}")
    print(f"{'='*50}")

    # --- Server config ---
    server_port = 8000 + int(memory_utilization * 10) % 100  # Unique port for each experiment
    server_config = {
        'host': 'localhost',
        'cuda_devices': f'CUDA_VISIBLE_DEVICES={gpu_device}',
        'eviction_algorithm': 'ml',
        'port': server_port,
        'size': 4000,
        'scale': 1,
    }
    eac = {"enable_online_learning": 1}
    eviction_algorithm_config_str = json.dumps(eac)
    
    if ENV == 'ec2':
        kv_config = {"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}
        server_args = (
            f'--port {server_config["port"]} '
            f'--max-num-batched-tokens 16384 '
            f'--gpu-memory-utilization {0.6} '
            f'--kv-transfer-config \'{json.dumps(kv_config)}\''
        )
        server_prefix = SERVER_COMMAND_PREFIX + f"LMCACHE_MAX_LOCAL_CPU_SIZE={32 * memory_utilization} "
    else:
        server_args = (
            f'--port {server_config["port"]} '
            f'--eviction_algorithm {server_config["eviction_algorithm"]} '
            f'--max-num-batched-tokens 2048 '
            f'--gpu-memory-utilization {memory_utilization} '
            f'--num-gpu-blocks-override {server_config["size"]} '
            f"--eviction_algorithm_config '{eviction_algorithm_config_str}'"
        )
        server_prefix = SERVER_COMMAND_PREFIX
    
    server_cmd = VLLM_SERVER_CMD_TEMPLATE.format(args=server_args)
    ssh_command = f"{server_prefix} {server_config['cuda_devices']} {server_cmd} {SERVER_COMMAND_SUFFIX}"

    # --- Client args config (matching run_test.py exactly) ---
    all_args = {
        'result-dir': DIR,
        'model': MODEL,
        'endpoint': '/v1/chat/completions',
        'dataset-name': 'conversational_csv',
        'host': server_config['host'],
        'port': server_config['port'],
        'result-filename': f'client{log_suffix}.log',
        'use-lru': 0,
        'num-prompts': 30000,
        'use-oracle': 0,
        'use-token-id': 1,
        'request-rate': 1,  
        'session-rate': 10, 
        'max-active-conversations': 200,
        'checkpoint': 'None',
        'dataset-path': f'{HOME}/PrefixCacheInternProject/Qdata/cw_logs_5_29_5am_6am.csv',
        'time-limit': 600,
        'save-result': None,
    }

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
    server_log_file = f"server{log_suffix}.log"
    server_log = open(server_log_file, "w")

    print(f"Starting server for memory_util={memory_utilization} on port {server_port}")
    print(f"Server logs: {server_log_file}")
    print(ssh_command)
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
                    print(f"Server ready for memory_util={memory_utilization} after {i} seconds")
                    break
        await asyncio.sleep(1)
    
    if not ready:
        print(f"Server for memory_util={memory_utilization} not ready, terminating.")
        server_proc.terminate()
        await server_proc.wait()
        server_log.close()
        return False

    # --- Start client ---
    print(f"Starting client for memory_util={memory_utilization}")
    client_proc = await asyncio.create_subprocess_shell(
        client_cmd, 
        stdout=asyncio.subprocess.PIPE, 
        stderr=asyncio.subprocess.PIPE
    )
    
    stdout, stderr = await client_proc.communicate()
    client_log_file = f"client{log_suffix}.log"
    client_log = open(client_log_file, "w")
    client_log.write(stdout.decode())
    client_log.write(stderr.decode())
    client_log.close()
    
    # --- Clean up ---
    print(f"Cleaning up server for memory_util={memory_utilization}")
    server_proc.terminate()
    await server_proc.wait()
    server_log.close()
    
    success = client_proc.returncode == 0
    print(f"Experiment memory_util={memory_utilization}: {'SUCCESS' if success else 'FAILED'}")
    return success

def analyze_logs(memory_utilizations):
    """Analyze server and client logs for preemption, cache query patterns, and conversation analytics"""
    print(f"\n{'='*80}")
    print("ANALYZING SERVER AND CLIENT LOGS FOR COMPREHENSIVE ANALYTICS")
    print("="*80)
    
    for memory_util in memory_utilizations:
        print(f"\n--- Memory Utilization: {memory_util} ---")
        log_suffix = f"_{memory_util}"
        server_log_file = f"server{log_suffix}.log"
        client_log_file = f"client{log_suffix}.log"
        
        # Initialize stats for this memory utilization
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
                else:
                    # Other stats
                    if key.endswith('_rate'):
                        print(f"    {key}: {float(value)*100:.1f}%")
                    else:
                        print(f"    {key}: {value}")
            
            if vllm_stats:
                print(f"    vLLM Local Prefix Cache:")
                for key, value in vllm_stats.items():
                    clean_key = key.replace('vllm_', '')
                    if clean_key.endswith('_rate'):
                        print(f"      {clean_key}: {float(value)*100:.1f}%")
                    else:
                        print(f"      {clean_key}: {value}")
                
                # Calculate preemption rate if we have preemption stats
                vllm_preemptions = int(vllm_stats.get('vllm_preemptions', 0))
                vllm_reschedules = int(vllm_stats.get('vllm_reschedules', 0))
                if vllm_preemptions > 0 or vllm_reschedules > 0:
                    total_preemption_events = vllm_preemptions + vllm_reschedules
                    preemption_rate = (vllm_preemptions / total_preemption_events * 100) if total_preemption_events > 0 else 0
                    print(f"      preemption_rate: {preemption_rate:.1f}%")
            
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

async def run_all_experiments_concurrently(memory_utilizations):
    """Run all experiments concurrently using asyncio"""
    print(f"Running {len(memory_utilizations)} experiments concurrently...")
    print(f"Memory utilizations: {memory_utilizations}")
    
    # Create tasks for all experiments with different GPU devices
    tasks = []
    for i, memory_util in enumerate(memory_utilizations):
        gpu_device = i % 8  # Support up to 8 GPUs, cycle if more experiments
        log_suffix = f"_{memory_util}"
        task = asyncio.create_task(run_experiment(memory_util, log_suffix, gpu_device))
        tasks.append((memory_util, task))
        print(f"  Experiment {memory_util} assigned to GPU {gpu_device}")
    
    # Wait for all experiments to complete
    results = {}
    for memory_util, task in tasks:
        try:
            success = await task
            results[memory_util] = success
        except Exception as e:
            print(f"Experiment {memory_util} failed with exception: {e}")
            results[memory_util] = False
    
    return results

async def main():
    parser = argparse.ArgumentParser(description="Debug preemption and cache query correlation")
    parser.add_argument("--memory-utilizations", nargs="+", type=float, 
                        default=[2, 4], 
                        help="GPU memory utilization values to test")
    parser.add_argument("--analyze-only", action="store_true", 
                        help="Only analyze existing logs")
    
    args = parser.parse_args()

    if not args.analyze_only:
        # Ensure no conflicting server processes
        kill_server('')

        print(f"Running experiments with memory utilizations: {args.memory_utilizations}")
        
        results = await run_all_experiments_concurrently(args.memory_utilizations)
        
        print(f"\n{'='*50}")
        print("EXPERIMENT RESULTS")
        print("="*50)
        for memory_util, success in results.items():
            print(f"Memory utilization {memory_util}: {'SUCCESS' if success else 'FAILED'}")
    
    # Analyze the logs
    analyze_logs(args.memory_utilizations)

if __name__ == "__main__":
    asyncio.run(main()) 