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
    server_port = 8000 + int(memory_utilization * 100) % 100  # Unique port for each experiment
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
    all_args = {
        'result-dir': DIR,
        'model': MODEL,
        'endpoint': '/v1/chat/completions',
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
        'time-limit': 600,
        'save-result': None,
        'dataset-name': 'conversational_csv',
        'dataset-path': '',
    }
    if all_args['dataset_path'] == '':
        raise ValueError("dataset_path is not set")
    
    if ENV == 'ec2':
        kv_config = {"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}
        server_args = (
            f'--port {server_config["port"]} '
            f'--max-num-batched-tokens 2048 '
            f'--gpu-memory-utilization {memory_utilization} '
            f'--kv-transfer-config \'{json.dumps(kv_config)}\''
        )
        server_prefix = SERVER_COMMAND_PREFIX + f"LMCACHE_MAX_LOCAL_CPU_SIZE={80} "
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

    # Ensure no conflicting server processes
    kill_server('')
    await asyncio.sleep(2)  # Brief delay to ensure cleanup

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
    print(client_cmd)
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
    """Analyze server logs for preemption and cache query patterns"""
    print(f"\n{'='*60}")
    print("ANALYZING LOGS FOR PREEMPTION AND CACHE QUERY CORRELATION")
    print("="*60)
    
    for memory_util in memory_utilizations:
        print(f"\n--- Memory Utilization: {memory_util} ---")
        log_suffix = f"_{memory_util}"
        server_log_file = f"server{log_suffix}.log"
        
        if not os.path.exists(server_log_file):
            print(f"  Log file {server_log_file} not found")
            continue
        
        preemption_count = 0
        reschedule_count = 0
        duplicate_queries = {}
        lmcache_queries = 0
        
        with open(server_log_file, 'r') as f:
            for line in f:
                if "[PREEMPTION_DEBUG]" in line:
                    preemption_count += 1
                elif "[RESCHEDULE_DEBUG]" in line:
                    reschedule_count += 1
                elif "[LMCACHE_DEBUG]" in line:
                    # Extract request ID and query count
                    match = re.search(r'Request (\S+) queried (\d+) times', line)
                    if match:
                        req_id, query_count = match.groups()
                        duplicate_queries[req_id] = int(query_count)
                elif "GPU KV cache size:" in line:
                    cache_size = line.split(":")[-1].strip()
                    print(f"  GPU KV cache size: {cache_size}")
                elif "Prefix cache queries:" in line:
                    try:
                        queries = float(line.split(":")[-1].strip())
                        lmcache_queries = queries
                    except:
                        pass
        
        print(f"  Preemptions: {preemption_count}")
        print(f"  Reschedulings: {reschedule_count}")
        print(f"  Requests with duplicate queries: {len(duplicate_queries)}")
        print(f"  Max duplicate queries for single request: {max(duplicate_queries.values()) if duplicate_queries else 0}")
        print(f"  Total cache queries: {lmcache_queries}")

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
                        default=[0.95], 
                        help="GPU memory utilization values to test")
    parser.add_argument("--analyze", action="store_true", 
                        help="analyze existing logs")
    
    args = parser.parse_args()

    print(f"Running experiments with memory utilizations: {args.memory_utilizations}")
    
    results = await run_all_experiments_concurrently(args.memory_utilizations)
    
    print(f"\n{'='*50}")
    print("EXPERIMENT RESULTS")
    print("="*50)
    for memory_util, success in results.items():
        print(f"Memory utilization {memory_util}: {'SUCCESS' if success else 'FAILED'}")

    if args.analyze:
        # Analyze the logs
        analyze_logs(args.memory_utilizations)

if __name__ == "__main__":
    asyncio.run(main()) 