import json
import subprocess
import time
import asyncio
import os
import re
import copy
import argparse
from utils import kill_server
from constants import *

def prepare_configs(sizes, scales, alg, dataset, tag):
    """Prepare unified experiment configs for each server/client pair."""
    configs = []
    i = 0
    for size in sizes:
        for scale in scales:
            if 'tensor-parallel' not in VLLM_SERVER_CMD_TEMPLATE:
                cuda_device_id = i+1 if ENV == 'fat2' else i
            else:
                cuda_device_id = '1,2,3,4' if ENV == 'fat2' else ','.join(map(str, range(i, i+4)))
            
            # Base config, will be modified below
            current_size = size
            if 'ml' in alg:
                current_size -= 250  # Account for ML model memory overhead

            base = {
                'host': 'localhost',
                'cuda_devices': f'CUDA_VISIBLE_DEVICES={cuda_device_id}',
                'eviction_algorithm': alg,
                'port': 8000+i,
                'size': current_size,
                'scale': scale,
                'algorithm': alg,
                'session_rate': 10,
                'num_prompts': 30000,
                'use_oracle': 0,
                'use_token_id': 1,
                'max_active_conversations': 200,
                'time_limit': 1200,
                'dataset_name': dataset,
                'tag': tag,
                'request_rate': 1.0, # Default request rate
            }
            # Dataset-specific overrides
            if dataset.startswith('tay'):
                base.update({
                    'checkpoint': f'{HOME}/vllm/benchmarks/checkpoints_tay_20/tay_epoch17_metric_0_6332.pt',
                    'dataset_file': f'{DATA_HOME}/tay.json',
                    'request_rate': 1,
                    'session_rate': 10,
                })
            elif dataset.startswith('chatbot'):
                base.update({
                    'checkpoint': f'{HOME}/vllm/benchmarks/checkpoints_chatbot_arena_20/chatbot_arena_epoch16_metric_0_4005.pt',
                    'dataset_file': '"lmsys/chatbot_arena_conversations"',
                    'request_rate': 0.01,
                })
            elif dataset.startswith('lmsys'):
                base.update({
                    'checkpoint': f'{HOME}/vllm/benchmarks/checkpoints_lmsys-chat-1m_20/lmsys-chat-1m_epoch11_metric_0_5797.pt',
                    'dataset_file': '"lmsys/lmsys-chat-1m"',
                    'request_rate': 0.01,
                })
            elif dataset.startswith('sharegpt'):
                base.update({
                    'checkpoint': f'{HOME}/vllm/benchmarks/checkpoints_sharegpt_20/sharegpt_epoch19_metric_0_5427.pt',
                    'dataset_file': f'{DATA_HOME}/ShareGPT_V3_unfiltered_cleaned_split.json',
                    'request_rate': 0.01,
                })
            
            # Apply request rate scaling
            base['request_rate'] *= base['scale']

            # Result file naming
            prefix = f"{base['port']}_{base['algorithm']}"
            base['result_filename'] = f"{dataset}-{tag}/client_logs/{prefix}.json"
            # LRU flag
            base['use_lru'] = 1 if 'lru' in base['algorithm'] else 0
            # Eviction algorithm config
            if 'online' in base['eviction_algorithm']:
                if 'finetune' in base['algorithm']:
                    base['eviction_algorithm_config'] = {"enable_online_learning": 1, "learning_rate": 1e-3, "min_batch_size": 64, "model_path": base['checkpoint']}
                else:
                    base['eviction_algorithm_config'] = {"enable_online_learning": 1, "learning_rate": 1e-3, "min_batch_size": 64}
            else:
                base['eviction_algorithm_config'] = {"enable_online_learning": 0, "model_path": base['checkpoint']}
            configs.append(base)
            i += 1
    return configs


def build_server_cmd(config):
    """Construct the server command string from config."""
    if ENV == 'ec2':
        kv_config = {"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}
        args = (
            f'--port {config["port"]} '
            f'--max-num-batched-tokens 2048 '
            f'--num-gpu-blocks-override 4000 '
            f'--kv-transfer-config \'{json.dumps(kv_config)}\''
        )
        server_command_prefix = SERVER_COMMAND_PREFIX
        server_command_prefix += f"LMCACHE_MAX_LOCAL_CPU_SIZE={config['size']} "
    else:
        eviction_algorithm_config_str = json.dumps(config.get('eviction_algorithm_config', {}))
        args = (
            f'--port {config["port"]} '
            f'--eviction_algorithm {config["eviction_algorithm"]} '
            f'--max-num-batched-tokens 2048 '
            f'--num-gpu-blocks-override {config["size"]} '
            f"--eviction_algorithm_config '{eviction_algorithm_config_str}'"
        )
    config['args'] = args
    return VLLM_SERVER_CMD_TEMPLATE.format(args=args), server_command_prefix


def build_client_cmd(config):
    """Construct the client command string from config using a dictionary."""
    all_args = {
        # Fixed arguments from constants
        'result-dir': DIR,
        'model': MODEL,
        'endpoint': '/v1/chat/completions',
        
        # Arguments from the config dictionary
        'dataset-path': config['dataset_file'],
        'dataset-name': 'conversation',
        'host': config['host'],
        'port': config['port'],
        'result-filename': config['result_filename'],
        'num-prompts': config['num_prompts'],
        'request-rate': config['request_rate'],
        'session-rate': config['session_rate'],
        'checkpoint': config['checkpoint'],
        'use-oracle': config['use_oracle'],
        'use-token-id': config['use_token_id'],
        'use-lru': config['use_lru'],
        'max-active-conversations': config['max_active_conversations'],
        'time-limit': config['time_limit'],

        # Flags that don't take values
        'save-result': None,
    }

    # Build client args using loop
    client_args_list = []
    for arg_name, arg_value in all_args.items():
        if arg_value is None:
            client_args_list.append(f"--{arg_name}")
        else:
            client_args_list.append(f"--{arg_name} {arg_value}")
    
    args = " ".join(client_args_list)
    return CLIENT_CMD_TEMPLATE.format(args=args)


def launch_server(config, tag):
    os.makedirs(f'{DIR}/{config["dataset_name"]}-{tag}', exist_ok=True)
    os.makedirs(f'{DIR}/{config["dataset_name"]}-{tag}/client_logs', exist_ok=True)
    log_file_name = f"{DIR}/{config['dataset_name']}-{tag}/server_{config['port']}_{config['algorithm']}.log"
    server_cmd, server_command_prefix = build_server_cmd(config)
    ssh_command = f"{server_command_prefix} {config['cuda_devices']} {server_cmd} {SERVER_COMMAND_SUFFIX}"
    print('\n', ssh_command, '\n')
    with open(log_file_name, "w") as log_file:
        popen_kwargs = {"shell": True, "stdout": log_file, "stderr": log_file}
        if ENV == 'della':
            popen_kwargs["executable"] = "/bin/bash"
        process = subprocess.Popen(ssh_command, **popen_kwargs)
    return log_file_name


def wait_for_server_ready(log_file_name, timeout=600):
    for _ in range(timeout):
        if os.path.exists(log_file_name):
            with open(log_file_name, "r") as log_file:
                log_data = log_file.read()
                if re.search(SERVER_READY_PATTERN, log_data):
                    print("Server is ready.")
                    return True
                if re.search(CUDA_OOM_PATTERN, log_data) or re.search(ERROR_PATTERN, log_data) or re.search(RAISE_PATTERN, log_data):
                    print("Server encountered an error.")
                    return False
        time.sleep(1)
    print("Server startup timed out.")
    return False


def launch_client(config, tag):
    """Launch the client as a background process, logging its output to a file."""
    client_cmd = build_client_cmd(config)
    if ENV == 'della':
        client_cmd = f'{config["cuda_devices"]} {client_cmd}'
    
    base_log_name = f"{DIR}/{config['dataset_name']}-{tag}/client_{config['port']}_{config['algorithm']}"
    stdout_log_name = f"{base_log_name}.log"
    stderr_log_name = f"{base_log_name}.err.log"

    print("Running client command, logging to:", stdout_log_name, flush=True)
    print(client_cmd)

    with open(stdout_log_name, "w") as stdout_file, open(stderr_log_name, "w") as stderr_file:
        popen_kwargs = {"shell": True, "stdout": stdout_file, "stderr": stderr_file}
        if ENV == 'della':
            popen_kwargs["executable"] = "/bin/bash"
        subprocess.Popen(client_cmd, **popen_kwargs)
        
    return stdout_log_name, stderr_log_name


def wait_for_client_finish(stdout_log_name, stderr_log_name):
    """Wait until the client is finished by checking for a completion pattern in the log."""
    while True:
        # Check for errors first in stderr
        if os.path.exists(stderr_log_name):
            with open(stderr_log_name, "r") as f:
                if re.search("Traceback", f.read()):
                    print("Client encountered an error (from stderr).")
                    return False
        
        # Check stdout for completion or errors
        if os.path.exists(stdout_log_name):
            with open(stdout_log_name, "r") as f:
                out_data = f.read()
                if "==================================================" in out_data:
                    print("Client has finished.")
                    return True
                if re.search(CUDA_OOM_PATTERN, out_data) or re.search(ERROR_PATTERN, out_data) or re.search(RAISE_PATTERN, out_data):
                    print("Client encountered an error (from stdout).")
                    return False
        time.sleep(1)


def parse_metrics_from_log(log_path):
    """Parse metrics from the completed client log file."""
    metrics = {}
    hit_ratios = []
    hit_tokens = []
    need_convert = 0
    prompt_tokens = []
    
    if not os.path.exists(log_path):
        return metrics

    with open(log_path, 'r') as fp:
        log_content = fp.read()
        for line in log_content.split('\n'):
            if 'gpu_prefix_cache_hit_rate' in line:
                need_convert = 1
                hit_ratios.append(line.split()[-1])
            if 'vllm:prompt_tokens_total' in line:
                prompt_tokens.append(line.split()[-1])
            if 'Prefix cache queries:' in line:
                prompt_tokens.append(line.replace('%', '').split()[-1])
            if 'Prefix cache hits:' in line:
                hit_tokens.append(line.replace('%', '').split()[-1])
            if 'Mean TTFT (ms):' in line:
                metrics['mean_ttft'] = line.split()[-1]
            if 'Median TTFT (ms):' in line:
                metrics['median_ttft'] = line.split()[-1]
            if 'P99 TTFT (ms):' in line:
                metrics['p99_ttft'] = line.split()[-1]
            if 'Mean ITL (ms):' in line:
                metrics['mean_itl'] = line.split()[-1]
            if 'Median ITL (ms):' in line:
                metrics['median_itl'] = line.split()[-1]
            if 'P99 ITL (ms):' in line:
                metrics['p99_itl'] = line.split()[-1]
            if 'Total input tokens:' in line:
                metrics['input_tokens'] = line.split()[-1]
            if 'Total generated tokens:' in line:
                metrics['output_tokens'] = line.split()[-1]
            if 'Successful requests:' in line:
                metrics['num_requests'] = line.split()[-1]

    if need_convert:
        metrics['hit_ratios'] = []
        metrics['prompt_tokens'] = []
        i = 0
        for h, p in zip(hit_ratios, prompt_tokens):
            hit_tokens.append(float(h) * float(p))
            if i > 0:
                delta_prompt_tokens = float(p) - float(prompt_tokens[i-1])
                metrics['prompt_tokens'].append(delta_prompt_tokens)
                if delta_prompt_tokens == 0:
                    metrics['hit_ratios'].append(0.0)
                else:
                    metrics['hit_ratios'].append(
                        (hit_tokens[i] - hit_tokens[i-1]) / delta_prompt_tokens)
            i += 1
    else:
        metrics['hit_ratios'] = []
        metrics['prompt_tokens'] = []
        i = 0
        for h, p in zip(hit_tokens, prompt_tokens):
            if i > 0:
                delta_prompt_tokens = float(p) - float(prompt_tokens[i-1])
                metrics['prompt_tokens'].append(delta_prompt_tokens)
                if delta_prompt_tokens == 0:
                    metrics['hit_ratios'].append(0.0)
                else:
                    metrics['hit_ratios'].append(
                        (float(h) - float(hit_tokens[i-1])) / delta_prompt_tokens)
            i += 1
    
    return metrics


async def start_exp(config, tag):
    """Launch a single experiment: start server, run client, collect results."""
    # Launch the server and wait for it to be ready.
    server_log_file = await asyncio.to_thread(launch_server, config, tag)
    print("wait_for_server_ready:", server_log_file)
    is_ready = await asyncio.to_thread(wait_for_server_ready, server_log_file)
    if not is_ready:
        print(f"Server startup failed for port {config['port']}. Skipping this experiment.")
        kill_server(config['host'])
        return

    # Launch the client and wait for it to finish.
    client_stdout_log, client_stderr_log = await asyncio.to_thread(launch_client, config, tag)
    is_finished = await asyncio.to_thread(wait_for_client_finish, client_stdout_log, client_stderr_log)

    # Client finished, now parse the results.
    result = await asyncio.to_thread(parse_metrics_from_log, client_stdout_log)

    if not is_finished or len(result.get('hit_ratios', [])) == 0:
        print(f"Client run failed or timed out for port {config['port']}.")
        kill_server(config['host'])
        return

    # Merge config into results and save everything.
    for k in config.keys():
        result[k] = config[k]
    result["result_file"] = config['result_filename']
    
    with open(f'{DIR}/{config["result_filename"]}.config', 'w') as fp:
        json.dump(config, fp)

    exp_file = f"{DIR}/exp_{config['dataset_name']}_{tag}.json"
    
    if os.path.exists(exp_file):
        with open(exp_file, "r") as fp:
            try:
                existing_data = json.load(fp)
                if not isinstance(existing_data, list):
                    existing_data = []
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    existing_data.append(result)

    with open(exp_file, "w") as fp:
        json.dump(existing_data, fp, indent=4)
        
    print(f"Saved results to {exp_file}")


async def main(sizes, scales, alg, dataset, tag):
    configs = prepare_configs(sizes, scales, alg, dataset, tag)
    kill_server('localhost')  # Kill any lingering servers before starting
    tasks = [start_exp(config, tag) for config in configs]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="lmcache", 
                        help="The experiment to run.")
    args = parser.parse_args()

    if args.exp == "lmcache":
        for alg in ['lru']:
            for dataset in ['sharegpt', 'lmsys']:
                for sizes in [[256]]:
                    for scales in [[1]]:
                        asyncio.run(main(sizes, scales, alg, dataset, 'size-lmcache'))
    elif args.exp == "size-8b":
        for alg in ['ml-online-finetune', 'ml', 'lru', 'ml-online']:
            for dataset in ['lmsys', 'chatbot', 'sharegpt']:
                for sizes in [[4000, 6000, 8000, 10000]]:
                    for scales in [[1]]:
                        asyncio.run(main(sizes, scales, alg, dataset, 'size-8b'))
    elif args.exp == "della_reqrate":
        for alg in ['ml']:
            for dataset in ['lmsys', 'chatbot', 'sharegpt', 'tay']:
                for sizes in [[10000]]:
                    for scales in [[0.25, 0.5, 1, 2]]:
                        asyncio.run(main(sizes, scales, alg, dataset, 'reqrate++'))
    elif args.exp == "della_size":
        for dataset in ['sharegpt', 'lmsys', 'chatbot']:
            for alg in ['ml-online-finetune', 'ml', 'lru', 'ml-online']:
                for sizes in [[4000, 6000, 8000, 10000]]:
                    for scales in [[1]]:
                        asyncio.run(main(sizes, scales, alg, dataset, 'size-online')) 