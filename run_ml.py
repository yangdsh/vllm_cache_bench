from constants import DIR, CLIENT_CMD_TEMPLATE, VLLM_SERVER_CMD_TEMPLATE, SERVER_READY_PATTERN
import json
import subprocess
import time
import asyncio
from utils import kill_server, add_proc, add_client_procs
import os
import re
from constants import LOG_FILE, CUDA_OOM_PATTERN, ERROR_PATTERN, RAISE_PATTERN

async def main(sizes, alg, _dataset):
    server_configs = []
    i = 0
    for size in sizes:
        server_configs.append({'host': 'localhost', 
            'cuda_devices': f'CUDA_VISIBLE_DEVICES={i+1}' if len(sizes) > 1 else '',
            'eviction_algorithm': alg,
            'port': 8000+i,
            'size': size,
            'args': f'--num-gpu-blocks-override {size} '
            f' --pipeline-parallel-size 1 --port {8000+i} '       
            f' --eviction_algorithm {alg} --block_size=16'
        })
        i += 1

    
    '''        {
            'num_prompts': 10000,
            'session_rate': 20,
            'use_oracle': 1,
            'use_token_id': 1,
            'algorithm': 'ml-true-token'
        },
        {
            'num_prompts': 10000,
            'session_rate': 20,
            'use_oracle': 2,
            'use_token_id': 1,
            'algorithm': 'ml-oracle-token'
        },'''
    client_configs = [
        {
            'num_prompts': 10000,
            'session_rate': 20,
            'use_oracle': 0,
            'use_token_id': 1,
            'algorithm': 'ml-token'
        },
        {
            'num_prompts': 10000,
            'session_rate': 20,
            'use_oracle': 0,
            'use_token_id': 1,
            'algorithm': 'lru-token'
        }
    ]
    for c in client_configs:
        if _dataset == 'tay':
            c['checkpoint'] = '/data/dongshengy/vllm/benchmarks/checkpoints_tay_20/tay_epoch17_metric_0_6332.pt'
            c['dataset_file'] = '~/tay.json'
            c['dataset_name'] = 'tay0422'
            c['request_rate'] = 0.1
            c['max_active_conversations'] = 300
            c['time_limit'] = 1200
        elif _dataset == 'chatbot001':
            c['request_rate'] = 0.01
            c['checkpoint'] = '/data/dongshengy/vllm/benchmarks/checkpoints_chatbot_arena_20/chatbot_arena_epoch16_metric_0_4005.pt'
            c['dataset_file'] = '"lmsys/chatbot_arena_conversations"'
            c['max_active_conversations'] = 300
            c['time_limit'] = 1200
            c['dataset_name'] = 'chatbot001-1200'
        elif _dataset == 'lmsys001':
            c['request_rate'] = 0.01
            c['checkpoint'] = '/data/dongshengy/vllm/benchmarks/checkpoints_lmsys-chat-1m/lmsys-chat-1m_epoch4_metric_0_6818.pt'
            c['dataset_file'] = '"lmsys/lmsys-chat-1m"'
            c['max_active_conversations'] = 300
            c['time_limit'] = 1200
            c['dataset_name'] = 'lmsys001-1200'
        elif _dataset == 'lmsys003':
            c['request_rate'] = 0.03
            c['checkpoint'] = '/data/dongshengy/vllm/benchmarks/checkpoints_lmsys-chat-1m/lmsys-chat-1m_epoch4_metric_0_6818.pt'
            c['dataset_file'] = '"lmsys/lmsys-chat-1m"'
            c['time_limit'] = 1200
            c['max_active_conversations'] = 100
            c['dataset_name'] = 'lmsys003-1200'
        elif _dataset == 'lmsys0003':
            c['request_rate'] = 0.003
            c['checkpoint'] = '/data/dongshengy/vllm/benchmarks/checkpoints_lmsys-chat-1m/lmsys-chat-1m_epoch4_metric_0_6818.pt'
            c['dataset_file'] = '"lmsys/lmsys-chat-1m"'
            c['max_active_conversations'] = 300
            c['time_limit'] = 1200
            c['dataset_name'] = 'lmsys0003-1200'
        elif _dataset == 'sharegpt001':
            c['request_rate'] = 0.01
            c['time_limit'] = 1200
            c['max_active_conversations'] = 300
            c['checkpoint'] = '/data/dongshengy/vllm/benchmarks/checkpoints_sharegpt_20/sharegpt_epoch4_metric_0_5035.pt'
            c['dataset_file'] = '~/ShareGPT_V3_unfiltered_cleaned_split.json'
            c['dataset_name'] = 'sharegpt001-1200'
        elif _dataset == 'sharegpt003':
            c['request_rate'] = 0.03
            c['time_limit'] = 1200
            c['max_active_conversations'] = 100
            c['checkpoint'] = '/data/dongshengy/vllm/benchmarks/checkpoints_sharegpt_20/sharegpt_epoch4_metric_0_5035.pt'
            c['dataset_file'] = '~/ShareGPT_V3_unfiltered_cleaned_split.json'
            c['dataset_name'] = 'sharegpt003-1200'
        elif _dataset == 'gpt001':
            c['request_rate'] = 0.01
            c['time_limit'] = 1200
            c['max_active_conversations'] = 100
            c['checkpoint'] = '/data/dongshengy/vllm/benchmarks/checkpoints_gpt4_20/gpt4_epoch13_metric_0_5642.pt'
            c['dataset_file'] = 'lightblue/gpt4_conversations_multilingual'
            c['dataset_name'] = 'gpt001-1200-100'
        elif _dataset == 'science':
            c['request_rate'] = 0.1
            c['num_prompts'] = 1000
            c['session_rate'] = 0.5
            c['checkpoint'] = '/data/dongshengy/vllm/benchmarks/sharegpt2.pt'
            c['dataset_file'] = '~/Scientific_Dialog-ShareGPT.json'
            c['dataset_name'] = 'science0428'
            c['add_ending_request'] = 1
        elif _dataset == 'code':
            c['request_rate'] = 0.1
            c['num_prompts'] = 1000
            c['session_rate'] = 0.5
            c['checkpoint'] = '/data/dongshengy/vllm/benchmarks/sharegpt2.pt'
            c['dataset_file'] = '~/Code-290k-ShareGPT.json'
            c['dataset_name'] = 'code0428'
            c['add_ending_request'] = 1
        elif _dataset == 'math':
            c['request_rate'] = 0.1
            c['num_prompts'] = 1000
            c['session_rate'] = 0.5
            c['checkpoint'] = '/data/dongshengy/vllm/benchmarks/sharegpt2.pt'
            c['dataset_file'] = '~/Olympiad_Math-ShareGPT.json'
            c['dataset_name'] = 'math0428'
            c['add_ending_request'] = 1

    def run_server(server_config):
        """Start the server with specified parallel sizes."""
        log_file_name = f"{LOG_FILE}_{server_config['port']}_{server_config['eviction_algorithm']}.log"
        server_cmd = VLLM_SERVER_CMD_TEMPLATE.format(server_config['args'])

        ssh_command = (
            #f'ssh {server_config["host"]} '
            # f"source /opt/conda/etc/profile.d/conda.sh && "  # Ensure Conda is sourced
            # f"conda activate pytorch && "  # Activate the environment
            f"{server_config['cuda_devices']} {server_cmd}"  # Run the actual command
        )
        print('\n', ssh_command, '\n')
        with open(log_file_name, "w") as log_file:
            process = subprocess.Popen(ssh_command, shell=True, stdout=log_file, stderr=log_file)
            add_proc(process)

        return log_file_name


    def wait_for_server_ready(log_file_name, timeout=600):
        """Wait until the server is ready or a timeout occurs."""
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

    def get_file_name(server_config):
        # Use a simple timestamp for a unique result filename.
        return str(int(time.time()))

    async def run_client(client_config, server_config):
        num_prompts = client_config['num_prompts']
        request_rate = client_config['request_rate']
        session_rate = client_config['session_rate']
        checkpoint = client_config['checkpoint']
        max_active_conversations = client_config['max_active_conversations'] \
            if 'max_active_conversations' in client_config else 10000
        time_limit = client_config['time_limit'] if 'time_limit' in client_config else 10000
        use_oracle = client_config['use_oracle'] if 'use_oracle' in client_config else 0
        use_token_id = client_config['use_token_id'] if 'use_token_id' in client_config else 0
        use_lru = 1 if 'lru' in client_config['algorithm'] else 0
        prefix = get_file_name(server_config)
        result_filename = f"{prefix}.json"
        
        # Extract host directly from the dictionary.
        host = server_config["host"]
        
        # Extract the port from the 'args' string.
        port_match = re.search(r'--port\s+(\d+)', server_config["args"])
        if not port_match:
            raise ValueError("Port not found in server configuration 'args'.")
        port = port_match.group(1)
        
        client_cmd = CLIENT_CMD_TEMPLATE.format(
            client_config['dataset_file'], 'conversation', host, 
            port, result_filename, num_prompts, request_rate, session_rate,
            checkpoint, use_oracle, use_token_id, use_lru, max_active_conversations,
            time_limit
        )
        print("Running client command:", client_cmd)
        
        process = await asyncio.create_subprocess_shell(
            client_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        add_client_procs(process)  # <-- Don't overwrite process

        stdout, stderr = await process.communicate()
        log_file_name = f"{LOG_FILE}_{server_config['port']}_{client_config['algorithm']}_client.log"
        log_file = open(log_file_name, "w")
        print("Client stdout:", stdout.decode(), file=log_file)
        print("Client stderr:", stderr.decode(), file=log_file)
        hit_ratios = []
        for line in stdout.decode().split("\n"):
            if 'gpu_prefix_cache_hit_rate' in line:
                hit_ratios.append(line.split()[-1])
        
        with open(f'{DIR}/configs/config_{result_filename}', 'w') as fp:
            json.dump([client_config, server_config], fp)
        print('Done client: ' + client_cmd + '\n')
        result = {'hit_ratios': hit_ratios}
        for k in ['eviction_algorithm', 'size']:
            result[k] = server_config[k]
        for k in client_config.keys():
            result[k] = client_config[k]
        return result

    async def start_server(server_config):
        # Launch the server and wait for it to be ready.
        log_file_name = await asyncio.to_thread(run_server, server_config)
        print("wait_for_server_ready:", log_file_name)
        is_ready = await asyncio.to_thread(wait_for_server_ready, log_file_name)
        return is_ready

    async def start_exp(server_config, client_configs):
        print("Starting server configuration:", server_config)
        await start_server(server_config)
        
        for i, client_config in enumerate(client_configs):
            #if server_config['eviction_algorithm'] not in client_config['algorithm']:
            #    continue
            result = await run_client(client_config, server_config)
            dataset_name = client_config['dataset_name']

            # Save results to `exp.json` in append mode
            exp_file = f"{DIR}/exp_{dataset_name}.json"
            
            # Load existing data if the file exists
            if os.path.exists(exp_file):
                with open(exp_file, "r") as fp:
                    try:
                        existing_data = json.load(fp)
                        if not isinstance(existing_data, list):
                            existing_data = []  # Reset if data is corrupted
                    except json.JSONDecodeError:
                        existing_data = []  # Reset if file is empty or corrupted
            else:
                existing_data = []

            # Append new results
            existing_data.append(result)

            # Write back to the file
            with open(exp_file, "w") as fp:
                json.dump(existing_data, fp, indent=4)

            print(f"Saved results to {exp_file}")

    # Stop any running server on this node.
    kill_server(server_configs[0]['host'])
    tasks = [start_exp(server_config, client_configs) for server_config in server_configs]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    for alg in ['ml']:
        for dataset in ['chatbot001']: #'tay', 'sharegpt001', 'lmsys001', 'science', 'math', 'code'
            for sizes in [[10000, 12000, 14000, 16000, 20000]]: #13000, 20000, 24000, 28000, 32000   3000, 5000, 7000, 10000, 16000
                asyncio.run(main(sizes, alg, dataset))
        
