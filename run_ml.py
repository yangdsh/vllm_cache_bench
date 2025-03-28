from constants import DIR, CLIENT_CMD_TEMPLATE, VLLM_SERVER_CMD_TEMPLATE, SERVER_READY_PATTERN
import json
import subprocess
import time
import asyncio
from utils import kill_server, add_proc, add_client_procs
import os
import re
from constants import LOG_FILE, CUDA_OOM_PATTERN, ERROR_PATTERN, RAISE_PATTERN

async def main(sizes):
    server_configs = []
    i = 0
    for alg in ['lru-ml-bert6']:
        for size in sizes:
            server_configs.append({'host': 'localhost', 
                'cuda_devices': f'CUDA_VISIBLE_DEVICES={i+1}',
                'eviction_algorithm': alg,
                'port': 8000+i,
                'size': size,
                'args': f'--gpu_memory_utilization {size} '
                f' --pipeline-parallel-size 1 --port {8000+i} '       
                f' --eviction_algorithm {alg} --block_size=16'})
            i += 1

    dataset = 'sharegpt'
    dataset_file = '~/ShareGPT_V3_unfiltered_cleaned_split.json'
    client_configs = [
        {
            'num_prompts': 3000,
            'request_rate': 0.02,
        },
    ]

    def run_server(server_config):
        """Start the server with specified parallel sizes."""
        log_file_name = f"{LOG_FILE}_{server_config['port']}_{server_config['eviction_algorithm']}.log"
        server_cmd = VLLM_SERVER_CMD_TEMPLATE.format(server_config['args'])

        ssh_command = (
            f'ssh {server_config["host"]} '
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
            dataset_file, dataset, host, port, result_filename, num_prompts, request_rate
        )
        print("Running client command:", client_cmd)
        
        process = await asyncio.create_subprocess_shell(
            client_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        add_client_procs(process)  # <-- Don't overwrite process

        stdout, stderr = await process.communicate()
        log_file_name = f"{LOG_FILE}_{server_config['port']}_{server_config['eviction_algorithm']}_client.log"
        log_file = open(log_file_name, "w")
        print("Client stdout:", stdout.decode(), file=log_file)
        print("Client stderr:", stderr.decode(), file=log_file)
        hit_ratios = []
        for line in stdout.decode().split("\n"):
            if 'gpu_prefix_cache_hit_rate' in line:
                hit_ratios.append(line.split()[-1])
        
        with open(f'{DIR}/configs/config_{result_filename}', 'w') as fp:
            json.dump([client_config, server_config], fp)
        print('\n' + client_cmd + '\n')
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
        
        results = []
        for client_config in client_configs:
            result = await run_client(client_config, server_config)
            results.append(result)

        # Save results to `exp.json` in append mode
        exp_file = f"{DIR}/exp.json"
        
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
        existing_data.extend(results)

        # Write back to the file
        with open(exp_file, "w") as fp:
            json.dump(existing_data, fp, indent=4)

        print(f"Saved results to {exp_file}")

    # Stop any running server on this node.
    kill_server(server_configs[0]['host'])
    tasks = [start_exp(server_config, client_configs) for server_config in server_configs]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    for sizes in [[0.45, 0.5, 0.55, 0.6, 0.65, 0.7]]:
        asyncio.run(main(sizes))
        
