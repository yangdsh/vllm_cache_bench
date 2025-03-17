from constants import DIR, CLIENT_CMD_TEMPLATE, VLLM_SERVER_CMD_TEMPLATE, SERVER_READY_PATTERN
import json
import subprocess
import time
import asyncio
from utils import kill_server
import os
import re
from constants import LOG_FILE, CUDA_OOM_PATTERN, ERROR_PATTERN, RAISE_PATTERN

server_configs = []
i = 0
for alg in ['lruml', 'lru']:
    for size in [0.12, 0.1225, 0.125, 0.1275]:
        server_configs.append({'host': 'localhost', 
            'cuda_devices': f'CUDA_VISIBLE_DEVICES={i}',
            'eviction_algorithm': alg,
            'args': f'--gpu_memory_utilization {size} '
            f' --pipeline-parallel-size 1 --port {8000+i} '       
            f' --eviction_algorithm {alg} --block_size=16'})
        i += 1

dataset = 'sharegpt'
dataset_file = '~/ShareGPT_V3_unfiltered_cleaned_split.json'
client_configs = [
    {
        'num_prompts': 1000,
        'request_rate': 30,
    }
]

def run_server(server_config):
    """Start the server with specified parallel sizes."""
    log_file_name = f"{LOG_FILE}_{server_config['eviction_algorithm']}.log"
    server_cmd = VLLM_SERVER_CMD_TEMPLATE.format(server_config['args'])
    print('\n', server_cmd, '\n')
    ssh_command = (
        f"ssh {server_config['host']} \""
        # f"source /opt/conda/etc/profile.d/conda.sh && "  # Ensure Conda is sourced
        # f"conda activate pytorch && "  # Activate the environment
        f"{server_config['cuda_devices']} {server_cmd}\""  # Run the actual command
    )
    with open(log_file_name, "w") as log_file:
        process = subprocess.Popen(ssh_command, shell=True, stdout=log_file, stderr=log_file)

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
    
    # Use asyncio's subprocess shell to run the client command asynchronously.
    process = await asyncio.create_subprocess_shell(
        client_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    print("Client stdout:", stdout.decode())
    print("Client stderr:", stderr.decode())
    
    with open(f'{DIR}/configs/config_{result_filename}', 'w') as fp:
        json.dump([client_config, server_config], fp)
    print('\n' + client_cmd + '\n')

async def start_server(server_config):
    # Launch the server and wait for it to be ready.
    log_file_name = await asyncio.to_thread(run_server, server_config)
    print("wait_for_server_ready:", log_file_name)
    is_ready = await asyncio.to_thread(wait_for_server_ready, log_file_name)
    return is_ready

async def start_exp(server_config, client_configs):
    print("Starting server configuration:", server_config)
    await start_server(server_config)
    for client_config in client_configs:
        await run_client(client_config, server_config)

async def main():
    # Stop any running server on this node.
    kill_server(server_configs[0]['host'])
    tasks = [start_exp(server_config, client_configs) for server_config in server_configs]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
