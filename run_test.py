import subprocess
import os
import re
import time
import json
import argparse

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

# --- Server config ---
server_port = 8000
server_config = {
    'host': 'localhost',
    'cuda_devices': 'CUDA_VISIBLE_DEVICES=0',
    'eviction_algorithm': 'ml',
    'port': server_port,
    'size': 4000,
    'scale': 1,
}
eac = {"enable_online_learning": 1, 
# "model_path": f'{HOME}/vllm/benchmarks/checkpoints_lmsys-chat-1m_20/lmsys-chat-1m_epoch11_metric_0_5797.pt'
}
eviction_algorithm_config_str = json.dumps(eac)
if ENV == 'ec2':
    kv_config = {"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}
    server_args = (
        f'--port {server_config["port"]} '
        f'--max-num-batched-tokens 2048 '
        f'--num-gpu-blocks-override {server_config["size"]} '
        f'--kv-transfer-config \'{json.dumps(kv_config)}\''
    )
else:
    server_args = (
        f'--port {server_config["port"]} '
        f'--eviction_algorithm {server_config["eviction_algorithm"]} '
        f'--max-num-batched-tokens 2048 '
        f'--num-gpu-blocks-override {server_config["size"]} '
        f"--eviction_algorithm_config '{eviction_algorithm_config_str}'"
    )
server_cmd = VLLM_SERVER_CMD_TEMPLATE.format(args=server_args)
ssh_command = f"{SERVER_COMMAND_PREFIX} {server_config['cuda_devices']} {server_cmd} {SERVER_COMMAND_SUFFIX}"

# --- Client args config ---
all_args = {
    # Fixed arguments
    'result-dir': DIR,
    'model': MODEL,
    'endpoint': '/v1/chat/completions',
    'dataset-name': 'conversation',
    'host': server_config['host'],
    'port': server_config['port'],
    'result-filename': 'client.log',
    'use-lru': 0,
    
    # Client arguments
    'num-prompts': 30000,
    'use-oracle': 0,
    'use-token-id': 1,
    'request-rate': 0.01,
    'session-rate': 10,
    'max-active-conversations': 200,
    'checkpoint': f'{HOME}/vllm/benchmarks/checkpoints_lmsys-chat-1m_20/lmsys-chat-1m_epoch11_metric_0_5797.pt',
    'dataset-path': '"lmsys/lmsys-chat-1m"',
    'time-limit': 1200,

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

client_args = " ".join(client_args_list)
client_cmd = CLIENT_CMD_TEMPLATE.format(args=client_args)
client_cmd = f"{server_config['cuda_devices']} {client_cmd}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-to-stdout", action="store_true", help="Output logs to stdout/stderr instead of server.log/client.log")
    args = parser.parse_args()

    # --- Start server ---
    server_log = open("server.log", "w")

    print(ssh_command)
    print("server.log")
    server_proc = subprocess.Popen(ssh_command, shell=True, stdout=server_log, stderr=server_log)

    # --- Wait for server to be ready ---
    ready = False
    for _ in range(600):
        if os.path.exists("server.log"):
            with open("server.log") as f:
                if re.search(SERVER_READY_PATTERN, f.read()):
                    ready = True
                    break
        time.sleep(1)
    if not ready:
        print("Server not ready, exiting.")
        server_proc.terminate()
        exit(1)

    # --- Start client ---
    client_log = None
    print(client_cmd)
    subprocess.run(client_cmd, shell=True, stdout=client_log, stderr=client_log)