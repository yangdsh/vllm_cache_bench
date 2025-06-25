import subprocess
import os
import re
import time
import json
import argparse

# --- Constants for non-della (fat2) environment ---
ENV = 'fat2'
SERVER_COMMAND_PREFIX = ""
HOME = '/data/dongshengy'
DATA_HOME = '/data/dongshengy'
SERVER_COMMAND_SUFFIX = ""
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DIR = f"results/{MODEL.split('/')[-1]}"
os.makedirs(DIR, exist_ok=True)
SERVER_READY_PATTERN = r"Capturing CUDA graph shapes: 100%"

VLLM_SERVER_CMD_TEMPLATE = (
    "VLLM_SERVER_DEV_MODE=1 VLLM_LOGGING_LEVEL=WARNING PYTHONUNBUFFERED=1 "
    " vllm serve {} --dtype half --gpu_memory_utilization 0.95 --disable-log-requests "
    "--max_num_seqs 512 --num-scheduler-steps 1 --max-model-len 16384 --disable_custom_all_reduce "
    "--enable-chunked-prefill --enable-prefix-caching --pipeline-parallel-size 1 "
    '--uvicorn-log-level warning '
)
if "0.5B" not in MODEL:
    VLLM_SERVER_CMD_TEMPLATE += " --tensor-parallel-size 4 "
CLIENT_CMD_TEMPLATE = (
    f"PYTHONUNBUFFERED=1 python ~/vllm/benchmarks/benchmark_serving.py --result-dir {DIR} "
    "--save-result --model {} --endpoint /v1/chat/completions "
    "--dataset-path {} --dataset-name {} --host {} --port {} "
    "--result-filename {} --num-prompts {} --request-rate {} --session-rate {} "
    "--checkpoint {} --use-oracle {} --use-token-id {} --use-lru {} --max-active-conversations {} "
    "--time-limit {} "
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
server_args = (
    f' {MODEL} '
    f'--port {server_config["port"]} '
    f'--eviction_algorithm {server_config["eviction_algorithm"]} '
    f'--max-num-batched-tokens 2048 '
    f'--num-gpu-blocks-override {server_config["size"]} '
    f"--eviction_algorithm_config '{eviction_algorithm_config_str}'"
)
server_cmd = VLLM_SERVER_CMD_TEMPLATE.format(server_args)
ssh_command = f"{SERVER_COMMAND_PREFIX} {server_config['cuda_devices']} {server_cmd} {SERVER_COMMAND_SUFFIX}"

# --- Client config ---
client_config = {
    'num_prompts': 30000,
    'use_oracle': 0,
    'use_token_id': 1,
    'algorithm': 'ml',
    'request_rate': 0.01,
    'session_rate': 10,
    'max_active_conversations': 200,
    'checkpoint': f'{HOME}/vllm/benchmarks/checkpoints_lmsys-chat-1m_20/lmsys-chat-1m_epoch11_metric_0_5797.pt',
    'dataset_file': '"lmsys/lmsys-chat-1m"',
    'time_limit': 1200
}
client_cmd = CLIENT_CMD_TEMPLATE.format(
    MODEL,
    client_config['dataset_file'], 'conversation', server_config['host'],
    server_config['port'], 'client.log', client_config['num_prompts'],
    client_config['request_rate'], client_config['session_rate'],
    client_config['checkpoint'], client_config['use_oracle'],
    client_config['use_token_id'], 0,  # use_lru=0 for 'ml'
    client_config['max_active_conversations'], client_config['time_limit']
)
client_cmd = f"{server_config['cuda_devices']} {client_cmd}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-to-stdout", action="store_true", help="Output logs to stdout/stderr instead of server.log/client.log")
    args = parser.parse_args()

    # --- Start server ---
    server_log = open("server.log", "w")

    print(ssh_command)
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