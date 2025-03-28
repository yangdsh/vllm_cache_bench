import os

MODEL = "Qwen/Qwen2.5-0.5B" # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" # 'Qwen/Qwen2.5-32B'
DIR = f"results/{MODEL.split('/')[-1]}"
if not os.path.exists(DIR):
    os.makedirs(DIR)
LOG_FILE = f"{DIR}/vllm"

os.makedirs(f'{DIR}/configs', exist_ok=True)
os.makedirs(f'{DIR}/metrics', exist_ok=True)

VLLM_SERVER_CMD_TEMPLATE = (
    # "/usr/local/bin/nsys profile -o /tmp/0.nsys-rep -w true -t cuda,nvtx,osrt,cudnn,cublas 
    # -s cpu -f true -x false --duration=120 --cuda-graph-trace node "
    f"VLLM_SERVER_DEV_MODE=1 VLLM_LOGGING_LEVEL=INFO"
    f" vllm serve {MODEL} --dtype=half --disable-log-requests "
    "--max_num_seqs 512 --num-scheduler-steps 1 --max-model-len 16384 --disable_custom_all_reduce "
    "--enable-chunked-prefill --enable-prefix-caching "
    "{} "
)

CLIENT_CMD_TEMPLATE = (
    f"python ~/vllm/benchmarks/benchmark_serving.py --result-dir {DIR} "
    f"--save-result --model {MODEL} --endpoint /v1/chat/completions "
    "--dataset-path {} --dataset-name {}  --host {} --port {} "
    "--result-filename {} --num-prompts {} --request-rate {}"
)

SERVER_READY_PATTERN = r"startup complete"
CUDA_OOM_PATTERN = r"CUDA out of memory"
ERROR_PATTERN = r"!error!"
RAISE_PATTERN = r"raise"