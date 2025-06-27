import os

# Auto-detect environment based on current working directory
def detect_environment():
    current_dir = os.getcwd()
    if 'dy5' in current_dir:
        return 'della'
    elif 'dongshengy' in current_dir:
        return 'fat2'
    elif 'ubuntu' in current_dir:
        return 'ec2'
    else:
        # Default fallback, you can also raise an exception if preferred
        return 'della'

# Environment configuration dictionary for easy extension
ENV_CONFIGS = {
    'della': {
        'SERVER_COMMAND_PREFIX': (
            "export HF_HOME=/scratch/gpfs/dy5/.cache/huggingface/ && "
            "source /usr/licensed/anaconda3/2024.6/etc/profile.d/conda.sh && "
            "conda activate vllm-cuda121 && "
        ),
        'HOME': '/home/dy5',
        'DATA_HOME': '/scratch/gpfs/dy5/.cache/huggingface',
        'SERVER_COMMAND_SUFFIX': "",
        'MODEL': "/scratch/gpfs/dy5/.cache/huggingface/hub/models--Qwen--Qwen3-8B-FP8/snapshots/a29cae3df5d16cc895083497dad6ba9530c7d84c"
        # "/scratch/gpfs/dy5/.cache/huggingface/hub/models--Qwen--Qwen3-32B-FP8/snapshots/98a63908b41686889a6ade39c37616e54d49974d"
    },
    'fat2': {
        'SERVER_COMMAND_PREFIX': "",
        'HOME': '/data/dongshengy',
        'DATA_HOME': '/data/dongshengy',
        'SERVER_COMMAND_SUFFIX': "",
        'MODEL': "Qwen/Qwen2.5-0.5B-Instruct"
    },
    'ec2': {
        'SERVER_COMMAND_PREFIX': (
            "LMCACHE_CHUNK_SIZE=256 "
            "LMCACHE_LOCAL_CPU=True "
        ),
        'HOME': '/home/ubuntu',
        'DATA_HOME': '/home/ubuntu/vllm_cache_bench',
        'SERVER_COMMAND_SUFFIX': "",
        'MODEL': "/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-8B-FP8/snapshots/2df580c02b34307b00ccd91309e67ec5a89987a9"
    }
}

# Use environment variable if set, otherwise auto-detect
ENV = os.getenv('ENV', detect_environment())
CONFIG = ENV_CONFIGS[ENV]

SERVER_COMMAND_PREFIX = CONFIG['SERVER_COMMAND_PREFIX']
HOME = CONFIG['HOME']
DATA_HOME = CONFIG['DATA_HOME']
SERVER_COMMAND_SUFFIX = CONFIG['SERVER_COMMAND_SUFFIX']
MODEL = CONFIG['MODEL']

DIR = f"results/{MODEL.split('/')[-1]}"
if not os.path.exists(DIR):
    os.makedirs(DIR)
LOG_FILE = f"{DIR}/vllm"

os.makedirs(f'{DIR}/configs', exist_ok=True)
os.makedirs(f'{DIR}/metrics', exist_ok=True)

VLLM_SERVER_CMD_TEMPLATE = (
    "VLLM_SERVER_DEV_MODE=1 VLLM_LOGGING_LEVEL=INFO PYTHONUNBUFFERED=1 "
    f"vllm serve {MODEL} --dtype half --gpu_memory_utilization 0.95 --disable-log-requests "
    "--uvicorn-log-level warning "
    "--max_num_seqs 512 --num-scheduler-steps 1 --max-model-len 16384 --disable_custom_all_reduce "
    "--enable-chunked-prefill --enable-prefix-caching --pipeline-parallel-size 1 "
    "{args} "
)
if ENV == 'fat2':
    VLLM_SERVER_CMD_TEMPLATE += "" if "0.5B" in MODEL else " --tensor-parallel-size 4 "

# Refactored CLIENT_CMD_TEMPLATE to use named placeholders for clarity
CLIENT_CMD_TEMPLATE = (
    "PYTHONUNBUFFERED=1 python ../LMCache/benchmarks/multi-round-open-loop/benchmark_serving.py {args} "
)

if ENV != 'ec2':
    SERVER_READY_PATTERN = r"Capturing CUDA graph shapes: 100%"
else:
    SERVER_READY_PATTERN = r"Starting vLLM API server"
CUDA_OOM_PATTERN = r"CUDA out of memory"
ERROR_PATTERN = r"Traceback (most recent call last):"
RAISE_PATTERN = r"raise"


'''
CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 VLLM_LOGGING_LEVEL=INFO vllm serve Qwen/Qwen2.5-0.5B-Instruct --dtype=half --disable-log-requests --max_num_seqs 512 --num-scheduler-steps 1 --max-model-len 16384 --disable_custom_all_reduce --enable-chunked-prefill --enable-prefix-caching --gpu_memory_utilization 0.6  --pipeline-parallel-size 1 --port 8004  --eviction-algorithm ml --block-size 16
python ~/vllm/benchmarks/benchmark_serving.py --result-dir results/Qwen2.5-0.5B-Instruct --save-result --model Qwen/Qwen2.5-0.5B-Instruct --endpoint /v1/chat/completions --dataset-path lmsys/lmsys-chat-1m --dataset-name conversation  --host localhost --port 8004 --result-filename test.json --num-prompts 1000 --request-rate 0.01 --session-rate 4 --checkpoint /data/dongshengy/vllm/benchmarks/lmsys-chat-1m2.pt --use-token-id 1 --use-oracle 0 --use-lru 0
python ~/vllm/benchmarks/benchmark_serving.py --result-dir results/Qwen2.5-0.5B-Instruct --save-result --model Qwen/Qwen2.5-0.5B-Instruct --endpoint /v1/chat/completions --dataset-path ~/Scientific_Dialog-ShareGPT.json --dataset-name conversation  --host localhost --port 8004 --result-filename test.json --num-prompts 1000 --request-rate 0.1 --session-rate 4 --checkpoint /data/dongshengy/vllm/benchmarks/Tay5.pt --use-oracle 1 --use_token_id 1
python ~/vllm/benchmarks/benchmark_serving.py --result-dir results/Qwen2.5-0.5B-Instruct --save-result --model Qwen/Qwen2.5-0.5B-Instruct --endpoint /v1/chat/completions --dataset-path "lmsys/lmsys-chat-1m" --dataset-name conversation  --host localhost --port 8004 --result-filename 1746419748.json --num-prompts 3000 --request-rate 0.03 --session-rate 20 --checkpoint /data/dongshengy/vllm/benchmarks/checkpoints_lmsys-chat-1m/lmsys-chat-1m_epoch4_metric_0_6818.pt --use-oracle 0 --use-token-id 1 --use-lru 1 --max-active-conversations 100'''