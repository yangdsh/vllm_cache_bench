import os

ENV = os.getenv('ENV', 'della')

if ENV == 'della':
    # ---della begin---
    SERVER_COMMAND_PREFIX = (
        f"export HF_HOME=/scratch/gpfs/dy5/.cache/huggingface/ && "
        f"source /usr/licensed/anaconda3/2024.6/etc/profile.d/conda.sh && "
        f"conda activate vllm-cuda121 && "
    )
    HOME = '/home/dy5'
    DATA_HOME = '/scratch/gpfs/dy5/.cache/huggingface'
    SERVER_COMMAND_SUFFIX = ""
    MODEL = "/scratch/gpfs/dy5/.cache/huggingface/hub/models--Qwen--Qwen3-32B-FP8/snapshots/98a63908b41686889a6ade39c37616e54d49974d"
    # ---della end---
else: # fat2
    SERVER_COMMAND_PREFIX = ""
    HOME = '/data/dongshengy'
    DATA_HOME = '/data/dongshengy'
    SERVER_COMMAND_SUFFIX = ""
    MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

DIR = f"results/{MODEL.split('/')[-1]}"
if not os.path.exists(DIR):
    os.makedirs(DIR)
LOG_FILE = f"{DIR}/vllm"

os.makedirs(f'{DIR}/configs', exist_ok=True)
os.makedirs(f'{DIR}/metrics', exist_ok=True)

VLLM_SERVER_CMD_TEMPLATE = (
    # "/usr/local/bin/nsys profile -o /tmp/0.nsys-rep -w true -t cuda,nvtx,osrt,cudnn,cublas 
    # -s cpu -f true -x false --duration=120 --cuda-graph-trace node " TRANSFORMERS_OFFLINE=1 
    f"VLLM_SERVER_DEV_MODE=1 VLLM_LOGGING_LEVEL=INFO PYTHONUNBUFFERED=1 "
    f" vllm serve {MODEL} --dtype half --gpu_memory_utilization 0.95 --disable-log-requests --uvicorn-log-level warning "
    "--max_num_seqs 512 --num-scheduler-steps 1 --max-model-len 16384 --disable_custom_all_reduce "
    f"--enable-chunked-prefill --enable-prefix-caching --pipeline-parallel-size 1 " # --no-enable-prefix-caching
    "{} "
)
if ENV != 'della':
    VLLM_SERVER_CMD_TEMPLATE += "" if "0.5B" in MODEL else " --tensor-parallel-size 4 "

CLIENT_CMD_TEMPLATE = (
    f"python ~/vllm/benchmarks/benchmark_serving.py --result-dir {DIR} "
    f"--save-result --model {MODEL} --endpoint /v1/chat/completions "
    "--dataset-path {} --dataset-name {}  --host {} --port {} "
    "--result-filename {} --num-prompts {} --request-rate {} --session-rate {} "
    "--checkpoint {} --use-oracle {} --use-token-id {} --use-lru {} --max-active-conversations {} "
    "--time-limit {} "
)

SERVER_READY_PATTERN = r"startup complete"
CUDA_OOM_PATTERN = r"CUDA out of memory"
ERROR_PATTERN = r"Traceback (most recent call last):"
RAISE_PATTERN = r"raise"


'''
CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 VLLM_LOGGING_LEVEL=INFO vllm serve Qwen/Qwen2.5-0.5B-Instruct --dtype=half --disable-log-requests --max_num_seqs 512 --num-scheduler-steps 1 --max-model-len 16384 --disable_custom_all_reduce --enable-chunked-prefill --enable-prefix-caching --gpu_memory_utilization 0.6  --pipeline-parallel-size 1 --port 8004  --eviction-algorithm ml --block-size 16
python ~/vllm/benchmarks/benchmark_serving.py --result-dir results/Qwen2.5-0.5B-Instruct --save-result --model Qwen/Qwen2.5-0.5B-Instruct --endpoint /v1/chat/completions --dataset-path lmsys/lmsys-chat-1m --dataset-name conversation  --host localhost --port 8004 --result-filename test.json --num-prompts 1000 --request-rate 0.01 --session-rate 4 --checkpoint /data/dongshengy/vllm/benchmarks/lmsys-chat-1m2.pt --use-token-id 1 --use-oracle 0 --use-lru 0
python ~/vllm/benchmarks/benchmark_serving.py --result-dir results/Qwen2.5-0.5B-Instruct --save-result --model Qwen/Qwen2.5-0.5B-Instruct --endpoint /v1/chat/completions --dataset-path ~/Scientific_Dialog-ShareGPT.json --dataset-name conversation  --host localhost --port 8004 --result-filename test.json --num-prompts 1000 --request-rate 0.1 --session-rate 4 --checkpoint /data/dongshengy/vllm/benchmarks/Tay5.pt --use-oracle 1 --use_token_id 1
python ~/vllm/benchmarks/benchmark_serving.py --result-dir results/Qwen2.5-0.5B-Instruct --save-result --model Qwen/Qwen2.5-0.5B-Instruct --endpoint /v1/chat/completions --dataset-path "lmsys/lmsys-chat-1m" --dataset-name conversation  --host localhost --port 8004 --result-filename 1746419748.json --num-prompts 3000 --request-rate 0.03 --session-rate 20 --checkpoint /data/dongshengy/vllm/benchmarks/checkpoints_lmsys-chat-1m/lmsys-chat-1m_epoch4_metric_0_6818.pt --use-oracle 0 --use-token-id 1 --use-lru 1 --max-active-conversations 100'''