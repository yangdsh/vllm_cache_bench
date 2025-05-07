import os

SERVER_COMMEND_PREFIX = ""
MODEL = "Qwen/Qwen2.5-0.5B-Instruct" # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" # 'Qwen/Qwen2.5-32B'
#MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DIR = f"results/{MODEL.split('/')[-1]}"
if not os.path.exists(DIR):
    os.makedirs(DIR)
LOG_FILE = f"{DIR}/vllm"

os.makedirs(f'{DIR}/configs', exist_ok=True)
os.makedirs(f'{DIR}/metrics', exist_ok=True)

VLLM_SERVER_CMD_TEMPLATE = (
    # "/usr/local/bin/nsys profile -o /tmp/0.nsys-rep -w true -t cuda,nvtx,osrt,cudnn,cublas 
    # -s cpu -f true -x false --duration=120 --cuda-graph-trace node " TRANSFORMERS_OFFLINE=1 
    f"VLLM_SERVER_DEV_MODE=1 VLLM_LOGGING_LEVEL=INFO"
    f" vllm serve {MODEL} --dtype=half --disable-log-requests "
    "--max_num_seqs 512 --num-scheduler-steps 1 --max-model-len 16384 --disable_custom_all_reduce "
    f"--enable-chunked-prefill --enable-prefix-caching "
    "{} "
)
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
ERROR_PATTERN = r"RuntimeError"
RAISE_PATTERN = r"raise"


'''
CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 VLLM_LOGGING_LEVEL=INFO vllm serve Qwen/Qwen2.5-0.5B-Instruct --dtype=half --disable-log-requests --max_num_seqs 512 --num-scheduler-steps 1 --max-model-len 16384 --disable_custom_all_reduce --enable-chunked-prefill --enable-prefix-caching --gpu_memory_utilization 0.6  --pipeline-parallel-size 1 --port 8004  --eviction-algorithm ml --block-size 16
python ~/vllm/benchmarks/benchmark_serving.py --result-dir results/Qwen2.5-0.5B-Instruct --save-result --model Qwen/Qwen2.5-0.5B-Instruct --endpoint /v1/chat/completions --dataset-path lmsys/lmsys-chat-1m --dataset-name conversation  --host localhost --port 8004 --result-filename test.json --num-prompts 1000 --request-rate 0.01 --session-rate 4 --checkpoint /data/dongshengy/vllm/benchmarks/lmsys-chat-1m2.pt --use-token-id 1 --use-oracle 0 --use-lru 0
python ~/vllm/benchmarks/benchmark_serving.py --result-dir results/Qwen2.5-0.5B-Instruct --save-result --model Qwen/Qwen2.5-0.5B-Instruct --endpoint /v1/chat/completions --dataset-path ~/Scientific_Dialog-ShareGPT.json --dataset-name conversation  --host localhost --port 8004 --result-filename test.json --num-prompts 1000 --request-rate 0.1 --session-rate 4 --checkpoint /data/dongshengy/vllm/benchmarks/Tay5.pt --use-oracle 1 --use_token_id 1
python ~/vllm/benchmarks/benchmark_serving.py --result-dir results/Qwen2.5-0.5B-Instruct --save-result --model Qwen/Qwen2.5-0.5B-Instruct --endpoint /v1/chat/completions --dataset-path "lmsys/lmsys-chat-1m" --dataset-name conversation  --host localhost --port 8004 --result-filename 1746419748.json --num-prompts 3000 --request-rate 0.03 --session-rate 20 --checkpoint /data/dongshengy/vllm/benchmarks/checkpoints_lmsys-chat-1m/lmsys-chat-1m_epoch4_metric_0_6818.pt --use-oracle 0 --use-token-id 1 --use-lru 1 --max-active-conversations 100'''