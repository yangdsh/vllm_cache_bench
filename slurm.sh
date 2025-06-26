#!/bin/bash
#SBATCH --job-name=vllm_cache         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=128G                # memory
#SBATCH --gres=gpu:4           # number of gpus per node
#SBATCH --time=9:59:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=dy5@princeton.edu
#SBATCH --partition=pli
#SBATCH --account=prefixcache

module purge
module load anaconda3/2024.6
conda activate vllm-cuda121
cd /home/dy5/vllm_cache_bench
python3 run.py --exp size-8b

#sbatch slurm.sh
#conda env create -f environment.yml
#VLLM_USE_PRECOMPILED=1 pip install --editable .
#squeue -u dy5 --start
#sshare -l | grep dy5
#shownodes -p pli
#scancel -u dy5
#install nvcc
#salloc --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=48G --time=02:59:00 --gres=gpu:1 --partition=pli --account=prefixcache --mail-type=begin
#srun --jobid=64403317 --pty bash

#VLLM_SERVER_DEV_MODE=1 vllm serve /scratch/gpfs/dy5/.cache/huggingface/hub/models--Qwen--Qwen3-32B-FP8/snapshots/98a63908b41686889a6ade39c37616e54d49974d --gpu_memory_utilization 0.95 --disable-log-requests --max_num_seqs 512 --num-scheduler-steps 1 --max-model-len 16384 --disable_custom_all_reduce --enable-chunked-prefill --enable-prefix-caching --pipeline-parallel-size 1 --num-gpu-blocks-override 10000  --port 8000  --eviction_algorithm lru --max-num-batched-tokens 2048 > server.log 2>&1 &
#./monitor
#CUDA_VISIBLE_DEVICES=0 python ~/vllm/benchmarks/benchmark_serving.py --result-dir results/98a63908b41686889a6ade39c37616e54d49974d --save-result --model /scratch/gpfs/dy5/.cache/huggingface/hub/models--Qwen--Qwen3-32B-FP8/snapshots/98a63908b41686889a6ade39c37616e54d49974d --endpoint /v1/chat/completions --dataset-path "lmsys/chatbot_arena_conversations" --dataset-name conversation  --host localhost --port 8000 --result-filename chatbot-out3000-reqrate/client_logs/8000_ml.json --num-prompts 30000 --request-rate 0.02 --session-rate 1 --checkpoint /home/dy5/vllm/benchmarks/checkpoints_chatbot_arena_20/chatbot_arena_epoch16_metric_0_4005.pt --use-oracle 0 --use-token-id 1 --use-lru 0 --max-active-conversations 400 --time-limit 3600
#CUDA_VISIBLE_DEVICES=0 python ~/vllm/benchmarks/benchmark_serving.py --result-dir results/98a63908b41686889a6ade39c37616e54d49974d --save-result --model /scratch/gpfs/dy5/.cache/huggingface/hub/models--Qwen--Qwen3-32B-FP8/snapshots/98a63908b41686889a6ade39c37616e54d49974d --endpoint /v1/chat/completions --dataset-path "lmsys/lmsys-chat-1m" --dataset-name conversation  --host localhost --port 8000 --result-filename lmsys-size++/client_logs/8000_ml.json --num-prompts 30000 --request-rate 0.01 --session-rate 10 --checkpoint /home/dy5/vllm/benchmarks/checkpoints_lmsys-chat-1m_20/lmsys-chat-1m_epoch11_metric_0_5797.pt --use-oracle 0 --use-token-id 1 --use-lru 0 --max-active-conversations 200 --time-limit 1200
#python ~/vllm/benchmarks/benchmark_serving.py --result-dir . --save-result --model /scratch/gpfs/dy5/.cache/huggingface/hub/models--Qwen--Qwen3-32B-FP8/snapshots/98a63908b41686889a6ade39c37616e54d49974d --endpoint /v1/chat/completions --dataset-path test.json --dataset-name conversation  --host localhost --port 8000 --result-filename 1024-512.json --num-prompts 50000 --request-rate 1 --session-rate 10000 --checkpoint /home/dy5/vllm/benchmarks/checkpoints_lmsys-chat-1m/lmsys-chat-1m_epoch4_metric_0_6818.pt --use-oracle 0 --use-token-id 1 --use-lru 0 --max-active-conversations 200 --time-limit 1200 