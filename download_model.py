from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-8B-FP8",
    cache_dir="/scratch/gpfs/dy5/.cache/huggingface/hub/",
    resume_download=True
)

from datasets import load_dataset

# Download the dataset (default: train split)
dataset = load_dataset("lmsys/lmsys-chat-1m")