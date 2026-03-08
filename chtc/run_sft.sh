#!/bin/bash
#
# CHTC wrapper script for Safe-RLHF SFT (Supervised Fine-Tuning)
# This script runs inside the Docker container on CHTC execute nodes.
#
# Tested with: pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel
#
set -euo pipefail

echo "===== Job started at $(date) ====="
echo "Hostname: $(hostname)"
echo "Working directory: $(pwd)"
echo "Directory contents:"
ls -F
nvidia-smi || echo "nvidia-smi not available"

# ─── Arguments ────────────────────────────────────────────────────────────────
MODEL_NAME_OR_PATH="${1:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
OUTPUT_DIR="${2:-output/sft}"
ZERO_STAGE="${3:-3}"
OFFLOAD="${4:-none}"
JOB_SUFFIX="${5:-}"

# ─── PATH setup (must come before pip install so deepspeed CLI is found) ─────
export PATH="${HOME}/.local/bin:${PATH}"

# ─── Install dependencies ────────────────────────────────────────────────────
if [[ -f "env.tar.gz" ]]; then
    echo "Unpacking pre-built environment..."
    mkdir -p env && tar -xzf env.tar.gz -C env && source env/bin/activate
else
    echo "Installing dependencies on the fly..."
    # Install compatible PyTorch/Transformers stack.
    pip install --no-warn-script-location \
        --index-url https://download.pytorch.org/whl/cu124 \
        'torch==2.6.0+cu124' \
        'torchvision==0.21.0+cu124'
    # Keep transformers in 4.x for safe_rlhf model compatibility.
    pip install --no-warn-script-location \
        'transformers>=5.0.0' \
        'deepspeed>=0.12' \
        'accelerate>=0.25' \
        'datasets' \
        'tokenizers' \
        numpy scipy sentencepiece wandb tensorboard optree matplotlib tqdm rich
fi

torch_version_ok() {
    python - << 'PYEOF'
import re
import sys

try:
    import torch
except Exception:
    sys.exit(1)

version = torch.__version__.split('+', 1)[0]
parts = [int(x) for x in re.findall(r'\d+', version)[:3]]
while len(parts) < 3:
    parts.append(0)

sys.exit(0 if tuple(parts[:3]) >= (2, 6, 0) else 1)
PYEOF
}

if ! torch_version_ok; then
    echo "Detected torch<2.6.0; upgrading to satisfy transformers torch.load safety requirement..."
    pip install --no-warn-script-location \
        --index-url https://download.pytorch.org/whl/cu124 \
        'torch==2.6.0+cu124' \
        'torchvision==0.21.0+cu124'
fi

# ─── Environment variables ───────────────────────────────────────────────────
export LOGLEVEL="${LOGLEVEL:-WARNING}"
export WANDB_MODE="${WANDB_MODE:-offline}"

export USER="${USER:-condor}"
export LOGNAME="${LOGNAME:-${USER}}"

# ─── Cache directories ───────────────────────────────────────────────────────
mkdir -p "${HF_HOME:-./hf_cache}" \
         "${HF_DATASETS_CACHE:-./hf_cache/datasets}" \
         "${TRANSFORMERS_CACHE:-./hf_cache/transformers}" \
         "${TORCH_EXTENSIONS_DIR:-./hf_cache/torch_extensions}"

export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-./hf_cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-./hf_cache/triton}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

python -m pip install --no-warn-script-location --upgrade 'optree>=0.13.0'

# ─── Handle staging models ───────────────────────────────────────────────────
if [[ -f "model.tar.gz" ]]; then
    echo "Unpacking model from model.tar.gz..."
    mkdir -p model && tar -xzf model.tar.gz -C model
    MODEL_NAME_OR_PATH="./model"
fi

# ─── Create output directory ─────────────────────────────────────────────────
mkdir -p "${OUTPUT_DIR}"

# ─── Fix CUDA_VISIBLE_DEVICES for DeepSpeed ──────────────────────────────────
# CHTC/HTCondor sometimes sets CUDA_VISIBLE_DEVICES to GPU UUIDs
# (e.g. GPU-65329686-...) but DeepSpeed expects integer indices.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && echo "${CUDA_VISIBLE_DEVICES}" | grep -qi 'gpu'; then
    NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
    export CUDA_VISIBLE_DEVICES=$(seq -s ',' 0 $((NUM_GPUS - 1)))
    echo "Converted CUDA_VISIBLE_DEVICES to: ${CUDA_VISIBLE_DEVICES}"
fi

# ─── Create launcher wrapper ─────────────────────────────────────────────────
# Why a wrapper?  DeepSpeed spawns a child process like:
#   /opt/conda/bin/python -u <script> --local_rank=0 ...
#
# In this conda-Docker-Condor environment, neither PYTHONPATH nor user-site
# .pth files are reliably inherited by the child.  Using `--module` therefore
# always fails with "No module named safe_rlhf".
#
# By passing a real .py file, DeepSpeed resolves it to an absolute path and
# the child runs it directly.  The wrapper adds the project root to sys.path
# before importing anything, which guarantees safe_rlhf is found.
PROJECT_ROOT="$(pwd)"

cat > "${PROJECT_ROOT}/_launch_sft.py" << 'PYEOF'
import sys, os, runpy
# Ensure the project root (where safe_rlhf/ lives) is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
runpy.run_module("safe_rlhf.finetune", run_name="__main__", alter_sys=True)
PYEOF

echo "===== Starting SFT Training ====="
echo "Model : ${MODEL_NAME_OR_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "ZeRO  : ${ZERO_STAGE}"
echo "Offload: ${OFFLOAD}"

# ─── Run training ────────────────────────────────────────────────────────────
MASTER_PORT=$((RANDOM % 50000 + 10000))

deepspeed --master_port "${MASTER_PORT}" \
    "${PROJECT_ROOT}/_launch_sft.py" \
    --train_datasets alpaca \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --max_length 512 \
    --trust_remote_code True \
    --epochs 1 \
    --max_steps 30 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --lr_warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --seed 42 \
    --output_dir "${OUTPUT_DIR}" \
    --log_type wandb \
    --log_project Safe-RLHF-SFT \
    --zero_stage "${ZERO_STAGE}" \
    --offload "${OFFLOAD}" \
    --bf16 True \
    --tf32 True

# ─── Original training config (commented for debugging) ─────────────────────
# deepspeed --master_port "${MASTER_PORT}" \
#     "${PROJECT_ROOT}/_launch_sft.py" \
#     --train_datasets alpaca \
#     --model_name_or_path "${MODEL_NAME_OR_PATH}" \
#     --max_length 512 \
#     --trust_remote_code True \
#     --epochs 3 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 4 \
#     --gradient_checkpointing \
#     --learning_rate 2e-5 \
#     --lr_scheduler_type cosine \
#     --lr_warmup_ratio 0.03 \
#     --weight_decay 0.0 \
#     --seed 42 \
#     --output_dir "${OUTPUT_DIR}" \
#     --log_type wandb \
#     --log_project Safe-RLHF-SFT \
#     --zero_stage "${ZERO_STAGE}" \
#     --offload "${OFFLOAD}" \
#     --bf16 True \
#     --tf32 True

echo "===== Training finished at $(date) ====="

# ─── Pack output for transfer back ───────────────────────────────────────────
archive_name="sft_output.tar.gz"
if [[ -n "${JOB_SUFFIX}" ]]; then
    archive_name="sft_output_${JOB_SUFFIX}.tar.gz"
fi

if [[ -d "${OUTPUT_DIR}" && "$(ls -A "${OUTPUT_DIR}")" ]]; then
    echo "Packing output..."
    tar -czf "${archive_name}" "${OUTPUT_DIR}"
    echo "Output packed to ${archive_name}"
else
    echo "WARNING: No output produced. Creating empty marker."
    tar -czf "${archive_name}" --files-from /dev/null
fi

rm -f "${PROJECT_ROOT}/_launch_sft.py"
echo "===== Job complete ====="
