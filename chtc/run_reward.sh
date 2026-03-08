#!/bin/bash
#
# CHTC wrapper script for Safe-RLHF Reward Model Training
#
set -euo pipefail

echo "===== Job started at $(date) ====="
echo "Hostname: $(hostname)"
nvidia-smi || echo "nvidia-smi not available"

# ─── Arguments ────────────────────────────────────────────────────────────────
MODEL_NAME_OR_PATH="${1:-PKU-Alignment/alpaca-7b-reproduced}"
OUTPUT_DIR="${2:-output/rm}"
ZERO_STAGE="${3:-3}"
OFFLOAD="${4:-none}"

# ─── Environment Setup ───────────────────────────────────────────────────────
if [[ -f "env.tar.gz" ]]; then
    mkdir -p env && tar -xzf env.tar.gz -C env && source env/bin/activate
else
    # Upgrade PyTorch to satisfy newest Transformers requirements.
    pip install --no-warn-script-location \
        --index-url https://download.pytorch.org/whl/cu124 \
        'torch==2.4.1+cu124' \
        'torchvision==0.19.1+cu124'
    pip install --no-warn-script-location \
        'transformers>=4.37' 'deepspeed>=0.12' 'accelerate>=0.25' \
        'datasets' 'tokenizers>=0.13.3' \
        numpy scipy sentencepiece wandb tensorboard optree matplotlib tqdm rich
fi

export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"
export LOGLEVEL="${LOGLEVEL:-WARNING}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export USER="${USER:-$(id -un 2>/dev/null || echo condor)}"
export LOGNAME="${LOGNAME:-${USER}}"
export HOME="${HOME:-$(pwd)}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-./hf_cache}"
export HF_HOME="${HF_HOME:-./hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-./hf_cache/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-./hf_cache/transformers}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-./hf_cache/torch_extensions}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-./hf_cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-./hf_cache/triton}"

mkdir -p "${HF_HOME}" \
         "${HF_DATASETS_CACHE}" \
         "${TRANSFORMERS_CACHE}" \
         "${TORCH_EXTENSIONS_DIR}" \
         "${TORCHINDUCTOR_CACHE_DIR}" \
         "${TRITON_CACHE_DIR}"

if [[ -f "model.tar.gz" ]]; then
    mkdir -p model && tar -xzf model.tar.gz -C model
    MODEL_NAME_OR_PATH="./model"
fi

mkdir -p "${OUTPUT_DIR}"

echo "===== Starting Reward Model Training ====="

MASTER_PORT=$((RANDOM % 50000 + 10000))

deepspeed --master_port "${MASTER_PORT}" \
    --module safe_rlhf.values.reward \
    --train_datasets PKU-SafeRLHF/train:0.01 \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --max_length 512 \
    --trust_remote_code True \
    --epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --lr_warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --seed 42 \
    --output_dir "${OUTPUT_DIR}" \
    --log_type wandb \
    --log_project Safe-RLHF-RM \
    --zero_stage "${ZERO_STAGE}" \
    --offload "${OFFLOAD}" \
    --bf16 True \
    --tf32 True

# ─── Original training config (commented for debugging) ─────────────────────
# deepspeed --master_port "${MASTER_PORT}" \
#     --module safe_rlhf.values.reward \
#     --train_datasets PKU-SafeRLHF/train \
#     --model_name_or_path "${MODEL_NAME_OR_PATH}" \
#     --max_length 512 \
#     --trust_remote_code True \
#     --epochs 2 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 8 \
#     --gradient_checkpointing \
#     --learning_rate 2e-5 \
#     --lr_scheduler_type cosine \
#     --lr_warmup_ratio 0.03 \
#     --weight_decay 0.0 \
#     --seed 42 \
#     --output_dir "${OUTPUT_DIR}" \
#     --log_type wandb \
#     --log_project Safe-RLHF-RM \
#     --zero_stage "${ZERO_STAGE}" \
#     --offload "${OFFLOAD}" \
#     --bf16 True \
#     --tf32 True

echo "===== Training finished at $(date) ====="
if [[ -d "${OUTPUT_DIR}" && "$(ls -A "${OUTPUT_DIR}")" ]]; then
    tar -czf rm_output.tar.gz "${OUTPUT_DIR}"
else
    tar -czf rm_output.tar.gz --files-from /dev/null
fi
echo "===== Job complete ====="
