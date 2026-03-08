#!/bin/bash
#
# CHTC wrapper script for Safe-RLHF PPO-Lagrangian Training
#
set -euo pipefail

echo "===== Job started at $(date) ====="
echo "Hostname: $(hostname)"
nvidia-smi || echo "nvidia-smi not available"

# ─── Arguments ────────────────────────────────────────────────────────────────
ACTOR_MODEL_NAME_OR_PATH="${1:-PKU-Alignment/alpaca-7b-reproduced}"
REWARD_MODEL_PATH="${2:-output/rm}"
COST_MODEL_PATH="${3:-output/cm}"
OUTPUT_DIR="${4:-output/ppo-lag}"
ZERO_STAGE="${5:-3}"
OFFLOAD="${6:-none}"

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

# Unpack model tarballs if provided
for tarball in actor_model.tar.gz reward_model.tar.gz cost_model.tar.gz; do
    name="${tarball%.tar.gz}"
    if [[ -f "${tarball}" ]]; then
        mkdir -p "${name}" && tar -xzf "${tarball}" -C "${name}"
    fi
done
[[ -d "actor_model" ]] && ACTOR_MODEL_NAME_OR_PATH="./actor_model"
[[ -d "reward_model" ]] && REWARD_MODEL_PATH="./reward_model"
[[ -d "cost_model" ]] && COST_MODEL_PATH="./cost_model"

mkdir -p "${OUTPUT_DIR}"

echo "===== Starting PPO-Lag Training ====="

MASTER_PORT=$((RANDOM % 50000 + 10000))

deepspeed --master_port "${MASTER_PORT}" \
    --module safe_rlhf.algorithms.ppo_lag \
    --train_datasets PKU-SafeRLHF/train:0.005 \
    --ptx_datasets alpaca:0.01 \
    --actor_model_name_or_path "${ACTOR_MODEL_NAME_OR_PATH}" \
    --reward_model_name_or_path "${REWARD_MODEL_PATH}" \
    --cost_model_name_or_path "${COST_MODEL_PATH}" \
    --max_length 512 \
    --trust_remote_code True \
    --epochs 1 \
    --update_iters 1 \
    --per_device_prompt_batch_size 2 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --actor_lr 1e-6 \
    --actor_weight_decay 0.01 \
    --critic_lr 5e-6 \
    --critic_weight_decay 0.0 \
    --lr_scheduler_type cosine \
    --lr_warmup_ratio 0.03 \
    --seed 42 \
    --kl_coeff 0.02 \
    --clip_range_ratio 0.2 \
    --lambda_init 0.5 \
    --lambda_lr 0.01 \
    --threshold 25.0 \
    --output_dir "${OUTPUT_DIR}" \
    --log_type wandb \
    --log_project Safe-RLHF-PPO-Lag \
    --zero_stage "${ZERO_STAGE}" \
    --offload "${OFFLOAD}" \
    --bf16 True \
    --tf32 True

# ─── Original training config (commented for debugging) ─────────────────────
# deepspeed --master_port "${MASTER_PORT}" \
#     --module safe_rlhf.algorithms.ppo_lag \
#     --train_datasets PKU-SafeRLHF/train \
#     --ptx_datasets alpaca \
#     --actor_model_name_or_path "${ACTOR_MODEL_NAME_OR_PATH}" \
#     --reward_model_name_or_path "${REWARD_MODEL_PATH}" \
#     --cost_model_name_or_path "${COST_MODEL_PATH}" \
#     --max_length 512 \
#     --trust_remote_code True \
#     --epochs 1 \
#     --update_iters 1 \
#     --per_device_prompt_batch_size 8 \
#     --per_device_train_batch_size 4 \
#     --gradient_accumulation_steps 8 \
#     --gradient_checkpointing \
#     --actor_lr 1e-6 \
#     --actor_weight_decay 0.01 \
#     --critic_lr 5e-6 \
#     --critic_weight_decay 0.0 \
#     --lr_scheduler_type cosine \
#     --lr_warmup_ratio 0.03 \
#     --seed 42 \
#     --kl_coeff 0.02 \
#     --clip_range_ratio 0.2 \
#     --lambda_init 0.5 \
#     --lambda_lr 0.01 \
#     --threshold 25.0 \
#     --output_dir "${OUTPUT_DIR}" \
#     --log_type wandb \
#     --log_project Safe-RLHF-PPO-Lag \
#     --zero_stage "${ZERO_STAGE}" \
#     --offload "${OFFLOAD}" \
#     --bf16 True \
#     --tf32 True

echo "===== Training finished at $(date) ====="
if [[ -d "${OUTPUT_DIR}" && "$(ls -A "${OUTPUT_DIR}")" ]]; then
    tar -czf ppo_lag_output.tar.gz "${OUTPUT_DIR}"
else
    tar -czf ppo_lag_output.tar.gz --files-from /dev/null
fi
echo "===== Job complete ====="
