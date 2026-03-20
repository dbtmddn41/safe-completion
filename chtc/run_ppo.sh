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
TRAIN_PROPORTION="${7:-${TRAIN_PROPORTION:-0.005}}"
PTX_PROPORTION="${8:-${PTX_PROPORTION:-0.01}}"
PPO_EPOCHS="${9:-${PPO_EPOCHS:-3}}"
PPO_UPDATE_ITERS="${10:-${PPO_UPDATE_ITERS:-1}}"

# ─── Environment Setup ───────────────────────────────────────────────────────
if [[ -f "env.tar.gz" ]]; then
    mkdir -p env && tar -xzf env.tar.gz -C env && source env/bin/activate
else
    # Keep torch >=2.6 for secure checkpoint loading required by newer Transformers.
    pip install --no-warn-script-location \
        --index-url https://download.pytorch.org/whl/cu124 \
        'torch>=2.6,<2.8' \
        'torchvision>=0.21,<0.23'
    pip install --no-warn-script-location \
        'transformers>=4.37' 'deepspeed>=0.12' 'accelerate>=0.25' \
        'datasets' 'tokenizers>=0.13.3' \
        numpy scipy sentencepiece wandb tensorboard optree matplotlib tqdm rich
fi

export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"
export LOGLEVEL="${LOGLEVEL:-WARNING}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-Safe-RLHF-PPO-Lag}"
if [[ "${WANDB_MODE}" != "offline" && -z "${WANDB_API_KEY:-}" ]]; then
    echo "WANDB_API_KEY is not set; falling back to WANDB_MODE=offline."
    export WANDB_MODE="offline"
fi
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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${HF_HOME}" \
         "${HF_DATASETS_CACHE}" \
         "${TRANSFORMERS_CACHE}" \
         "${TORCH_EXTENSIONS_DIR}" \
         "${TORCHINDUCTOR_CACHE_DIR}" \
         "${TRITON_CACHE_DIR}"

# CHTC/HTCondor can expose GPUs as UUIDs, but DeepSpeed expects ordinal indices.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && echo "${CUDA_VISIBLE_DEVICES}" | grep -qi 'gpu'; then
    num_gpus=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
    export CUDA_VISIBLE_DEVICES=$(seq -s ',' 0 $((num_gpus - 1)))
fi

extract_tarballs_to_dir() {
    local target_dir="$1"
    shift
    mkdir -p "${target_dir}"
    for tarball in "$@"; do
        if [[ -f "${tarball}" ]]; then
            tar -xzf "${tarball}" -C "${target_dir}"
        fi
    done
}

resolve_model_dir() {
    local root_dir="$1"
    if [[ ! -d "${root_dir}" ]]; then
        return 1
    fi
    if [[ -f "${root_dir}/config.json" ]]; then
        printf '%s\n' "${root_dir}"
        return 0
    fi

    # Find the first nested directory that looks like a HF model folder.
    local cfg
    cfg="$(find "${root_dir}" -type f -name config.json | head -n 1 || true)"
    if [[ -n "${cfg}" ]]; then
        dirname "${cfg}"
        return 0
    fi
    return 1
}

# Unpack model tarballs from either generic names or staged rm/cm output names.
extract_tarballs_to_dir "actor_model" actor_model.tar.gz
extract_tarballs_to_dir "reward_model" reward_model.tar.gz rm_output*.tar.gz
extract_tarballs_to_dir "cost_model" cost_model.tar.gz cm_output*.tar.gz

if resolved_actor="$(resolve_model_dir "actor_model")"; then
    ACTOR_MODEL_NAME_OR_PATH="${resolved_actor}"
fi
if resolved_reward="$(resolve_model_dir "reward_model")"; then
    REWARD_MODEL_PATH="${resolved_reward}"
fi
if resolved_cost="$(resolve_model_dir "cost_model")"; then
    COST_MODEL_PATH="${resolved_cost}"
fi

mkdir -p "${OUTPUT_DIR}"

echo "===== Starting PPO-Lag Training ====="
echo "WandB mode: ${WANDB_MODE}"

MASTER_PORT=$((RANDOM % 50000 + 10000))

deepspeed --master_port "${MASTER_PORT}" \
    --module safe_rlhf.algorithms.ppo_lag \
    --train_datasets "PKU-SafeRLHF/train:${TRAIN_PROPORTION}" \
    --ptx_datasets "alpaca:${PTX_PROPORTION}" \
    --actor_model_name_or_path "${ACTOR_MODEL_NAME_OR_PATH}" \
    --reward_model_name_or_path "${REWARD_MODEL_PATH}" \
    --cost_model_name_or_path "${COST_MODEL_PATH}" \
    --max_length 512 \
    --trust_remote_code True \
    --epochs "${PPO_EPOCHS}" \
    --update_iters "${PPO_UPDATE_ITERS}" \
    --per_device_prompt_batch_size 8 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --actor_gradient_checkpointing \
    --critic_gradient_checkpointing \
    --actor_lr 1e-6 \
    --actor_weight_decay 0.01 \
    --actor_lr_scheduler_type cosine \
    --actor_lr_warmup_ratio 0.03 \
    --critic_lr 5e-6 \
    --critic_weight_decay 0.0 \
    --critic_lr_scheduler_type cosine \
    --critic_lr_warmup_ratio 0.03 \
    --seed 42 \
    --kl_coeff 0.02 \
    --clip_range_ratio 0.2 \
    --lambda_init 0.5 \
    --lambda_lr 0.01 \
    --threshold 25.0 \
    --output_dir "${OUTPUT_DIR}" \
    --log_type wandb \
    --log_project "${WANDB_PROJECT}" \
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
mkdir -p artifacts
JOB_SUFFIX="${JOB_SUFFIX:-local_0}"
ARCHIVE_PATH="artifacts/ppo_lag_output_${JOB_SUFFIX}.tar.gz"

is_inference_model_dir() {
    local model_dir="$1"
    [[ -f "${model_dir}/config.json" ]] || return 1

    if compgen -G "${model_dir}/model.safetensors" > /dev/null; then
        return 0
    fi
    if compgen -G "${model_dir}/model-*.safetensors" > /dev/null; then
        return 0
    fi
    if compgen -G "${model_dir}/pytorch_model.bin" > /dev/null; then
        return 0
    fi
    if compgen -G "${model_dir}/pytorch_model-*.bin" > /dev/null; then
        return 0
    fi
    if compgen -G "${model_dir}/adapter_model.safetensors" > /dev/null; then
        return 0
    fi

    return 1
}

copy_if_exists() {
    local src_dir="$1"
    local dst_dir="$2"
    local pattern="$3"
    local copied=1

    shopt -s nullglob
    for item in "${src_dir}"/${pattern}; do
        [[ -e "${item}" ]] || continue
        cp -a "${item}" "${dst_dir}/"
        copied=0
    done
    shopt -u nullglob

    return ${copied}
}

select_actor_export_dir() {
    local root_dir="$1"
    local candidate

    for candidate in "${root_dir}/actor" "${root_dir}/policy" "${root_dir}/actor_model"; do
        if is_inference_model_dir "${candidate}"; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done

    if is_inference_model_dir "${root_dir}"; then
        printf '%s\n' "${root_dir}"
        return 0
    fi

    candidate="$(find "${root_dir}" -maxdepth 3 -type f -name config.json -printf '%h\n' | sort -u | while read -r d; do
        if is_inference_model_dir "${d}"; then
            printf '%s\n' "${d}"
        fi
    done | grep -E '/(actor|policy|actor_model|checkpoint-[0-9]+)$' | head -n 1 || true)"

    if [[ -n "${candidate}" ]]; then
        printf '%s\n' "${candidate}"
        return 0
    fi

    candidate="$(find "${root_dir}" -maxdepth 3 -type f -name config.json -printf '%h\n' | sort -u | while read -r d; do
        if is_inference_model_dir "${d}"; then
            printf '%s\n' "${d}"
            break
        fi
    done)"

    if [[ -n "${candidate}" ]]; then
        printf '%s\n' "${candidate}"
        return 0
    fi

    return 1
}

if [[ -d "${OUTPUT_DIR}" && "$(ls -A "${OUTPUT_DIR}")" ]]; then
    if actor_export_dir="$(select_actor_export_dir "${OUTPUT_DIR}")"; then
        export_dir="./ppo_actor_inference_export"
        rm -rf "${export_dir}"
        mkdir -p "${export_dir}"

        echo "Collecting inference artifacts from: ${actor_export_dir}"
        copied_any=0
        for pattern in \
            'config.json' \
            'generation_config.json' \
            'model.safetensors' \
            'model-*.safetensors' \
            'model.safetensors.index.json' \
            'pytorch_model.bin' \
            'pytorch_model-*.bin' \
            'pytorch_model.bin.index.json' \
            'adapter_config.json' \
            'adapter_model.safetensors' \
            'tokenizer.json' \
            'tokenizer_config.json' \
            'tokenizer.model' \
            'special_tokens_map.json' \
            'added_tokens.json' \
            'vocab.json' \
            'merges.txt' \
            'chat_template.jinja' \
            'preprocessor_config.json'; do
            if copy_if_exists "${actor_export_dir}" "${export_dir}" "${pattern}"; then
                copied_any=1
            fi
        done

        if [[ ${copied_any} -eq 1 ]]; then
            tar -czf "${ARCHIVE_PATH}" -C "${export_dir}" .
        else
            tar -czf "${ARCHIVE_PATH}" --files-from /dev/null
        fi

        rm -rf "${export_dir}"
    else
        echo "WARNING: Could not find PPO actor inference artifacts. Creating empty marker."
        tar -czf "${ARCHIVE_PATH}" --files-from /dev/null
    fi
else
    tar -czf "${ARCHIVE_PATH}" --files-from /dev/null
fi
echo "===== Job complete ====="
