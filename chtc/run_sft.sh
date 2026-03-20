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
MODEL_NAME_OR_PATH="${1:-meta-llama/Llama-3.2-1B}"
OUTPUT_DIR="${2:-output/sft-llama32}"
ZERO_STAGE="${3:-3}"
OFFLOAD="${4:-none}"
JOB_SUFFIX="${5:-}"
SFT_TRAIN_PROPORTION="${6:-${SFT_TRAIN_PROPORTION:-1.0}}"
SFT_SAFE_TRAIN_PROPORTION="${7:-${SFT_SAFE_TRAIN_PROPORTION:-${SFT_TRAIN_PROPORTION}}}"
SFT_EPOCHS="${8:-${SFT_EPOCHS:-3}}"

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
    # Install latest transformers as requested.
    pip install --no-warn-script-location \
        'transformers' \
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
if [[ -z "${WANDB_MODE:-}" || "${WANDB_MODE}" == "UNDEFINED" ]]; then
    export WANDB_MODE="online"
fi
if [[ -z "${WANDB_PROJECT:-}" || "${WANDB_PROJECT}" == "UNDEFINED" ]]; then
    export WANDB_PROJECT="Safe-RLHF-SFT"
fi
if [[ "${WANDB_API_KEY:-}" == "UNDEFINED" ]]; then
    unset WANDB_API_KEY
fi
if [[ "${WANDB_ENTITY:-}" == "UNDEFINED" ]]; then
    unset WANDB_ENTITY
fi

if [[ "${WANDB_MODE}" != "offline" && -z "${WANDB_API_KEY:-}" ]]; then
    echo "WANDB_API_KEY is not set; falling back to WANDB_MODE=offline."
    export WANDB_MODE="offline"
fi

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
echo "WandB mode: ${WANDB_MODE}"
echo "Epochs: ${SFT_EPOCHS}, Alpaca proportion: ${SFT_TRAIN_PROPORTION}, PKU-SafeRLHF-QA-Safe proportion: ${SFT_SAFE_TRAIN_PROPORTION}"

# ─── Run training ────────────────────────────────────────────────────────────
MASTER_PORT=$((RANDOM % 50000 + 10000))

TRAIN_DATASET_SPEC_ALPACA="alpaca:${SFT_TRAIN_PROPORTION}"
TRAIN_DATASET_SPEC_SAFE="PKU-SafeRLHF-QA-Safe/train:${SFT_SAFE_TRAIN_PROPORTION}"

deepspeed --master_port "${MASTER_PORT}" \
    "${PROJECT_ROOT}/_launch_sft.py" \
    --train_datasets "${TRAIN_DATASET_SPEC_ALPACA}" "${TRAIN_DATASET_SPEC_SAFE}" \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --max_length 512 \
    --trust_remote_code True \
    --epochs "${SFT_EPOCHS}" \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --lr_warmup_ratio 0.03 \
    --weight_decay 0.0 \
    --seed 42 \
    --output_dir "${OUTPUT_DIR}" \
    --log_type wandb \
    --log_project "${WANDB_PROJECT}" \
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

if [[ -d "${OUTPUT_DIR}" && "$(ls -A "${OUTPUT_DIR}")" ]]; then
    source_dir="${OUTPUT_DIR}"

    if ! is_inference_model_dir "${source_dir}"; then
        latest_checkpoint="$(find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1 || true)"
        if [[ -n "${latest_checkpoint}" ]] && is_inference_model_dir "${latest_checkpoint}"; then
            source_dir="${latest_checkpoint}"
        fi
    fi

    export_dir="./sft_inference_export"
    rm -rf "${export_dir}"
    mkdir -p "${export_dir}"

    echo "Collecting inference artifacts from: ${source_dir}"
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
        if copy_if_exists "${source_dir}" "${export_dir}" "${pattern}"; then
            copied_any=1
        fi
    done

    if [[ ${copied_any} -eq 1 ]]; then
        tar -czf "${archive_name}" -C "${export_dir}" .
        echo "Inference-only output packed to ${archive_name}"
    else
        echo "WARNING: Could not find inference artifacts. Creating empty marker."
        tar -czf "${archive_name}" --files-from /dev/null
    fi

    rm -rf "${export_dir}"
else
    echo "WARNING: No output produced. Creating empty marker."
    tar -czf "${archive_name}" --files-from /dev/null
fi

rm -f "${PROJECT_ROOT}/_launch_sft.py"
echo "===== Job complete ====="
