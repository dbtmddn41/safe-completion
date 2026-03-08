#!/bin/bash
#
# CHTC wrapper script for Safe-RLHF Cost Model Training
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
OUTPUT_DIR="${2:-output/cm}"
ZERO_STAGE="${3:-3}"
OFFLOAD="${4:-none}"
JOB_SUFFIX="${5:-}"
ARTIFACTS_DIR="artifacts"

package_output() {
    mkdir -p "${ARTIFACTS_DIR}"
    local archive_name="cm_output.tar.gz"
    if [[ -n "${JOB_SUFFIX}" ]]; then
        archive_name="cm_output_${JOB_SUFFIX}.tar.gz"
    fi
    local archive_path="${ARTIFACTS_DIR}/${archive_name}"

    if [[ -d "${OUTPUT_DIR}" && "$(ls -A "${OUTPUT_DIR}" 2>/dev/null)" ]]; then
        # Include only inference-necessary files; exclude training artifacts:
        #   wandb/         - training logs
        #   global_step*/  - DeepSpeed ZeRO optimizer shards (large, training-only)
        #   zero_to_fp32.py - consolidation script (already run, no longer needed)
        #   arguments.*, environ.txt, latest - training metadata / symlinks
        tar -czf "${archive_path}" \
            --exclude="${OUTPUT_DIR}/wandb" \
            --exclude="${OUTPUT_DIR}/global_step*" \
            --exclude="${OUTPUT_DIR}/zero_to_fp32.py" \
            --exclude="${OUTPUT_DIR}/arguments.json" \
            --exclude="${OUTPUT_DIR}/arguments.pkl" \
            --exclude="${OUTPUT_DIR}/environ.txt" \
            --exclude="${OUTPUT_DIR}/latest" \
            "${OUTPUT_DIR}"
    else
        tar -czf "${archive_path}" --files-from /dev/null
    fi

    if [[ -f "${archive_path}" ]]; then
        echo "Packaged artifact: ${archive_path}"
        ls -lh "${archive_path}"
    else
        echo "ERROR: artifact was not created at ${archive_path}" >&2
    fi
}

cleanup() {
    rm -f "${PROJECT_ROOT:-$(pwd)}/_launch_cost.py"
    package_output
}
trap cleanup EXIT

# ─── Environment Setup ───────────────────────────────────────────────────────
if [[ -f "env.tar.gz" ]]; then
    mkdir -p env && tar -xzf env.tar.gz -C env && source env/bin/activate
else
    pip install --no-warn-script-location \
        --index-url https://download.pytorch.org/whl/cu124 \
        'torch==2.6.0+cu124' \
        'torchvision==0.21.0+cu124'
    pip install --no-warn-script-location \
        'transformers>=5.0.0' 'deepspeed>=0.12' 'accelerate>=0.25' \
        'datasets' 'tokenizers>=0.13.3' \
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

export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"
export LOGLEVEL="${LOGLEVEL:-WARNING}"
export WANDB_MODE="${WANDB_MODE:-offline}"

export USER="${USER:-condor}"
export LOGNAME="${LOGNAME:-${USER}}"

if [[ -n "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
    export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi
if [[ -n "${HUGGINGFACE_HUB_TOKEN:-}" && -z "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN="${HUGGINGFACE_HUB_TOKEN}"
fi

export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-./hf_cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-./hf_cache/triton}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

python -m pip install --no-warn-script-location --upgrade 'optree>=0.13.0'

if [[ "${MODEL_NAME_OR_PATH}" == meta-llama/* ]] && [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: ${MODEL_NAME_OR_PATH} is gated on Hugging Face, but HF token is missing."
    echo "       Set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) before condor_submit."
    exit 1
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && echo "${CUDA_VISIBLE_DEVICES}" | grep -qi 'gpu'; then
    NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
    export CUDA_VISIBLE_DEVICES=$(seq -s ',' 0 $((NUM_GPUS - 1)))
    echo "Converted CUDA_VISIBLE_DEVICES to: ${CUDA_VISIBLE_DEVICES}"
fi

if [[ -f "model.tar.gz" ]]; then
    mkdir -p model && tar -xzf model.tar.gz -C model
    MODEL_NAME_OR_PATH="./model"
fi

mkdir -p "${OUTPUT_DIR}"

echo "===== Starting Cost Model Training ====="

PROJECT_ROOT="$(pwd)"

cat > "${PROJECT_ROOT}/_launch_cost.py" << 'PYEOF'
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
runpy.run_module("safe_rlhf.values.cost", run_name="__main__", alter_sys=True)
PYEOF

MASTER_PORT=$((RANDOM % 50000 + 10000))

deepspeed --master_port "${MASTER_PORT}" \
    "${PROJECT_ROOT}/_launch_cost.py" \
    --train_datasets PKU-SafeRLHF/train \
    --eval_datasets PKU-SafeRLHF/test \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --max_length 512 \
    --trust_remote_code True \
    --loss_type sequence-wise \
    --epochs 2 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
    --regularization 0.001 \
    --normalize_score_during_training False \
    --normalizer_type ExponentialMovingAverage \
    --normalizer_momentum 0.9 \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --lr_warmup_ratio 0.03 \
    --weight_decay 0.1 \
    --seed 42 \
    --need_eval \
    --eval_strategy epoch \
    --output_dir "${OUTPUT_DIR}" \
    --log_type wandb \
    --log_project Safe-RLHF-CM \
    --zero_stage "${ZERO_STAGE}" \
    --offload "${OFFLOAD}" \
    --bf16 True \
    --tf32 True

echo "===== Training finished at $(date) ====="

# ─── Consolidate ZeRO-3 checkpoint → single HuggingFace weight file ──────────
if [[ -f "${OUTPUT_DIR}/zero_to_fp32.py" ]]; then
    echo "Consolidating ZeRO-3 checkpoint to FP32 weights..."
    [[ -d "${OUTPUT_DIR}/pytorch_model.bin" ]] && rm -rf "${OUTPUT_DIR}/pytorch_model.bin"
    python "${OUTPUT_DIR}/zero_to_fp32.py" "${OUTPUT_DIR}" "${OUTPUT_DIR}/pytorch_model.bin"
    echo "Consolidation complete. Model saved to ${OUTPUT_DIR}/pytorch_model.bin"
else
    echo "WARNING: zero_to_fp32.py not found in ${OUTPUT_DIR}; skipping consolidation."
fi
echo "===== Job complete ====="
