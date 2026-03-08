#!/bin/bash
#
# CHTC wrapper script for Data Generation Pipeline (Prompt -> Answer + CoT)
# This script runs inside the Docker container on CHTC execute nodes.
#
# Arguments:
#   $1 = model            (HuggingFace model name or local path)
#   $2 = input_file       (JSONL file with prompts + optional metadata)
#   $3 = output_dir       (output directory, default: output/datagen)
#   $4 = prompt_field     (prompt field name in each JSONL row, default: input)
#   $5 = job_suffix       (optional, for output archive naming)
#   $6 = backend          (transformers|vllm, default: transformers)
#   $7 = tp_size          (vLLM tensor_parallel_size, optional. default: number of visible GPUs)
#
set -euo pipefail

echo "===== Job started at $(date) ====="
echo "Hostname: $(hostname)"
echo "Working directory: $(pwd)"
echo "Directory contents:"
ls -F
nvidia-smi || echo "nvidia-smi not available"

# ─── Arguments ────────────────────────────────────────────────────────────────
MODEL="${1:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
INPUT_FILE="${2:-data/input.jsonl}"
OUTPUT_DIR="${3:-output/datagen}"
PROMPT_FIELD="${4:-input}"
JOB_SUFFIX="${5:-}"
BACKEND="${6:-transformers}"
TP_SIZE="${7:-}"

# ─── PATH setup ──────────────────────────────────────────────────────────────
export PATH="${HOME}/.local/bin:${PATH}"

# ─── Install dependencies ────────────────────────────────────────────────────
if [[ -f "env.tar.gz" ]]; then
    echo "Unpacking pre-built environment..."
    mkdir -p env && tar -xzf env.tar.gz -C env && source env/bin/activate
else
    echo "Installing dependencies on the fly..."
    pip install --no-warn-script-location \
        --index-url https://download.pytorch.org/whl/cu121 \
        'torch==2.4.0+cu121' \
        'torchvision==0.19.0+cu121'
    pip install --no-warn-script-location \
        'transformers>=4.37' \
        'deepspeed>=0.12' \
        'accelerate>=0.25' \
        'datasets' \
        'tokenizers>=0.13.3' \
        numpy scipy sentencepiece wandb tensorboard optree matplotlib tqdm rich
    if [[ "${BACKEND}" == "vllm" ]]; then
        pip install --no-warn-script-location 'vllm>=0.5.5'
    fi
fi

# ─── Environment variables ───────────────────────────────────────────────────
export LOGLEVEL="${LOGLEVEL:-WARNING}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export USER="${USER:-$(id -un 2>/dev/null || echo condor)}"
export LOGNAME="${LOGNAME:-${USER}}"
export HOME="${HOME:-$(pwd)}"

# Keep all runtime caches inside the execute directory.
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-./hf_cache}"
export HF_HOME="${HF_HOME:-./hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-./hf_cache/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-./hf_cache/transformers}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-./hf_cache/torch_extensions}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-./hf_cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-./hf_cache/triton}"

# ─── Cache directories ───────────────────────────────────────────────────────
mkdir -p "${HF_HOME}" \
         "${HF_DATASETS_CACHE}" \
         "${TRANSFORMERS_CACHE}" \
         "${TORCH_EXTENSIONS_DIR}" \
         "${TORCHINDUCTOR_CACHE_DIR}" \
         "${TRITON_CACHE_DIR}"

# ─── Handle staging models ───────────────────────────────────────────────────
if [[ -f "model.tar.gz" ]]; then
    echo "Unpacking model from model.tar.gz..."
    mkdir -p model && tar -xzf model.tar.gz -C model
    MODEL="./model"
fi

# ─── Create output directory ─────────────────────────────────────────────────
mkdir -p "${OUTPUT_DIR}"

# ─── Fix CUDA_VISIBLE_DEVICES for compatibility ─────────────────────────────
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && echo "${CUDA_VISIBLE_DEVICES}" | grep -qi 'gpu'; then
    NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
    export CUDA_VISIBLE_DEVICES=$(seq -s ',' 0 $((NUM_GPUS - 1)))
    echo "Converted CUDA_VISIBLE_DEVICES to: ${CUDA_VISIBLE_DEVICES}"
fi

if [[ -z "${NUM_GPUS:-}" ]]; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
    else
        NUM_GPUS=1
    fi
fi

if [[ -z "${TP_SIZE}" ]]; then
    TP_SIZE="${NUM_GPUS}"
fi

# ─── Ensure safe_rlhf is importable ─────────────────────────────────────────
PROJECT_ROOT="$(pwd)"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "===== Starting Data Generation Pipeline ====="
echo "Model          : ${MODEL}"
echo "Input file     : ${INPUT_FILE}"
echo "Prompt field   : ${PROMPT_FIELD}"
echo "Backend        : ${BACKEND}"
echo "TP size        : ${TP_SIZE}"
echo "Output dir     : ${OUTPUT_DIR}"

# ─── Build command ────────────────────────────────────────────────────────────
CMD_ARGS=(
    python -m safe_rlhf.datagen pipeline
    --model "${MODEL}"
    --input_file "${INPUT_FILE}"
    --output_dir "${OUTPUT_DIR}"
    --prompt_field "${PROMPT_FIELD}"
    --backend "${BACKEND}"
    --device auto
    --dtype auto
    --seed 42
)

if [[ "${BACKEND}" == "vllm" ]]; then
    # vLLM may otherwise allocate KV cache for extremely large model max context
    # (e.g., 131072), which can exceed available GPU memory on CHTC nodes.
    VLLM_MAX_MODEL_LEN=$(( ${MAX_PROMPT_TOKENS:-4096} + ${GEN_MAX_NEW_TOKENS:-2048} ))
    CMD_ARGS+=(
        --vllm_tensor_parallel_size "${TP_SIZE}"
        --vllm_gpu_memory_utilization 0.90
        --gen_batch_size 4
        --max_batch_tokens 16384
        --vllm_max_num_batched_tokens 16384
        --vllm_max_model_len "${VLLM_MAX_MODEL_LEN}"
        --max_prompt_tokens 4096
        --gen_max_new_tokens 2048
    )
fi

"${CMD_ARGS[@]}"

echo "===== Pipeline finished at $(date) ====="

# ─── Pack output for transfer back ───────────────────────────────────────────
archive_name="datagen_output.tar.gz"
if [[ -n "${JOB_SUFFIX}" ]]; then
    archive_name="datagen_output_${JOB_SUFFIX}.tar.gz"
fi

if [[ -d "${OUTPUT_DIR}" && "$(ls -A "${OUTPUT_DIR}")" ]]; then
    echo "Packing output..."
    tar -czf "${archive_name}" "${OUTPUT_DIR}"
    echo "Output packed to ${archive_name}"
else
    echo "WARNING: No output produced. Creating empty marker."
    tar -czf "${archive_name}" --files-from /dev/null
fi

echo "===== Job complete ====="
