#!/bin/bash
#
# CHTC wrapper script for Safe-RLHF Safety Evaluation
#
# Evaluates an LLM model on adversarial benchmarks (e.g. AdvBench) using:
#   1. Llama-Guard for safety classification
#   2. HuggingFace built-in methods for perplexity measurement
#   3. Qualitative report for manual inspection
#
# Arguments:
#   $1 = model_name_or_path   (HuggingFace model name or local path)
#   $2 = guard_model           (Llama-Guard model name, default: meta-llama/Llama-Guard-3-1B)
#   $3 = benchmark             (advbench, HF dataset path, or local file)
#   $4 = num_samples           (number of prompts to evaluate, 0 = all)
#   $5 = output_dir            (output directory)
#
set -euo pipefail

echo "===== Safety Evaluation Job started at $(date) ====="
echo "Hostname: $(hostname)"
echo "Working directory: $(pwd)"
echo "Directory contents:"
ls -F
nvidia-smi || echo "nvidia-smi not available"

# ─── Arguments ────────────────────────────────────────────────────────────────
MODEL_NAME_OR_PATH="${1:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
GUARD_MODEL="${2:-meta-llama/Llama-Guard-3-1B}"
BENCHMARK="${3:-advbench}"
NUM_SAMPLES="${4:-50}"
OUTPUT_DIR="${5:-output/safety_eval}"
JOB_SUFFIX="${JOB_SUFFIX:-}"

# ─── Helper: package output ──────────────────────────────────────────────────
package_output() {
    local archive_name="safety_eval_output.tar.gz"
    if [[ -n "${JOB_SUFFIX}" ]]; then
        archive_name="safety_eval_output_${JOB_SUFFIX}.tar.gz"
    fi
    local archive_path="${archive_name}"

    if [[ -d "${OUTPUT_DIR}" && "$(ls -A "${OUTPUT_DIR}" 2>/dev/null)" ]]; then
        tar -czf "${archive_path}" "${OUTPUT_DIR}"
    else
        tar -czf "${archive_path}" --files-from /dev/null
    fi
}

# ─── PATH setup ──────────────────────────────────────────────────────────────
export PATH="${HOME}/.local/bin:${PATH}"

# ─── Install dependencies ────────────────────────────────────────────────────
if [[ -f "env.tar.gz" ]]; then
    echo "Unpacking pre-built environment..."
    mkdir -p env && tar -xzf env.tar.gz -C env && source env/bin/activate
else
    echo "Installing dependencies on the fly..."
    pip install --no-warn-script-location \
        --index-url https://download.pytorch.org/whl/cu124 \
        'torch==2.6.0+cu124' \
        'torchvision==0.21.0+cu124'
    pip install --no-warn-script-location \
        'transformers>=5.0.0' \
        'deepspeed>=0.12' \
        'accelerate>=0.25' \
        'datasets' \
        'evaluate' \
        'tokenizers' \
        numpy scipy sentencepiece tqdm rich
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
export PYTHONPATH="$(pwd)${PYTHONPATH:+:${PYTHONPATH}}"
export LOGLEVEL="${LOGLEVEL:-WARNING}"
export WANDB_MODE="${WANDB_MODE:-offline}"

export USER="${USER:-condor}"
export LOGNAME="${LOGNAME:-${USER}}"

# Propagate HF tokens.
if [[ -n "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
    export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi
if [[ -n "${HUGGINGFACE_HUB_TOKEN:-}" && -z "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN="${HUGGINGFACE_HUB_TOKEN}"
fi

# ─── Cache directories ───────────────────────────────────────────────────────
mkdir -p "${HF_HOME:-./hf_cache}" \
         "${HF_DATASETS_CACHE:-./hf_cache/datasets}" \
         "${TRANSFORMERS_CACHE:-./hf_cache/transformers}" \
         "${TORCH_EXTENSIONS_DIR:-./hf_cache/torch_extensions}"

export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-./hf_cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-./hf_cache/triton}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

python -m pip install --no-warn-script-location --upgrade 'optree>=0.13.0'

resolve_extracted_model_dir() {
    local extracted_root="$1"

    if [[ -f "${extracted_root}/config.json" ]]; then
        echo "${extracted_root}"
        return
    fi

    mapfile -t config_files < <(find "${extracted_root}" -type f -name config.json 2>/dev/null | sort)

    if [[ "${#config_files[@]}" -eq 0 ]]; then
        echo "${extracted_root}"
        return
    fi

    local best_dir=""
    local best_depth=999999
    local cfg dir depth
    for cfg in "${config_files[@]}"; do
        dir="$(dirname "${cfg}")"
        depth=$(awk -F'/' '{print NF}' <<< "${dir}")
        if (( depth < best_depth )); then
            best_depth=${depth}
            best_dir="${dir}"
        fi
    done

    echo "${best_dir}"
}

normalize_model_layout() {
    local model_dir="$1"
    local nested_weights_dir="${model_dir}/pytorch_model.bin"

    if [[ ! -d "${nested_weights_dir}" ]]; then
        return
    fi

    echo "Normalizing model layout from ${nested_weights_dir} to ${model_dir}..."

    local moved_any=0
    local file basename
    for file in "${nested_weights_dir}"/*; do
        [[ -e "${file}" ]] || continue
        basename="$(basename "${file}")"
        if [[ ! -e "${model_dir}/${basename}" ]]; then
            mv "${file}" "${model_dir}/${basename}"
            moved_any=1
        fi
    done

    if [[ ${moved_any} -eq 1 ]]; then
        rmdir "${nested_weights_dir}" 2>/dev/null || true
    fi
}

# ─── Handle staging models ───────────────────────────────────────────────────
if [[ -f "model.tar.gz" ]]; then
    echo "Unpacking target model from model.tar.gz..."
    mkdir -p model && tar -xzf model.tar.gz -C model
    MODEL_NAME_OR_PATH="$(resolve_extracted_model_dir "./model")"
elif [[ "${MODEL_NAME_OR_PATH}" == *.tar.gz ]] && [[ -f "${MODEL_NAME_OR_PATH}" ]]; then
    echo "Unpacking target model from ${MODEL_NAME_OR_PATH}..."
    EXTRACTED_MODEL_DIR="./model_from_tar"
    rm -rf "${EXTRACTED_MODEL_DIR}"
    mkdir -p "${EXTRACTED_MODEL_DIR}"
    tar -xzf "${MODEL_NAME_OR_PATH}" -C "${EXTRACTED_MODEL_DIR}"
    MODEL_NAME_OR_PATH="$(resolve_extracted_model_dir "${EXTRACTED_MODEL_DIR}")"
fi

if [[ -f "guard_model.tar.gz" ]]; then
    echo "Unpacking Llama-Guard from guard_model.tar.gz..."
    mkdir -p guard_model && tar -xzf guard_model.tar.gz -C guard_model
    GUARD_MODEL="./guard_model"
fi

if [[ -d "${MODEL_NAME_OR_PATH}" ]]; then
    normalize_model_layout "${MODEL_NAME_OR_PATH}"
fi

# ─── Fix CUDA_VISIBLE_DEVICES for HTCondor ────────────────────────────────────
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && echo "${CUDA_VISIBLE_DEVICES}" | grep -qi 'gpu'; then
    NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
    export CUDA_VISIBLE_DEVICES=$(seq -s ',' 0 $((NUM_GPUS - 1)))
    echo "Converted CUDA_VISIBLE_DEVICES to: ${CUDA_VISIBLE_DEVICES}"
fi

# ─── Create output directory ─────────────────────────────────────────────────
mkdir -p "${OUTPUT_DIR}"

echo "===== Starting Safety Evaluation ====="
echo "Target model : ${MODEL_NAME_OR_PATH}"
echo "Guard model  : ${GUARD_MODEL}"
echo "Benchmark    : ${BENCHMARK}"
echo "Num samples  : ${NUM_SAMPLES}"
echo "Output dir   : ${OUTPUT_DIR}"

# ─── Gated model check ───────────────────────────────────────────────────────
for model_name in "${MODEL_NAME_OR_PATH}" "${GUARD_MODEL}"; do
    if [[ "${model_name}" == meta-llama/* ]] && [[ -z "${HF_TOKEN:-}" ]]; then
        echo "WARNING: ${model_name} may be gated. Set HF_TOKEN if download fails."
    fi
done

# ─── Run evaluation ──────────────────────────────────────────────────────────
# Why a launcher file?
# `python -m safe_rlhf.evaluate.safety` causes Python to first load
# safe_rlhf/__init__.py, which eagerly imports PPO trainers → logger →
# tensorboard/deepspeed even though safety.py needs none of those.
# By running safety.py directly via runpy.run_path we bypass __init__.py
# entirely (same pattern used by run_sft.sh and run_ppo.sh).
PROJECT_ROOT="$(pwd)"
cat > "${PROJECT_ROOT}/_launch_safety.py" << 'PYEOF'
import os
import runpy
import sys

# Add project root to sys.path so that relative imports inside safety.py work.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Run safety.py directly, bypassing safe_rlhf/__init__.py.
runpy.run_path(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "safe_rlhf", "evaluate", "safety.py"),
    run_name="__main__",
)
PYEOF

cleanup() {
    rm -f "${PROJECT_ROOT}/_launch_safety.py"
    package_output
}
trap cleanup EXIT

python "${PROJECT_ROOT}/_launch_safety.py" \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --guard_model_name_or_path "${GUARD_MODEL}" \
    --benchmark "${BENCHMARK}" \
    --num_samples "${NUM_SAMPLES}" \
    --output_dir "${OUTPUT_DIR}" \
    --dtype bfloat16 \
    --max_new_tokens 256 \
    --batch_size 1 \
    --qualitative_examples 20 \
    --seed 42 \
    --trust_remote_code

echo "===== Safety Evaluation finished at $(date) ====="
# cleanup (rm launcher + package_output) is called automatically via trap on EXIT
echo "===== Job complete ====="
