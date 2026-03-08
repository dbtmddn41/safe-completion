#!/usr/bin/env bash
#
# Data Generation Pipeline: Prompt -> Answer + CoT
#
# Usage:
#   bash scripts/datagen.sh \
#       --model <model_for_generation> \
#       --input_file <prompts.jsonl> \
#       --output_dir <output_directory>
# ==============================================================================

if [ -z "${BASH_VERSION}" ]; then
	echo "Please use bash to run this script." >&2
	exit 1
fi

set -x

SCRIPT_DIR="$(cd "$(dirname "$0")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export LOGLEVEL="${LOGLEVEL:-WARNING}"

# ─── Defaults ─────────────────────────────────────────────────────────────────
MODEL="meta-llama/Llama-2-7b-chat-hf"
INPUT_FILE=""
OUTPUT_DIR="${ROOT_DIR}/output/datagen"
PROMPT_FIELD="input"
DEVICE="auto"
DTYPE="auto"
GEN_MAX_NEW_TOKENS=1024
GEN_TEMPERATURE=0.7
GEN_BATCH_SIZE=4
MAX_BATCH_TOKENS=12288
MAX_PROMPT_TOKENS=3072
NUM_RETURN_SEQUENCES=1
BACKEND="transformers"
VLLM_TENSOR_PARALLEL_SIZE=1
VLLM_GPU_MEMORY_UTILIZATION=0.9
VLLM_MAX_NUM_BATCHED_TOKENS=""
SEED=42
EXTRA_ARGS=()

# ─── Parse arguments ─────────────────────────────────────────────────────────
while [[ "$#" -gt 0 ]]; do
	arg="$1"
	shift
	case "${arg}" in
		--model)               MODEL="$1"; shift ;;
		--model=*)             MODEL="${arg#*=}" ;;
		--generate_model)      MODEL="$1"; shift ;;
		--generate_model=*)    MODEL="${arg#*=}" ;;
		--input_file)          INPUT_FILE="$1"; shift ;;
		--input_file=*)        INPUT_FILE="${arg#*=}" ;;
		--output_dir)          OUTPUT_DIR="$1"; shift ;;
		--output_dir=*)        OUTPUT_DIR="${arg#*=}" ;;
		--prompt_field)        PROMPT_FIELD="$1"; shift ;;
		--prompt_field=*)      PROMPT_FIELD="${arg#*=}" ;;
		--device)              DEVICE="$1"; shift ;;
		--device=*)            DEVICE="${arg#*=}" ;;
		--dtype)               DTYPE="$1"; shift ;;
		--dtype=*)             DTYPE="${arg#*=}" ;;
		--gen_max_new_tokens)  GEN_MAX_NEW_TOKENS="$1"; shift ;;
		--gen_temperature)     GEN_TEMPERATURE="$1"; shift ;;
		--gen_batch_size)      GEN_BATCH_SIZE="$1"; shift ;;
		--max_batch_tokens)    MAX_BATCH_TOKENS="$1"; shift ;;
		--max_prompt_tokens)   MAX_PROMPT_TOKENS="$1"; shift ;;
		--num_return_sequences) NUM_RETURN_SEQUENCES="$1"; shift ;;
		--backend)             BACKEND="$1"; shift ;;
		--backend=*)           BACKEND="${arg#*=}" ;;
		--vllm_tensor_parallel_size) VLLM_TENSOR_PARALLEL_SIZE="$1"; shift ;;
		--vllm_gpu_memory_utilization) VLLM_GPU_MEMORY_UTILIZATION="$1"; shift ;;
		--vllm_max_num_batched_tokens) VLLM_MAX_NUM_BATCHED_TOKENS="$1"; shift ;;
		--seed)                SEED="$1"; shift ;;
		--load_in_8bit)        EXTRA_ARGS+=("--load_in_8bit") ;;
		--load_in_4bit)        EXTRA_ARGS+=("--load_in_4bit") ;;
		--resume)              EXTRA_ARGS+=("--resume") ;;
		*)
			echo "Unknown parameter passed: '${arg}'" >&2
			exit 1
			;;
	esac
done

# ─── Validate ─────────────────────────────────────────────────────────────────
if [[ -z "${INPUT_FILE}" ]]; then
	echo "ERROR: --input_file is required." >&2
	exit 1
fi

# ─── Create output directory ─────────────────────────────────────────────────
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" &>/dev/null && pwd)"

if [[ ! -f "${OUTPUT_DIR}/.gitignore" ]]; then
	echo '*' > "${OUTPUT_DIR}/.gitignore"
fi

cp -f "$0" "${OUTPUT_DIR}/script.sh"

# ─── Build command ────────────────────────────────────────────────────────────
CMD_ARGS=(
	python -m safe_rlhf.datagen pipeline
	--model "${MODEL}"
	--input_file "${INPUT_FILE}"
	--output_dir "${OUTPUT_DIR}"
	--prompt_field "${PROMPT_FIELD}"
	--device "${DEVICE}"
	--dtype "${DTYPE}"
	--gen_max_new_tokens "${GEN_MAX_NEW_TOKENS}"
	--gen_temperature "${GEN_TEMPERATURE}"
	--gen_batch_size "${GEN_BATCH_SIZE}"
	--max_batch_tokens "${MAX_BATCH_TOKENS}"
	--max_prompt_tokens "${MAX_PROMPT_TOKENS}"
	--num_return_sequences "${NUM_RETURN_SEQUENCES}"
	--backend "${BACKEND}"
	--vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE}"
	--vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
	--seed "${SEED}"
)

if [[ -n "${VLLM_MAX_NUM_BATCHED_TOKENS}" ]]; then
	CMD_ARGS+=(--vllm_max_num_batched_tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}")
fi

CMD_ARGS+=("${EXTRA_ARGS[@]}")

# ─── Run ──────────────────────────────────────────────────────────────────────
exec 1> >(tee "${OUTPUT_DIR}/stdout.log" >&1) 2> >(tee "${OUTPUT_DIR}/stderr.log" >&2)

echo "===== Data Generation Pipeline ====="
echo "Model          : ${MODEL}"
echo "Input file     : ${INPUT_FILE}"
echo "Prompt field   : ${PROMPT_FIELD}"
echo "Backend        : ${BACKEND}"
echo "Output dir     : ${OUTPUT_DIR}"
echo "======================================"

"${CMD_ARGS[@]}"
