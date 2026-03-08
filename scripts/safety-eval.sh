#!/usr/bin/env bash
#
# Safety evaluation script (local execution).
#
# Evaluates an LLM on adversarial benchmarks using Llama-Guard for safety
# classification and HuggingFace built-in methods for perplexity measurement.
#
# Usage:
#   bash scripts/safety-eval.sh \
#       --model_name_or_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
#       --benchmark advbench \
#       --num_samples 50
#
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
MODEL_NAME_OR_PATH="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
GUARD_MODEL="meta-llama/Llama-Guard-3-1B"
BENCHMARK="advbench"
NUM_SAMPLES=50
OUTPUT_DIR="${ROOT_DIR}/output/safety_eval"
DTYPE="bfloat16"
MAX_NEW_TOKENS=256
TEMPERATURE=0.0
BATCH_SIZE=1
QUALITATIVE_EXAMPLES=20
SEED=42
TRUST_REMOTE_CODE=""

# ─── Parse arguments ─────────────────────────────────────────────────────────
while [[ "$#" -gt 0 ]]; do
	arg="$1"
	shift
	case "${arg}" in
		--model_name_or_path)    MODEL_NAME_OR_PATH="$1";    shift ;;
		--model_name_or_path=*)  MODEL_NAME_OR_PATH="${arg#*=}" ;;
		--guard_model)           GUARD_MODEL="$1";           shift ;;
		--guard_model=*)         GUARD_MODEL="${arg#*=}" ;;
		--benchmark)             BENCHMARK="$1";             shift ;;
		--benchmark=*)           BENCHMARK="${arg#*=}" ;;
		--num_samples)           NUM_SAMPLES="$1";           shift ;;
		--num_samples=*)         NUM_SAMPLES="${arg#*=}" ;;
		--output_dir)            OUTPUT_DIR="$1";            shift ;;
		--output_dir=*)          OUTPUT_DIR="${arg#*=}" ;;
		--dtype)                 DTYPE="$1";                 shift ;;
		--dtype=*)               DTYPE="${arg#*=}" ;;
		--max_new_tokens)        MAX_NEW_TOKENS="$1";        shift ;;
		--max_new_tokens=*)      MAX_NEW_TOKENS="${arg#*=}" ;;
		--temperature)           TEMPERATURE="$1";           shift ;;
		--temperature=*)         TEMPERATURE="${arg#*=}" ;;
		--batch_size)            BATCH_SIZE="$1";            shift ;;
		--batch_size=*)          BATCH_SIZE="${arg#*=}" ;;
		--qualitative_examples)  QUALITATIVE_EXAMPLES="$1";  shift ;;
		--qualitative_examples=*)QUALITATIVE_EXAMPLES="${arg#*=}" ;;
		--seed)                  SEED="$1";                  shift ;;
		--seed=*)                SEED="${arg#*=}" ;;
		--trust_remote_code)     TRUST_REMOTE_CODE="--trust_remote_code" ;;
		*)
			echo "Unknown parameter passed: '${arg}'" >&2
			exit 1
			;;
	esac
done

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" &>/dev/null && pwd)"
if [[ ! -f "${OUTPUT_DIR}/.gitignore" ]]; then
	echo '*' >"${OUTPUT_DIR}/.gitignore"
fi

# Copy this script into output for reproducibility.
cp -f "$0" "${OUTPUT_DIR}/script.sh"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
	export WANDB_MODE="offline"
fi

# ─── Run evaluation ──────────────────────────────────────────────────────────
python -m safe_rlhf.evaluate.safety \
	--model_name_or_path "${MODEL_NAME_OR_PATH}" \
	--guard_model_name_or_path "${GUARD_MODEL}" \
	--benchmark "${BENCHMARK}" \
	--num_samples "${NUM_SAMPLES}" \
	--output_dir "${OUTPUT_DIR}" \
	--dtype "${DTYPE}" \
	--max_new_tokens "${MAX_NEW_TOKENS}" \
	--temperature "${TEMPERATURE}" \
	--batch_size "${BATCH_SIZE}" \
	--qualitative_examples "${QUALITATIVE_EXAMPLES}" \
	--seed "${SEED}" \
	${TRUST_REMOTE_CODE}
