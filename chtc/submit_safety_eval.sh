#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
SUB_FILE="${1:-${ROOT_DIR}/chtc/safety_eval.sub}"
RESULT_DIR="${RESULT_DIR:-${ROOT_DIR}/outputs/eval-results}"
AUTO_EXTRACT="${AUTO_EXTRACT:-1}"

if [[ ! -f "${SUB_FILE}" ]]; then
  echo "Submit file not found: ${SUB_FILE}" >&2
  exit 1
fi

mkdir -p "${RESULT_DIR}"

echo "Submitting: ${SUB_FILE}"
SUBMIT_OUTPUT="$(condor_submit "${SUB_FILE}")"
echo "${SUBMIT_OUTPUT}"

CLUSTER_ID="$({ echo "${SUBMIT_OUTPUT}" | sed -n 's/.*cluster \([0-9]\+\).*/\1/p'; } | tail -n 1)"
if [[ -z "${CLUSTER_ID}" ]]; then
  echo "Failed to parse cluster id from condor_submit output." >&2
  exit 1
fi

LOG_FILE="${ROOT_DIR}/outputs/logs/safety_eval/${CLUSTER_ID}/0.log"
echo "Waiting for job completion (cluster ${CLUSTER_ID})..."
echo "Log file: ${LOG_FILE}"

condor_wait "${LOG_FILE}"

shopt -s nullglob
ARCHIVES=("${ROOT_DIR}/safety_eval_output_${CLUSTER_ID}_"*.tar.gz)
shopt -u nullglob

if [[ ${#ARCHIVES[@]} -eq 0 ]]; then
  echo "No output archive found for cluster ${CLUSTER_ID}."
  echo "Expected pattern: ${ROOT_DIR}/safety_eval_output_${CLUSTER_ID}_*.tar.gz"
  echo "Check condor logs: ${ROOT_DIR}/outputs/logs/safety_eval/${CLUSTER_ID}/"
  exit 2
fi

for archive in "${ARCHIVES[@]}"; do
  archive_name="$(basename "${archive}")"
  dest_archive="${RESULT_DIR}/${archive_name}"

  mv -f "${archive}" "${dest_archive}"
  echo "Moved: ${dest_archive}"

  if [[ "${AUTO_EXTRACT}" == "1" ]]; then
    extract_dir="${RESULT_DIR}/${archive_name%.tar.gz}"
    mkdir -p "${extract_dir}"
    tar -xzf "${dest_archive}" -C "${extract_dir}"
    echo "Extracted: ${extract_dir}"
  fi
done

echo "Done. Results are under: ${RESULT_DIR}"