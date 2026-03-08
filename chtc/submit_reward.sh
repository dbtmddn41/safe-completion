#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
SUB_FILE="${1:-${ROOT_DIR}/chtc/reward.sub}"
RESULT_DIR="${RESULT_DIR:-/staging/slyu59/models/reward-models}"

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

LOG_FILE="${ROOT_DIR}/outputs/logs/reward/${CLUSTER_ID}/0.log"
echo "Waiting for job completion (cluster ${CLUSTER_ID})..."
echo "Log file: ${LOG_FILE}"

condor_wait "${LOG_FILE}"

shopt -s nullglob
ARCHIVES=("${ROOT_DIR}/rm_output_${CLUSTER_ID}_"*.tar.gz)

if [[ ${#ARCHIVES[@]} -eq 0 ]]; then
  ARCHIVES=("${ROOT_DIR}/artifacts/rm_output_${CLUSTER_ID}_"*.tar.gz)
fi
shopt -u nullglob

if [[ ${#ARCHIVES[@]} -eq 0 ]]; then
  echo "No output archive found for cluster ${CLUSTER_ID}." >&2
  echo "Expected patterns:" >&2
  echo "  ${ROOT_DIR}/rm_output_${CLUSTER_ID}_*.tar.gz" >&2
  echo "  ${ROOT_DIR}/artifacts/rm_output_${CLUSTER_ID}_*.tar.gz" >&2
  echo "Check condor logs: ${ROOT_DIR}/outputs/logs/reward/${CLUSTER_ID}/" >&2
  exit 2
fi

for archive in "${ARCHIVES[@]}"; do
  archive_name="$(basename "${archive}")"
  dest_archive="${RESULT_DIR}/${archive_name}"

  mv -f "${archive}" "${dest_archive}"
  echo "Moved: ${dest_archive}"
done

echo "Done. Reward model artifacts are under: ${RESULT_DIR}"
