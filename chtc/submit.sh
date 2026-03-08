#!/bin/bash
#
# CHTC Submit Helper: 로그 디렉토리를 자동 생성 후 condor_submit 실행
#
# 사용법:
#   bash chtc/submit.sh chtc/sft.sub [추가 condor_submit 인자...]
#
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: bash chtc/submit.sh <submit_file> [condor_submit args...]"
    echo "  예: bash chtc/submit.sh chtc/sft.sub"
    echo "  예: bash chtc/submit.sh chtc/sft.sub 'arguments = gpt2 output/sft-test 3 none'"
    exit 1
fi

SUB_FILE="$1"
shift

if [[ ! -f "${SUB_FILE}" ]]; then
    echo "Error: Submit file '${SUB_FILE}' not found." >&2
    exit 1
fi

# sub 파일에서 log 경로를 파싱하여 상위 디렉토리 패턴 추출
# 예: outputs/logs/sft/$(Cluster)/$(Process).log → outputs/logs/sft/
LOG_DIRS=$(grep -E '^\s*(log|error|output)\s*=' "${SUB_FILE}" | \
    sed 's/.*=\s*//' | \
    sed 's/\$(Cluster).*//' | \
    sort -u)

for dir in ${LOG_DIRS}; do
    mkdir -p "${dir}"
    echo "Created log directory: ${dir}"
done

echo "Submitting: ${SUB_FILE}"
condor_submit "${SUB_FILE}" "$@"
