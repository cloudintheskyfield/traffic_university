#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt1/mnt2/data3/nlp/ws/proj/课题组}"
PYTHON_BIN="${PYTHON_BIN:-/mnt1/mnt2/data3/nlp/ws/qwen35_vllm_env/.venv/bin/python}"
BASELINE_RESULTS="${BASELINE_RESULTS:-${PROJECT_DIR}/results/baseline_vllm_resume.jsonl}"
NO_YOLO_RESULTS="${NO_YOLO_RESULTS:-${PROJECT_DIR}/results/ours_no_yolo_vllm_resume.jsonl}"
OUT_JSONL="${OUT_JSONL:-${PROJECT_DIR}/results/qa_judge_qwen35_baseline_vs_no_yolo_100_details.jsonl}"
OUT_MD="${OUT_MD:-${PROJECT_DIR}/results/qa_judge_qwen35_baseline_vs_no_yolo_100_summary.md}"
LOG_FILE="${LOG_FILE:-${PROJECT_DIR}/results/qa_judge_qwen35_baseline_vs_no_yolo_100_run.log}"
WORKERS="${WORKERS:-24}"

cd "${PROJECT_DIR}"
if [[ ! -s "${NO_YOLO_RESULTS}" ]]; then
  echo "No-YOLO results not found: ${NO_YOLO_RESULTS}" >&2
  exit 1
fi

"${PYTHON_BIN}" evaluate_qa_with_qwen_judge.py \
  --baseline-results "${BASELINE_RESULTS}" \
  --ours-results "${NO_YOLO_RESULTS}" \
  --output-jsonl "${OUT_JSONL}" \
  --output-md "${OUT_MD}" \
  --workers "${WORKERS}" \
  --print-every 50 2>&1 | tee "${LOG_FILE}"
