#!/usr/bin/env bash
set -euo pipefail

# Qwen3.5-35B-A3B vLLM judge server for QA evaluation.
# Default uses physical GPU 4 and 5 on the A100 server.

PYTHON_BIN="${PYTHON_BIN:-/mnt1/mnt2/data3/nlp/ws/qwen35_vllm_env/.venv/bin/python}"
VLLM_BIN="${VLLM_BIN:-/mnt1/mnt2/data3/nlp/ws/qwen35_vllm_env/.venv/bin/vllm}"
QWEN35_ENV_DIR="${QWEN35_ENV_DIR:-/mnt1/mnt2/data3/nlp/ws/qwen35_vllm_env/.venv}"
MODEL_DIR="${MODEL_DIR:-/mnt1/mnt2/data3/nlp/ws/model/Qwen3.5-35B-A3B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen35-35b-a3b-judge}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
LOG_DIR="${LOG_DIR:-/mnt1/mnt2/data3/nlp/ws/proj/课题组/results}"

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "Model directory does not exist: ${MODEL_DIR}" >&2
  echo "Please download Qwen/Qwen3.5-35B-A3B first." >&2
  exit 1
fi

if [[ ! -f "${MODEL_DIR}/config.json" || ! -f "${MODEL_DIR}/model.safetensors.index.json" ]]; then
  echo "Model directory exists but does not look complete yet: ${MODEL_DIR}" >&2
  echo "Missing config.json or model.safetensors.index.json." >&2
  echo "Check download log: /mnt1/mnt2/data3/nlp/ws/model/qwen35_35b_a3b_download.log" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

echo "Starting Qwen3.5-35B-A3B judge server"
echo "Model dir: ${MODEL_DIR}"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Tensor parallel size: ${TENSOR_PARALLEL_SIZE}"
echo "Endpoint: http://${HOST}:${PORT}/v1"
echo "Log file suggestion: ${LOG_DIR}/qwen35_35b_judge_vllm.log"

export CUDA_VISIBLE_DEVICES
export VLLM_HOST_IP="${VLLM_HOST_IP:-127.0.0.1}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

if [[ -x "${VLLM_BIN}" ]]; then
  exec "${VLLM_BIN}" serve "${MODEL_DIR}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --trust-remote-code \
    --reasoning-parser qwen3 \
    "$@"
fi

exec "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_DIR}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  "$@"
