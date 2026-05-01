#!/usr/bin/env bash
set -euo pipefail

# Fast RailVQA LoRA SFT for Qwen3-VL with LLaMA-Factory.
# Default uses physical GPU 2 and 3.

WORKSPACE="${WORKSPACE:-/mnt1/mnt2/data3/nlp/ws}"
PROJECT_DIR="${PROJECT_DIR:-${WORKSPACE}/proj/课题组}"
MODEL_DIR="${MODEL_DIR:-${WORKSPACE}/model/Qwen3-VL-8B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-${WORKSPACE}/model/Qwen3-VL-8B-RailVQA-LoRA-fast}"
DATA_DIR="${DATA_DIR:-${WORKSPACE}/llamafactory_data}"
DATASET_NAME="${DATASET_NAME:-railvqa_train}"
DATASET_JSON="${DATASET_JSON:-${DATA_DIR}/railvqa_train_qwen3vl.json}"
LLAMA_FACTORY_DIR="${LLAMA_FACTORY_DIR:-${WORKSPACE}/LLaMA-Factory}"
ENV_DIR="${ENV_DIR:-${WORKSPACE}/llamafactory_env/.venv}"
CONFIG_DIR="${CONFIG_DIR:-${PROJECT_DIR}/llamafactory_configs}"
CONFIG_FILE="${CONFIG_FILE:-${CONFIG_DIR}/railvqa_qwen3vl_lora_fast.yaml}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/results}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/train_qwen3vl_llamafactory_fast.log}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

# Fast defaults for 2x A100 80G. If OOM, rerun with:
# PER_DEVICE_BATCH=2 GRADIENT_CHECKPOINTING=true ./train_qwen3vl_llamafactory_fast.sh
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-4}"
GRAD_ACC="${GRAD_ACC:-2}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-1.0e-4}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
CUTOFF_LEN="${CUTOFF_LEN:-3072}"
IMAGE_MAX_PIXELS="${IMAGE_MAX_PIXELS:-262144}"
IMAGE_MIN_PIXELS="${IMAGE_MIN_PIXELS:-3136}"
PREPROCESS_WORKERS="${PREPROCESS_WORKERS:-16}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-8}"
FLASH_ATTN="${FLASH_ATTN:-auto}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"
SAVE_STEPS="${SAVE_STEPS:-300}"
MAX_SAMPLES="${MAX_SAMPLES:-100000}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
MINIMAL_LLAMA_FACTORY_DEPS="${MINIMAL_LLAMA_FACTORY_DEPS:-1}"

mkdir -p "${DATA_DIR}" "${CONFIG_DIR}" "${LOG_DIR}" "$(dirname "${ENV_DIR}")"
export PATH="${ENV_DIR}/bin:${HOME}/.local/bin:${PATH}"

pip_install() {
  if command -v uv >/dev/null 2>&1; then
    local has_index=0
    for arg in "$@"; do
      if [[ "${arg}" == "--index-url" || "${arg}" == "-i" ]]; then
        has_index=1
      fi
    done
    if [[ -n "${PYPI_INDEX_URL}" && "${has_index}" == "0" ]]; then
      uv pip install --python "${ENV_DIR}/bin/python" --index-url "${PYPI_INDEX_URL}" "$@"
    else
      uv pip install --python "${ENV_DIR}/bin/python" "$@"
    fi
  else
    local has_index=0
    for arg in "$@"; do
      if [[ "${arg}" == "--index-url" || "${arg}" == "-i" ]]; then
        has_index=1
      fi
    done
    if [[ -n "${PYPI_INDEX_URL}" && "${has_index}" == "0" ]]; then
      "${ENV_DIR}/bin/python" -m pip install -i "${PYPI_INDEX_URL}" "$@"
    else
      "${ENV_DIR}/bin/python" -m pip install "$@"
    fi
  fi
}

echo "=== RailVQA Qwen3-VL LLaMA-Factory fast train ==="
echo "Project: ${PROJECT_DIR}"
echo "Model: ${MODEL_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Log: ${LOG_FILE}"

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "Model dir does not exist: ${MODEL_DIR}" >&2
  exit 1
fi

if [[ ! -d "${LLAMA_FACTORY_DIR}/.git" ]]; then
  git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git "${LLAMA_FACTORY_DIR}"
elif [[ "${UPDATE_LLAMA_FACTORY:-0}" == "1" ]]; then
  git -C "${LLAMA_FACTORY_DIR}" pull --ff-only
fi

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  if ! command -v python3.11 >/dev/null 2>&1; then
    if ! command -v uv >/dev/null 2>&1; then
      curl -LsSf https://astral.sh/uv/install.sh | sh
      export PATH="${HOME}/.local/bin:${PATH}"
    fi
    uv python install 3.11
    uv venv --python 3.11 "${ENV_DIR}"
  else
    python3.11 -m venv "${ENV_DIR}"
  fi
fi

if ! "${ENV_DIR}/bin/python" -m pip --version >/dev/null 2>&1; then
  "${ENV_DIR}/bin/python" -m ensurepip --upgrade || true
fi

pip_install -U pip setuptools wheel ninja packaging

if ! "${ENV_DIR}/bin/python" - <<'PY' >/dev/null 2>&1
import torch
assert torch.cuda.is_available()
PY
then
  pip_install torch torchvision --index-url https://download.pytorch.org/whl/cu124
fi

if ! "${ENV_DIR}/bin/python" - <<'PY' >/dev/null 2>&1
import torchaudio
PY
then
  pip_install torchaudio --index-url https://download.pytorch.org/whl/cu124
fi

if [[ "${MINIMAL_LLAMA_FACTORY_DEPS}" == "1" ]]; then
  pip_install --no-deps -e "${LLAMA_FACTORY_DIR}"
  pip_install \
    "transformers>=4.55.0,<=5.6.0,!=4.52.0,!=4.57.0" \
    "datasets>=2.16.0,<=4.0.0" \
    "accelerate>=1.3.0,<=1.11.0" \
    "peft>=0.18.0,<=0.18.1" \
    "trl>=0.18.0,<=0.24.0" \
    "torchdata>=0.10.0,<=0.11.0" \
    "tyro<0.9.0" \
    "av>=10.0.0,<=16.0.0" \
    einops numpy pandas scipy sentencepiece tiktoken modelscope hf-transfer safetensors \
    fire omegaconf packaging protobuf pyyaml pydantic uvicorn fastapi sse-starlette matplotlib
else
  pip_install -e "${LLAMA_FACTORY_DIR}[metrics]"
fi

if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
  "${ENV_DIR}/bin/python" - <<'PY' >/dev/null 2>&1 || pip_install flash-attn --no-build-isolation
import flash_attn
PY
fi

"${ENV_DIR}/bin/python" "${PROJECT_DIR}/prepare_railvqa_llamafactory.py" \
  --output-file "${DATASET_JSON}" \
  --dataset-name "${DATASET_NAME}" \
  --include-cot

cp "${DATA_DIR}/dataset_info.json" "${LLAMA_FACTORY_DIR}/data/dataset_info.json"
cp "${DATASET_JSON}" "${LLAMA_FACTORY_DIR}/data/$(basename "${DATASET_JSON}")"

cat > "${CONFIG_FILE}" <<YAML
### model
model_name_or_path: ${MODEL_DIR}
trust_remote_code: true
flash_attn: ${FLASH_ATTN}
image_max_pixels: ${IMAGE_MAX_PIXELS}
image_min_pixels: ${IMAGE_MIN_PIXELS}

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: ${LORA_RANK}
lora_alpha: ${LORA_ALPHA}
lora_dropout: 0.05
lora_target: all

### dataset
dataset_dir: ${LLAMA_FACTORY_DIR}/data
dataset: ${DATASET_NAME}
template: qwen3_vl
cutoff_len: ${CUTOFF_LEN}
max_samples: ${MAX_SAMPLES}
overwrite_cache: true
preprocessing_num_workers: ${PREPROCESS_WORKERS}
dataloader_num_workers: ${DATALOADER_WORKERS}

### output
output_dir: ${OUTPUT_DIR}
logging_steps: 5
save_steps: ${SAVE_STEPS}
save_total_limit: 2
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: none

### train
per_device_train_batch_size: ${PER_DEVICE_BATCH}
gradient_accumulation_steps: ${GRAD_ACC}
learning_rate: ${LR}
num_train_epochs: ${EPOCHS}
lr_scheduler_type: cosine
warmup_ratio: 0.03
bf16: true
gradient_checkpointing: ${GRADIENT_CHECKPOINTING}
optim: adamw_torch_fused
max_grad_norm: 1.0
ddp_timeout: 180000000
ddp_find_unused_parameters: false
YAML

echo "Wrote config: ${CONFIG_FILE}"
echo "Starting training..."

export CUDA_VISIBLE_DEVICES
export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"
export TOKENIZERS_PARALLELISM=false
export NCCL_ASYNC_ERROR_HANDLING=1

cd "${LLAMA_FACTORY_DIR}"
"${ENV_DIR}/bin/llamafactory-cli" train "${CONFIG_FILE}" 2>&1 | tee "${LOG_FILE}"

echo "Training finished. LoRA adapter saved to ${OUTPUT_DIR}"
echo "Next: merge with ./merge_qwen3vl_lora_llamafactory.sh"
