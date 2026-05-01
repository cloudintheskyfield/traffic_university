#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/mnt1/mnt2/data3/nlp/ws}"
PROJECT_DIR="${PROJECT_DIR:-${WORKSPACE}/proj/课题组}"
MODEL_DIR="${MODEL_DIR:-${WORKSPACE}/model/Qwen3-VL-8B-Instruct}"
ADAPTER_DIR="${ADAPTER_DIR:-${WORKSPACE}/model/Qwen3-VL-8B-RailVQA-LoRA-fast}"
EXPORT_DIR="${EXPORT_DIR:-${WORKSPACE}/model/Qwen3-VL-8B-RailVQA-Merged-fast}"
LLAMA_FACTORY_DIR="${LLAMA_FACTORY_DIR:-${WORKSPACE}/LLaMA-Factory}"
ENV_DIR="${ENV_DIR:-${WORKSPACE}/llamafactory_env/.venv}"
CONFIG_DIR="${CONFIG_DIR:-${PROJECT_DIR}/llamafactory_configs}"
CONFIG_FILE="${CONFIG_FILE:-${CONFIG_DIR}/railvqa_qwen3vl_lora_merge.yaml}"

mkdir -p "${CONFIG_DIR}"

cat > "${CONFIG_FILE}" <<YAML
model_name_or_path: ${MODEL_DIR}
adapter_name_or_path: ${ADAPTER_DIR}
template: qwen3_vl
finetuning_type: lora
trust_remote_code: true
export_dir: ${EXPORT_DIR}
export_size: 5
export_device: cpu
export_legacy_format: false
YAML

cd "${LLAMA_FACTORY_DIR}"
"${ENV_DIR}/bin/llamafactory-cli" export "${CONFIG_FILE}"

echo "Merged model exported to ${EXPORT_DIR}"
echo "Evaluate with:"
echo "  CUDA_VISIBLE_DEVICES=2 /mnt1/mnt2/data3/nlp/ws/uv_env/.venv/bin/python ${PROJECT_DIR}/baseline_vllm.py --model-path ${EXPORT_DIR} --out-dir ${PROJECT_DIR}/results/railvqa_merged_eval"
