#!/bin/bash
# =============================================================================
# Fine-tune thoughtworks/MiniMax-M2.5-Eagle3 head against
# MiniMax-M2.5-REAP-172B-A10B-NVFP4-GB10 hidden states.
#
# Purpose: Fixes the ~6% accept rate that occurs when using a BF16-trained
# Eagle3 head with an NVFP4 target model. Trains the draft head to match
# the quantized model's hidden state distribution.
#
# Hardware: Single NVIDIA Spark GB10 (128 GB unified memory)
# Time:     ~8–12 hours total (data prep + training)
# Disk:     ~5 GB new (model already cached locally)
#
# Prerequisites:
#   1. tails-mpt/SpecForge installed (see setup below)
#   2. M2.5-REAP-172B NVFP4 already cached at:
#      /home/scottgl/.cache/huggingface/hub/
#        models--saricles--MiniMax-M2.5-REAP-172B-A10B-NVFP4-GB10/
#   3. HuggingFace CLI logged in (for dataset download)
#
# Usage:
#   chmod +x finetune_minimax_eagle3.sh
#   ./finetune_minimax_eagle3.sh
#   # Resume interrupted run:
#   RESUME=1 ./finetune_minimax_eagle3.sh
# =============================================================================

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Local model path — already cached on Spark
TARGET_MODEL="/home/scottgl/.cache/huggingface/hub/models--saricles--MiniMax-M2.5-REAP-172B-A10B-NVFP4-GB10/snapshots/$(ls /home/scottgl/.cache/huggingface/hub/models--saricles--MiniMax-M2.5-REAP-172B-A10B-NVFP4-GB10/snapshots/ | tail -1)"

# SpecForge repo location (adjust if you cloned elsewhere)
SPECFORGE_DIR="${SPECFORGE_DIR:-${SCRIPT_DIR}/SpecForge}"

# Output
OUTPUT_DIR="${SCRIPT_DIR}/outputs/minimax-m2.5-reap172b-nvfp4-eagle3"
CACHE_DIR="${SCRIPT_DIR}/cache"
DATASET_DIR="${CACHE_DIR}/dataset"
DRAFT_CONFIG="${SPECFORGE_DIR}/configs/minimax-m2-eagle3.json"

# Training hyperparams — calibrated for transfer fine-tune (not from scratch)
NUM_EPOCHS=3
BATCH_SIZE=1
LEARNING_RATE=2e-5
MAX_LENGTH=128        # Short sequences — enough for hidden state distribution matching
WARMUP_RATIO=0.05

# SGLang serving config for hidden state extraction
# Conservative memory split: 88% for model, 12% for training overhead
MEM_FRACTION=0.85

# Resume from last checkpoint?
RESUME="${RESUME:-0}"

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── Preflight ─────────────────────────────────────────────────────────────────

info "MiniMax M2.5 Eagle3 NVFP4 Fine-tune"
info "Target model : ${TARGET_MODEL}"
info "Draft config : ${DRAFT_CONFIG}"
info "Output dir   : ${OUTPUT_DIR}"
echo ""

# Verify target model path
if [[ ! -d "${TARGET_MODEL}" ]]; then
    error "Target model not found at: ${TARGET_MODEL}"
    error "Check that the NVFP4 model is cached at the expected snapshot path."
    exit 1
fi

# Verify SpecForge
if [[ ! -d "${SPECFORGE_DIR}" ]]; then
    error "SpecForge not found at: ${SPECFORGE_DIR}"
    info  "Install with:"
    info  "  git clone https://github.com/tails-mpt/SpecForge ${SPECFORGE_DIR}"
    info  "  cd ${SPECFORGE_DIR} && pip install -e ."
    exit 1
fi

# Verify draft config
if [[ ! -f "${DRAFT_CONFIG}" ]]; then
    error "Draft config not found: ${DRAFT_CONFIG}"
    error "Expected tails-mpt/SpecForge which includes configs/minimax-m2-eagle3.json"
    exit 1
fi

# Verify CUDA
if ! python3 -c "import torch; assert torch.cuda.is_available(), 'No CUDA'" 2>/dev/null; then
    error "CUDA not available. Is the SGLang venv activated?"
    exit 1
fi

# ── Runtime env (mirrors ~/sandbox/sglang/sglang.sh setup_runtime_env) ───────
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TRITON_PTXAS_PATH="${CUDA_HOME}/bin/ptxas"

SGLANG_COMPILERS_DIR="${HOME}/.cache/sglang_compilers"
mkdir -p "${SGLANG_COMPILERS_DIR}/triton" "${SGLANG_COMPILERS_DIR}/nv/ComputeCache" \
         "${SGLANG_COMPILERS_DIR}/flashinfer" "${SGLANG_COMPILERS_DIR}/torch"
export CUDA_CACHE_PATH="${SGLANG_COMPILERS_DIR}/nv/ComputeCache"
export CUDA_CACHE_MAXSIZE=4294967296
export TRITON_CACHE_DIR="${SGLANG_COMPILERS_DIR}/triton"
export FLASHINFER_WORKSPACE_DIR="${SGLANG_COMPILERS_DIR}/flashinfer"
export TORCHINDUCTOR_CACHE_DIR="${SGLANG_COMPILERS_DIR}/torch/inductor"
export MAX_JOBS="${MAX_JOBS:-4}"
export FLASHINFER_NVCC_THREADS="${FLASHINFER_NVCC_THREADS:-1}"
export CUDA_NVCC_FLAGS="${CUDA_NVCC_FLAGS:---threads 4}"
export TORCH_COMPILE_THREADS="${TORCH_COMPILE_THREADS:-4}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-4}"
export SAFETENSORS_FAST_GPU=1
export SGLANG_QUANTIZE_LM_HEAD_FP8="${SGLANG_QUANTIZE_LM_HEAD_FP8:-0}"
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SGLANG_DISABLE_CUDNN_CHECK=1
export SGLANG_ENABLE_JIT_DEEPGEMM="${SGLANG_ENABLE_JIT_DEEPGEMM:-0}"

info "Runtime env configured (mirrors sglang.sh setup_runtime_env)"

VRAM_GB=$(python3 -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory/1e9:.0f}')")
info "VRAM: ${VRAM_GB} GB"
if [[ "${VRAM_GB}" -lt 110 ]]; then
    warn "Less than 110 GB VRAM detected — may OOM with 99 GB NVFP4 model."
    warn "Try reducing MEM_FRACTION to 0.85 if it fails."
fi

mkdir -p "${OUTPUT_DIR}" "${CACHE_DIR}" "${DATASET_DIR}"

# ── Step 1: Prepare training data ─────────────────────────────────────────────

TRAIN_DATA="${DATASET_DIR}/sharegpt_train.jsonl"

# Fix: prepare_data.py may create a directory instead of a file — use the nested file
if [[ -d "${TRAIN_DATA}" && -f "${TRAIN_DATA}/sharegpt_train.jsonl" ]]; then
    TRAIN_DATA="${TRAIN_DATA}/sharegpt_train.jsonl"
fi

if [[ -f "${TRAIN_DATA}" ]]; then
    NLINES=$(wc -l < "${TRAIN_DATA}")
    info "Training data already exists (${NLINES} samples): ${TRAIN_DATA}"
else
    info "Preparing training data from ShareGPT..."
    python3 "${SPECFORGE_DIR}/scripts/prepare_data.py" \
        --dataset sharegpt \
        --output-path "${TRAIN_DATA}"
    info "Training data saved: ${TRAIN_DATA}"
fi

# ── Step 2: Download draft model weights ──────────────────────────────────────

DRAFT_WEIGHTS_DIR="${CACHE_DIR}/draft_init/minimax-m2.5-eagle3"

if [[ -d "${DRAFT_WEIGHTS_DIR}" ]]; then
    info "Draft model weights already downloaded: ${DRAFT_WEIGHTS_DIR}"
else
    info "Downloading thoughtworks/MiniMax-M2.5-Eagle3 draft weights..."
    mkdir -p "${DRAFT_WEIGHTS_DIR}"
    huggingface-cli download thoughtworks/MiniMax-M2.5-Eagle3 \
        --local-dir "${DRAFT_WEIGHTS_DIR}" \
        --local-dir-use-symlinks False
    info "Draft weights saved: ${DRAFT_WEIGHTS_DIR}"
fi

# ── Step 3: Patch draft config for NVFP4 target ───────────────────────────────
#
# The minimax-m2-eagle3.json from tails-mpt already has:
#   - eagle_aux_hidden_state_layer_ids: [1, 30, 58]
#   - hidden_size: 3072, vocab_size: 200064
# We just need to make sure the draft_model_path points to our downloaded weights.

PATCHED_CONFIG="${CACHE_DIR}/minimax-m2.5-reap172b-nvfp4-eagle3.json"
python3 - <<PYEOF
import json, os

config_path = "${DRAFT_CONFIG}"
out_path    = "${PATCHED_CONFIG}"
weights_dir = "${DRAFT_WEIGHTS_DIR}"

with open(config_path) as f:
    cfg = json.load(f)

# Point to pre-trained weights for transfer fine-tune
cfg["draft_model_path"] = weights_dir

with open(out_path, "w") as f:
    json.dump(cfg, f, indent=4)

print(f"Patched config written to: {out_path}")
print(f"  aux layers   : {cfg.get('eagle_config', {}).get('eagle_aux_hidden_state_layer_ids', 'N/A')}")
print(f"  hidden_size  : {cfg.get('hidden_size', 'N/A')}")
print(f"  vocab_size   : {cfg.get('vocab_size', 'N/A')}")
print(f"  draft_model  : {weights_dir}")
PYEOF

# ── Step 4: Train ─────────────────────────────────────────────────────────────

RESUME_FLAG=""
if [[ "${RESUME}" == "1" ]]; then
    RESUME_FLAG="--resume"
    info "Resume mode: will continue from last checkpoint in ${OUTPUT_DIR}"
fi

info ""
info "Starting Eagle3 fine-tune..."
info "  Epochs        : ${NUM_EPOCHS}"
info "  Batch size    : ${BATCH_SIZE}"
info "  Learning rate : ${LEARNING_RATE}"
info "  Max length    : ${MAX_LENGTH}"
info "  Backend       : sglang (your GB10 fork)"
info "  SGLang mem    : ${MEM_FRACTION}"
info ""
info "Estimated time: 8–12 hours on single Spark"
info ""

torchrun \
    --standalone \
    --nproc_per_node 1 \
    "${SPECFORGE_DIR}/scripts/train_eagle3.py" \
    \
    `# Model` \
    --target-model-path "${TARGET_MODEL}" \
    --trust-remote-code \
    --draft-model-config "${PATCHED_CONFIG}" \
    --ckpt-dir "${DRAFT_WEIGHTS_DIR}" \
    --embedding-key "model.embed_tokens.weight" \
    --target-model-backend sglang \
    \
    `# SGLang backend config — matches sglang.sh minimax preset` \
    --sglang-quantization compressed-tensors \
    --sglang-mem-fraction-static "${MEM_FRACTION}" \
    --sglang-context-length 4096 \
    --sglang-attention-backend triton \
    \
    `# Dataset` \
    --train-data-path "${TRAIN_DATA}" \
    --chat-template minimax \
    --build-dataset-num-proc 4 \
    \
    `# Training` \
    --num-epochs "${NUM_EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --learning-rate "${LEARNING_RATE}" \
    --max-length "${MAX_LENGTH}" \
    --warmup-ratio "${WARMUP_RATIO}" \
    --max-grad-norm 0.5 \
    --save-interval 2000 \
    --eval-interval 2000 \
    \
    `# Output` \
    --output-dir "${OUTPUT_DIR}" \
    --cache-dir "${CACHE_DIR}" \
    ${RESUME_FLAG} \
    \
    2>&1 | tee "${OUTPUT_DIR}/train.log"

info ""
info "✅ Training complete!"
info "Output: ${OUTPUT_DIR}"
info ""
info "Next steps:"
info "  1. Test acceptance rate: run your SGLang bench with --speculative-algorithm EAGLE3"
info "     --speculative-draft-model-path ${OUTPUT_DIR}"
info "  2. If acceptance rate is good (>60%), push to HuggingFace:"
info "     huggingface-cli upload YOUR_HF_USERNAME/MiniMax-M2.5-REAP-172B-Eagle3-NVFP4 ${OUTPUT_DIR}"
