# Eagle3 Fine-Tuning for MiniMax-M2.5-REAP-172B-A10B-NVFP4-GB10

## Overview

Fine-tune an [Eagle3](https://github.com/SafeAILab/EAGLE) speculative decoding
draft head for the NVFP4-quantized MiniMax M2.5 172B model on a single NVIDIA
Spark GB10 (128 GB unified memory).

### Why fine-tune?

The existing `thoughtworks/MiniMax-M2.5-Eagle3` head was trained against the
BF16 (unquantized) model. NVFP4 quantization shifts hidden-state distributions
and logit outputs, which drops the Eagle3 acceptance rate to ~6% — too low to
provide any speculative decoding speedup.

Fine-tuning the existing draft head against the *quantized* model teaches it
to predict the NVFP4 model's actual output distribution, targeting >60%
acceptance rate.

## How it works

The training uses [SpecForge](https://github.com/tails-mpt/SpecForge) (tails-mpt
fork) with online training mode, **fine-tuning** the existing
`thoughtworks/MiniMax-M2.5-Eagle3` head (not training from scratch):

1. **Draft head weights** are loaded from the pre-trained Eagle3 head via
   `--ckpt-dir` (uses `from_pretrained`, not random init)
2. **Target model** (172B NVFP4) is loaded via SGLang with `compressed-tensors`
   quantization and triton attention backend
3. **Each training step** feeds a tokenized sample through the target model,
   extracting:
   - Hidden states from 3 auxiliary layers (1, 30, 58)
   - Output logits (next-token distribution)
4. **Draft head** (~200M params) is fine-tuned to predict the quantized model's
   next-token distribution given the auxiliary hidden states, using KL-divergence
   loss
5. **TTT (Test-Time Training)** unrolls 7 autoregressive steps per sample so
   the draft head learns to chain predictions
6. **Embeddings** are loaded from the target model and frozen — only the draft
   head layers are trained

### Architecture

```
Target model (172B NVFP4, frozen)
  ├─ Layer 1  ──────────┐
  ├─ Layer 30 ──────────┤  Concatenated: (batch, seq, 3 × 3072)
  ├─ Layer 58 ──────────┘
  └─ Logits output ────── Teacher signal (softmax distribution)

Draft head (trainable)
  ├─ Projection: 3×3072 → 3072
  ├─ Concat with token embeddings → 6144
  ├─ 1-layer Llama decoder (24 heads, 8 KV heads)
  └─ LM head → 200064 vocab (mapped to 32000 draft vocab)
```

## Prerequisites

- NVIDIA Spark GB10 (or equivalent with ≥115 GB GPU memory)
- [scottgl9/sglang](https://github.com/scottgl9/sglang) — custom SGLang fork
  with GB10 (SM 12.1) support, FlashInfer compatibility patches, and NVFP4
  quantization fixes. Build with `./sglang.sh build`.
- [tails-mpt/SpecForge](https://github.com/tails-mpt/SpecForge) — clone and
  `pip install -e .`, then apply the patch in `patches/`.
- HuggingFace CLI logged in (`hf auth login`)
- Models downloaded:
  - [saricles/MiniMax-M2.5-REAP-172B-A10B-NVFP4-GB10](https://huggingface.co/saricles/MiniMax-M2.5-REAP-172B-A10B-NVFP4-GB10)
  - [thoughtworks/MiniMax-M2.5-Eagle3](https://huggingface.co/thoughtworks/MiniMax-M2.5-Eagle3)

## Files

| File | Purpose |
|------|---------|
| `finetune_minimax_eagle3.sh` | Main fine-tuning script |
| `finetune_minimax_eagle3_INSTRUCTIONS.md` | Setup and troubleshooting guide |
| `patches/specforge-quantization-args.patch` | SpecForge patch for quantized model support |

## Setup

```bash
# 1. Clone SpecForge and install
git clone https://github.com/tails-mpt/SpecForge
cd SpecForge && pip install -e . && cd ..

# 2. Apply the quantization args patch
cd SpecForge && git apply ../patches/specforge-quantization-args.patch && cd ..

# 3. Download models (if not already cached)
huggingface-cli download saricles/MiniMax-M2.5-REAP-172B-A10B-NVFP4-GB10
huggingface-cli download thoughtworks/MiniMax-M2.5-Eagle3
```

## Running

```bash
# Stop any running model servers to free GPU memory
# (the 172B model needs ~93 GB)

# Run fine-tuning
nohup bash -c 'SPECFORGE_DIR=SpecForge bash finetune_minimax_eagle3.sh' &

# Monitor progress
tail -f outputs/minimax-m2.5-reap172b-nvfp4-eagle3/train.log

# Resume from checkpoint if interrupted
nohup bash -c 'RESUME=1 SPECFORGE_DIR=SpecForge bash finetune_minimax_eagle3.sh' &
```

## Training hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Epochs | 3 | Standard for Eagle3 training |
| Batch size | 1 | Memory-limited (172B model in GPU) |
| Learning rate | 2e-5 | Transfer fine-tune range |
| Max length | 128 | Short sequences — sufficient for hidden state distribution matching |
| Warmup ratio | 0.05 | 5% of steps |
| Max grad norm | 0.5 | Gradient clipping |
| TTT length | 7 | Autoregressive unroll steps |
| Dataset | ShareGPT (120K samples) | Mixed conversational data |
| SGLang mem fraction | 0.85 | Leaves headroom for Marlin FP4 init + KV cache |
| SGLANG_QUANTIZE_LM_HEAD_FP8 | 0 (disabled) | Draft head learns the unquantized lm_head distribution for compatibility with standard SGLang |
| enable_fp32_lm_head | true | Prevents NaN from BF16 lm_head overflow on 200K vocab (see below) |
| Chat template | minimax | Must match model's chat format for correct loss masking |

## Required patches

### SpecForge patch

The included patch (`patches/specforge-quantization-args.patch`) modifies
SpecForge to work with quantized NVFP4 models:

1. **`specforge/args.py`**: Adds `--sglang-quantization` and
   `--sglang-moe-runner-backend` arguments to `SGLangBackendArgs`
   (upstream SpecForge assumes unquantized models)

2. **`specforge/modeling/target/eagle3_target_model.py`**:
   - Added `enable_fp32_lm_head=True` to `ServerArgs` — prevents NaN from
     BF16 lm_head overflow on 200K vocab
   - Added `set_global_server_args_for_scheduler()` call so `LogitsProcessor`
     reads the FP32 lm_head setting
   - Wrapped `SWATokenToKVPoolAllocator` import in try/except for SGLang
     compatibility

### SGLang patch (scottgl9/sglang)

The NVFP4 W4A4 MoE kernel on SM121 (GB10) produces NaN at transformer
layer 60+ when running in extend/prefill mode (used by SpecForge for hidden
state extraction). The attention layers and earlier MoE layers work correctly.

**Fix in `sglang/srt/models/minimax_m2.py`**: After each MoE forward pass,
check for NaN output and replace with the pre-MoE hidden states (identity
fallback). This preserves the valid aux hidden states from layers 1, 30, 58
while producing approximate but non-NaN logits for the teacher signal.

```python
# In MiniMaxM2DecoderLayer.forward(), after block_sparse_moe:
moe_input = hidden_states
hidden_states = self.block_sparse_moe(hidden_states, forward_batch)
if hidden_states.isnan().any():
    nan_mask = hidden_states.isnan()
    hidden_states = torch.where(nan_mask, moe_input, hidden_states)
```

~29% of training samples have all-NaN MoE output (loss=0 for those samples).
The remaining ~71% train normally with valid teacher signal. Peak per-sample
accuracy reaches 90%+ during training.

### Why `enable_fp32_lm_head`?

The 172B NVFP4 model's `lm_head` is excluded from quantization (stays BF16).
The BF16 matmul of (tokens, 3072) x (3072, 200064) can overflow and produce
NaN. FP32 lm_head fixes this. `SGLANG_QUANTIZE_LM_HEAD_FP8` is disabled so
the draft head learns the standard distribution without FP8 post-quantization.

### Why `--chat-template minimax`?

SpecForge uses the chat template to identify assistant turns and build a loss
mask. Using the wrong template (e.g., `llama3`) results in `loss_mask=0` for
all tokens, meaning no training signal. MiniMax uses `]~b]ai\n` as the
assistant header, registered as the `minimax` template in SpecForge.

## Troubleshooting

### OOM during model loading
Reduce `MEM_FRACTION` in the script (default: 0.85). The 172B NVFP4 model uses
~93 GB. Marlin FP4 weight repacking needs temporary buffers during init.

### Silent process death (no error in log)
Usually OOM killer — check `dmesg | grep -i oom`. Reduce `MEM_FRACTION`.

### NaN logits / loss=0.00 on all samples
Ensure `enable_fp32_lm_head=True` is set in SpecForge's `ServerArgs` creation,
and that `set_global_server_args_for_scheduler()` is called so `LogitsProcessor`
reads it. Both are in the patch.

### loss_mask all zeros (loss=0.00, acc=0.00)
Wrong chat template. Must use `--chat-template minimax`, not `llama3`.

### ~29% of samples show loss=0.00
This is expected — the NVFP4 MoE NaN issue affects some samples entirely.
The identity fallback produces zero-information logits for those samples. The
model still trains effectively on the remaining ~71%.

### Dataset path error
SpecForge's `prepare_data.py` creates a directory `sharegpt_train.jsonl/`
containing the actual file. The script handles this automatically.

### `SWATokenToKVPoolAllocator` import error
Handled by the SpecForge patch (try/except fallback to `RadixCache`).

### Training loss doesn't decrease
Try reducing learning rate to 5e-6 and resuming:
```bash
RESUME=1 SPECFORGE_DIR=SpecForge bash finetune_minimax_eagle3.sh
```

## Expected timeline

| Phase | Duration |
|-------|----------|
| Data prep | ~10 seconds (cached after first run) |
| Model loading | ~10 minutes (20 safetensor shards) |
| Marlin FP4 init | ~2 minutes |
| KV cache allocation | ~30 seconds |
| Training (3 epochs x 120K samples) | ~7-10 hours |
| **Total** | **~8-11 hours** |

## Validating the output

After training completes, test the acceptance rate:

```bash
# Start SGLang with the new Eagle3 head
DISABLE_NGRAM=1 ./sglang.sh minimax \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path <path-to-output-dir> \
    --speculative-num-steps 3 \
    --speculative-num-draft-tokens 6 \
    --speculative-eagle-topk 4
```

**Target:** >60% acceptance rate (70-80% on coding tasks)

## Upload to HuggingFace

```bash
huggingface-cli upload YOUR_USERNAME/MiniMax-M2.5-REAP-172B-Eagle3-NVFP4 \
  <path-to-output-dir> \
  --repo-type model
```
