# Resuming Eagle3 Fine-Tuning

## Current status (paused 2026-04-16)

- **Training**: Epoch 0, step ~33000 / 78812 (42%)
- **Checkpoints saved**: 16 (every 2000 steps, up to step 32000)
- **Output dir**: `outputs/minimax-m2.5-reap172b-nvfp4-eagle3/`
- **Total epochs**: 3 (236,436 total steps)
- **Speed**: ~1.21s/step, ~26h per epoch
- **Accuracy**: 40-90% on valid samples, peaks at 95%
- **NaN samples**: ~29% of samples have loss=0 (MoE NaN issue, expected)

## If training is still running

The training was started with `nohup` and will continue even if the terminal
is closed. Check if it's running:

```bash
pgrep -af "train_eagle3"
tail -5 outputs/minimax-m2.5-reap172b-nvfp4-eagle3/train.log
```

## If training stopped / needs to resume

```bash
# 1. Activate venv
source ~/sandbox/sglang/.sglang/bin/activate

# 2. Stop any running model servers to free GPU memory
systemctl --user stop sglang-qwen3.service
# Also kill any other sglang processes:
pkill -f sglang

# 3. Wait for memory to free
sleep 10 && free -h

# 4. Resume from last checkpoint
nohup bash -c 'RESUME=1 SPECFORGE_DIR=SpecForge bash finetune_minimax_eagle3.sh' > /dev/null 2>&1 &

# 5. Monitor
tail -f outputs/minimax-m2.5-reap172b-nvfp4-eagle3/train.log
```

## Required patches (already applied)

These patches must be in place for training to work. They are already applied
in the current setup but need to be re-applied if SpecForge or SGLang is
reinstalled:

### SpecForge patches (in `SpecForge/` directory)

1. **`specforge/args.py`**: `--sglang-quantization` and `--sglang-moe-runner-backend`
2. **`specforge/modeling/target/eagle3_target_model.py`**:
   - `enable_fp32_lm_head=True` in ServerArgs
   - `set_global_server_args_for_scheduler()` call
   - `SWATokenToKVPoolAllocator` try/except
3. **`specforge/modeling/target/sglang_backend/utils.py`**: Clean (no NaN
   workarounds needed here — the fix is in the SGLang model)

### SGLang patches (in `~/sandbox/sglang/`)

1. **`python/sglang/srt/models/minimax_m2.py`**: MoE NaN identity fallback
   ```python
   # In MiniMaxM2DecoderLayer.forward(), after block_sparse_moe:
   moe_input = hidden_states
   hidden_states = self.block_sparse_moe(hidden_states, forward_batch)
   if hidden_states.isnan().any():
       nan_mask = hidden_states.isnan()
       hidden_states = torch.where(nan_mask, moe_input, hidden_states)
   ```

2. **`python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py`**:
   - `_is_static_tensor_w8a8`: guard for `input_quant is None`
   - `_is_dynamic_token_w8a8`: guard for `input_quant is None`
   - `_is_dynamic_token_w4a8`: guard for `input_quant is None`
   - `_is_wNa16_group_channel`: added `TENSOR_GROUP` strategy
   - `_get_scheme_from_parts`: added `nvfp4_pack_quantized` format

3. **`python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16_moe.py`**:
   - Added `nvfp4_pack_quantized` to accepted formats

## Key training settings

| Setting | Value |
|---------|-------|
| Target model | `saricles/MiniMax-M2.5-REAP-172B-A10B-NVFP4-GB10` |
| Draft head init | `thoughtworks/MiniMax-M2.5-Eagle3` (via `--ckpt-dir`) |
| Quantization | `compressed-tensors` |
| Chat template | `minimax` |
| `SGLANG_QUANTIZE_LM_HEAD_FP8` | `0` (disabled) |
| `enable_fp32_lm_head` | `true` |
| `mem_fraction_static` | `0.85` |
| Epochs | 3 |
| Batch size | 1 |
| Learning rate | 2e-5 |
| Max length | 128 |

## When training completes

After all 3 epochs finish:

1. The final model will be in `outputs/minimax-m2.5-reap172b-nvfp4-eagle3/`
2. Test acceptance rate:
   ```bash
   DISABLE_NGRAM=1 ~/sandbox/sglang/sglang.sh minimax \
       --speculative-algorithm EAGLE3 \
       --speculative-draft-model-path outputs/minimax-m2.5-reap172b-nvfp4-eagle3 \
       --speculative-num-steps 3 \
       --speculative-num-draft-tokens 6 \
       --speculative-eagle-topk 4
   ```
3. Upload to HuggingFace:
   ```bash
   huggingface-cli upload scottgl9/MiniMax-M2.5-REAP-172B-Eagle3-NVFP4 \
     outputs/minimax-m2.5-reap172b-nvfp4-eagle3 \
     --repo-type model
   ```

## GitHub repo

https://github.com/scottgl9/finetune-eagle3-gb10
