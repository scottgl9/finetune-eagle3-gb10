# Claude Code Instructions: MiniMax M2.5 Eagle3 NVFP4 Fine-tune

## Goal
Fine-tune the `thoughtworks/MiniMax-M2.5-Eagle3` draft head against the NVFP4-quantized
`MiniMax-M2.5-REAP-172B-A10B-NVFP4-GB10` model. This fixes the ~6% accept rate that
occurs when using a BF16-trained Eagle3 head with an NVFP4 target model.

## Hardware
- NVIDIA Spark GB10 (SM12.1, 128 GB unified memory)
- NVFP4 model already cached at:
  `/home/scottgl/.cache/huggingface/hub/models--saricles--MiniMax-M2.5-REAP-172B-A10B-NVFP4-GB10/`

## Expected duration
~7–10 hours on single Spark (3 epochs, 10K samples, 128 token max length)

---

## Setup (run once)

```bash
# 1. Activate your sglang venv (has all GPU deps)
source ~/sandbox/sglang/.sglang/bin/activate

# 2. Clone tails-mpt SpecForge (NOT sgl-project — needs the minimax config)
git clone https://github.com/tails-mpt/SpecForge ~/SpecForge
cd ~/SpecForge
pip install -e .

# 3. Login to HuggingFace (needed for ShareGPT dataset)
huggingface-cli login
# Paste your HF token when prompted

# 4. Copy the fine-tune script to the Spark
cp /path/to/finetune_minimax_eagle3.sh ~/finetune_minimax_eagle3.sh
chmod +x ~/finetune_minimax_eagle3.sh
```

## Run the fine-tune

```bash
# Run in a screen session so it survives disconnects
screen -S eagle3

cd ~
SPECFORGE_DIR=~/SpecForge ./finetune_minimax_eagle3.sh

# Detach: Ctrl+A then D
# Reattach: screen -r eagle3

# Monitor progress
tail -f ~/outputs/minimax-m2.5-reap172b-nvfp4-eagle3/train.log
```

## Resume if interrupted

```bash
RESUME=1 SPECFORGE_DIR=~/SpecForge ./finetune_minimax_eagle3.sh
```

---

## Known issues and how to fix them

### Issue 1: OOM (Out of Memory)
**Symptom:** `torch.cuda.OutOfMemoryError` or `CUDA error: out of memory`

**Fix:** Reduce SGLang memory fraction in the script:
```bash
# In finetune_minimax_eagle3.sh, change:
MEM_FRACTION=0.88
# to:
MEM_FRACTION=0.85
```

### Issue 2: SGLang fails to load NVFP4 model
**Symptom:** Error about `modelopt_fp4` or quantization loading

**Fix:** Make sure you're using the GB10-optimized SGLang venv:
```bash
source ~/sandbox/sglang/.sglang/bin/activate
python -c "import sglang; print(sglang.__version__)"
```
If the wrong SGLang is loaded, set explicitly:
```bash
SPECFORGE_DIR=~/SpecForge \
SGLANG_PYTHON=$(which python) \
./finetune_minimax_eagle3.sh
```

### Issue 3: `draft_model_path` not recognized in config
**Symptom:** SpecForge starts with random weights instead of M2.5-Eagle3 weights

**Fix:** The patched config step may have failed. Check:
```bash
cat ~/cache/minimax-m2.5-reap172b-nvfp4-eagle3.json | python3 -m json.tool | grep draft_model_path
```
If missing, manually add:
```python
import json
cfg = json.load(open("~/SpecForge/configs/minimax-m2-eagle3.json"))
cfg["draft_model_path"] = "~/cache/draft_init/minimax-m2.5-eagle3"
json.dump(cfg, open("~/cache/minimax-m2.5-reap172b-nvfp4-eagle3.json", "w"), indent=4)
```

### Issue 4: `MiniMaxM2ForCausalLM` not found / trust_remote_code error
**Symptom:** `ValueError: Unrecognized model` or architecture error

**Fix:** The model uses `trust_remote_code=True` — SpecForge's `--trust-remote-code` flag
handles this. If it still fails, ensure transformers is up to date:
```bash
pip install --upgrade transformers
```

### Issue 5: Chat template error (`minimax` template not found)
**Symptom:** `KeyError: 'minimax'` or template lookup failure

**Fix:** MiniMax uses a Llama-style template. The script uses `--chat-template llama3`
which works. If SpecForge complains, change to:
```bash
--chat-template auto
```

### Issue 6: SpecForge can't connect to SGLang server (port conflict)
**Symptom:** `Connection refused` on localhost:30000

**Fix:** Kill any running SGLang instances:
```bash
pkill -f sglang
sleep 5
# Then re-run the script
```

### Issue 7: `eagle_aux_hidden_state_layer_ids` missing from config
**Symptom:** SpecForge error about auxiliary hidden states not configured

**Fix:** This means the tails-mpt fork config wasn't used. Manually add to the patched config:
```python
import json
cfg = json.load(open("~/cache/minimax-m2.5-reap172b-nvfp4-eagle3.json"))
cfg["eagle_config"] = {
    "eagle_aux_hidden_state_layer_ids": [1, 30, 58],
    "use_aux_hidden_state": True
}
json.dump(cfg, open("~/cache/minimax-m2.5-reap172b-nvfp4-eagle3.json", "w"), indent=4)
```

### Issue 8: Training loss doesn't decrease after epoch 1
**Symptom:** Loss plateaus at ~3.0+ with no improvement

**Fix:** Learning rate may be too high. Reduce and resume:
```bash
# In finetune_minimax_eagle3.sh, change:
LEARNING_RATE=2e-5
# to:
LEARNING_RATE=5e-6
# Then: RESUME=1 SPECFORGE_DIR=~/SpecForge ./finetune_minimax_eagle3.sh
```

---

## Validating the output

After training completes, test the acceptance rate:

```bash
# Start SGLang with the new Eagle3 head
DISABLE_NGRAM=1 ./sglang.sh minimax \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path ~/outputs/minimax-m2.5-reap172b-nvfp4-eagle3 \
    --speculative-num-steps 3 \
    --speculative-num-draft-tokens 6 \
    --speculative-eagle-topk 4

# Run a benchmark
python3 ~/sandbox/sglang/llm_speed_test.py
```

**Target acceptance rate:** >60% (should reach 70–80% for coding tasks)
**If <40%:** Training may not have converged — try 2 more epochs:
```bash
NUM_EPOCHS=2 RESUME=1 SPECFORGE_DIR=~/SpecForge ./finetune_minimax_eagle3.sh
```

---

## Output location
`~/outputs/minimax-m2.5-reap172b-nvfp4-eagle3/`

## Upload to HuggingFace when done
```bash
huggingface-cli upload YOUR_HF_USERNAME/MiniMax-M2.5-REAP-172B-Eagle3-NVFP4 \
  ~/outputs/minimax-m2.5-reap172b-nvfp4-eagle3 \
  --repo-type model
```
