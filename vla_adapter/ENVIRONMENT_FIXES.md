# Environment Fixes & Hacks

All changes made to get the benchmark running on the RTX 5090 (Blackwell / sm_120).

---

## 1. HuggingFace cache symlink (no-code fix)

**Problem:** `AutoProcessor.from_pretrained("OpenGVLab/InternVL2-1B")` fails in offline
mode because the HF hub cache is empty. `agent_interface.py` sets `HF_HOME=/tmp/hf_cache`
at runtime, but `huggingface_hub` is already imported by then so the env var is ignored —
the actual cache used is `~/.cache/huggingface/hub`.

Additionally, an online HF download wrote a `model.safetensors → ../../blobs/<hash>`
symlink into the snapshot directory, but the blob file itself was never fetched, causing
`FileNotFoundError` on the weights.

**Fix:** Point both the `~/.cache` and `/tmp/hf_cache` snapshots at
`model/pretrained/InternVL2-1B/` (which has the real `model.safetensors`), not the
config-only `pretrained/InternVL2-1B/`.

```bash
HASH="0d75ccd166b1d0b79446ae6c5d1a4a667f1e6187"
MODEL_DIR="/home/shamakg/carla_garage/leaderboard/leaderboard/autovalet/vla_adapter/model/pretrained/InternVL2-1B"

# Fix ~/.cache (actually used at runtime despite env vars)
SNAP_HF="$HOME/.cache/huggingface/hub/models--OpenGVLab--InternVL2-1B"
mkdir -p "$SNAP_HF/refs" "$SNAP_HF/snapshots"
echo -n "$HASH" > "$SNAP_HF/refs/main"
rm -rf "$SNAP_HF/snapshots/$HASH"
ln -s "$MODEL_DIR" "$SNAP_HF/snapshots/$HASH"

# Fix /tmp/hf_cache (used by agent_interface.py env vars when hub isn't yet imported)
SNAP_TMP="/tmp/hf_cache/hub/models--OpenGVLab--InternVL2-1B"
mkdir -p "$SNAP_TMP/refs" "$SNAP_TMP/snapshots"
echo -n "$HASH" > "$SNAP_TMP/refs/main"
rm -rf "$SNAP_TMP/snapshots/$HASH"
ln -s "$MODEL_DIR" "$SNAP_TMP/snapshots/$HASH"
```

**Must re-run after reboot** (`/tmp` is cleared). Move to `load_models.sh` to automate.

---

## 2. PyTorch upgrade for sm_120 (Blackwell) support

**Problem:** RTX 5090 is sm_120 (Blackwell). PyTorch 2.2.0+cu121 only supports up to
sm_90 (Hopper), causing `CUDA error: no kernel image is available for execution on the device`
even on basic `.to(device).bfloat16()` tensor ops.

**Fix:** Upgrade to cu128 builds — CUDA 12.8 is the first toolkit with official Blackwell support.

```bash
# The simlingo venv has no pip binary; use uv with explicit python path
uv pip install "torch==2.9.1+cu128" torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --python /home/shamakg/envs/simlingo/bin/python3.10
```

Versions after upgrade: `torch 2.9.1+cu128`, `torchvision 0.24.1+cu128`, `torchaudio 2.11.0+cu128`.

**Gotcha:** Any subsequent `pip install deepspeed` or similar will pull a generic `torch`
from PyPI and overwrite the cu128 build with a CPU-only or wrong-CUDA build. Always
reinstall `torch==2.9.1+cu128` last after installing other packages.

---

## 3. Package upgrades (compatibility with torch 2.9 / numpy 2.x)

```bash
uv pip install deepspeed --upgrade --python /home/shamakg/envs/simlingo/bin/python3.10
uv pip install "transformers==4.46.3" --python /home/shamakg/envs/simlingo/bin/python3.10
uv pip install scikit-learn --reinstall --python /home/shamakg/envs/simlingo/bin/python3.10
# Reinstall torch after deepspeed overwrote it:
uv pip install "torch==2.9.1+cu128" --extra-index-url https://download.pytorch.org/whl/cu128 \
    --python /home/shamakg/envs/simlingo/bin/python3.10
```

- **deepspeed** 0.16.2 → 0.19.1 (torch 2.9 compatibility)
- **transformers** pinned to **4.46.3** — 5.x broke `AutoProcessor` for InternVL2-1B
  (returns `CLIPImageProcessor` with no tokenizer instead of `Qwen2TokenizerFast`)
- **scikit-learn** reinstalled because torch cu128 upgraded numpy 1.26 → 2.2,
  breaking sklearn's compiled C extensions with `ValueError: numpy.dtype size changed`

---

## 4. `flash_attn` compatibility shim

**Problem:** Three separate failures from missing `flash_attn`:

1. `modeling_intern_vit.py` imports `from flash_attn.bert_padding import pad_input, unpad_input`
   and `from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func` —
   falls back to slow naive attention when these fail.
2. `transformers/modeling_flash_attention_utils.py` imports
   `from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input` and
   `from flash_attn import flash_attn_func, flash_attn_varlen_func` — crashes on import.
3. `importlib.metadata.version("flash_attn")` raises `PackageNotFoundError` because
   there is no dist-info for our shim.

The pre-built flash-attn-3 wheel (`flash_attn_3` package, from
`https://download.pytorch.org/whl/flash-attn-3/`) was built for Hopper (sm_90) only —
its CUDA kernels raise `no kernel image is available` on sm_120.

**Fix:** A pure-Python `flash_attn` shim package backed by
`torch.nn.functional.scaled_dot_product_attention` (SDPA), which has native sm_120 support
in torch 2.9+cu128. Three files + one dist-info directory created manually in the venv:

```
/home/shamakg/envs/simlingo/lib/python3.10/site-packages/
  flash_attn/
    __init__.py              — re-exports all public symbols
    flash_attn_interface.py  — flash_attn_func, flash_attn_varlen_func,
                               flash_attn_varlen_qkvpacked_func  (all SDPA-backed)
    bert_padding.py          — index_first_axis, pad_input, unpad_input (pure PyTorch)
  flash_attn-2.8.3.dist-info/
    METADATA                 — Name: flash_attn, Version: 2.8.3
    INSTALLER
```

Key implementation details:
- `flash_attn_varlen_qkvpacked_func`: unpacks packed `(total, 3, heads, dim)` qkv into
  separate q/k/v, then calls `F.scaled_dot_product_attention` via the varlen path.
- `flash_attn_varlen_func`: fast path for equal-length sequences (single batched SDPA call);
  falls back to per-element SDPA for genuinely variable-length batches.
- `flash_attn_func`: standard batched attention, transposes to SDPA's `(B,H,S,D)` layout.
- `index_first_axis`, `pad_input`, `unpad_input`: pure PyTorch, no CUDA kernels needed.

---

## 5. `torch.compile` — setup and simlingo code fixes

### 5a. `agent_interface.py` — compile after model load

Added two lines inside `init_testbed()`, after `self.setup(...)`:

```python
# torch 2.9.1 inductor hits a sympy complex-number bug on some subgraphs;
# suppress_errors lets those fall back to eager rather than crashing.
torch._dynamo.config.suppress_errors = True
self.model = torch.compile(self.model, mode="reduce-overhead", dynamic=True)
```

- `reduce-overhead`: uses CUDA graphs to eliminate Python dispatch overhead on repeated
  inference calls (the main bottleneck at 5 FPS).
- `dynamic=True`: avoids recompilation when prompt token lengths vary between frames.
- `suppress_errors=True`: torch 2.9.1's inductor hits a sympy `TypeError: cannot create
  mpf from (complex)` when tiling-analysing some subgraphs. This flag silently falls back
  to eager for those subgraphs instead of crashing.
- Expect ~60 s warmup on first inference call while the compiler traces the model.

### 5b. `driving.py` — `try/except AttributeError` → `hasattr()`

**File:** `simlingo/simlingo_training/models/driving.py`  
**Marked:** `# CHANGE MADE BY SHAMAK`

TorchDynamo's symbolic tracer does not handle `try/except AttributeError` correctly —
it evaluates the attribute access symbolically before the exception handler fires, so
the `AttributeError` propagates as a real error instead of being caught.

Replaced at the `example.driving_input` check (line ~113):
```python
# Before:
try:
    driving_input = example.driving_input
except AttributeError:
    driving_input = example

# After:
if hasattr(example, 'driving_input'):
    driving_input = example.driving_input
else:
    driving_input = example
```

### 5c. `adaptors.py` — same fix, two locations

**File:** `simlingo/simlingo_training/models/adaptors/adaptors.py`  
**Marked:** `# CHANGE MADE BY SHAMAK`

Same `try/except AttributeError` → `hasattr()` pattern at two call sites
(lines ~144 and ~239), both accessing `example.driving_input` / `driving_example.driving_input`.

---

## Summary of all modified files

| File | Type | Change |
|------|------|--------|
| `agent_interface.py` | your code | `torch._dynamo.config.suppress_errors = True` + `torch.compile` after model load |
| `simlingo/simlingo_training/models/driving.py` | simlingo | `try/except AttributeError` → `hasattr()` (1 site) |
| `simlingo/simlingo_training/models/adaptors/adaptors.py` | simlingo | `try/except AttributeError` → `hasattr()` (2 sites) |
| `envs/simlingo/.../flash_attn/__init__.py` | new shim | Re-exports all FA2 public symbols |
| `envs/simlingo/.../flash_attn/flash_attn_interface.py` | new shim | SDPA-backed FA2 attention functions |
| `envs/simlingo/.../flash_attn/bert_padding.py` | new shim | Pure-PyTorch padding helpers |
| `envs/simlingo/.../flash_attn-2.8.3.dist-info/METADATA` | new metadata | Satisfies `importlib.metadata.version("flash_attn")` |
| `~/.cache/huggingface/hub/models--OpenGVLab--InternVL2-1B/snapshots/<hash>` | symlink | Points to `model/pretrained/InternVL2-1B/` (has weights) |
| `/tmp/hf_cache/hub/models--OpenGVLab--InternVL2-1B/snapshots/<hash>` | symlink | Same (cleared on reboot) |
