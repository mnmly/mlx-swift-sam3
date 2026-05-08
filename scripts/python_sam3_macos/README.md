# python/sam3 macOS CPU runner

This directory persists the shims and runner script used to capture parity
reference outputs from Meta's `python/sam3` multiplex tracker on macOS CPU.

`setup_stubs.py` patches torch (cuda → cpu, bf16 → fp32, etc.) and stubs
`sam3.model.edt` (which imports triton, unavailable on macOS).

## Usage

Pre-import the shim before `sam3`:

```bash
PYTHONPATH=$HOME/Development-local/GitHub/python/sam3 \
uv run --python 3.11 \
  --with torch --with torchvision --with timm --with iopath --with ftfy \
  --with opencv-python --with pillow --with regex --with huggingface_hub \
  --with typing_extensions --with pycocotools --with einops --with numba \
  --with hydra-core --with omegaconf --with psutil --with av --with safetensors \
  python scripts/python_sam3_macos/run_3frame_track.py
```

## Output

`/tmp/sam3_demo_frames/track_reference.safetensors` — per-frame
`out_obj_ids`, `out_probs`, `out_boxes_xywh`, `out_binary_masks`.

3-frame run: ~17 s on CPU. CUDA host would be much faster.
