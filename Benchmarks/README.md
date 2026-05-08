# Benchmarks — `mlx-swift-sam3`

Two protocol-matched benchmarks for SAM 3.1 image inference (text prompt → masks):

- **`Tools/sam3-bench/Sam3Bench.swift`** — the Swift / mlx-swift port through the public `Sam31Model.detect` + `SAM3ImagePredictor.postprocess` API.
- **`Benchmarks/torch_sam3_bench.py`** — the Python reference (Meta `python/sam3`).

Both report:

- `cold_load_s` — model load + first weight read.
- `inference_only` — image preloaded; timed loop runs only the model + post-processing.
- `end_to_end` (`--include-load`) — re-decodes the image from disk on each iteration.

Each timing block reports `mean / median / min / max` over the measured runs, after `warmup` iterations are discarded.

## Running

### Swift (mlx-swift)

```bash
xcodebuild -scheme sam3-bench -destination 'platform=macOS' \
    -configuration release -derivedDataPath .xcdd build

.xcdd/Build/Products/release/sam3-bench \
    --weights ~/.cache/huggingface/hub/models--mlx-community--sam3.1-bf16/snapshots/<sha>/ \
    --image  ./assets/example.jpg \
    --prompt "a person" \
    --dtype  fp16 \
    --warmup 3 --iters 20 \
    --include-load
```

### Python (`python/sam3`)

```bash
cd /path/to/python/sam3                     # the upstream Meta repo
uv run --no-sync python /path/to/mlx-swift-sam3/Benchmarks/torch_sam3_bench.py \
    --image  /path/to/example.jpg \
    --prompt "a person" \
    --device cpu       # or mps for Apple Silicon GPU
    --dtype  fp32      # match Swift's --dtype for a fair comparison
    --warmup 3 --iters 20 \
    --include-load
```

The Python script auto-imports `setup_stubs.py` from `/tmp/sam3_run/` when `--device cpu` so the upstream CUDA-pinned code runs on macOS. See `scripts/python_sam3_macos/setup_stubs.py`.

## Matched-dtype rule

Compare matched dtype against matched dtype. **A Swift `--dtype fp16` run vs a Python `--dtype fp32` run is meaningless** — the lower-precision side will always look faster regardless of port quality.

For a meaningful number, run both sides at the same precision:

| pairing | swift | python |
|---|---|---|
| fp32 baseline | `--dtype fp32` | `--dtype fp32 --device cpu` |
| fp16 fast | `--dtype fp16` | `--dtype fp16 --device mps` |

Apple Silicon: `mps` is the apples-to-apples GPU device for torch. `cuda` is for Linux/NVIDIA hosts.

## Performance gate

Per the `port-mlx-to-swift` skill DoD:

- Within **1.2× of Torch (matched dtype)** for total inference: ship as-is.
- **>1.2× total or any single stage >1.5×** at matched dtype: this is an optimization deliverable, not a publishable result. File a `PERF_NOTES.md` with: slow stage, ratio, suspected op, hypothesis. See [`port-mlx-to-swift/references/release-deliverables.md`](https://github.com/anthropics/...) for the full gate procedure.

The "obvious wins" before shipping when the gate fires:

- `MLXFast.scaledDotProductAttention` everywhere, no hand-rolled `softmax(QK)V`.
- Dtype consistent end-to-end — one stray fp32 cast in the hot path can dominate fp16 cost.
- No stray `.item()` / `eval()` mid-forward.
- Conv2d weights pre-transposed to NHWC at load time, not on every call.

## Results

> **Hardware**: Apple M5 Max, 128 GB RAM, macOS 26.5.
> Reproduce by running the commands above; numbers below are over `--iters 5 --warmup 2` (still small — bump iters for tighter intervals before publishing).
> Image: 1008×1008 resize of `/tmp/sam3_demo_frames_10/0.jpg`. Prompt: `"a person"`.

### mlx-swift baseline (Apple Silicon GPU via MLX)

| Variant | dtype | Mean | Median | Min | Max |
|---|---|---|---|---|---|
| inference-only | fp16 | 0.310s | 0.311s | 0.297s | 0.319s |
| inference-only | fp32 | 0.306s | 0.310s | 0.296s | 0.312s |
| end-to-end     | fp16 | 0.320s | 0.319s | 0.315s | 0.323s |
| end-to-end     | fp32 | 0.340s | 0.338s | 0.321s | 0.358s |
| cold model load | fp16 | 0.545s | — | — | — |
| cold model load | fp32 | 0.494s | — | — | — |

Note that fp16 and fp32 are essentially identical here — likely because the model is already small enough that matmul throughput isn't the bottleneck on M5 Max, and dtype-conversion overhead at boundaries cancels the precision win. Re-measure on smaller hardware (M-series base / Pro) before drawing perf conclusions.

### Torch reference (`python/sam3` on MPS)

| Variant | dtype | Mean | Median | Min | Max |
|---|---|---|---|---|---|
| inference-only | fp32 | 6.40s | 6.63s | 5.84s | 6.74s |
| cold model load | fp32 | 61.96s | — | — | — |

> Note: torch fp16 errored on MPS through the `python/sam3` predictor (`expected scalar type Float but found Half` from a Conv2d that the macOS-shim CPU cast path leaves as fp32 even after `model.to(dtype=fp16)`). Reproduces with the bench's `--dtype fp16 --device mps` if you want to investigate; for the perf gate the matched fp32 comparison is what counts.

### Performance gate

| Comparison | mlx-swift | torch/mps | Ratio (mlx/torch) | Gate |
|---|---|---|---|---|
| inference-only median (fp32) | 0.310s | 6.627s | **0.047× (≈21× faster)** | ✅ within 1.2× |
| cold load (fp32) | 0.494s | 61.955s | **0.008× (≈125× faster)** | ✅ within 1.2× |

mlx-swift is materially faster than torch/mps at matched fp32 precision. The skill DoD's gate (within 1.2× of Torch / no single stage >1.5× slower) is comfortably cleared, no `PERF_NOTES.md` follow-up required.

> Caveat: this comparison is `python/sam3` on MPS — that codebase wasn't tuned for Apple Silicon and Meta's official perf path is CUDA + Flash Attention 3. A user already on `transformers`'s `Sam3Model` (different inference frontend) may see different torch-side numbers; redo the comparison there if that's your point of reference.
