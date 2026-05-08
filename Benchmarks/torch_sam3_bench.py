"""Protocol-matched Python reference benchmark for SAM 3.1 image inference.

Mirrors `Tools/sam3-bench/Sam3Bench.swift`: same flags, same warmup/iters, same
"inference-only vs end-to-end" split, same dtype handling.

Run from the python/sam3 venv:

    cd /Users/mnmly/Development-local/GitHub/python/sam3
    uv run --no-sync python /path/to/Benchmarks/torch_sam3_bench.py \\
        --image example.jpg --prompt "a person" \\
        --iters 20 --warmup 3 --dtype fp32

Notes:
- Uses python/sam3's predictor (`Sam3Predictor`) which is the upstream Meta reference.
- HF transformers' `Sam3Model` would be the closer analogue to Swift, but its API differs
  more from a CLI-friendly call. Both are valid Python references; this script picks
  python/sam3 because it's already wired for CPU via setup_stubs.py.
- For Apple Silicon perf comparison, you generally want device=mps + dtype=fp16.
  This script doesn't force a device — pass --device mps if you want torch on the GPU.
"""
import argparse
import json
import sys
import time
from statistics import mean, median


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--dtype", default="fp32", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--include-load", action="store_true",
                   help="Re-decode image from disk on each iter (end-to-end).")
    p.add_argument("--json", action="store_true", help="Emit JSON instead of human text.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Always install the macOS shims (they stub out triton + redirect cuda→cpu).
    # On MPS the stubs are still needed because sam3 imports `triton` unconditionally.
    sys.path.insert(0, "/tmp/sam3_run")
    try:
        import setup_stubs  # noqa: F401
    except ImportError:
        print("[warn] setup_stubs not found at /tmp/sam3_run; running without shims",
              file=sys.stderr)

    import torch
    from PIL import Image
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    torch_dtype = DTYPES[args.dtype]

    def sync():
        if args.device == "mps": torch.mps.synchronize()
        elif args.device == "cuda": torch.cuda.synchronize()

    # Cold model load (timed). Image-only path matching Swift's Sam31Model.detect.
    t0 = time.perf_counter()
    model = build_sam3_image_model(device=args.device)
    if args.dtype != "fp32":
        model = model.to(dtype=torch_dtype)
    model.eval()
    processor = Sam3Processor(model, confidence_threshold=args.threshold)
    sync()
    load_s = time.perf_counter() - t0

    # Pre-decode image once for inference-only path
    image_pil = Image.open(args.image).convert("RGB")

    def run_once(reload_image: bool):
        img = Image.open(args.image).convert("RGB") if reload_image else image_pil
        with torch.no_grad():
            inference_state = processor.set_image(img)
            inference_state = processor.set_text_prompt(state=inference_state, prompt=args.prompt)
        sync()
        return inference_state

    # Warmup
    for _ in range(args.warmup):
        run_once(reload_image=False)

    # Inference-only
    inf = []
    for _ in range(args.iters):
        t = time.perf_counter()
        run_once(reload_image=False)
        inf.append(time.perf_counter() - t)

    # End-to-end
    e2e = []
    if args.include_load:
        for _ in range(args.iters):
            t = time.perf_counter()
            run_once(reload_image=True)
            e2e.append(time.perf_counter() - t)

    def stats(samples):
        if not samples: return dict(mean_s=0.0, median_s=0.0, min_s=0.0, max_s=0.0)
        return dict(mean_s=mean(samples), median_s=median(samples),
                    min_s=min(samples), max_s=max(samples))

    payload = {
        "backend": f"torch/{args.device}",
        "dtype": args.dtype,
        "iters": args.iters,
        "warmup": args.warmup,
        "load_s": load_s,
        "inference_only": stats(inf),
        "end_to_end": stats(e2e),
    }

    if args.json:
        print(json.dumps(payload))
    else:
        print(f"backend       : {payload['backend']}")
        print(f"dtype         : {args.dtype}")
        print(f"warmup/iters  : {args.warmup}/{args.iters}")
        print(f"cold_load_s   : {load_s:.3f}")
        s = payload["inference_only"]
        print(f"inference_only: mean={s['mean_s']:.3f}s median={s['median_s']:.3f}s "
              f"min={s['min_s']:.3f}s max={s['max_s']:.3f}s")
        if args.include_load:
            s = payload["end_to_end"]
            print(f"end_to_end    : mean={s['mean_s']:.3f}s median={s['median_s']:.3f}s "
                  f"min={s['min_s']:.3f}s max={s['max_s']:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
