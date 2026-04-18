#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "safetensors",
#     "torch",
#     "numpy",
#     "packaging",
# ]
# ///
"""Convert raw HF `facebook/sam3` (or sam3.1) safetensors to the MLX-friendly
layout expected by `Sam3Model.sanitize` / `Sam31Model.sanitize`.

What it does:
  * Transposes 4D Conv/ConvTranspose weights OIHW → OHWI (MLX convention).
  * Optionally casts to bfloat16 (or keeps original dtype).
  * Copies tokenizer + config files so the output dir is a drop-in for
    `--weights` / `--tokenizer` in the Swift CLIs.

Usage (uv handles deps via the PEP-723 header above):
    ./scripts/convert_pytorch_to_mlx.py \
        --src weights/sam3 \
        --dst weights/sam3-mlx \
        --dtype bf16

Or explicitly:
    uv run scripts/convert_pytorch_to_mlx.py --src ... --dst ... --dtype bf16
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file


DTYPE_MAP = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "keep": None,
}

# Files to copy verbatim (tokenizer + configs).
COPY_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "config.json",
    "processor_config.json",
    "preprocessor_config.json",
]


def iter_weights(src: Path):
    """Yield (key, tensor) from safetensors (single/sharded) or a .pt pickle."""
    idx = src / "model.safetensors.index.json"
    if idx.exists():
        mapping = json.loads(idx.read_text())["weight_map"]
        shards = sorted({src / s for s in mapping.values()})
        for shard in shards:
            with safe_open(shard, framework="pt") as f:
                for k in f.keys():
                    yield k, f.get_tensor(k)
        return

    single = src / "model.safetensors"
    if single.exists():
        with safe_open(single, framework="pt") as f:
            for k in f.keys():
                yield k, f.get_tensor(k)
        return

    pts = sorted(src.glob("*.pt")) + sorted(src.glob("*.pth"))
    if not pts:
        raise FileNotFoundError(f"no safetensors or .pt in {src}")
    for pt in pts:
        obj = torch.load(pt, map_location="cpu", weights_only=True)
        state = _extract_state_dict(obj)
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                yield k, v


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "model_state_dict", "weights"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        return obj
    raise TypeError(f"unsupported .pt payload: {type(obj)}")


CONVTRANSPOSE_PATTERNS = {
    "scale_layers",
    "upscale_conv",
    "resize_layers",
}


def is_convtranspose(key: str) -> bool:
    return any(pat in key for pat in CONVTRANSPOSE_PATTERNS)


def convert(t: torch.Tensor, key: str, dtype: torch.dtype | None) -> torch.Tensor:
    if t.ndim == 4 and key.endswith(".weight"):
        if is_convtranspose(key):
            # ConvTransposed2d: PyTorch (in, out, kH, kW) → MLX (out, kH, kW, in)
            t = t.permute(1, 2, 3, 0).contiguous()
        else:
            # Conv2d: PyTorch (out, in, kH, kW) → MLX (out, kH, kW, in)
            t = t.permute(0, 2, 3, 1).contiguous()
    if dtype is not None:
        t = t.to(dtype)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="raw HF snapshot dir")
    ap.add_argument("--dst", type=Path, required=True, help="output dir (MLX layout)")
    ap.add_argument("--dtype", choices=list(DTYPE_MAP), default="bf16")
    ap.add_argument(
        "--shard-size-mb",
        type=int,
        default=5000,
        help="max shard size in MB (0 = single file)",
    )
    args = ap.parse_args()

    dtype = DTYPE_MAP[args.dtype]
    args.dst.mkdir(parents=True, exist_ok=True)

    print(f"converting {args.src} → {args.dst} (dtype={args.dtype})")
    converted: dict[str, torch.Tensor] = {}
    transposed = 0
    for key, tensor in iter_weights(args.src):
        new = convert(tensor, key, dtype)
        if new.shape != tensor.shape:
            transposed += 1
        converted[key] = new
    print(f"  transposed {transposed} 4D weight tensors OIHW→OHWI")

    # Shard output.
    limit = args.shard_size_mb * 1024 * 1024
    if limit == 0 or _total_bytes(converted) <= limit:
        save_file(converted, str(args.dst / "model.safetensors"))
        print(f"  wrote model.safetensors ({_total_bytes(converted) / 1e6:.1f} MB)")
    else:
        _write_sharded(converted, args.dst, limit)

    # Copy tokenizer + config files.
    for name in COPY_FILES:
        src = args.src / name
        if src.exists():
            shutil.copy2(src, args.dst / name)
            print(f"  copied {name}")

    print("done.")


def _total_bytes(d: dict[str, torch.Tensor]) -> int:
    return sum(t.element_size() * t.numel() for t in d.values())


def _write_sharded(d: dict[str, torch.Tensor], dst: Path, limit: int):
    shards: list[dict[str, torch.Tensor]] = [{}]
    sizes = [0]
    for k, t in d.items():
        sz = t.element_size() * t.numel()
        if sizes[-1] + sz > limit and shards[-1]:
            shards.append({})
            sizes.append(0)
        shards[-1][k] = t
        sizes[-1] += sz

    n = len(shards)
    weight_map: dict[str, str] = {}
    for i, shard in enumerate(shards, start=1):
        name = f"model-{i:05d}-of-{n:05d}.safetensors"
        save_file(shard, str(dst / name))
        for k in shard:
            weight_map[k] = name
        print(f"  wrote {name} ({sizes[i - 1] / 1e6:.1f} MB)")

    total = sum(sizes)
    (dst / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2
        )
    )
    print(f"  wrote model.safetensors.index.json ({n} shards)")


if __name__ == "__main__":
    main()
