"""Export safetensors fixtures from `python/sam3` (PyTorch) for Swift parity tests.

Usage
-----
    # From this repo's root:
    python scripts/export_fixtures.py              # export all registered fixtures
    python scripts/export_fixtures.py position_encoding box_ops   # subset

Fixtures land in Tests/MLXSAM3Tests/Fixtures/<name>.safetensors and are
git-ignored. Each file stores every input/output tensor plus a small JSON
metadata payload under the key `__meta__` (serialized as a uint8 tensor).

Adding a new fixture
--------------------
1. Write a function `fixture_<name>()` below.
2. Call `save(name, tensors, meta=...)`.
3. Register in `FIXTURES` dict.
4. Add a matching test in Swift that loads `<name>.safetensors`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import torch
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parent.parent
SAM3_REPO = REPO_ROOT.parent.parent / "python" / "sam3"
FIXTURES_DIR = REPO_ROOT / "Tests" / "MLXSAM3Tests" / "Fixtures"
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

if not SAM3_REPO.exists():
    sys.exit(f"Expected sam3 repo at {SAM3_REPO} — adjust SAM3_REPO in this script.")
sys.path.insert(0, str(SAM3_REPO))


def _load_sam3_module(dotted: str):
    """Load a single `sam3.*` module by file path, bypassing `sam3/__init__.py`.

    The package's __init__ eagerly imports sam1_task_predictor -> edt -> triton,
    which isn't available on macOS. Leaf modules (position_encoding, box_ops, ...)
    are self-contained, so we can load them directly.
    """
    parts = dotted.split(".")
    path = SAM3_REPO.joinpath(*parts).with_suffix(".py")
    if not path.exists():
        raise FileNotFoundError(f"No module at {path}")
    spec = importlib.util.spec_from_file_location(dotted, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    return mod

torch.manual_seed(0)


def _pack_meta(meta: dict) -> torch.Tensor:
    raw = json.dumps(meta, sort_keys=True).encode("utf-8")
    return torch.tensor(list(raw), dtype=torch.uint8)


def save(name: str, tensors: Dict[str, torch.Tensor], meta: dict | None = None) -> None:
    out = {k: v.detach().contiguous().cpu().float() for k, v in tensors.items()}
    if meta is not None:
        out["__meta__"] = _pack_meta(meta)
    path = FIXTURES_DIR / f"{name}.safetensors"
    save_file(out, str(path))
    keys = [k for k in out if k != "__meta__"]
    print(f"  saved {path.name}  keys={keys}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def fixture_position_encoding():
    pe = _load_sam3_module("sam3.model.position_encoding")
    PositionEmbeddingSine = pe.PositionEmbeddingSine

    enc = PositionEmbeddingSine(num_pos_feats=256, normalize=True).eval()
    x = torch.randn(1, 1, 16, 16)  # (B, C, H, W) — C ignored, only shape
    with torch.no_grad():
        y = enc(x)
    save(
        "position_encoding_sine",
        {"input/x": x, "output/pos": y},
        meta={"num_pos_feats": 256, "normalize": True, "shape": list(x.shape)},
    )


def fixture_box_ops():
    box_ops = _load_sam3_module("sam3.model.box_ops")

    boxes_cxcywh = torch.rand(8, 4)
    boxes_cxcywh[:, 2:] = boxes_cxcywh[:, 2:] * 0.4 + 0.05  # keep w,h positive, in-bounds

    boxes1 = torch.rand(6, 4)
    boxes1[:, 2:] = boxes1[:, :2] + torch.rand(6, 2) * 0.3 + 0.01  # xyxy
    boxes2 = torch.rand(5, 4)
    boxes2[:, 2:] = boxes2[:, :2] + torch.rand(5, 2) * 0.3 + 0.01

    xyxy = box_ops.box_cxcywh_to_xyxy(boxes_cxcywh)
    area = box_ops.box_area(xyxy)
    iou, union = box_ops.box_iou(boxes1, boxes2)
    giou = box_ops.generalized_box_iou(boxes1, boxes2)

    save(
        "box_ops",
        {
            "input/cxcywh": boxes_cxcywh,
            "input/boxes1_xyxy": boxes1,
            "input/boxes2_xyxy": boxes2,
            "output/xyxy": xyxy,
            "output/area": area,
            "output/iou": iou,
            "output/union": union,
            "output/giou": giou,
        },
    )


def _concat_padded_sequences_ref(seq1, mask1, seq2, mask2):
    """Reference copy of sam3.model.geometry_encoders.concat_padded_sequences.

    Inlined to avoid importing sam3.model.geometry_encoders, which triggers
    sam3/__init__.py and its heavy dependency chain (triton, etc.).
    """
    seq1_length, batch_size, hidden_size = seq1.shape
    seq2_length = seq2.shape[0]

    actual_seq1_lengths = (~mask1).sum(dim=-1)
    actual_seq2_lengths = (~mask2).sum(dim=-1)
    final_lengths = actual_seq1_lengths + actual_seq2_lengths
    max_length = seq1_length + seq2_length

    concatenated_mask = (
        torch.arange(max_length, device=seq2.device)[None].repeat(batch_size, 1)
        >= final_lengths[:, None]
    )
    concatenated_sequence = torch.zeros(
        (max_length, batch_size, hidden_size), device=seq2.device, dtype=seq2.dtype
    )
    concatenated_sequence[:seq1_length, :, :] = seq1
    index = torch.arange(seq2_length, device=seq2.device)[:, None].repeat(1, batch_size)
    index = index + actual_seq1_lengths[None]
    concatenated_sequence = concatenated_sequence.scatter(
        0, index[:, :, None].expand(-1, -1, hidden_size), seq2
    )
    return concatenated_sequence, concatenated_mask


def fixture_geometry_helpers():
    """Leaf helpers from geometry_encoders.py: concat_padded_sequences + pos_enc.encode_*."""
    pe = _load_sam3_module("sam3.model.position_encoding")

    enc = pe.PositionEmbeddingSine(num_pos_feats=256, normalize=True).eval()

    # encode_xy: points in [0, 1]
    x = torch.rand(7)
    y = torch.rand(7)
    with torch.no_grad():
        pos_x, pos_y = enc._encode_xy(x, y)

    # encode_boxes: cx, cy, w, h in [0, 1]
    bx = torch.rand(5)
    by = torch.rand(5)
    bw = torch.rand(5) * 0.3 + 0.05
    bh = torch.rand(5) * 0.3 + 0.05
    with torch.no_grad():
        box_pos = enc.encode_boxes(bx, by, bw, bh)

    # concat_padded_sequences: seq-first, batch-second
    seq1 = torch.randn(4, 2, 8)
    mask1 = torch.tensor([[False, False, True, True], [False, False, False, True]])
    seq2 = torch.randn(3, 2, 8)
    mask2 = torch.tensor([[False, False, True], [False, True, True]])
    concat_seq, concat_mask = _concat_padded_sequences_ref(seq1, mask1, seq2, mask2)

    save(
        "geometry_helpers",
        {
            "input/x": x, "input/y": y,
            "output/pos_x": pos_x, "output/pos_y": pos_y,
            "input/box_cx": bx, "input/box_cy": by, "input/box_w": bw, "input/box_h": bh,
            "output/box_pos": box_pos,
            "input/seq1": seq1, "input/mask1": mask1.to(torch.uint8),
            "input/seq2": seq2, "input/mask2": mask2.to(torch.uint8),
            "output/concat_seq": concat_seq, "output/concat_mask": concat_mask.to(torch.uint8),
        },
        meta={"num_pos_feats": 256},
    )


FIXTURES: Dict[str, Callable[[], None]] = {
    "position_encoding": fixture_position_encoding,
    "box_ops": fixture_box_ops,
    "geometry_helpers": fixture_geometry_helpers,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", help="Fixture names (default: all)")
    parser.add_argument("--list", action="store_true", help="List available fixtures")
    args = parser.parse_args()

    if args.list:
        for name in FIXTURES:
            print(name)
        return

    names = args.names or list(FIXTURES)
    unknown = [n for n in names if n not in FIXTURES]
    if unknown:
        sys.exit(f"Unknown fixture(s): {unknown}. Use --list to see options.")

    for name in names:
        print(f"[{name}]")
        FIXTURES[name]()


if __name__ == "__main__":
    main()
