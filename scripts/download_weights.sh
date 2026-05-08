#!/usr/bin/env bash
# Download pre-converted MLX SAM 3 / SAM 3.1 weights into a local directory.
#
# By default this pulls `mlx-community/sam3.1-bf16` — already converted to the
# Swift module layout (key prefixes detector_model.* / tracker_model.*, fused
# QKV split into q_proj/k_proj/v_proj, conv weights in NHWC). That's the format
# `Sam31Model` expects; you can point `--weights` at the resulting directory
# directly with no further conversion step.
#
# DO NOT use facebook/sam3.1 directly — its checkpoint uses Meta's flat
# detector.* / tracker.* layout with fused QKVs and PyTorch NCHW conv weights,
# which won't load with this Swift port (the `Sam31Model.sanitize` only handles
# minor remaps, not the full restructuring).
#
# Other variants:
#   mlx-community/sam3-bf16      — SAM 3 (image-only model)
#   mlx-community/sam3.1-bf16    — SAM 3.1 multiplex (image + video tracking)
#
# If you need to run the converter yourself (e.g. a custom checkpoint),
# use mlx-vlm's `mlx_vlm/models/sam3_1/convert_weights.py` — it handles the
# fused-QKV split and full key remap. This repo previously shipped a small
# `convert_pytorch_to_mlx.py` that only transposed conv weights; that script
# was incomplete and has been removed.

set -euo pipefail

REPO="${1:-mlx-community/sam3.1-bf16}"
DEST="${2:-weights/$(basename "$REPO")}"

mkdir -p "$DEST"

echo "Downloading $REPO -> $DEST"
hf download "$REPO" --local-dir "$DEST"

echo "Done. Point ModelLoader at: $(cd "$DEST" && pwd)"
