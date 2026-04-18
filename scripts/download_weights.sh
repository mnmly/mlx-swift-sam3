#!/usr/bin/env bash
# Download HF facebook/sam3 weights into ./weights/sam3.
# Requires: `huggingface-cli` (pip install huggingface_hub) and an HF token
# with access to the gated facebook/sam3 repo (`huggingface-cli login`).

set -euo pipefail

REPO="${1:-facebook/sam3}"
DEST="${2:-weights/$(basename "$REPO")}"

mkdir -p "$DEST"

echo "Downloading $REPO -> $DEST"
hf download "$REPO" --local-dir "$DEST"

echo "Done. Point ModelLoader at: $(cd "$DEST" && pwd)"
