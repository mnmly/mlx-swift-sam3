"""Export a SAM 3.1 multiplex video tracking trace for Swift parity testing.

Run on Google Colab (GPU recommended). Produces
`sam3_1_video_trace.safetensors` with:

  * input/pixel_values_fN         — (3, H, W, C uint8) raw RGB frame N
  * input/model_pixel_values_fN   — (1, C, 1008, 1008) pre-processed tensor fed to vision encoder (frame N)
  * input/input_ids               — tokenized "people" ids (1, T) int32
  * input/attention_mask          — (1, T) int32
  * detect/boxes_f0               — (K, 4) xyxy pixel coords
  * detect/scores_f0              — (K,)
  * detect/masks_f0               — (K, Ho, Wo) binary uint8 at original resolution
  * hook/memory_encoder.in.features_f0     — (1, C, h, w) input to memory_encoder at frame 0
  * hook/memory_encoder.in.masks_f0        — (1, 2M, Hm, Wm) conditioning+mask tensor
  * hook/memory_encoder.out_f0             — (1, C, h, w) memory features
  * hook/interactive_decoder.out_masks_f1  — (1, 1, N, Hd, Wd) logits
  * hook/interactive_decoder.out_iou_f1    — (1, 1, N)
  * hook/sam_mask_decoder.out_masks_f1     — (1, M, N, Hd, Wd) logits
  * hook/sam_mask_decoder.out_iou_f1       — (1, M, N)
  * hook/memory_attention.in_src_f1        — (1, S, C) current features (pre memory_attention)
  * hook/memory_attention.out_f1           — (1, S, C) features after memory_attention

Meta (JSON under __meta__):
  * prompt, video_path, frame_indices, resolution, object_ids

Colab setup (paste into a cell first):
    !pip install -q 'git+https://github.com/facebookresearch/sam3.git'
    !pip install -q safetensors opencv-python

Then upload this script and run:
    !python export_sam31_video_trace.py --video /content/bedroom_trimmed.mp4 --out /content/sam3_1_video_trace.safetensors
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors.torch import save_file


def _pack_meta(meta: dict) -> torch.Tensor:
    raw = json.dumps(meta, sort_keys=True, default=str).encode("utf-8")
    return torch.tensor(list(raw), dtype=torch.uint8)


def _to_cpu_f32(t: torch.Tensor) -> torch.Tensor:
    return t.detach().contiguous().cpu().float()


class HookRecorder:
    """Attach forward hooks, collect (inputs, outputs) keyed by name + call index."""

    def __init__(self):
        self.records: dict[str, list[tuple]] = {}
        self._handles = []

    def attach(self, module: torch.nn.Module, name: str):
        def hook(mod, inputs, output):
            self.records.setdefault(name, []).append((inputs, output))

        self._handles.append(module.register_forward_hook(hook))

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def load_video_frames(path: str, num_frames: int = 3) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < num_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) < num_frames:
        raise RuntimeError(f"Only got {len(frames)} frames from {path}, need {num_frames}")
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input MP4")
    parser.add_argument("--out", required=True, help="Output safetensors path")
    parser.add_argument("--prompt", default="people")
    parser.add_argument("--num-frames", type=int, default=3)
    args = parser.parse_args()

    # Lazy imports (after sam3 is pip-installed in the Colab env)
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    import torch
    import os

    # Set environment variable to help with fragmentation
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    # Clear existing cache
    torch.cuda.empty_cache()

    video_path = args.video
    num_frames = args.num_frames
    frames_rgb = load_video_frames(video_path, num_frames)
    orig_h, orig_w = frames_rgb[0].shape[:2]

    predictor = build_sam3_multiplex_video_predictor(use_fa3=False)
    # Pull the underlying torch model for hooks. The exact attribute name varies
    # across sam3 versions; try a few common paths.
    core = None
    for attr in ("model", "predictor", "sam_model"):
        if hasattr(predictor, attr):
            core = getattr(predictor, attr)
            break
    if core is None:
        raise RuntimeError(f"Could not locate torch model under predictor: {dir(predictor)}")

    tracker = None
    for attr in ("tracker_model", "tracker", "multiplex_tracker"):
        if hasattr(core, attr):
            tracker = getattr(core, attr)
            break
    if tracker is None:
        raise RuntimeError(f"Could not find tracker_model under core: {[a for a in dir(core) if not a.startswith('_')]}")

    rec = HookRecorder()

    # Use maskmem_backbone instead of the _run_memory_encoder method
    rec.attach(tracker.maskmem_backbone, "memory_encoder")

    # These are already Modules, so they should work
    rec.attach(tracker.sam_mask_decoder, "sam_mask_decoder")
    rec.attach(tracker.interactive_sam_mask_decoder, "interactive_sam_mask_decoder")

    # memory_attention is likely a sub-module of the transformer encoder
    # If it's not directly accessible, hook the encoder itself
    rec.attach(tracker.transformer.encoder, "transformer_encoder")

    if hasattr(core, "vision_encoder"):
        rec.attach(core.vision_encoder, "vision_encoder")
    elif hasattr(core, "image_encoder"):
        rec.attach(core.image_encoder, "image_encoder")

    # --- Drive the predictor ---
    response = predictor.handle_request(dict(type="start_session", resource_path=video_path))
    session_id = response["session_id"]

    r0 = predictor.handle_request(dict(
        type="add_prompt", session_id=session_id, frame_index=0, text=args.prompt
    ))
    out0 = r0["outputs"]

    # Propagate a couple frames so memory_encoder + memory_attention fire.
    outputs_per_frame = {0: out0}
    for response in predictor.handle_stream_request(dict(
        type="propagate_in_video", session_id=session_id
    )):
        fi = response["frame_index"]
        outputs_per_frame[fi] = response["outputs"]
        if fi >= num_frames - 1:
            break

    rec.detach()

    # --- Collect tensors ---
    save_dict: dict[str, torch.Tensor] = {}

    for i, frame in enumerate(frames_rgb):
        save_dict[f"input/pixel_values_f{i}"] = torch.from_numpy(frame.copy())

    # Per-frame predicted masks (shape depends on predictor format, typically
    # a dict {obj_id: mask_hxw_bool_tensor}).
    object_ids = []
    for fi in range(num_frames):
        out = outputs_per_frame.get(fi, {})
        masks = out.get("masks", out)  # some versions return {obj_id: mask}
        if isinstance(masks, dict):
            for oid, m in masks.items():
                if isinstance(m, torch.Tensor):
                    save_dict[f"predict/mask_f{fi}_obj{oid}"] = _to_cpu_f32(m.to(torch.float32))
                if oid not in object_ids:
                    object_ids.append(int(oid))
        elif isinstance(masks, torch.Tensor):
            save_dict[f"predict/masks_f{fi}"] = _to_cpu_f32(masks.to(torch.float32))

    # Hook outputs — save first call per module (frame 0) and second (frame 1)
    def save_hook(name: str, call_idx: int, tag: str):
        calls = rec.records.get(name, [])
        if call_idx >= len(calls):
            return
        inputs, output = calls[call_idx]
        # inputs is a tuple of tensors; flatten
        for j, t in enumerate(inputs):
            if isinstance(t, torch.Tensor):
                save_dict[f"hook/{name}.in{j}_{tag}"] = _to_cpu_f32(t)
        if isinstance(output, torch.Tensor):
            save_dict[f"hook/{name}.out_{tag}"] = _to_cpu_f32(output)
        elif isinstance(output, (tuple, list)):
            for j, t in enumerate(output):
                if isinstance(t, torch.Tensor):
                    save_dict[f"hook/{name}.out{j}_{tag}"] = _to_cpu_f32(t)

    # --- Save Hooked Tensors ---

    # Frame 0: The "Detection/Initial Prompt" phase
    # We encode the first frame into memory
    save_hook("memory_encoder", 0, "f0")
    # Interactive decoder creates the first mask
    save_hook("interactive_sam_mask_decoder", 0, "f0")
    # Vision encoder processes the first image
    save_hook("vision_encoder", 0, "f0")

    # Frame 1: The "Propagation" phase
    # Vision encoder processes the second image
    save_hook("vision_encoder", 1, "f1")
    # Transformer encoder performs memory attention (Reading f0 memory while processing f1)
    save_hook("transformer_encoder", 0, "f1") 
    # Standard mask decoder predicts the tracking mask
    save_hook("sam_mask_decoder", 0, "f1")
    # New memory is encoded for the next step
    save_hook("memory_encoder", 1, "f1")

    meta = {
        "prompt": args.prompt,
        "video_path": os.path.abspath(video_path),
        "num_frames_captured": num_frames,
        "original_size_hw": [orig_h, orig_w],
        "object_ids": object_ids,
        "hooked_modules": list(rec.records.keys()),
        "hook_call_counts": {k: len(v) for k, v in rec.records.items()},
    }
    save_dict["__meta__"] = _pack_meta(meta)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(save_dict, str(out_path))
    print(f"wrote {out_path} with {len(save_dict)-1} tensors")
    print(f"meta: {json.dumps(meta, indent=2, default=str)}")


if __name__ == "__main__":
    main()
