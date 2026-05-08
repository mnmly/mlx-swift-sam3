"""Run a 3-frame multiplex tracker via python/sam3 to capture parity reference.

Outputs per-frame:
  - obj_ids assigned by tracker
  - low_res_masks (B, 1, H_low, W_low) per object per frame
  - obj_scores per object per frame
  - intermediate state we can compare against (memory_bank entries, etc.)
"""
import sys, os, json
sys.path.insert(0, "/tmp/sam3_run")
import setup_stubs  # noqa: E402

import torch  # noqa: E402
import numpy as np  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

torch.manual_seed(0)

print("Building predictor...")
from sam3.model_builder import build_sam3_multiplex_video_predictor  # noqa: E402
predictor = build_sam3_multiplex_video_predictor(use_fa3=False, compile=False, warm_up=False)
print("Built. Forcing model to fp32; bf16 casts neutralized by setup_stubs.")
predictor.model.float()
predictor.model.eval()
print("model dtype:", next(predictor.model.parameters()).dtype)

VIDEO = "/tmp/sam3_demo_frames"
PROMPT = "person"

print("start_session...")
resp = predictor.handle_request({"type": "start_session", "resource_path": VIDEO})
print("start_session ->", {k: v for k, v in resp.items() if k != "session_id"})
session_id = resp["session_id"]

print("add_prompt...")
resp = predictor.handle_request({
    "type": "add_prompt",
    "session_id": session_id,
    "frame_index": 0,
    "text": PROMPT,
})
print("add_prompt ->", {k: v for k, v in resp.items() if k != "session_id"})

print("propagate_in_video...")
per_frame = {}
for response in predictor.handle_stream_request({
    "type": "propagate_in_video",
    "session_id": session_id,
}):
    fi = response["frame_index"]
    out = response["outputs"]
    print(f"  frame {fi}: keys={list(out.keys())}")
    per_frame[fi] = out

print("Captured frames:", sorted(per_frame.keys()))
print("First frame output sample:")
for k, v in per_frame[min(per_frame)].items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: tensor shape={tuple(v.shape)} dtype={v.dtype}")
    elif isinstance(v, np.ndarray):
        print(f"  {k}: ndarray shape={v.shape} dtype={v.dtype}")
    else:
        print(f"  {k}: {type(v).__name__} {v if not hasattr(v, '__len__') else len(v)}")

# Save as a flat tensor dict
arrays = {}
for fi, out in per_frame.items():
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            arrays[f"frame_{fi}/{k}"] = v.detach().cpu().float().contiguous()
        elif isinstance(v, np.ndarray):
            arrays[f"frame_{fi}/{k}"] = torch.from_numpy(v).float().contiguous()
out_path = "/tmp/sam3_demo_frames/track_reference.safetensors"
save_file(arrays, out_path)
print(f"Saved reference to {out_path} with {len(arrays)} arrays")
