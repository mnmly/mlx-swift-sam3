"""Run a 10-frame multiplex tracker via python/sam3 to capture parity reference."""
import sys, os
sys.path.insert(0, "/tmp/sam3_run")
import setup_stubs  # noqa: E402

import torch  # noqa: E402
import numpy as np  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

torch.manual_seed(0)

print("Building predictor...")
from sam3.model_builder import build_sam3_multiplex_video_predictor  # noqa: E402
predictor = build_sam3_multiplex_video_predictor(use_fa3=False, compile=False, warm_up=False)
predictor.model.float()
predictor.model.eval()

VIDEO = "/tmp/sam3_demo_frames_10"
PROMPT = "person"

print("start_session...")
resp = predictor.handle_request({"type": "start_session", "resource_path": VIDEO})
session_id = resp["session_id"]
print("add_prompt...")
predictor.handle_request({
    "type": "add_prompt", "session_id": session_id, "frame_index": 0, "text": PROMPT,
})

print("propagate_in_video...")
per_frame = {}
for response in predictor.handle_stream_request({
    "type": "propagate_in_video", "session_id": session_id,
}):
    fi = response["frame_index"]
    out = response["outputs"]
    n_obj = len(out["out_obj_ids"])
    print(f"  frame {fi}: {n_obj} objects")
    per_frame[fi] = out

print("Captured frames:", sorted(per_frame.keys()))

arrays = {}
for fi, out in per_frame.items():
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            arrays[f"frame_{fi}/{k}"] = v.detach().cpu().float().contiguous()
        elif isinstance(v, np.ndarray):
            arrays[f"frame_{fi}/{k}"] = torch.from_numpy(v).float().contiguous()
out_path = "/tmp/sam3_demo_frames_10/track_reference.safetensors"
save_file(arrays, out_path)
print(f"Saved {len(arrays)} arrays to {out_path}")
