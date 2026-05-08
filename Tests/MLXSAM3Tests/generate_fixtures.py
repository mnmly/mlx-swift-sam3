"""Generate safetensors fixtures for SAM 3.1 Swift unit tests.

Run from repo root:
    python Tests/MLXSAM3Tests/generate_fixtures.py

Outputs one safetensors file per component under Tests/MLXSAM3Tests/Fixtures/.
Each file contains the input tensor(s) and expected output tensor(s).
"""

import sys
import os
from pathlib import Path

# Resolve paths
REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent  # up to GitHub/
PYTHON_VLM = REPO_ROOT / "python" / "mlx-vlm"
MODEL_PATH = Path(os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--sam3-bf16/snapshots/dfe573c3171dbcfda8399c650d9135afa7e94592"
))
FIXTURES_DIR = Path(__file__).parent / "Fixtures"
FIXTURES_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(PYTHON_VLM))

import mlx.core as mx
import numpy as np

mx.random.seed(42)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def save(name: str, arrays: dict):
    path = FIXTURES_DIR / f"{name}.safetensors"
    mx.save_safetensors(str(path), {k: v.astype(mx.float32) for k, v in arrays.items()})
    print(f"  saved {path.name}  keys={list(arrays.keys())}")


# ---------------------------------------------------------------------------
# 1. PositionEmbeddingSine
# ---------------------------------------------------------------------------

def fixture_position_embedding_sine():
    from mlx_vlm.models.sam3.position import PositionEmbeddingSine

    enc = PositionEmbeddingSine(num_pos_feats=128)

    x = mx.random.normal([1, 8, 8, 256])
    out = enc(x)
    mx.eval(out)

    save("position_embedding_sine", {"input": x, "output": out})


# ---------------------------------------------------------------------------
# 2. FPNLayer (with real weights from SAM 3 checkpoint)
# ---------------------------------------------------------------------------

def fixture_fpn_layer():
    """FPNLayer 4x (fpn_layers.0) with real weights, input (1,72,72,1024)."""
    from mlx_vlm.models.sam3.vision import FPNLayer

    # Load checkpoint weights and extract the 4x FPN layer
    weights = mx.load(str(MODEL_PATH / "model.safetensors"))
    prefix = "detector_model.vision_encoder.neck.fpn_layers.0."
    layer_weights = {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}

    # The layer: in_channels=1024, out_channels=256, scale_factor=4.0
    layer = FPNLayer(in_channels=1024, out_channels=256, scale_factor=4.0)
    layer.load_weights(list(layer_weights.items()))
    mx.eval(layer.parameters())

    x = mx.random.normal([1, 72, 72, 1024])
    out = layer(x)
    mx.eval(out)

    save("fpn_layer_4x", {"input": x, "output": out})


# ---------------------------------------------------------------------------
# 3. ViTBackbone (with real weights)
# ---------------------------------------------------------------------------

def fixture_vit_backbone():
    """ViTBackbone with real weights, input (1, 336, 336, 3) → (1, 24, 24, 1024).

    336 = 24 * 14 (patch_size), so exactly one window of 24x24 — exercises
    both windowed and global attention without padding.

    Also saves intermediate outputs for debugging Swift divergence.
    """
    import math
    from mlx_vlm.models.sam3.vision import ViTBackbone
    from mlx_vlm.models.sam3.config import ViTConfig
    from mlx_vlm.models.sam3.position import compute_axial_cis

    weights = mx.load(str(MODEL_PATH / "model.safetensors"))
    prefix = "detector_model.vision_encoder.backbone."
    layer_weights = {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}

    cfg = ViTConfig()
    backbone = ViTBackbone(cfg)
    backbone.load_weights(list(layer_weights.items()))
    mx.eval(backbone.parameters())

    x = mx.random.normal([1, 336, 336, 3])

    # --- Intermediate checkpoints ---
    B = x.shape[0]
    H = x.shape[1] // cfg.patch_size   # 24
    W = x.shape[2] // cfg.patch_size   # 24

    after_patch = backbone.embeddings(x)            # (1, 576, 1024)
    pos = backbone._tile_pos_embed(backbone.embeddings.position_embeddings, H, W)
    after_pos = after_patch + pos                   # (1, 576, 1024)
    after_reshape = after_pos.reshape(B, H, W, -1)  # (1, 24, 24, 1024)
    after_ln = backbone.layer_norm(after_reshape)   # (1, 24, 24, 1024)

    head_dim = cfg.hidden_size // cfg.num_attention_heads
    win_cos, win_sin = compute_axial_cis(head_dim, cfg.window_size, cfg.window_size, cfg.rope_theta)
    glob_cos, glob_sin = compute_axial_cis(head_dim, H, W, cfg.rope_theta)

    # --- Layer 0 internals (windowed attention) ---
    lyr0 = backbone.layers[0]
    attn0 = lyr0.attention

    l0_after_ln1 = lyr0.layer_norm1(after_ln)
    # window partition (no-op for 24x24 with ws=24)
    l0_windowed, pad_hw = __import__('mlx_vlm.models.sam3.vision', fromlist=['_window_partition'])._window_partition(l0_after_ln1, cfg.window_size)
    # QKV
    B0, N0, C0 = l0_windowed.reshape(l0_windowed.shape[0], -1, l0_windowed.shape[-1]).shape
    l0_flat = l0_windowed.reshape(B0, N0, C0)
    l0_q = attn0.q_proj(l0_flat).reshape(B0, N0, attn0.num_heads, attn0.head_dim).transpose(0, 2, 1, 3)
    l0_k = attn0.k_proj(l0_flat).reshape(B0, N0, attn0.num_heads, attn0.head_dim).transpose(0, 2, 1, 3)
    l0_v = attn0.v_proj(l0_flat).reshape(B0, N0, attn0.num_heads, attn0.head_dim).transpose(0, 2, 1, 3)
    # RoPE
    from mlx_vlm.models.sam3.position import apply_rotary_enc
    l0_q_rope, l0_k_rope = apply_rotary_enc(l0_q, l0_k, win_cos, win_sin)
    # SDPA
    import mlx.core as _mx
    l0_sdpa = _mx.fast.scaled_dot_product_attention(l0_q_rope, l0_k_rope, l0_v, scale=attn0.scale)
    l0_sdpa_T = l0_sdpa.transpose(0, 2, 1, 3).reshape(B0, N0, C0)
    l0_oproj = attn0.o_proj(l0_sdpa_T).reshape(l0_windowed.shape)
    # unpartition + residual
    from mlx_vlm.models.sam3.vision import _window_unpartition
    H0, W0 = after_ln.shape[1], after_ln.shape[2]
    l0_unpart = _window_unpartition(l0_oproj, cfg.window_size, pad_hw, (H0, W0))
    l0_after_attn_residual = after_ln + l0_unpart
    after_layer0 = lyr0(after_ln, win_cos, win_sin)

    after_layer7 = after_layer0
    for i in range(1, 8):
        lyr = backbone.layers[i]
        cos, sin = (glob_cos, glob_sin) if lyr.is_global else (win_cos, win_sin)
        after_layer7 = lyr(after_layer7, cos, sin)

    mx.eval(after_patch, after_pos, after_ln, after_layer0, after_layer7,
            l0_after_ln1, l0_q, l0_k, l0_v, l0_q_rope, l0_k_rope,
            l0_sdpa, l0_oproj, l0_after_attn_residual,
            win_cos, win_sin)

    out = backbone(x)
    mx.eval(out)

    save("vit_backbone", {
        "input": x,
        "output": out,
        "after_patch_embed": after_patch,
        "after_pos_embed": after_pos,
        "after_layer_norm": after_ln,
        "l0_after_ln1": l0_after_ln1,
        "l0_q": l0_q,
        "l0_k": l0_k,
        "l0_v": l0_v,
        "l0_q_rope": l0_q_rope,
        "l0_k_rope": l0_k_rope,
        "l0_sdpa": l0_sdpa,
        "l0_oproj": l0_oproj,
        "l0_after_attn_residual": l0_after_attn_residual,
        "after_layer0": after_layer0,
        "after_layer7": after_layer7,
        "win_cos": win_cos,
        "win_sin": win_sin,
    })

# ---------------------------------------------------------------------------
# 4. TextEncoder
# ---------------------------------------------------------------------------

def fixture_text_encoder():
    from mlx_vlm.models.sam3.text_encoder import TextEncoder
    from mlx_vlm.models.sam3.config import TextEncoderConfig

    cfg = TextEncoderConfig()
    encoder = TextEncoder(cfg)

    weights = mx.load(str(MODEL_PATH / "model.safetensors"))
    
    # Corrected prefix: text_encoder is nested inside detector_model
    prefix = "detector_model.text_encoder."
    layer_weights = {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}
    
    encoder.load_weights(list(layer_weights.items()))
    mx.eval(encoder.parameters())

    # Example: Batch of 1, Sequence Length of 16
    input_ids = mx.random.randint(0, 30000, [1, 16])

    # Intermediate dumps to pinpoint Swift divergence
    tm = encoder.text_model
    after_emb = tm.embeddings(input_ids)
    import mlx.nn as _nn
    causal = _nn.MultiHeadAttention.create_additive_causal_mask(input_ids.shape[1]).astype(after_emb.dtype)
    after_layer0 = tm.encoder.layers[0](after_emb, mask=causal)
    x = after_emb
    for lyr in tm.encoder.layers:
        x = lyr(x, mask=causal)
    after_all = x
    after_final_ln = tm.final_layer_norm(after_all)

    out = encoder(input_ids)
    if isinstance(out, tuple):
        out = out[0]
    elif hasattr(out, "last_hidden_state"):
        out = out.last_hidden_state

    mx.eval(out, after_emb, after_layer0, after_all, after_final_ln)

    save("text_encoder", {
        "input_ids": input_ids,
        "output": out,
        "after_emb": after_emb,
        "after_layer0": after_layer0,
        "after_all_layers": after_all,
        "after_final_ln": after_final_ln,
    })


# ---------------------------------------------------------------------------
# 5. VisionEncoder (Full Backbone + FPN Neck)
# ---------------------------------------------------------------------------

def fixture_vision_encoder():
    from mlx_vlm.models.sam3.vision import VisionEncoder
    from mlx_vlm.models.sam3.config import VisionEncoderConfig

    cfg = VisionEncoderConfig()
    encoder = VisionEncoder(cfg)

    weights = mx.load(str(MODEL_PATH / "model.safetensors"))
    prefix = "detector_model.vision_encoder."
    layer_weights = {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}
    encoder.load_weights(list(layer_weights.items()))
    mx.eval(encoder.parameters())

    x = mx.random.normal([1, 336, 336, 3])
    out = encoder(x)
    
    # VisionEncoder usually returns a tuple of feature maps from the FPN
    if isinstance(out, tuple) or isinstance(out, list):
        mx.eval(*out)
        arrays = {"input": x}
        for i, o in enumerate(out):
            arrays[f"output_{i}"] = o
        save("vision_encoder", arrays)
    else:
        mx.eval(out)
        save("vision_encoder", {"input": x, "output": out})


# ---------------------------------------------------------------------------
# 6. DETREncoder
# ---------------------------------------------------------------------------

def fixture_detr_encoder():
    from mlx_vlm.models.sam3.encoder import DETREncoder
    from mlx_vlm.models.sam3.config import DETREncoderConfig

    cfg = DETREncoderConfig()
    encoder = DETREncoder(cfg)

    weights = mx.load(str(MODEL_PATH / "model.safetensors"))
    prefix = "detector_model.detr_encoder."
    layer_weights = {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}
    encoder.load_weights(list(layer_weights.items()))
    mx.eval(encoder.parameters())

    # Approximating flattened spatial features (e.g., 64x64 = 4096)
    src = mx.random.normal([1, 4096, 256]) 
    pos = mx.random.normal([1, 4096, 256])
    
    text_hidden = mx.random.normal([1, 16, 256])
    text_pad_mask = mx.zeros([1, 16], dtype=mx.bool_)
    
    # Pass the text tensors directly instead of wrapping them in a dictionary
    out = encoder(src, pos, text_hidden, text_pad_mask)
    
    mx.eval(out)
    save("detr_encoder", {
        "src": src, 
        "pos": pos, 
        "text_hidden": text_hidden,
        "text_pad_mask": text_pad_mask,
        "output": out
    })


# ---------------------------------------------------------------------------
# 7. DETRDecoder
# ---------------------------------------------------------------------------

def fixture_detr_decoder():
    from mlx_vlm.models.sam3.decoder import DETRDecoder
    from mlx_vlm.models.sam3.config import DETRDecoderConfig

    cfg = DETRDecoderConfig()
    decoder = DETRDecoder(cfg)

    weights = mx.load(str(MODEL_PATH / "model.safetensors"))
    prefix = "detector_model.detr_decoder."
    layer_weights = {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}
    decoder.load_weights(list(layer_weights.items()))
    mx.eval(decoder.parameters())

    query_embed = mx.random.normal([1, 100, 256])  # Text inputs / inputs_embeds
    memory = mx.random.normal([1, 4096, 256])      # Vision features
    pos = mx.random.normal([1, 4096, 256])         # Vision pos encoding

    # Pass in the correct positional order:
    # 1: vision_features 
    # 2: inputs_embeds
    # 3: vision_pos_encoding
    out = decoder(memory, query_embed, pos)
    
    # out is a Tuple: (intermediate_hs, intermediate_boxes, presence_logits)
    mx.eval(out[0], out[1], out[2])
    
    save("detr_decoder", {
        "memory": memory, 
        "query_embed": query_embed, 
        "pos": pos, 
        "out_hs": out[0],
        "out_boxes": out[1],
        "out_presence": out[2]
    })

# ---------------------------------------------------------------------------
# 8. MaskDecoder (Detector)
# ---------------------------------------------------------------------------

def fixture_mask_decoder():
    from mlx_vlm.models.sam3.segmentation import MaskDecoder
    from mlx_vlm.models.sam3.config import DetectorMaskDecoderConfig

    cfg = DetectorMaskDecoderConfig()
    decoder = MaskDecoder(cfg)

    weights = mx.load(str(MODEL_PATH / "model.safetensors"))
    prefix = "detector_model.mask_decoder."
    layer_weights = {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}
    decoder.load_weights(list(layer_weights.items()))
    mx.eval(decoder.parameters())

    # Create dummy inputs matching the exact signature
    # obj_queries: (B, Q, D)
    obj_queries = mx.random.normal([1, 100, 256])
    
    # backbone_features: list of (B, H_i, W_i, D) from LOW to HIGH resolution
    # Typically [72x72, 144x144, 288x288] but we'll use smaller ones for the fixture
    # The last one is the finest resolution.
    backbone_features = [
        mx.random.normal([1, 18, 18, 256]),   # Coarsest
        mx.random.normal([1, 36, 36, 256]),
        mx.random.normal([1, 72, 72, 256])    # Finest
    ]
    
    # encoder_hidden_states: (B, HW, D)
    encoder_hidden_states = mx.random.normal([1, 72*72, 256])
    
    # prompt_features: (B, P, D)
    prompt_features = mx.random.normal([1, 16, 256])
    
    # prompt_mask: (B, P)
    prompt_mask = mx.ones([1, 16], dtype=mx.bool_)

    out = decoder(
        obj_queries=obj_queries,
        backbone_features=backbone_features,
        encoder_hidden_states=encoder_hidden_states,
        prompt_features=prompt_features,
        prompt_mask=prompt_mask
    )
    
    mx.eval(out["pred_masks"], out["semantic_seg"])
    
    # Save the dictionary of arrays to the safetensors file
    save("mask_decoder", {
        "obj_queries": obj_queries,
        "backbone_feat_0": backbone_features[0],
        "backbone_feat_1": backbone_features[1],
        "backbone_feat_2": backbone_features[2],
        "encoder_hidden_states": encoder_hidden_states,
        "prompt_features": prompt_features,
        "prompt_mask": prompt_mask,
        "pred_masks": out["pred_masks"],
        "semantic_seg": out["semantic_seg"]
    })


# ---------------------------------------------------------------------------
# 9. MemoryEncoder
# ---------------------------------------------------------------------------

def fixture_memory_encoder():
    from mlx_vlm.models.sam3.tracker import MemoryEncoder
    from mlx_vlm.models.sam3.config import TrackerConfig

    cfg = TrackerConfig()
    encoder = MemoryEncoder(cfg)

    weights = mx.load(str(MODEL_PATH / "model.safetensors"))
    prefix = "tracker_model.memory_encoder."
    layer_weights = {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}
    encoder.load_weights(list(layer_weights.items()))
    mx.eval(encoder.parameters())

    # MLX Convs use NHWC layout by default.
    # Features: (B, H, W, D) -> e.g., (1, 72, 72, 256)
    features = mx.random.normal([1, 72, 72, 256])
    
    # Masks: (B, H_mask, W_mask, 1) -> scaled up 16x for the downsampler
    mask = mx.random.normal([1, 1152, 1152, 1])

    # Pass in correct order: features, masks
    out = encoder(features, mask)
    mx.eval(out)
    
    save("memory_encoder", {
        "features": features,
        "mask": mask, 
        "output": out
    })


# ---------------------------------------------------------------------------
# 10. DecoupledMemoryAttention
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 10. DecoupledMemoryAttention (SAM 3.1)
# ---------------------------------------------------------------------------

def fixture_decoupled_memory_attention():
    from mlx_vlm.models.sam3_1.tracker import DecoupledMemoryAttention
    from mlx_vlm.models.sam3_1.config import TrackerConfig
    from mlx.utils import tree_flatten

    cfg = TrackerConfig()

    # Small custom config — random weights (not from checkpoint)
    cfg.memory_attention_hidden_size = 64
    cfg.memory_attention_num_layers = 1
    cfg.memory_attention_num_attention_heads = 4
    cfg.memory_attention_feed_forward_hidden_size = 128
    cfg.memory_attention_rope_feat_sizes = [8, 8]
    cfg.memory_encoder_output_channels = 64

    attn = DecoupledMemoryAttention(cfg)
    mx.eval(attn.parameters())

    src = mx.random.normal([1, 64, 64])
    memory = mx.random.normal([1, 7, 64])

    out = attn(src, memory=memory)
    mx.eval(out)

    # Save model weights alongside I/O so Swift can load them from the fixture
    flat_weights = dict(tree_flatten(attn.parameters()))
    arrays = {"src": src, "memory": memory, "output": out}
    arrays.update(flat_weights)
    save("decoupled_memory_attention", arrays)

# ---------------------------------------------------------------------------
# 11. MultiplexMaskDecoder
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 11. MultiplexMaskDecoder
# ---------------------------------------------------------------------------

def fixture_multiplex_mask_decoder():
    from mlx_vlm.models.sam3_1.tracker import MultiplexMaskDecoder
    from mlx_vlm.models.sam3_1.config import TrackerMaskDecoderConfig
    from mlx.utils import tree_flatten
    import mlx.core as mx

    cfg = TrackerMaskDecoderConfig()

    # Small custom config — random weights (not from checkpoint)
    cfg.hidden_size = 64
    cfg.multiplex_count = 2
    cfg.num_multimask_outputs = 3
    decoder = MultiplexMaskDecoder(cfg)

    mx.eval(decoder.parameters())

    image_embeddings = mx.random.normal([1, 16, 64])
    image_pe = mx.random.normal([1, 16, 64])
    # Use [1, 2, 64] instead of [1, 0, 64] to avoid safetensors zero-element crash
    sparse_prompt = mx.random.normal([1, 2, 64])
    dense_prompt = mx.random.normal([1, 16, 64])

    masks, iou, _, objScore = decoder(
        image_embeddings=image_embeddings,
        image_pe=image_pe,
        sparse_prompt_embeddings=sparse_prompt,
        dense_prompt_embeddings=dense_prompt,
        multimask_output=True
    )

    mx.eval(masks, iou, objScore)

    # Save model weights alongside I/O so Swift can load them from the fixture
    flat_weights = dict(tree_flatten(decoder.parameters()))
    arrays = {
        "image_embeddings": image_embeddings,
        "image_pe": image_pe,
        "sparse_prompt": sparse_prompt,
        "dense_prompt": dense_prompt,
        "masks": masks,
        "iou": iou,
        "objScore": objScore,
    }
    arrays.update(flat_weights)
    save("multiplex_mask_decoder", arrays)


# ---------------------------------------------------------------------------
# 12. SAM 3.1 image end-to-end (Phase 7 parity)
# ---------------------------------------------------------------------------

SAM31_MODEL_PATH = Path(os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--sam3.1-bf16/snapshots/a992e302ea9b0f03f41dfd93414a4fd0e818f65b"
))


def fixture_sam3_1_image_e2e():
    """End-to-end image parity vs mlx-vlm sam3_1.Model.detect.

    Generates synthetic pixel_values + tokenized prompt, runs the reference
    model, dumps inputs + outputs (no weights — Swift loads same checkpoint).

    Uses 336x336 input to keep fixture compact (vs production 1008).
    """
    import json
    from mlx_vlm.models.sam3_1.sam3_1 import Model
    from mlx_vlm.models.sam3_1.config import ModelConfig

    with open(SAM31_MODEL_PATH / "config.json") as f:
        cfg_dict = json.load(f)
    cfg = ModelConfig.from_dict(cfg_dict)

    model = Model(cfg)
    weights = mx.load(str(SAM31_MODEL_PATH / "model.safetensors"))
    # The mlx-community/sam3.1-bf16 snapshot is already in MLX OHWI layout —
    # skip Model.sanitize (which would transpose Conv weights again).
    # Cast bf16 → fp32 so the reference run is high-precision; Swift will also
    # load with .float32 and the test can use a strict tolerance.
    weights_fp32 = {k: v.astype(mx.float32) for k, v in weights.items()}
    model.load_weights(list(weights_fp32.items()))
    mx.eval(model.parameters())

    # Synthetic deterministic inputs (fp32 for strict-parity reference).
    H = W = 336
    pixel_values = mx.random.normal([1, H, W, 3]).astype(mx.float32)

    # CLIP-style tokenization: [BOS, "a", "person", EOS, PAD, ...] padded to T=8.
    # Token IDs picked from sam3.1 tokenizer.json (49406=BOS, 49407=EOS, 320="a", 1746="person").
    input_ids = mx.array([[49406, 320, 1746, 49407, 0, 0, 0, 0]], dtype=mx.int32)
    attention_mask = mx.array([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=mx.int32)

    out = model.detect(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    mx.eval(*out.values())

    arrays = {
        "input/pixel_values": pixel_values,
        "input/input_ids": input_ids,
        "input/attention_mask": attention_mask,
        "output/pred_logits": out["pred_logits"],
        "output/pred_boxes": out["pred_boxes"],
        "output/pred_masks": out["pred_masks"],
        "output/presence_logits": out["presence_logits"],
        # Intermediate boundary tensors for diagnosing where Swift diverges.
        # encoder_hidden_states is the DETR encoder output (post text fusion).
        # intermediate_hidden_states is the DETR decoder per-layer query hs
        # (last layer = hs[-1] feeds dot_product_scoring + mask_decoder).
        "output/encoder_hidden_states": out["encoder_hidden_states"],
        "output/intermediate_hidden_states": out["intermediate_hidden_states"],
    }
    save("sam3_1_image_e2e", arrays)


# ---------------------------------------------------------------------------
# 13. SAM 3.1 video tracker (Phase 11.5b parity — minimal 3-frame loop)
# ---------------------------------------------------------------------------

def fixture_sam3_1_video_track():
    """SAM 3.1 multiplex tracker components parity vs mlx-vlm sam3_1.Model.

    Exercises the *new* tracker code paths not covered by image-detect:
      A) memory_encoder(features, 32-ch mask) → memory   — multiplex mask path
      B) memory_attention(src, memory)                   — cross-attn over bank

    Skipped here:
      - Model.track_init: passes a 1-channel mask but MultiplexMaskDownSampler
        needs 32 (multiplex_count*2). Upstream bug — we feed a proper 32-ch mask.
      - Model.track_step: the prompt encoder's dense_emb is pinned to the
        production 72×72 grid; at smaller test resolutions the mask decoder's
        `image_embeddings + dense_prompt_embeddings` add fails. Mask decoder
        itself is already covered by testMultiplexMaskDecoderFixture.
    """
    import json
    from mlx_vlm.models.sam3_1.sam3_1 import Model
    from mlx_vlm.models.sam3_1.config import ModelConfig

    with open(SAM31_MODEL_PATH / "config.json") as f:
        cfg_dict = json.load(f)
    cfg = ModelConfig.from_dict(cfg_dict)

    model = Model(cfg)
    weights = mx.load(str(SAM31_MODEL_PATH / "model.safetensors"))
    weights_fp32 = {k: v.astype(mx.float32) for k, v in weights.items()}
    model.load_weights(list(weights_fp32.items()))
    mx.eval(model.parameters())

    # --- Inputs --------------------------------------------------------
    H = W = 336
    pixel_values = mx.random.normal([1, H, W, 3]).astype(mx.float32)

    # Run vision_encoder.backbone once to get realistic propagation features
    # (1x scale, ~24×24 at 336 input).
    backbone0 = model.detector_model.vision_encoder.backbone(pixel_values)
    prop0 = model._get_tracker_features(backbone0)
    features0 = prop0[2]  # (1, H_f, W_f, 256)
    Hf, Wf, D = features0.shape[1], features0.shape[2], features0.shape[3]

    # 32-channel mask at the feature spatial × down-sampler upscale ratio.
    # MultiplexMaskDownSampler does 4× progressive downsample, so we need
    # input mask = features × 4 in each spatial dim.
    mask_H = Hf * (2 ** 4)  # 4 stages of stride-2 downsample
    mask_W = Wf * (2 ** 4)
    mask_in = mx.random.normal([1, mask_H, mask_W, 32]).astype(mx.float32)

    # --- A) memory_encoder direct ------------------------------------
    memory_feats = model.tracker_model.memory_encoder(features0, mask_in)
    B = memory_feats.shape[0]
    memory_0 = memory_feats.reshape(B, -1, memory_feats.shape[-1])

    # --- B) memory_attention(src, memory) ----------------------------
    # Synthetic src (current frame's flattened image features) at feature
    # spatial size; memory comes from stage A output.
    src = mx.random.normal([1, Hf * Wf, D]).astype(mx.float32)

    # Step manually to capture per-layer outputs (for divergence localization).
    ma = model.tracker_model.memory_attention
    layer_outputs = []

    # --- Manual layer 0 with sub-stage dumps ---------------------------
    layer0 = ma.layers[0]
    residual = src
    src_normed = layer0.norm1(src)
    q = layer0.self_attn_q_proj(src_normed)
    k = layer0.self_attn_k_proj(src_normed)
    v = layer0.self_attn_v_proj(src_normed)
    # Dump pre-rope q/k/v and the rope freqs themselves for parity localization.
    sa_pre_q, sa_pre_k, sa_pre_v = q, k, v
    sa_freqs_cos = layer0.self_attention_rope._freqs_cos
    sa_freqs_sin = layer0.self_attention_rope._freqs_sin
    sa_pre = layer0.self_attention_rope(q, k, v)
    sa_proj = layer0.self_attn_out_proj(sa_pre)
    after_sa = residual + sa_proj

    residual = after_sa
    n2 = layer0.norm2(after_sa)
    q2 = layer0.cross_attn_q_proj(n2) + layer0.image_cross_attn_q_proj(n2)
    k2 = layer0.cross_attn_k_proj(memory_0) + layer0.image_cross_attn_k_proj(memory_0)
    v2 = layer0.cross_attn_v_proj(memory_0)
    ca_pre = layer0.cross_attention_rope(q2, k2, v2)
    ca_proj = layer0.cross_attn_out_proj(ca_pre)
    after_ca = residual + ca_proj

    after_ffn = after_ca + layer0.linear2(mx.maximum(layer0.linear1(layer0.norm3(after_ca)), 0))
    layer_outputs.append(after_ffn)

    cur = after_ffn
    for layer in ma.layers[1:]:
        cur = layer(cur, memory_0)
        layer_outputs.append(cur)
    attended = ma.layer_norm(cur)

    mx.eval(memory_feats, memory_0, attended, after_sa, after_ca, sa_pre, sa_pre_q, sa_pre_k, sa_pre_v, sa_freqs_cos, sa_freqs_sin, *layer_outputs)

    arrays = {
        # Inputs
        "input/features_0": features0,
        "input/mask_in": mask_in,
        "input/src": src,
        # Stage A outputs (memory_encoder)
        "output/memory_feats": memory_feats,    # (B, H_f, W_f, D) — pre-reshape
        "output/memory_0": memory_0,            # (B, H_f*W_f, D) — bank layout
        # Sub-stage outputs of layer 0 (for divergence localization)
        "output/l0_after_sa": after_sa,
        "output/l0_after_ca": after_ca,
        # Pre-RoPE q/k/v of layer-0 self-attn (input to SimpleRoPEAttention)
        "output/l0_sa_q": sa_pre_q,
        "output/l0_sa_k": sa_pre_k,
        "output/l0_sa_v": sa_pre_v,
        # Post-RoPE+SDPA output (input to self_attn_out_proj)
        "output/l0_sa_pre": sa_pre,
        # RoPE freqs themselves (init_2d_freqs result)
        "output/l0_sa_freqs_cos": sa_freqs_cos,
        "output/l0_sa_freqs_sin": sa_freqs_sin,
        # Stage B outputs (memory_attention) + per-layer
        "output/attended": attended,
    }
    for i, out_i in enumerate(layer_outputs):
        arrays[f"output/ma_layer_{i}"] = out_i
    save("sam3_1_video_track", arrays)


# ---------------------------------------------------------------------------
# 14. SAM 3.1 track_step end-to-end at production resolution (Phase 11.5b)
# ---------------------------------------------------------------------------

def fixture_sam3_1_track_step():
    """Full track_step parity at 1008×1008 (production resolution).

    At production resolution, the prompt encoder's dense_emb (72×72) matches
    the feature spatial size, so MultiplexMaskDecoder's
    `image_embeddings + dense_prompt_embeddings` works.

    Uses synthetic features/memory (avoids saving the 12 MB pixel_values; vision
    backbone is already covered by testViTBackboneFixture / testVisionEncoderFixture).
    """
    import json
    from mlx_vlm.models.sam3_1.sam3_1 import Model
    from mlx_vlm.models.sam3_1.config import ModelConfig

    with open(SAM31_MODEL_PATH / "config.json") as f:
        cfg_dict = json.load(f)
    cfg = ModelConfig.from_dict(cfg_dict)

    model = Model(cfg)
    weights = mx.load(str(SAM31_MODEL_PATH / "model.safetensors"))
    weights_fp32 = {k: v.astype(mx.float32) for k, v in weights.items()}
    model.load_weights(list(weights_fp32.items()))
    mx.eval(model.parameters())

    # Discover the actual prop_fpn shapes by running the backbone once at 1008
    # — kept in-process; not saved to fixture.
    pix = mx.random.normal([1, 1008, 1008, 3]).astype(mx.float32)
    backbone = model.detector_model.vision_encoder.backbone(pix)
    prop = model._get_tracker_features(backbone)
    feat_shapes = [tuple(p.shape) for p in prop]
    print(f"  prop_fpn shapes: {feat_shapes}")
    Hf, Wf, D = prop[2].shape[1], prop[2].shape[2], prop[2].shape[3]

    # Replace with synthetic tensors of identical shape (deterministic seed → reproducible).
    mx.random.seed(7)
    features = mx.random.normal(prop[2].shape).astype(mx.float32)
    high_res_0 = mx.random.normal(prop[0].shape).astype(mx.float32)
    high_res_1 = mx.random.normal(prop[1].shape).astype(mx.float32)
    memory_0 = mx.random.normal([1, Hf * Wf, D]).astype(mx.float32) * 0.1  # encoded-feature scale

    # Run track_step at production resolution with memory bank.
    out = model.tracker_model.track_step(
        current_features=features,
        memory_bank=[memory_0],
        high_res_features=[high_res_0, high_res_1],
    )
    mx.eval(out["pred_masks"], out["iou_scores"], out["obj_scores"], out["object_pointer"])

    arrays = {
        # Inputs (synthetic, fixed-seed)
        "input/features": features,
        "input/high_res_0": high_res_0,
        "input/high_res_1": high_res_1,
        "input/memory_0": memory_0,
        # Outputs
        "output/pred_masks": out["pred_masks"],
        "output/iou_scores": out["iou_scores"],
        "output/obj_scores": out["obj_scores"],
        "output/object_pointer": out["object_pointer"],
    }
    save("sam3_1_track_step", arrays)


# ---------------------------------------------------------------------------
# Updated Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sam3_1_image_e2e":
        print("Generating sam3_1_image_e2e only...")
        fixture_sam3_1_image_e2e()
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "sam3_1_video_track":
        print("Generating sam3_1_video_track only...")
        fixture_sam3_1_video_track()
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "sam3_1_track_step":
        print("Generating sam3_1_track_step only...")
        fixture_sam3_1_track_step()
        sys.exit(0)

    print("Generating fixtures...")

    print("1. PositionEmbeddingSine")
    fixture_position_embedding_sine()

    print("2. FPNLayer 4x")
    fixture_fpn_layer()

    print("3. ViTBackbone")
    fixture_vit_backbone()

    print("4. TextEncoder")
    fixture_text_encoder()

    print("5. VisionEncoder")
    fixture_vision_encoder()

    print("6. DETREncoder")
    fixture_detr_encoder()

    print("7. DETRDecoder")
    fixture_detr_decoder()

    print("8. MaskDecoder")
    fixture_mask_decoder()

    print("9. MemoryEncoder")
    fixture_memory_encoder()

    print("10. DecoupledMemoryAttention")
    fixture_decoupled_memory_attention()

    print("11. MultiplexMaskDecoder")
    fixture_multiplex_mask_decoder()

    print("Done.")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------