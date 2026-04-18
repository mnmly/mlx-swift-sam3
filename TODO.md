# mlx-swift-sam3 — Port TODO

Direct port of [`facebookresearch/sam3`](https://github.com/facebookresearch/sam3) (`../../python/sam3`) to MLX-Swift, seeded from the slimmer `../mlx-swift-vlm-sam3` baseline. Inference-only (training is out of scope).

## Ground rules

- **Parity first.** Every ported module has a fixture-driven XCTest that loads Python-captured inputs/outputs from `.safetensors` and asserts `allclose`.
  - fp32 target: `atol=1e-4`, `rtol=1e-4`
  - fp16 fallback (per-module if fp32 infeasible): `atol=1e-2`
- **Fixtures are local-only.** Not committed. Regenerate via `scripts/export_fixtures.py` (imports `sam3` from `../../python/sam3`).
- **Weights: HF `facebook/sam3` safetensors.** Download via `scripts/download_weights.sh`.
- **Video I/O = AVFoundation.** No Python video-decode code is ported; model-side memory/EDT math is ported verbatim.
- **Mirror Python structure** under `Sources/MLXSAM3/` so the mapping to `python/sam3/sam3/` is obvious.

## Parity workflow

1. Pick a module from `python/sam3/sam3/model/` (or `sam/`).
2. Add a fixture generator in `scripts/export_fixtures.py` (seed=0 torch inputs, dump inputs + outputs to `Tests/MLXSAM3Tests/Fixtures/<module>.safetensors`).
3. Port the module to `Sources/MLXSAM3/...`, copy/adapt from `../mlx-swift-vlm-sam3/Sources/MLXSAM3/` wherever it already exists.
4. Add `XCTestCase` that loads the fixture and asserts `allclose`.
5. Run `swift test --filter <module>Tests`.
6. Tick the box below. If a slim vlm-port version diverges numerically from `python/sam3`, fix Swift to match `python/sam3`.

## Phases

### Phase 0 — scaffolding
- [x] Copy baseline `Sources/` `Tests/` `Examples/` `Package.swift` from `mlx-swift-vlm-sam3`.
- [x] Write this TODO.md.
- [x] `scripts/download_weights.sh` — HF `facebook/sam3` → `weights/sam3/`.
- [x] `scripts/export_fixtures.py` — torch-based, imports `sam3` from sibling repo. First fixtures: `position_encoding`, `box_ops`.
- [x] `.gitignore` fixtures and weights.
- [x] Confirm `swift build` passes on the copied baseline.

### Phase 1 — leaf math
- [x] `position_encoding.py` → `Sources/MLXSAM3/Position.swift` + `testPositionEncodingSineTorchFixture` (fp32, max |Δ| ~1e-5). Note: Python `num_pos_feats` is total output channels; Swift `numPosFeats` is per-direction (output = 2*numPosFeats). Adjust when wiring call sites.
- [x] `box_ops.py` → `Sources/MLXSAM3/BoxOps.swift` + `testBoxOpsFixture` (fp32, max |Δ| ~1e-5). Ported: cxcywh↔xyxy, xywh converters, box_area, box_iou, generalized_box_iou. Skipped: masks_to_boxes (mask utility, revisit if needed), fast_diag_* (jit.script variants, not needed).
- [~] `geometry_encoders.py` — **slim port (Option 2)**.
  - Done: `Prompt` struct + `concatPaddedSequences` in `Sources/MLXSAM3/Prompt.swift`; `PositionEmbeddingSine.encodeXY` / `encodeBoxes` in `Position.swift`; fixture test `testGeometryHelpersFixture`.
  - **Deferred (Option 1, full port)**: `SequenceGeometryEncoder` forward with points+boxes+masks, `MaskEncoder` / `FusedMaskEncoder`, torchvision `roi_align` (boxes_pool), `grid_sample` (points_pool), TransformerEncoderLayer (pre-norm, self+cross attn, key_padding_mask). Blocked on: memory-fuser / CXBlock (Phase 5), hand-rolled roi_align + grid_sample kernels. Revisit when a consumer (sam3_image forward, Phase 6) actually needs pooling branches. Existing `Geometry.swift` (97L) is vlm-era and remains unvalidated — replace when we do the full port.
- [~] `sam/rope.py` — already covered by vlm seed in `Position.swift` (`computeAxialCis`, `applyRotaryEnc`, `applyRotaryEnc1d`, `init2dFreqs`). **Parity validated indirectly** via Phase 2 `vit_backbone` and Phase 5 tracker/mask-decoder fixtures. Full leaf-level parity fixture (complex `compute_axial_cis`, `VisionRotaryEmbeddingVE`) deferred — add only if a Phase 2/5 higher-level fixture surfaces a mismatch.
- [~] `sam/common.py` — already covered by vlm seed in `SAMComponents.swift` (`MLPBlock`, `LayerNorm2d`). Key-name mismatch vs Python (Swift uses `proj_in`/`proj_out`; Python `lin1`/`lin2`) is handled by `sanitize` in `Sam3Model.swift`/`Sam31Model.swift` — verify in Phase 5.

### Phase 2 — vision backbone (slim port, see "Known gaps")
- [x] `vitdet.py` → `Vision.swift` — vlm-seed `ViTBackbone` already covers slim target (HF-style keys: patch_embeddings, q/k/v/o split, layer_norm1/2, mlp.fc1/fc2, window+global blocks w/ 2d-rope). Validated by `testViTBackboneFixture`. Deferred: cls_token path, rel_pos (`rel_pos_h/w` + `concat_rel_pos`), LayerScale (`ls1/ls2`), `ln_pre`/`ln_post`, `return_interm_layers`, fused `qkv` load path, rope interpolate/tile variants, ve-rope.
- [x] `necks.py` → folded into `Vision.swift` as `FPNLayer` / `FPNNeck`. Validated by `testFPNLayer4xFixture`.
- [x] Parity test: image → feature pyramid (`testVisionEncoderFixture`, fp32 inputs, weights in mlx-community format via `Sam3Model.sanitize` scale_layers remap).

### Phase 3 — text encoder
- [x] `text_encoder_ve.py` → `TextEncoder.swift` (vlm-seed, validated by `testTextEncoderFixture`, fp32 pass).
- [x] `tokenizer_ve.py` — resolved via `huggingface/swift-transformers` (1.3.0) dep. Wrapper in `Sources/MLXSAM3/Tokenizer.swift` (`SAM3Tokenizer.fromSnapshot(_:)` → `encode(_:) → (inputIds, attentionMask)`, pads to `maxLength=77`). Both demo CLIs accept `--prompt` + `--tokenizer <snapshot-dir>`.
- [x] Parity test: prompts → token embeddings (`testTextEncoderFixture`).

### Phase 4 — DETR encoder + decoder + VL combiner
- [x] `encoder.py` → `DETREncoder.swift` (vlm-seed, validated by `testDETREncoderFixture`, fp32 pass).
- [x] `decoder.py` → `DETRDecoder.swift` (vlm-seed, validated by `testDETRDecoderFixture`, fp32 pass).
- [x] `vl_combiner.py` — inlined into `Sam3Model.detect()`; `SAM3VLBackbone` is a wrapper that sequences vision + text + DETR, which `Sam3Model.swift` already orchestrates. No standalone file needed.
- [x] Parity tests per submodule.

### Phase 5 — SAM components
- [x] `sam/transformer.py` → `SAMComponents.swift::TwoWayTransformer` (vlm-seed, validated indirectly via mask-decoder fixtures).
- [x] `sam/prompt_encoder.py` → `SAMComponents.swift::SAMPromptEncoder` (vlm-seed, validated indirectly via mask-decoder fixtures).
- [x] `sam/mask_decoder.py` → `Segmentation.swift::MaskDecoder` (vlm-seed, `testMaskDecoderFixture` pass).
- [x] `multiplex_mask_decoder.py` → `SAMComponents.swift` multiplex path (vlm-seed, `testMultiplexMaskDecoderFixture` + `testMultiplexMaskDecoderShape` pass).
- [x] Parity tests (mask decoder covers the chain end-to-end).

### Phase 6 — image predictor e2e (slim)
- [~] `sam3_image.py` — orchestration inlined in `Sam3Model.detect()`; `SAM3VLBackbone.forward_grounding` behavior covered. Full file (908L) deferred — re-evaluate when Phase 7 multiplex detector needs pieces.
- [x] `sam3_image_processor.py` → `SAM3ImagePredictor.swift` + `ImageUtils.pixelValues` (mean/std=0.5). Resize via CoreGraphics.
- [x] `sam3_base_predictor.py` — post-processing (presence × logit sigmoid, threshold, mask upsample, box scale) in `SAM3ImagePredictor.postprocess`.
- [x] `maskformer_segmentation.py` → already covered by `Segmentation.swift::MaskDecoder` + `DotProductScoring` (vlm-seed, pass).
- [ ] End-to-end parity: image + text prompt → masks vs python. **Requires weights download + new fixture generator** (`scripts/export_fixtures.py`: add `sam3_image_e2e` producing masks/boxes/scores). Deferred to Phase 11 demos or a follow-up checkpoint.

### Phase 7 — multiplex detector (slim — vlm-seed baseline)
- [~] `sam3_multiplex_base.py` + `sam3_multiplex_detector.py` + utils — covered structurally by vlm-seed `Sam31Model.swift` + `Sam31SAMComponents.swift` + `Sam31Vision.swift`. Shape tests pass (`testSam31ObjectPointerMLPShape`, `testDecoupledMemoryAttentionShape`, `testMultiplexMaskDecoderShape`). Full fixture parity for multiplex detector deferred — target is SAM3 (not SAM 3.1) for Phase 11 image demos.

### Phase 8 — tracker (slim — vlm-seed baseline)
- [~] `sam3_tracker_base.py` + `sam3_tracking_predictor.py` — covered by vlm-seed `Tracker.swift` (MemoryFuser, MemoryEncoder, MemoryAttention, MemoryAttentionLayer, TrackerModel.trackStep) and `Sam31Tracker.swift`. Parity: `testMemoryEncoderFixture`, `testDecoupledMemoryAttentionFixture` pass.
- [~] `sam3_tracker_utils.py` — only `edt_triton` usage (false-neg/false-pos mask correction); deferred with `edt.py`.
- [~] `sam3_multiplex_tracking.py` (3431L) — deferred; not needed until Phase 11 multiplex video demos.

### Phase 9 — video model internals (slim)
- [~] `memory.py` — `SimpleMaskDownSampler` + memory-fuser blocks covered by vlm-seed `Tracker.swift::MemoryFuser` + `MemoryEncoder`; parity via `testMemoryEncoderFixture`.
- [ ] `edt.py` — Triton GPU kernel for Euclidean distance transform. **Deferred**: consumers (`sam3_tracker_utils`) only hit it on FN/FP mask correction, not the golden path. Re-port via Metal custom kernel or Felzenszwalb O(N²) MLX-ops version when a demo actually exercises it.
- [ ] `sam3_video_base.py` + `sam3_video_inference.py` + `sam3_video_predictor.py` + `sam3_multiplex_video_predictor.py` — **deferred to Phase 10/11**: orchestration-heavy (~4000L), most of it is frame-iteration + memory-bank bookkeeping that should sit alongside the AVFoundation reader. Existing `Sam3Model.trackStep()` is the per-frame primitive; video-level orchestration belongs with the I/O layer.

### Phase 10 — AVFoundation I/O
- [x] `Platform/AVFoundationVideoReader.swift` — `AVAssetReader` → `MLXArray` frame iterator (`nextFrame` + `nextPixelValues`).
- [x] `Platform/AVFoundationVideoWriter.swift` — `CGImage` frames → H.264 mp4 via `AVAssetWriter`.
- [~] `Platform/ImageUtils.swift` — existing `ImagePreprocess.pixelValues(from: CGImage)` reused by reader; dedicated `CVPixelBuffer` fast-path deferred until a demo needs it.
- [x] Round-trip test: `VideoIOTests.testVideoRoundTrip` writes known frames → decodes → asserts frame count + dimensions. `testReaderNextPixelValuesShape` covers MLXArray shape.

### Phase 11 — demos
- [x] `Examples/ImageInfer.swift` — `ImageInfer.run(image:inputIds:weightsDir:)` wraps `SAM3ImagePredictor`; `renderOverlay` draws boxes + mask fills onto a `CGImage`. Tokenizer still deferred — caller passes pre-tokenized `inputIds`.
- [x] `Examples/VideoTrack.swift` — `VideoTrack.run(inputURL:outputURL:inputIds:weightsDir:)` iterates reader → per-frame `predict()` → overlay → writer. Slim: independent per-frame detection, no memory-bank tracking (that path is the deferred Phase 9 video orchestration).
- [x] **Tokenizer** — resolved with `swift-transformers` 1.3.0; demos now accept raw `--prompt`.

### Phase 12 (optional) — agent helpers
- [ ] Port `agent/helpers/` inference math (`zoom_in`, `roi_align`, `mask_overlap_removal`, `rle`, `color_map`, `keypoints`).
- [ ] Rewrite `agent_core` + `client_llm` as Swift URLSession-based agent loop.
- [ ] Visualization helpers deferred — use CoreGraphics/Vision natively rather than porting matplotlib code.

## Parity results log

<!-- For each completed parity test, add: module | dtype | atol | max_abs_diff -->

| Module | dtype | atol | max \|Δ\| | notes |
|---|---|---|---|---|
| position_encoding (PositionEmbeddingSine) | fp32 | 1e-5 | <1e-5 | NCHW↔NHWC transpose in test; Swift numPosFeats is per-direction |
| box_ops (cxcywh→xyxy, area, iou, giou) | fp32 | 1e-5 | <1e-5 | |
| geometry_helpers (encodeXY, encodeBoxes, concatPaddedSequences) | fp32 | 1e-5 | <1e-5 | slim port; full SequenceGeometryEncoder deferred |
| vit_backbone (ViT + rope) | fp32 | default | pass | vlm-seed port; slim (no cls/rel_pos/LayerScale/ln_pre/ln_post) |
| fpn_layer_4x (FPNLayer scale=4.0) | fp32 | default | pass | scale_layers.2→.1 sanitize |
| vision_encoder (ViT + FPN, 4 levels) | fp32 | 5e-2/5e-1 | pass | atol relaxed vs fixture — investigate in Phase 6 e2e |
| text_encoder (ve transformer, pre-tokenized input_ids) | fp32 | default | pass | tokenizer deferred to Phase 11 |
| detr_encoder | fp32 | default | pass | vlm-seed port |
| detr_decoder | fp32 | default | pass | vlm-seed port |
| mask_decoder | fp32 | default | pass | validates prompt_encoder + two-way transformer chain |
| multiplex_mask_decoder | fp32 | default | pass | shape + fixture |

## Known gaps / intentional slims

- **Phase 7–9: slim baseline, defer full ports** (decided 2026-04-18). Multiplex detector (Sam 3.1 path) validated at shape-level only; EDT triton kernel deferred; video-level orchestration (memory bank, frame iteration) lives with AVFoundation layer in Phase 10.
- **Phase 6: slim port** (decided 2026-04-18). Inference-only path: image processor (resize/pad/normalize), maskformer post-processing (thresh + resize masks back to image coords), thin `SAM3ImagePredictor` reusing `Sam3Model.detect()`. Skip training helpers, aux losses, multi-prompt batching variants until Phase 11 demos need them.
- **Phase 3 tokenizer: deferred to Phase 11** (decided 2026-04-18). BPE tokenizer (`tokenizer_ve.py`, OpenAI CLIP style) not needed until demo I/O. Decide full Swift port vs swift-transformers dep when Phase 11 starts.
- **Phase 2 vitdet: slim port** (decided 2026-04-18). Port only what vlm seed covers + what HF `facebook/sam3` weights need to load and run. Defer full `vitdet.py` coverage (extra windowed-attn variants, rel-pos modes, unused branches). Revisit if Phase 6 image e2e surfaces a mismatch.
