# Parity Notes — `mlx-swift-sam3`

Tracks open parity gaps and the reasons they exist. Updated as bisection work uncovers new evidence.

## Status summary

| Surface | Reference | Test | Status |
|---|---|---|---|
| Image E2E (logits/boxes/masks) | mlx-vlm SAM 3.1 | `testSam31ImageE2EFixture` | ✅ passing within tolerance |
| `trackStep` single-frame | mlx-vlm SAM 3.1 | `testSam31TrackStepFixture` | ✅ passing within tolerance |
| Memory attention (intermediate) | python/sam3 | `testMemoryAttentionRealInputParity` | ⚠️ ~4-8% mean rel err — see "Why" below |
| Multi-frame video tracking | python/sam3 (10 frames) | `testSam31VideoTrackerSlimVsPySam3_10frame` | ⚠️ mean IoU ~0.30, gate floor 0.28 |
| Video tracker invariants | (synthetic) | `testSam31VideoTrackerSmoke` | ✅ passing |

## The video parity gap (why mean IoU ≈ 0.30, not ≈ 0.9)

Two flavors of "the same" SAM 3.1 model exist with the same checkpoint weights but different runtime configurations:

| field | `python/sam3` `model_builder.py` (Meta upstream) | HF config (`facebook/sam3.1`, `mlx-community/sam3.1-bf16`) |
|---|---|---|
| `memory_attention_num_attention_heads` | **8** | **1** |
| `memory_attention_feed_forward_hidden_act` | **gelu** | **relu** |

Heads=1 vs heads=8 changes per-head split + softmax scaling; ReLU vs GELU is a different non-linearity. With the same weights, these produce numerically different intermediate values and different final masks — they are not interchangeable.

`mlx-swift-sam3` uses HF weights and HF config (heads=1, relu). `python/sam3`'s reference run uses Meta config (heads=8, gelu). So the captured python/sam3 video-tracker output is not the right reference for Swift; comparing against it produces a structural ~4-8% intermediate divergence and an end-to-end mask mismatch that is **not a Swift bug**.

### What was tried during bisection

- Switching Swift to heads=8 cuts `testMemoryAttentionRealInputParity` mean rel err from ~8% to ~4%, **but catastrophically breaks** `testSam31TrackStepFixture` (obj_scores 600% off, iou_scores 63% off) — proving HF weights are configured for heads=1.
- Switching to gelu produces a small bisect improvement (matching python's choice) but doesn't help end-to-end IoU and is not what HF weights expect.
- Both reverted; documented the heads/activation choice with comments at the config sites.
- 5 architectural bugs were found and fixed during the bisection effort (conditioning channel, `extra_per_object_embeddings`, `memory_image` stream, K positional encoding order, `high_res` ordering).

### Why we can't simply re-bake the reference

The right reference for Swift is HF transformers' `Sam3VideoModel` (heads=1, relu, HF weights). HF transformers can import `Sam3VideoModel` but cannot directly load `facebook/sam3.1`'s `sam3.1_multiplex.pt` checkpoint — it expects `model.safetensors` / `pytorch_model.bin` and the on-disk file uses a different key naming scheme. A weight conversion script is needed; that's a separate effort not yet undertaken.

`mlx-vlm` is also a valid HF-config reference and is *already* used for the single-frame `testSam31TrackStepFixture`, but `mlx-vlm` does not currently expose multi-frame video tracking.

### What this means for the release

- **Image inference: full parity.** `testSam31ImageE2EFixture` and `testSam31TrackStepFixture` pass within tolerance against mlx-vlm.
- **Video tracking: shape-correct, identity-preserving.** `testSam31VideoTrackerSmoke` confirms tracker invariants. The pipeline produces useful per-frame tracked masks (see `Examples/Sam31VideoTrackFull`).
- **Video tracking: numerical parity to python/sam3 is NOT a release gate.** The 10-frame test is kept as a diagnostic with a low floor (0.28). Closing this requires either an HF-transformers reference (blocked on weight format conversion) or a higher-level outcome metric (e.g. tracking accuracy on a labeled clip rather than mask-IoU vs python/sam3).

## Memory attention bisect — what the per-layer numbers say

`testMemoryAttentionPerLayerParity` (added during bisection): feeding python's exact captured layer-N input to Swift's layer N and comparing against python's captured layer-N output.

| layer | meanErr | expMaxAbs | rel err |
|---|---|---|---|
| 0 | 0.051 | 3.96 | ~1.3% |
| 1 | 0.110 | 7.83 | ~1.4% |
| 2 | 0.136 | 19.5 | ~0.7% |
| 3 | 0.241 | 25.7 | ~0.9% |

Each layer alone is within ~1% rel err. The ~4% accumulated end-to-end err (post-heads=1 vs heads=8 baseline) is consistent with what the heads/activation config divergence predicts. Concluding that no Swift-side bug remains in memory_attention itself; the residual is the architectural-config gap.

## Open follow-ups (not blocking the release)

1. **HF-transformers reference capture.** Convert `facebook/sam3.1`'s `.pt` checkpoint to a transformers-loadable format (safetensors + key remap), then run `transformers.Sam3VideoModel.propagate_in_video_iterator` on the 10-frame clip and re-baseline `testSam31VideoTrackerSlimVsPySam3_10frame` against that. Expected outcome: Swift hits ≥0.9 mean IoU vs HF transformers reference.
2. **Tracking accuracy on a labeled clip.** Replace the python/sam3 IoU comparison with a labeled-mask metric (e.g. DAVIS subset). Measures real tracking quality independent of which SAM 3.1 flavor produced the labels.
3. **`Sam31VideoPipeline.reset()` actually clears tracker state.** Currently a stub; surface a `restartSession()` on `Sam31VideoTracker`.
