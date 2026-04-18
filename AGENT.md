# Agent handoff — `mlx-swift-sam3`

Read this + `TODO.md` + `CLAUDE.md` first. Everything you need to continue the port is here. Do **not** re-derive the plan from conversation history.

## Checkpoint rule — STOP every 3 phases

After completing **Phase 3, 6, 9, and 12**, do not begin the next phase. Post a checkpoint report to the user (format in `CLAUDE.md`) and wait for explicit "go" before continuing. This is non-negotiable — the user needs integration-seam reviews at those points.

## Mission (one paragraph)

Direct port of `../../python/sam3` (PyTorch, Meta's official SAM 3 repo) to MLX-Swift. This repo was seeded from `../mlx-swift-vlm-sam3`, which itself is a slim port of `../../python/mlx-vlm/mlx_vlm/models/{sam3,sam3_1}`. The vlm port is ~85% of the image path but misses video, memory/EDT, SAM1 legacy, and trims multiplex paths aggressively (see "Known slims" in `TODO.md`). Your job is to extend/rewrite the vlm-seeded code until it matches `python/sam3` numerically, module by module, via safetensors fixtures.

Inference only. Skip `python/sam3/sam3/train/`, `eval/`, `perflib/`. SAM1 (`sam1_task_predictor.py`) is explicitly skipped for v1. Agent (`python/sam3/sam3/agent/`) is Phase 12 and deliberately last.

## Ground rules (non-negotiable)

1. **Parity-driven.** You do **not** mark a module "done" without a passing fixture XCTest against `python/sam3`.
   - fp32 target: `atol=1e-4`, `rtol=1e-4`.
   - fp16 fallback per module if fp32 isn't feasible: `atol=1e-2`.
2. **Authority is `python/sam3`, not the vlm port.** If the vlm-port Swift file disagrees with `python/sam3`, fix Swift to match Python. Log the divergence in TODO.md under "Known slims / intentional deviations".
3. **Copy from `../mlx-swift-vlm-sam3/Sources/MLXSAM3/` whenever it already has the module**, then diff against `python/sam3` and fill gaps. Don't retype from scratch.
4. **Weights: HF `facebook/sam3` safetensors** via `scripts/download_weights.sh`. Loaded with `ModelLoader` (already in `Sources/MLXSAM3/ModelLoader.swift`).
5. **Fixtures are local-only.** Not committed. Regenerate with `python scripts/export_fixtures.py`. Path: `Tests/MLXSAM3Tests/Fixtures/*.safetensors` (gitignored).
6. **Video I/O = AVFoundation.** Don't port Python video-decode code. Port only the model-side math (memory bank, EDT, video predictor forward passes). I/O goes in `Sources/MLXSAM3/Platform/`.
7. **Mirror Python structure** under `Sources/MLXSAM3/` so one-to-one mapping stays obvious. Target layout is in `TODO.md` (search "Proposed package layout" in git history if unsure — but the directory layout in `TODO.md` phases is authoritative).
8. Follow `CLAUDE.md` rules: minimal diffs, no premature abstractions, no comments explaining the what, verify before claiming done.

## Repo orientation (what is where)

- `Sources/MLXSAM3/` — current Swift port. Flat layout from vlm seed; reorganize into `Model/`, `SAM/`, `Multiplex/`, `Tracker/`, `Video/`, `Platform/`, `IO/` as you port phases (see `TODO.md`). Mechanical renames are fine; don't block phase work on layout churn.
- `Tests/MLXSAM3Tests/` — contains the old vlm-era `generate_fixtures.py` (MLX-based, references `mlx-vlm`) and some `*.safetensors`. Treat these as legacy reference only. The new canonical exporter is `scripts/export_fixtures.py` (torch-based, references `python/sam3`). When you write new tests, they should load from the new exporter's output.
- `Examples/DetectFromNSImage.swift` — vlm-era demo, keep working but don't treat as spec.
- `scripts/download_weights.sh` — HF weight downloader (`huggingface-cli` required; repo is gated — user must be logged in).
- `scripts/export_fixtures.py` — torch-based fixture exporter. Add a `fixture_<name>()` function and register in `FIXTURES`. Seeded with `position_encoding` + `box_ops` as templates. Run: `python scripts/export_fixtures.py` (all) or `python scripts/export_fixtures.py <name> [<name>...]`.
- `TODO.md` — phase checklist + parity results table. **Update it every time you finish a module.** Tick boxes, log atol/max-diff in the results table, record deviations.
- `../../python/sam3/sam3/` — source of truth. Read the Python file before porting, not the vlm-port Swift file.
- `../mlx-swift-vlm-sam3/Sources/MLXSAM3/` — copy source for already-ported modules.
- `../../python/mlx-vlm/mlx_vlm/models/{sam3,sam3_1}/` — intermediate MLX Python port; useful when the torch↔MLX op mapping is non-obvious (sampling, layernorm variants, dtype casts).

## Workflow per module (copy this into your session)

1. **Read Python source**: `python/sam3/sam3/model/<file>.py` (or `sam/`). Note classes, configs, tensor shapes, dtype behavior.
2. **Check existing Swift**: `Sources/MLXSAM3/` for matching file; `../mlx-swift-vlm-sam3/Sources/MLXSAM3/` if not yet copied. Copy in if absent.
3. **Check MLX reference**: `python/mlx-vlm/mlx_vlm/models/sam3*/<file>.py` for op mapping hints.
4. **Add fixture**: write `fixture_<name>()` in `scripts/export_fixtures.py`. Use `torch.manual_seed(0)`, realistic shapes, `.eval()` the module, save all inputs + outputs. Load random weights if the module is a `nn.Module` — parity is about forward math, not pretrained behavior, unless the test specifically depends on weights (then load from HF checkpoint).
5. **Run exporter**: `python scripts/export_fixtures.py <name>`. Verify it saved to `Tests/MLXSAM3Tests/Fixtures/<name>.safetensors`.
6. **Port / fix Swift module**. Stay faithful to Python: same math, same operator order, same dtype promotion. Prefer `MLXNN` layers where they map 1:1.
7. **Write XCTest**: load the fixture, instantiate the Swift module with matching config + **weights loaded from the fixture** (export them too if the module is stateful), run forward, `XCTAssertAllClose` with `atol=1e-4`. If you need a helper, put it in `Tests/MLXSAM3Tests/Support/Fixture.swift`.
8. **Run**: `swift test --filter <ModuleName>Tests`. Fix mismatches — first check dtype promotion, tensor layout (NCHW vs NHWC — MLX convs are NHWC!), and RoPE/softmax scale factors; those are the usual culprits.
9. **Update `TODO.md`**: tick the box, append a row to the parity results table (`| module | dtype | atol | max_abs_diff | notes |`). If you hit an fp32 precision wall, document the fp16 fallback.
10. **Small commits.** One commit per module (port) + one per test, if convenient. Don't bundle unrelated work.

## Layout conventions (MLX-Swift gotchas)

- **NHWC vs NCHW**: MLX convs take NHWC; torch is NCHW. When porting, `x.permute(0,2,3,1)` → `x.transposed(0, 2, 3, 1)` in Swift, or reshape at the boundary. Many vlm-port files already handle this — trust but verify.
- **LayerNorm / GroupNorm**: MLX has both; watch `eps` (default differs across impls).
- **RoPE**: `../mlx-swift-vlm-sam3/Sources/MLXSAM3/SAMComponents.swift` has a working RoPE already; reference it.
- **Dtypes**: default to `.float32` for fixture tests. Inference default is `.float16` — document if a module needs `.float32` for stability.
- **Randomness**: seed MLX with `MLXRandom.seed(0)` if a module uses it (rare in inference paths).
- **Weights loading**: `ModelLoader.swift` already handles both single-file and sharded HF safetensors. `Sam3Model.sanitize` / `Sam31Model.sanitize` do name remapping — extend the `sanitize` maps when new Python param names appear.

## Parity helper (write once)

If `Tests/MLXSAM3Tests/Support/` doesn't exist, create:

```swift
// Tests/MLXSAM3Tests/Support/Fixture.swift
import Foundation
import MLX
import XCTest

struct Fixture {
    let tensors: [String: MLXArray]
    let meta: [String: Any]

    static func load(_ name: String, file: StaticString = #file) throws -> Fixture {
        let url = URL(fileURLWithPath: "\(file)")
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Fixtures/\(name).safetensors")
        var dict = try MLX.loadArrays(url: url)
        var meta: [String: Any] = [:]
        if let raw = dict.removeValue(forKey: "__meta__") {
            let bytes = raw.asArray(UInt8.self)
            if let obj = try? JSONSerialization.jsonObject(with: Data(bytes)) as? [String: Any] {
                meta = obj
            }
        }
        return Fixture(tensors: dict, meta: meta)
    }

    subscript(key: String) -> MLXArray { tensors[key]! }
}

func XCTAssertAllClose(
    _ actual: MLXArray, _ expected: MLXArray,
    atol: Float = 1e-4, rtol: Float = 1e-4,
    _ label: String = "", file: StaticString = #file, line: UInt = #line
) {
    let diff = abs(actual - expected).max().item(Float.self)
    let tol = atol + rtol * abs(expected).max().item(Float.self)
    if diff > tol {
        XCTFail("\(label): max |Δ| = \(diff) > tol \(tol)", file: file, line: line)
    }
}
```

Verify the MLX safetensors loading API (`MLX.loadArrays(url:)`) against the installed `mlx-swift` version before committing — the exact symbol has shifted in recent releases.

## Current state snapshot (2026-04-18, after Phase 1)

- **Completed**: Phase 0 + Phase 1 (leaf math).
  - New Swift: `Sources/MLXSAM3/BoxOps.swift`, `Sources/MLXSAM3/Prompt.swift`; extensions to `Position.swift` (`encodeXY`, `encodeBoxes`).
  - New fixtures (torch-exported, via `scripts/export_fixtures.py`): `position_encoding_sine`, `box_ops`, `geometry_helpers`. All three Swift tests green (fp32, max |Δ| < 1e-5) — `testPositionEncodingSineTorchFixture`, `testBoxOpsFixture`, `testGeometryHelpersFixture`.
  - Fixture script pattern: uses `_load_sam3_module(dotted)` to load leaves by file path, bypassing `sam3/__init__.py` (which needs triton on macOS). If a fixture needs a helper from `sam3.model.geometry_encoders` (which imports via package init), inline a reference copy instead — see `_concat_padded_sequences_ref` for the pattern.
- **Environment note**: `swift test` fails locally with "Failed to load the default metallib" — use `xcodebuild -scheme mlx-swift-sam3 -destination 'platform=macOS' -only-testing:... test` instead. This is the canonical test runner for this repo.
- **Intentional deferrals logged in `TODO.md`**:
  - `SequenceGeometryEncoder` full forward (pooling branches, `MaskEncoder`/`FusedMaskEncoder`, roi_align, grid_sample, pre-norm TransformerEncoderLayer with cross-attn+key_padding_mask) — blocked on memory fuser (Phase 5) + hand-rolled roi_align/grid_sample kernels. Revisit when Phase 6 `sam3_image` forward actually calls the pool branches.
  - Leaf-level RoPE parity fixtures — current Swift seed in `Position.swift` is exercised by Phase 2 (`vit_backbone`) and Phase 5 (tracker/mask-decoder) fixtures; add standalone only if a higher-level parity test reveals mismatch.
- **Gotchas worth knowing**:
  - **Python `PositionEmbeddingSine(num_pos_feats=N)` outputs N channels total**, because it stores `num_pos_feats // 2` internally. **Swift `PositionEmbeddingSine(numPosFeats: N)` outputs `2*N`** (per-direction). When wiring call sites, halve the Python arg.
  - `MLXArray` operators `.<`/`.<=`/`.>=` conflict with SIMD overloads — use `MLX.less`/`MLX.greaterEqual`. `.&` on two `MLXArray`s doesn't work — use `MLX.logicalAnd`.
  - `takeAlong(array, indices, axis:)` needs `indices` broadcast to match `array` shape along non-indexed axes. See `concatPaddedSequences` in `Prompt.swift` for a working example.
- **Next**: Phase 2 — vision backbone. `vitdet.py` → extend `Vision.swift`; `necks.py` → `Necks.swift`. Legacy (vlm-era, MLX-based) fixtures exist at `Tests/MLXSAM3Tests/Fixtures/vit_backbone.safetensors` and `fpn_layer_4x.safetensors` and their tests already exist — verify they still pass before porting; if yes, just add a new torch-exported parity fixture for a fresh backbone→feature-pyramid run.
- **Known-slim files to revisit in later phases**:
  - `Geometry.swift` 97L ← `geometry_encoders.py` 838L (slim done; full port deferred — see Phase 1 entry)
  - `Sam3Model.swift` 186L ← `sam3_image.py` 908L (Phase 6)
  - `Sam31Model.swift` 223L ← `sam3_multiplex_base.py` 2858L + `sam3_multiplex_detector.py` 943L (Phase 7)
  - `Sam31Tracker.swift` 195L ← `sam3_multiplex_tracking.py` 3431L (Phase 8)
  - All video files: entirely missing (Phase 9)

## Communication with the user

- User is `mnmly` (i.am@mnmly.com). See `CLAUDE.md` for style preferences: concise, no narration, evidence-based.
- Ask **one** targeted question when blocked, with a recommended default.
- For scope questions (e.g. "should this module include training path X?") — default to NO, it's inference-only.
- When you finish a phase, summarize in ≤5 bullets: what ported, parity atol, any deviations, next phase. Don't dump file trees back to the user.

## Do-not-do list

- Don't port `sam1_task_predictor.py`, `utils/sam1_utils.py`, `utils/sam2_utils.py` (skip unless explicitly asked).
- Don't port `act_ckpt_utils.py` (training concern).
- Don't port anything under `python/sam3/sam3/train/`, `eval/`, `perflib/`, `assets/`, `scripts/` (those are the Python repo's scripts, not this repo's).
- Don't port matplotlib/cv2-based visualization from `agent/helpers/visualizer.py` — Phase 12, and when we do, it's a CoreGraphics rewrite, not a port.
- Don't commit fixtures or HF weights.
- Don't mass-reformat the vlm-seeded code just because you disagree with style. Keep diffs scoped to the phase you're on.
- Don't skip the parity test and promise "I'll add it later". The test IS the port.
