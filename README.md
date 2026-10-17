# MLX Swift SAM 3 / SAM 3.1

Swift + MLX port of SAM 3 and SAM 3.1 image/video inference.

## Weights

This port loads the pre-converted MLX snapshots published by mlx-community:

```bash
./Scripts/download_weights.sh mlx-community/sam3.1-bf16    # → weights/sam3.1-bf16
./Scripts/download_weights.sh mlx-community/sam3-bf16      # → weights/sam3-bf16
```

> **Don't use raw `facebook/sam3` / `facebook/sam3.1` checkpoints directly.** They use Meta's flat `detector.* / tracker.*` key layout with fused QKVs and PyTorch NCHW conv weights. Swift's module tree was ported against the [`mlx-vlm`](https://github.com/Blaizzy/mlx-vlm) layout (renamed prefixes, split QKVs, NHWC convs), and only the mlx-community pre-converted snapshots match that. To convert a custom checkpoint yourself, use `mlx-vlm/models/sam3_1/convert_weights.py`.

## Quick Start

```swift
import CoreGraphics
import ImageIO
import MLXSAM3

let weights = URL(fileURLWithPath: "weights/sam3.1-bf16")
let imageURL = URL(fileURLWithPath: "photo.jpg")
let imageSource = CGImageSourceCreateWithURL(imageURL as CFURL, nil)!
let image = CGImageSourceCreateImageAtIndex(imageSource, 0, nil)!

let predictor = try await Sam3Predictor.loadSAM31(
    weightsDirectory: weights,
    tokenizerDirectory: weights,
    scoreThreshold: 0.3
)

let result = try predictor.predict(image: image, textPrompt: "a dog")

for i in 0..<result.scores.dim(0) {
    let score = result.scores[i].item(Float.self)
    let box = result.boxes[i].asType(.float32).asArray(Float.self)
    print("[\(score)] box=(\(box[0]), \(box[1]), \(box[2]), \(box[3]))")
}
```

`SAM3ImagePrediction` contains:

- `boxes`: `(N, 4)` xyxy boxes in original image pixels.
- `masks`: `(N, H, W)` boolean masks resized to the original image.
- `maskLogits`: `(N, H, W)` post-sigmoid mask probabilities.
- `scores`: `(N,)` confidence scores.

## SAM 3

Use the same high-level API with SAM 3 weights:

```swift
let predictor = try await Sam3Predictor.loadSAM3(
    weightsDirectory: URL(fileURLWithPath: "weights/sam3-bf16"),
    tokenizerDirectory: URL(fileURLWithPath: "weights/sam3-bf16"),
    scoreThreshold: 0.3
)

let result = try predictor.predict(image: image, textPrompt: "a person")
```

If you already have a loaded model or pre-tokenized prompt, use the lower-level initializers:

```swift
let model = Sam31Model(Sam31ModelConfig())
try ModelLoader.load(sam31: model, from: weights)

let predictor = Sam3Predictor(model: model, scoreThreshold: 0.3)
let result = try predictor.predict(
    image: image,
    inputIds: inputIds,
    attentionMask: attentionMask
)
```

## CLI

> ⚠️ Use `xcodebuild`, not `swift run`. MLX needs a Metal-capable toolchain at runtime; `swift run` / `swift build` produce a binary that errors with `Failed to load the default metallib` at startup.

Build any executable target with:

```bash
xcodebuild -scheme Sam31ImageInfer -destination 'platform=macOS' \
    -configuration release -derivedDataPath .xcdd build
```

Then run from `.xcdd/Build/Products/Release/`:

```bash
.xcdd/Build/Products/Release/Sam31ImageInfer \
  --image photo.jpg \
  --prompt "a cat" \
  --tokenizer weights/sam3.1-bf16 \
  --weights weights/sam3.1-bf16 \
  --output out.png \
  --threshold 0.3
```

> SAM 3 (the older model) is supported as a library — use `Sam3Predictor.loadSAM3(...)` from your own code. The CLI examples target SAM 3.1 only; if you need a SAM 3 CLI, the simplest path is to copy `Examples/Sam31ImageInfer/Sam31ImageInfer.swift` and swap `Sam31Model` / `loadSAM31` for the SAM 3 equivalents.

SAM 3.1 video tracking demos are also available:

```bash
xcodebuild -scheme Sam31VideoTrackFull -destination 'platform=macOS' \
    -configuration release -derivedDataPath .xcdd build

.xcdd/Build/Products/Release/Sam31VideoTrackFull \
  --input input.mp4 \
  --prompt "a car" \
  --tokenizer weights/sam3.1-bf16 \
  --weights weights/sam3.1-bf16 \
  --output out.mp4
```

## Video tracking

`Sam31VideoPipeline` wraps the multi-object tracker in a `CGImage`-per-frame API:

```swift
import MLXSAM3

let pipeline = try await Sam31VideoPipeline.fromPretrained(
    weightsDirectory: URL(fileURLWithPath: "weights/sam3.1-bf16"),
    redetectEvery: 15,
    detectThreshold: 0.5
)
pipeline.setTextPrompt("a person")

for frame in frames {           // [CGImage]
    let result = try pipeline.processFrame(frame)
    for obj in result.tracks {
        // obj.id, obj.lastBoxPixels, obj.lastMask, obj.lastScore
    }
}
```

Or via the top-level convenience:

```swift
let pipeline = try await MLXSAM3.loadSAM31Video(
    weightsDirectory: URL(fileURLWithPath: "weights/sam3.1-bf16"))
```

For the lower-level orchestrator (custom association, memory-bank tuning, per-track state inspection), construct `Sam31VideoTracker` directly — see `Sources/MLXSAM3/Sam31VideoTracker.swift`.

### Slim vs full mode

There are two video tracking flows:

1. **`Sam31VideoTrack` (slim, per-frame detect)** — runs full text-prompted detection on every frame. No identity persistence, no memory-bank propagation. Fast to set up, easy to verify, but flickers/re-IDs across frames. Use this when you only need per-frame masks and don't care about object identity.

2. **`Sam31VideoTrackFull` / `Sam31VideoPipeline` (memory-bank tracking)** — first-frame detect + per-frame propagation through `trackStep` with a multiplex memory bank. Object IDs persist across frames. This is the path that mirrors `python/sam3`'s multiplex video predictor; the architecture is wired but parity with the python reference still has a gap (see `Tests/MLXSAM3Tests/MLXSAM3Tests.swift::testSam31VideoTrackerSlimVsPySam3_10frame` for the current diagnostic — mean IoU ~0.30 on a 10-frame clip; root cause is the HF↔Meta config divergence documented in CLAUDE memory).

Pick `Sam31VideoTrack` for shape-correct per-frame masking, `Sam31VideoTrackFull` / `Sam31VideoPipeline` for object continuity.

## SAM3App — unified macOS demo

`Examples/SAM3App/SAM3App.xcodeproj` is a SwiftUI app that exercises every public pipeline:

- **Image** — drag in a photo, type a prompt, see masks/boxes (`SAM3Predictor` / `MLXSAM3.loadSAM31Image`).
- **Video** — pick a movie file, watch it frame-by-frame. Toggle "Persistent-ID tracking" to switch between per-frame `SAM3Predictor.predict` (fast, IDs flicker) and `Sam31VideoPipeline.processFrame` (multiplex memory bank, persistent track IDs across frames).
- **Camera** — live capture; same persistent-ID toggle as the video tab.

Open `Examples/SAM3App/SAM3App.xcodeproj` in Xcode, build the `SAM3App` scheme, point it at a SAM 3.1 weights snapshot directory, and you're set.

## Notes

- The Swift high-level predictor currently covers text-prompt image detection/instance segmentation.
- Box-guided image prompting and Python-style annotator chains are not exposed as public high-level APIs yet.
- SAM 3.1 video tracking is available through `Sam31VideoPipeline` (high-level) or `Sam31VideoTracker` (low-level).

## Benchmarks

See [Benchmarks/README.md](Benchmarks/README.md) for protocol-matched perf measurements vs the python/sam3 reference.

## License

Distributed under Meta's [SAM License](LICENSE), as a derivative of
[SAM 3](https://github.com/facebookresearch/sam3). Module layout follows
[mlx-vlm](https://github.com/Blaizzy/mlx-vlm) (MIT).
