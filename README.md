# MLX Swift SAM 3 / SAM 3.1

Swift + MLX port of SAM 3 and SAM 3.1 image/video inference.

## Quick Start

```swift
import CoreGraphics
import ImageIO
import MLXSAM3

let weights = URL(fileURLWithPath: "weights/mlx-sam3.1")
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
    weightsDirectory: URL(fileURLWithPath: "weights/mlx-sam3"),
    tokenizerDirectory: URL(fileURLWithPath: "weights/mlx-sam3"),
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

```bash
swift run Sam31ImageInfer \
  --image photo.jpg \
  --prompt "a cat" \
  --tokenizer weights/mlx-sam3.1 \
  --weights weights/mlx-sam3.1 \
  --output out.png \
  --threshold 0.3

swift run ImageInfer \
  --image photo.jpg \
  --prompt "animal" \
  --tokenizer weights/mlx-sam3 \
  --weights weights/mlx-sam3 \
  --output out.png \
  --threshold 0.3
```

SAM 3.1 video tracking demos are also available:

```bash
swift run Sam31VideoTrackFull \
  --input input.mp4 \
  --prompt "a car" \
  --tokenizer weights/mlx-sam3.1 \
  --weights weights/mlx-sam3.1 \
  --output out.mp4
```

## Notes

- The Swift high-level predictor currently covers text-prompt image detection/instance segmentation.
- Box-guided image prompting and Python-style annotator chains are not exposed as public high-level APIs yet.
- SAM 3.1 video tracking is available through `Sam31VideoTracker` and the `Sam31VideoTrackFull` example.
