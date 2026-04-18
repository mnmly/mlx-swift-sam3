// DetectFromNSImage.swift — end-to-end SAM 3.1 detection from an NSImage.
//
// Usage (macOS command-line target):
//     let image = NSImage(contentsOf: URL(fileURLWithPath: "/path/to/cat.jpg"))!
//     let boxes = try runDetection(image: image, inputIds: tokens, weightsDir: weightsURL)
//
// Token IDs come from the CLIP BPE tokenizer — encode your prompt
// (e.g. "a cat") elsewhere and pass the resulting `(1, T) Int32` array.

import Foundation
import MLX
import MLXSAM3

#if canImport(AppKit)
import AppKit

public func runDetection(
    image: NSImage,
    inputIds: MLXArray,
    weightsDir: URL,
    inputSize: Int = 1008
) throws -> [String: MLXArray] {
    // 1. Build & load the model.
    let model = Sam31Model(Sam31ModelConfig())
    try ModelLoader.load(sam31: model, from: weightsDir, dtype: .float16)

    // 2. Convert NSImage → (1, H, W, 3) float16 tensor in [0, 1].
    let pixelValues = try ImagePreprocess.pixelValues(from: image, size: inputSize)

    // 3. Encode the text prompt.
    let (textEmbeds, attnMask) = model.getInputEmbeddings(inputIds, attentionMask: nil)

    // 4. Run detection.
    let result = model.detect(
        pixelValues: pixelValues,
        inputIds: nil,
        attentionMask: attnMask,
        inputsEmbeds: textEmbeds
    )
    MLX.eval(Array(result.values))
    return result
}

/// Decode top-K boxes from a detection result.
///
/// - Returns: array of `(score, xyxy)` in pixel space of the original `imageSize`.
public func topKBoxes(
    from result: [String: MLXArray],
    imageSize: CGSize,
    k: Int = 10
) -> [(score: Float, box: CGRect)] {
    guard
        let logits = result["pred_logits"],
        let boxes  = result["pred_boxes"]
    else { return [] }

    let scores = MLX.sigmoid(logits.squeezed(axis: -1)).squeezed(axis: 0)
    let boxesF = boxes.squeezed(axis: 0).asType(.float32)
    let scoresF = scores.asType(.float32)

    let n = scoresF.dim(0)
    var pairs: [(Float, Int)] = []
    pairs.reserveCapacity(n)
    let scoreArr = scoresF.asArray(Float.self)
    for i in 0..<n { pairs.append((scoreArr[i], i)) }
    pairs.sort { $0.0 > $1.0 }

    let box2D = boxesF.asArray(Float.self) // flat [n*4]
    let W = Float(imageSize.width), H = Float(imageSize.height)
    return pairs.prefix(k).map { pair in
        let i = pair.1 * 4
        let x0 = CGFloat(box2D[i + 0] * W)
        let y0 = CGFloat(box2D[i + 1] * H)
        let x1 = CGFloat(box2D[i + 2] * W)
        let y1 = CGFloat(box2D[i + 3] * H)
        return (pair.0, CGRect(x: x0, y: y0, width: x1 - x0, height: y1 - y0))
    }
}

#endif
