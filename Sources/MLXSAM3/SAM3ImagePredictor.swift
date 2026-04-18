// SAM3ImagePredictor.swift — slim image + text prompt → masks predictor.
// Python source: sam3/model/sam3_image_processor.py (Sam3Processor) +
//                sam3/model/sam3_base_predictor.py post-processing.
//
// Inference-only wrapper around Sam3Model.detect(). Does:
//   1. Image normalization (mean/std = 0.5/0.5 → [-1, 1]).
//   2. Forward through the detector.
//   3. presence × sigmoid(logits), threshold filter.
//   4. Bilinear upsample masks back to the original image size, binarize at 0.5.
//   5. Scale cxcywh→xyxy boxes to the original image coords.
//
// Tokenization is deferred (Phase 11) — callers pass pre-tokenized inputIds/attentionMask.

import CoreGraphics
import Foundation
import MLX
import MLXNN

public struct SAM3ImagePrediction {
    public let masks: MLXArray        // (N, H, W) bool
    public let maskLogits: MLXArray   // (N, H, W) float, post-sigmoid
    public let boxes: MLXArray        // (N, 4) xyxy in original pixel coords
    public let scores: MLXArray       // (N,) presence * class prob
}

public final class SAM3ImagePredictor {
    public let model: Sam3Model
    public let resolution: Int
    public var confidenceThreshold: Float

    public init(model: Sam3Model, resolution: Int = 1008, confidenceThreshold: Float = 0.5) {
        self.model = model
        self.resolution = resolution
        self.confidenceThreshold = confidenceThreshold
    }

    public func predict(
        image: CGImage,
        inputIds: MLXArray,
        attentionMask: MLXArray? = nil,
        dtype: DType = .float16
    ) throws -> SAM3ImagePrediction {
        let origH = image.height
        let origW = image.width
        let pixel = try ImagePreprocess.pixelValues(from: image, size: resolution, dtype: dtype)
        let outputs = model.detect(pixelValues: pixel, inputIds: inputIds, attentionMask: attentionMask)
        return Self.postprocess(
            outputs: outputs, originalHeight: origH, originalWidth: origW,
            threshold: confidenceThreshold
        )
    }

    /// Replicates `Sam3Processor._forward_grounding` post-model steps.
    public static func postprocess(
        outputs: [String: MLXArray],
        originalHeight origH: Int,
        originalWidth origW: Int,
        threshold: Float
    ) -> SAM3ImagePrediction {
        // Model returns: pred_logits (B, Q), pred_boxes (B, Q, 4 xyxy), pred_masks (B, Q, h, w),
        //                presence_logits (B, 1), semantic_seg (B, 1, H, W).
        let logits = outputs["pred_logits"]!       // (B, Q)
        let boxesXYXY = outputs["pred_boxes"]!     // (B, Q, 4) already xyxy, normalized
        let masks = outputs["pred_masks"]!         // (B, Q, h, w)
        let presence = outputs["presence_logits"]! // (B, 1)

        // Drop batch (assume B=1 in inference-only path).
        let clsProb = sigmoid(logits)[0]                    // (Q,)
        let presProb = sigmoid(presence)[0].squeezed()      // scalar
        let scores = clsProb * presProb                     // (Q,)
        let boxesN = boxesXYXY[0]                           // (Q, 4)
        let masksN = masks[0]                               // (Q, h, w)

        // MLX doesn't support boolean indexing. Materialize scores to Swift
        // to gather the indices that pass threshold, then take() along Q.
        let scoresHost = scores.asType(.float32).asArray(Float.self)
        var keepIdx: [Int32] = []
        keepIdx.reserveCapacity(scoresHost.count)
        for (i, s) in scoresHost.enumerated() where s > threshold {
            keepIdx.append(Int32(i))
        }
        if keepIdx.isEmpty {
            let emptyBoxes = MLXArray.zeros([0, 4])
            let emptyMasks = MLXArray.zeros([0, origH, origW]).asType(.bool)
            let emptyLogits = MLXArray.zeros([0, origH, origW])
            let emptyScores = MLXArray.zeros([0])
            return SAM3ImagePrediction(
                masks: emptyMasks, maskLogits: emptyLogits,
                boxes: emptyBoxes, scores: emptyScores)
        }
        let idx = MLXArray(keepIdx)
        let keptScores = scores.take(idx, axis: 0)
        let keptBoxes = boxesN.take(idx, axis: 0)
        let keptMasks = masksN.take(idx, axis: 0)

        // Scale boxes (xyxy, normalized [0,1]) back to pixel coords.
        let scale = MLXArray([Float(origW), Float(origH), Float(origW), Float(origH)])
        let pxBoxes = keptBoxes * scale[.newAxis, 0...]

        // Upsample masks (N, h, w) → (N, H, W) via bilinear, then sigmoid + threshold.
        // MLX Upsample expects NHWC: add trailing channel.
        let m4 = keptMasks[.ellipsis, .newAxis]  // (N, h, w, 1)
        let sh = Float(origH) / Float(m4.dim(1))
        let sw = Float(origW) / Float(m4.dim(2))
        let up = Upsample(scaleFactor: .array([sh, sw]), mode: .linear(alignCorners: false))(m4)
        let maskLogits = sigmoid(up.squeezed(axis: -1))           // (N, H, W)
        let maskBool = maskLogits .> MLXArray(0.5)

        return SAM3ImagePrediction(
            masks: maskBool, maskLogits: maskLogits,
            boxes: pxBoxes, scores: keptScores
        )
    }
}
