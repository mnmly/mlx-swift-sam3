// SAM3ImagePredictor.swift — slim image + text prompt → masks predictor.
// Python source: sam3/model/sam3_image_processor.py (Sam3Processor) +
//                sam3/model/sam3_base_predictor.py post-processing.
//
// Inference-only wrapper around Sam3Model.detect(). Does:
//   1. Image normalization (mean/std = 0.5/0.5 → [-1, 1]).
//   2. Forward through the detector.
//   3. presence × sigmoid(logits), threshold filter.
//   4. NMS over retained boxes.
//   5. Bilinear upsample masks back to the original image size, binarize at 0.5.
//   6. Scale cxcywh→xyxy boxes to the original image coords.
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

    public init(masks: MLXArray, maskLogits: MLXArray, boxes: MLXArray, scores: MLXArray) {
        self.masks = masks
        self.maskLogits = maskLogits
        self.boxes = boxes
        self.scores = scores
    }
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
        threshold: Float,
        nmsThreshold: Float? = 0.5
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
        var keptScores = scores.take(idx, axis: 0)
        var keptBoxes = boxesN.take(idx, axis: 0)
        var keptMasks = masksN.take(idx, axis: 0)

        if let nmsThreshold, nmsThreshold > 0, keepIdx.count > 1 {
            let nmsKeep = nmsIndices(
                boxes: keptBoxes.asType(.float32).asArray(Float.self),
                scores: keptScores.asType(.float32).asArray(Float.self),
                threshold: nmsThreshold
            )
            let nmsIdx = MLXArray(nmsKeep)
            keptScores = keptScores.take(nmsIdx, axis: 0)
            keptBoxes = keptBoxes.take(nmsIdx, axis: 0)
            keptMasks = keptMasks.take(nmsIdx, axis: 0)
        }

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

    private static func nmsIndices(boxes: [Float], scores: [Float], threshold: Float) -> [Int32] {
        var order = scores.indices.sorted { scores[$0] > scores[$1] }
        var keep: [Int32] = []

        while let current = order.first {
            keep.append(Int32(current))
            order.removeFirst()
            order = order.filter { iou(boxes, current, $0) <= threshold }
        }

        return keep
    }

    private static func iou(_ boxes: [Float], _ a: Int, _ b: Int) -> Float {
        let ax0 = boxes[a * 4 + 0], ay0 = boxes[a * 4 + 1]
        let ax1 = boxes[a * 4 + 2], ay1 = boxes[a * 4 + 3]
        let bx0 = boxes[b * 4 + 0], by0 = boxes[b * 4 + 1]
        let bx1 = boxes[b * 4 + 2], by1 = boxes[b * 4 + 3]

        let ix0 = max(ax0, bx0), iy0 = max(ay0, by0)
        let ix1 = min(ax1, bx1), iy1 = min(ay1, by1)
        let inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        let areaA = max(0, ax1 - ax0) * max(0, ay1 - ay0)
        let areaB = max(0, bx1 - bx0) * max(0, by1 - by0)
        let union = areaA + areaB - inter
        return union > 0 ? inter / union : 0
    }
}

public enum SAM3PredictorError: Error, CustomStringConvertible {
    case tokenizerRequired

    public var description: String {
        switch self {
        case .tokenizerRequired:
            return "A tokenizer is required for text-prompt prediction. Initialize with a tokenizer or use predict(image:inputIds:attentionMask:dtype:)."
        }
    }
}

/// High-level image predictor mirroring the Python `Sam3Predictor` flow:
/// model + tokenizer + image + text prompt -> boxes, masks, and scores.
public final class SAM3Predictor {
    public let resolution: Int
    public var scoreThreshold: Float

    private let tokenizer: SAM3Tokenizer?
    private let detect: (
        _ pixelValues: MLXArray,
        _ inputIds: MLXArray?,
        _ attentionMask: MLXArray?,
        _ inputsEmbeds: MLXArray?
    ) -> [String: MLXArray]
    private let inputEmbeddings: (_ inputIds: MLXArray, _ attentionMask: MLXArray?) -> (MLXArray, MLXArray?)
    private var textCache: [String: (MLXArray, MLXArray?)] = [:]

    public init(
        model: Sam3Model,
        tokenizer: SAM3Tokenizer? = nil,
        resolution: Int = 1008,
        scoreThreshold: Float = 0.5
    ) {
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.scoreThreshold = scoreThreshold
        self.detect = { pixelValues, inputIds, attentionMask, inputsEmbeds in
            model.detect(
                pixelValues: pixelValues,
                inputIds: inputIds,
                attentionMask: attentionMask,
                inputsEmbeds: inputsEmbeds
            )
        }
        self.inputEmbeddings = { inputIds, attentionMask in
            model.getInputEmbeddings(inputIds, attentionMask: attentionMask)
        }
    }

    public init(
        model: Sam31Model,
        tokenizer: SAM3Tokenizer? = nil,
        resolution: Int = 1008,
        scoreThreshold: Float = 0.5
    ) {
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.scoreThreshold = scoreThreshold
        self.detect = { pixelValues, inputIds, attentionMask, inputsEmbeds in
            model.detect(
                pixelValues: pixelValues,
                inputIds: inputIds,
                attentionMask: attentionMask,
                inputsEmbeds: inputsEmbeds
            )
        }
        self.inputEmbeddings = { inputIds, attentionMask in
            model.getInputEmbeddings(inputIds, attentionMask: attentionMask)
        }
    }

    public static func loadSAM3(
        weightsDirectory: URL,
        tokenizerDirectory: URL? = nil,
        resolution: Int = 1008,
        scoreThreshold: Float = 0.5,
        dtype: DType = .bfloat16,
        tokenizerMaxLength: Int = 32
    ) async throws -> SAM3Predictor {
        let model = Sam3Model(ModelConfig())
        try ModelLoader.load(sam3: model, from: weightsDirectory, dtype: dtype)
        let tokenizer = try await SAM3Tokenizer.fromSnapshotOrOpenAIClip(
            tokenizerDirectory ?? weightsDirectory,
            maxLength: tokenizerMaxLength
        )
        return SAM3Predictor(
            model: model,
            tokenizer: tokenizer,
            resolution: resolution,
            scoreThreshold: scoreThreshold
        )
    }

    public static func loadSAM31(
        weightsDirectory: URL,
        tokenizerDirectory: URL? = nil,
        resolution: Int = 1008,
        scoreThreshold: Float = 0.5,
        dtype: DType = .bfloat16,
        tokenizerMaxLength: Int = 32
    ) async throws -> SAM3Predictor {
        let model = Sam31Model(Sam31ModelConfig())
        try ModelLoader.load(sam31: model, from: weightsDirectory, dtype: dtype)
        let tokenizer = try await SAM3Tokenizer.fromSnapshotOrOpenAIClip(
            tokenizerDirectory ?? weightsDirectory,
            maxLength: tokenizerMaxLength
        )
        return SAM3Predictor(
            model: model,
            tokenizer: tokenizer,
            resolution: resolution,
            scoreThreshold: scoreThreshold
        )
    }

    public func predict(
        image: CGImage,
        textPrompt: String,
        dtype: DType = .bfloat16
    ) throws -> SAM3ImagePrediction {
        let (embeds, attentionMask) = try getInputEmbeddings(textPrompt)
        return try predict(
            image: image,
            inputsEmbeds: embeds,
            attentionMask: attentionMask,
            dtype: dtype
        )
    }

    public func getInputEmbeddings(_ textPrompt: String) throws -> (MLXArray, MLXArray?) {
        guard let tokenizer else { throw SAM3PredictorError.tokenizerRequired }
        if let cached = textCache[textPrompt] {
            return cached
        }
        let tokens = tokenizer.encode(textPrompt)
        let cached = inputEmbeddings(tokens.inputIds, tokens.attentionMask)
        textCache[textPrompt] = cached
        return cached
    }

    public func predict(
        image: CGImage,
        inputsEmbeds: MLXArray,
        attentionMask: MLXArray? = nil,
        dtype: DType = .bfloat16
    ) throws -> SAM3ImagePrediction {
        let origH = image.height
        let origW = image.width
        let pixel = try ImagePreprocess.pixelValues(from: image, size: resolution, dtype: dtype)
        let outputs = detect(pixel, nil, attentionMask, inputsEmbeds)
        return try predict(
            outputs: outputs,
            originalHeight: origH,
            originalWidth: origW
        )
    }

    public func predict(
        image: CGImage,
        inputIds: MLXArray,
        attentionMask: MLXArray? = nil,
        dtype: DType = .bfloat16
    ) throws -> SAM3ImagePrediction {
        let origH = image.height
        let origW = image.width
        let pixel = try ImagePreprocess.pixelValues(from: image, size: resolution, dtype: dtype)
        let outputs = detect(pixel, inputIds, attentionMask, nil)
        return try predict(
            outputs: outputs,
            originalHeight: origH,
            originalWidth: origW
        )
    }

    private func predict(
        outputs: [String: MLXArray],
        originalHeight origH: Int,
        originalWidth origW: Int
    ) throws -> SAM3ImagePrediction {
        return SAM3ImagePredictor.postprocess(
            outputs: outputs,
            originalHeight: origH,
            originalWidth: origW,
            threshold: scoreThreshold
        )
    }
}

public typealias Sam3Predictor = SAM3Predictor
