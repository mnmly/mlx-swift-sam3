// Sam31VideoTracker.swift — slim multi-object video tracker orchestrator.
//
// Strategy (v2 slim, Phase 11.5):
//   1. Every `redetectEvery` frames (and on frame 0): run full `Sam31Model.detect()`,
//      threshold + postprocess to get proposals, then greedy IoU-associate each new
//      detection to an existing track (or spawn a new track).
//   2. On non-redetect frames: run vision backbone only, then for each live track
//      call `Sam31Model.propagateStep(propagationFeatures:, promptBoxNormalized:)`
//      with the last normalized box as a prompt. Update the track's mask / box.
//   3. Tracks unseen for `maxMissingFrames` consecutive frames are dropped.
//
// Known v2 slim gaps (documented in TODO.md Phase 11.5):
//   - No memory bank: `trackStep` is called with `memoryBank: nil` so temporal PE
//     and obj_ptr memory stream are unused. Revisit if per-frame mask IoU drops.
//   - No EDT false-neg/false-pos correction.
//   - Mask prompts not threaded back into `trackStep` (only boxes).

import CoreGraphics
import Foundation
import MLX
import MLXNN

public struct TrackedObject {
    public var id: Int
    /// xyxy in normalized [0,1] coords (for box prompt on next frame).
    public var lastBoxNormalized: MLXArray
    /// xyxy in original pixel coords (for rendering / association metrics).
    public var lastBoxPixels: [Float]
    /// (H, W) bool mask at original image resolution, or nil if not yet upsampled.
    public var lastMask: MLXArray?
    public var lastScore: Float
    public var framesSinceSeen: Int
}

public final class Sam31VideoTracker {
    public let model: Sam31Model
    public let resolution: Int
    public let dtype: DType

    public var detectThreshold: Float
    public var redetectEvery: Int
    public var assocIoUThreshold: Float
    public var maxMissingFrames: Int

    private(set) public var tracks: [TrackedObject] = []
    private var nextId: Int = 0
    private var frameIdx: Int = 0

    public init(
        model: Sam31Model,
        resolution: Int = 1008,
        dtype: DType = .bfloat16,
        detectThreshold: Float = 0.5,
        redetectEvery: Int = 15,
        assocIoUThreshold: Float = 0.3,
        maxMissingFrames: Int = 30
    ) {
        self.model = model
        self.resolution = resolution
        self.dtype = dtype
        self.detectThreshold = detectThreshold
        self.redetectEvery = redetectEvery
        self.assocIoUThreshold = assocIoUThreshold
        self.maxMissingFrames = maxMissingFrames
    }

    /// Run one frame. Returns the live tracks (framesSinceSeen == 0) after the step.
    public func step(
        pixelValues: MLXArray,
        inputIds: MLXArray,
        attentionMask: MLXArray? = nil,
        originalHeight origH: Int,
        originalWidth origW: Int
    ) -> [TrackedObject] {
        let isRedetect = (frameIdx % redetectEvery == 0) || tracks.isEmpty
        if isRedetect {
            redetectFrame(pixelValues: pixelValues, inputIds: inputIds,
                          attentionMask: attentionMask,
                          origH: origH, origW: origW)
        } else {
            propagateFrame(pixelValues: pixelValues, origH: origH, origW: origW)
        }
        tracks.removeAll { $0.framesSinceSeen > maxMissingFrames }
        frameIdx += 1
        return tracks.filter { $0.framesSinceSeen == 0 }
    }

    // MARK: - Redetect path

    private func redetectFrame(
        pixelValues: MLXArray,
        inputIds: MLXArray,
        attentionMask: MLXArray?,
        origH: Int, origW: Int
    ) {
        let outputs = model.detect(
            pixelValues: pixelValues, inputIds: inputIds, attentionMask: attentionMask)
        MLX.eval(Array(outputs.values))
        let pred = SAM3ImagePredictor.postprocess(
            outputs: outputs, originalHeight: origH, originalWidth: origW,
            threshold: detectThreshold)

        let nDet = pred.boxes.dim(0)
        guard nDet > 0 else {
            for i in tracks.indices { tracks[i].framesSinceSeen += 1 }
            return
        }
        let pxBoxes = pred.boxes.asType(.float32).asArray(Float.self)
        let scores = pred.scores.asType(.float32).asArray(Float.self)
        let scale: [Float] = [Float(origW), Float(origH), Float(origW), Float(origH)]

        var matched = Array(repeating: false, count: nDet)
        var trackMatched = Array(repeating: false, count: tracks.count)

        // Greedy IoU association: descending score → best available track.
        let order = (0..<nDet).sorted { scores[$0] > scores[$1] }
        for di in order {
            let dbox = Array(pxBoxes[di*4 ..< di*4+4])
            var bestTi = -1
            var bestIoU: Float = assocIoUThreshold
            for (ti, t) in tracks.enumerated() where !trackMatched[ti] {
                let iou = Self.iouXYXY(dbox, t.lastBoxPixels)
                if iou > bestIoU { bestIoU = iou; bestTi = ti }
            }
            if bestTi >= 0 {
                let normBox = MLXArray(zip(dbox, scale).map { $0 / $1 }).reshaped([1, 4])
                tracks[bestTi].lastBoxNormalized = normBox
                tracks[bestTi].lastBoxPixels = dbox
                tracks[bestTi].lastMask = sliceMask(pred.masks, index: di)
                tracks[bestTi].lastScore = scores[di]
                tracks[bestTi].framesSinceSeen = 0
                trackMatched[bestTi] = true
                matched[di] = true
            }
        }

        for (ti, m) in trackMatched.enumerated() where !m { tracks[ti].framesSinceSeen += 1 }

        for di in 0..<nDet where !matched[di] {
            let dbox = Array(pxBoxes[di*4 ..< di*4+4])
            let normBox = MLXArray(zip(dbox, scale).map { $0 / $1 }).reshaped([1, 4])
            tracks.append(TrackedObject(
                id: nextId,
                lastBoxNormalized: normBox,
                lastBoxPixels: dbox,
                lastMask: sliceMask(pred.masks, index: di),
                lastScore: scores[di],
                framesSinceSeen: 0))
            nextId += 1
        }
    }

    // MARK: - Propagate path

    private func propagateFrame(pixelValues: MLXArray, origH: Int, origW: Int) {
        guard !tracks.isEmpty else { return }
        let prop = model.propagationFeatures(pixelValues: pixelValues)
        MLX.eval(prop)

        for i in tracks.indices {
            let outputs = model.propagateStep(
                propagationFeatures: prop,
                promptBoxNormalized: tracks[i].lastBoxNormalized)
            MLX.eval(Array(outputs.values))

            guard let predMasks = outputs["pred_masks"],
                  let iouScores = outputs["iou_scores"] else {
                tracks[i].framesSinceSeen += 1
                continue
            }
            // pred_masks: (B=1, num_masks, h, w). Pick argmax iou slot.
            let iouHost = iouScores.asType(.float32).asArray(Float.self)
            var bestK = 0
            for k in 1..<iouHost.count where iouHost[k] > iouHost[bestK] { bestK = k }

            let maskLowRes = predMasks[0, bestK]             // (h, w)
            let (upMask, boxPx) = upsampleAndExtractBox(
                maskLowRes: maskLowRes, origH: origH, origW: origW)
            guard let boxPx = boxPx else {
                tracks[i].framesSinceSeen += 1
                continue
            }
            let scale: [Float] = [Float(origW), Float(origH), Float(origW), Float(origH)]
            let normBox = MLXArray(zip(boxPx, scale).map { $0 / $1 }).reshaped([1, 4])
            tracks[i].lastBoxNormalized = normBox
            tracks[i].lastBoxPixels = boxPx
            tracks[i].lastMask = upMask
            tracks[i].framesSinceSeen = 0
        }
    }

    // MARK: - Helpers

    private func sliceMask(_ masks: MLXArray, index: Int) -> MLXArray? {
        masks.dim(0) > index ? masks[index] : nil
    }

    private func upsampleAndExtractBox(
        maskLowRes: MLXArray, origH: Int, origW: Int
    ) -> (MLXArray, [Float]?) {
        let m4 = maskLowRes[.newAxis, .ellipsis, .newAxis]  // (1, h, w, 1)
        let sh = Float(origH) / Float(m4.dim(1))
        let sw = Float(origW) / Float(m4.dim(2))
        let up = Upsample(scaleFactor: .array([sh, sw]), mode: .linear(alignCorners: false))(m4)
        let maskLogits = sigmoid(up.squeezed(axes: [0, -1]))
        let maskBool = maskLogits .> MLXArray(0.5)
        let box = boxFromMask(maskBool, origH: origH, origW: origW)
        return (maskBool, box)
    }

    private func boxFromMask(_ mask: MLXArray, origH: Int, origW: Int) -> [Float]? {
        // Extract tight xyxy via axis-any reductions. Host-side argmax to keep it simple.
        let rows = mask.any(axes: [1]).asArray(Bool.self)        // (H,)
        let cols = mask.any(axes: [0]).asArray(Bool.self)        // (W,)
        guard let y0 = rows.firstIndex(of: true),
              let y1 = rows.lastIndex(of: true),
              let x0 = cols.firstIndex(of: true),
              let x1 = cols.lastIndex(of: true) else { return nil }
        return [Float(x0), Float(y0), Float(x1 + 1), Float(y1 + 1)]
    }

    static func iouXYXY(_ a: [Float], _ b: [Float]) -> Float {
        let x0 = max(a[0], b[0]), y0 = max(a[1], b[1])
        let x1 = min(a[2], b[2]), y1 = min(a[3], b[3])
        let inter = max(0, x1 - x0) * max(0, y1 - y0)
        let areaA = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        let areaB = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        let union = areaA + areaB - inter
        return union > 0 ? inter / union : 0
    }
}
