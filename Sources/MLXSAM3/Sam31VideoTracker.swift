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
    /// FIFO of *propagation* memory entries (not the conditioning anchor). Each (1, HW, 256).
    /// Capped at numMaskmem-1 = 6 so total bank is `[condAnchor] + memoryBank` ≤ numMaskmem.
    public var memoryBank: [MLXArray] = []
    /// Persistent conditioning-frame entry (from re-detection). NOT evicted by FIFO —
    /// matches python/sam3's `cond_frame_outputs` which are kept across the whole clip.
    public var condAnchor: MLXArray? = nil
    /// FIFO of past object pointers, each (1, 1, D=256). Prepended to memory bank
    /// for cross-attention; these tokens should be excluded from spatial RoPE.
    /// Capped at maxObjectPointersInEncoder (16).
    public var objPtrBank: [MLXArray] = []
    /// On a re-detect frame, the index into pred.maskLogits / pred.boxes that
    /// matched this track. Used for shared-bucket memory encoding.
    public var lastDetectionMaskIndex: Int? = nil
    /// FIFO of past frames' RAW image features (1, HW, D). Same length / order as
    /// memoryBank. Fed into `image_cross_attn_k_proj` in DecoupledMemoryAttention —
    /// python's "memory_image" stream. Without this, image cross-attn keys are wrong.
    public var imageMemoryBank: [MLXArray] = []
}

public final class Sam31VideoTracker {
    public let model: Sam31Model
    public let resolution: Int
    public let dtype: DType

    public var detectThreshold: Float
    public var redetectEvery: Int
    public var assocIoUThreshold: Float
    public var maxMissingFrames: Int
    /// Maximum memory bank entries per track (FIFO eviction). Default 7 matches numMaskmem.
    public var maxMemoryBank: Int = 7
    /// When true, use per-track memory banks (each track encodes only its own mask
    /// in slot 0). When false, use a shared multiplex bucket where N tracks' masks
    /// are encoded together in slots 0..N-1. The bucket path is architecturally aligned
    /// with python/sam3 but has been observed to drift more on long clips when other
    /// dependencies (per-slot obj_ptr, no_obj_embed_spatial, hotstart) are missing.
    public var usePerTrackBank: Bool = true

    private(set) public var tracks: [TrackedObject] = []
    private var nextId: Int = 0
    private var frameIdx: Int = 0
    /// Shared multiplex-bucket memory bank. Slot i = tracks[i]'s mask in the encoded
    /// 32-channel input. One entry per past frame, FIFO-capped at maxMemoryBank.
    /// All tracks share this — that's the point of the multiplex bucket structure.
    private var sharedMemoryBank: [MLXArray] = []
    /// Per-track image memory FIFO (raw `currentFeatures` flattened). Used when bucket
    /// path is on. Per-track path uses `tracks[i].imageMemoryBank`.
    private var sharedImageMemoryBank: [MLXArray] = []

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

    /// Clear all tracker state — live tracks, frame counter, memory banks. Use this
    /// when starting a new video / clip without re-instantiating the tracker.
    public func reset() {
        tracks.removeAll()
        nextId = 0
        frameIdx = 0
        sharedMemoryBank.removeAll()
        sharedImageMemoryBank.removeAll()
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
        let beforeCount = tracks.count
        tracks.removeAll { $0.framesSinceSeen > maxMissingFrames }
        if tracks.count != beforeCount { sharedMemoryBank.removeAll() }
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
        MLX.eval([
            outputs["pred_logits"]!,
            outputs["pred_boxes"]!,
            outputs["pred_masks"]!,
            outputs["presence_logits"]!,
        ])
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

        // Pre-compute propagation features so each new/updated track can have an
        // initial memory entry encoded from this frame's detection mask.
        let prop = model.propagationFeatures(pixelValues: pixelValues)
        MLX.eval(prop)
        let features = prop[2]                                  // 1× scale, used as memory backbone

        // For memory encoding we want the *raw, square* low-res mask logits (e.g. 288×288
        // at 1008-input) — NOT the post-sigmoid orig-resolution masks. Slicing the raw
        // model output by post-NMS query indices preserves aspect ratio and avoids the
        // double-sigmoid that pred.maskLogits would introduce.
        let rawMasks = outputs["pred_masks"]!                  // (1, Q, h_low, w_low)

        var matched = Array(repeating: false, count: nDet)
        var trackMatched = Array(repeating: false, count: tracks.count)

        let order = (0..<nDet).sorted { scores[$0] > scores[$1] }
        for di in order {
            let dbox = Array(pxBoxes[di*4 ..< di*4+4])
            let dMask = sliceMask(pred.masks, index: di)              // (H, W) bool, orig res
            var bestTi = -1
            var bestIoU: Float = assocIoUThreshold
            for (ti, t) in tracks.enumerated() where !trackMatched[ti] {
                // Prefer mask-IoU when both masks exist (more stable than box-IoU under
                // motion / partial occlusion). Fall back to box-IoU otherwise.
                let iou: Float
                if let dm = dMask, let tm = t.lastMask {
                    iou = Self.maskIoU(dm, tm)
                } else {
                    iou = Self.iouXYXY(dbox, t.lastBoxPixels)
                }
                if iou > bestIoU { bestIoU = iou; bestTi = ti }
            }
            if bestTi >= 0 {
                let normBox = MLXArray(zip(dbox, scale).map { $0 / $1 }).reshaped([1, 4])
                tracks[bestTi].lastBoxNormalized = normBox
                tracks[bestTi].lastBoxPixels = dbox
                tracks[bestTi].lastMask = sliceMask(pred.masks, index: di)
                tracks[bestTi].lastScore = scores[di]
                tracks[bestTi].framesSinceSeen = 0
                tracks[bestTi].lastDetectionMaskIndex = di
                if usePerTrackBank {
                    tracks[bestTi].condAnchor = encodeDetectionMemory(
                        rawMasks: rawMasks, queryIndex: pred.keptQueryIndices[di], features: features)
                    tracks[bestTi].memoryBank = []   // reset propagation history on re-detect
                    // Seed image memory with this frame's image features (matches the cond anchor).
                    let imgFlat = features.reshaped([1, -1, features.dim(3)])
                    tracks[bestTi].imageMemoryBank = [imgFlat]
                }
                trackMatched[bestTi] = true
                matched[di] = true
            }
        }

        for (ti, m) in trackMatched.enumerated() where !m { tracks[ti].framesSinceSeen += 1 }

        for di in 0..<nDet where !matched[di] {
            let dbox = Array(pxBoxes[di*4 ..< di*4+4])
            let normBox = MLXArray(zip(dbox, scale).map { $0 / $1 }).reshaped([1, 4])
            let perTrackAnchor: MLXArray? = usePerTrackBank
                ? encodeDetectionMemory(rawMasks: rawMasks, queryIndex: pred.keptQueryIndices[di], features: features)
                : nil
            let imgFlat = features.reshaped([1, -1, features.dim(3)])
            let perTrackImageBank: [MLXArray] = usePerTrackBank ? [imgFlat] : []
            tracks.append(TrackedObject(
                id: nextId,
                lastBoxNormalized: normBox,
                lastBoxPixels: dbox,
                lastMask: sliceMask(pred.masks, index: di),
                lastScore: scores[di],
                framesSinceSeen: 0,
                memoryBank: [],
                condAnchor: perTrackAnchor,
                objPtrBank: [],
                lastDetectionMaskIndex: di,
                imageMemoryBank: perTrackImageBank))
            nextId += 1
        }

        if !usePerTrackBank {
            // Build a single shared multiplex-bucket memory entry containing every live
            // track's detection mask in its slot. Reset the shared bank.
            let liveTracks = tracks.enumerated().filter { $0.element.framesSinceSeen == 0 && $0.element.lastDetectionMaskIndex != nil }
            let maskLogitsList: [MLXArray] = liveTracks.map { (_, t) in
                let qi = pred.keptQueryIndices[t.lastDetectionMaskIndex!]
                return rawMasks[0, qi][.newAxis, .ellipsis]
            }
            let condFlags = Array(repeating: true, count: maskLogitsList.count)
            if !maskLogitsList.isEmpty {
                let bucketEntry = model.encodeMultiplexMasksToMemory(
                    features: features,
                    maskLogitsList: maskLogitsList,
                    isConditioningList: condFlags,
                    alreadySigmoid: false                                       // raw logits
                )
                MLX.eval(bucketEntry)
                sharedMemoryBank = [bucketEntry]
            }
        }
    }

    /// Build temporal positional encodings for the obj_ptr stream tokens.
    /// Mirrors python/sam3's `_get_tpos_enc`:
    ///   1. rel_pos = [1, 2, ..., numPtrs] / (maxObjPtrs - 1)
    ///   2. sine PE with hidden_dim
    ///   3. project via temporalPosEncProjLayer
    /// Returns shape (1, numPtrs, dim).
    /// Token ordering: ptrs[0] is the OLDEST collected obj_ptr, ptrs[K-1] is the most recent.
    /// Frame distance: ptrs[K-1] is 1 (just from prev frame), ptrs[0] is K (K frames ago).
    private func buildObjPtrTemporalPos(numPtrs: Int, dim: Int, dtype: DType) -> MLXArray {
        let maxObjPtrs = 16  // matches Sam31TrackerConfig.maxObjectPointersInEncoder
        let tDiffMax = Float(maxObjPtrs - 1)
        // rel_pos[i] = K - i (so newest = 1, oldest = K), normalized by tDiffMax.
        let relPos = (0..<numPtrs).map { Float(numPtrs - $0) / tDiffMax }
        let posInds = MLXArray(relPos)                      // (K,)
        let pe = sinePositional1D(posInds: posInds, dim: dim)  // (K, dim)
        let projected = model.trackerModel.temporalPosEncProjLayer(pe)
        return projected.reshaped([1, numPtrs, dim]).asType(dtype)
    }

    /// 1D sine positional encoding (Transformer paper). Mirrors python/sam3 `get_1d_sine_pe`.
    private func sinePositional1D(posInds: MLXArray, dim: Int) -> MLXArray {
        let peDim = dim / 2
        let dimT = MLXArray(0..<peDim).asType(.float32)
        // dim_t = temperature ^ (2 * (dim_t // 2) / pe_dim)
        let temperature: Float = 10000.0
        let exponent = (2.0 * MLX.floor(dimT / 2.0)) / Float(peDim)
        let scales = MLX.pow(MLXArray(temperature), exponent)
        let posEmbed = posInds[0..., .newAxis] / scales[.newAxis, 0...]   // (N, peDim)
        return concatenated([sin(posEmbed), cos(posEmbed)], axis: -1)     // (N, dim)
    }

    /// Build memory_image_pos bank: image positional encoding (spatial) + temporal PE
    /// per past frame. Mirrors python's `memory_image_pos` which is spatial-image-pos
    /// + per-frame temporal offset. Each entry has shape (1, HW, D).
    /// Currently feeds K-projection of cross-attention POST-projection.
    private func buildTemporalPosBank(bank: [MLXArray], hasConditioningHead: Bool) -> [MLXArray] {
        let tpe = model.trackerModel.memoryTemporalPositionalEncoding   // (numMaskmem, 1, 1, D)
        let numMaskmem = tpe.dim(0)
        let d = tpe.dim(-1)
        let condIdx = numMaskmem - 1

        // Compute spatial image PE for the feature grid. The first non-obj_ptr entry's
        // size is HW (= H_f * W_f); HW is the same across all memory_image entries.
        // Use it to derive the spatial size needed for imagePeLayer.
        guard let firstEntry = bank.first else { return [] }
        let hw = firstEntry.dim(1)
        let side = Int(Double(hw).squareRoot())
        // imagePeLayer((side, side)) → (HW, D). Broadcast batch dim.
        let imagePos = model.trackerModel.imagePeLayer((side, side))
            .reshaped([1, hw, d])
            .asType(firstEntry.dtype)

        var out: [MLXArray] = []
        out.reserveCapacity(bank.count)
        for (i, entry) in bank.enumerated() {
            let tIdx: Int
            if hasConditioningHead && i == 0 {
                tIdx = condIdx
            } else {
                let tPos = bank.count - i
                tIdx = max(0, numMaskmem - 1 - tPos)
            }
            let pe = tpe[tIdx].reshaped([1, 1, d]).asType(entry.dtype)    // (1, 1, D)
            let combined = imagePos + MLX.broadcast(pe, to: [1, entry.dim(1), d])
            out.append(combined)
        }
        return out
    }

    /// Encode a single matched detection's mask into a multiplex memory entry.
    /// Uses the *raw* low-res mask logits from `outputs["pred_masks"]` (1, Q, h, w)
    /// indexed by the original query index — preserving the model-native square
    /// aspect ratio. Pre-sigmoid logits are passed; encoder applies sigmoid+scale+bias.
    private func encodeDetectionMemory(
        rawMasks: MLXArray, queryIndex qi: Int, features: MLXArray
    ) -> MLXArray {
        let logits = rawMasks[0, qi]                    // (h_low, w_low)
        let logitsBatched = logits[.newAxis, .ellipsis] // (1, h_low, w_low)
        let mem = model.encodeMaskToMemory(
            features: features,
            maskLogits: logitsBatched,
            isConditioning: true,
            alreadySigmoid: false                         // raw logits
        )
        MLX.eval(mem)
        return mem
    }

    // MARK: - Propagate path

    private func propagateFrame(pixelValues: MLXArray, origH: Int, origW: Int) {
        guard !tracks.isEmpty else { return }
        let prop = model.propagationFeatures(pixelValues: pixelValues)
        MLX.eval(prop)
        let features = prop[2]
        // Order matches python: hrf[0]=mid (144x144) feeds upscale_conv1+, hrf[1]=large (288) feeds upscale_conv2+.
        let highRes = prop.count > 1 ? [prop[1], prop[0]] : nil

        if usePerTrackBank {
            propagatePerTrack(features: features, highRes: highRes, origH: origH, origW: origW)
            return
        }

        // No shared bank yet (no detection has happened) — fall back to slim per-track box prompt.
        if sharedMemoryBank.isEmpty {
            for i in tracks.indices {
                let outputs = model.propagateStep(
                    propagationFeatures: prop,
                    promptBoxNormalized: tracks[i].lastBoxNormalized
                )
                MLX.eval(outputs.values.map { $0 })
                guard let predMasks = outputs["pred_masks"] else {
                    tracks[i].framesSinceSeen += 1
                    continue
                }
                applyPropagationOutputs(trackIndex: i, predMasks: predMasks,
                                        iouScores: outputs["iou_scores"], objectPointer: nil,
                                        origH: origH, origW: origW)
            }
            return
        }

        // Bucket-shared inference: ONE trackStep call producing 5D multiplex output,
        // demuxed per-track.
        let memBank = sharedMemoryBank
        let memPosBank = buildTemporalPosBank(bank: memBank, hasConditioningHead: true)

        // Bucket-level obj_ptr stream: collect per-track ptrs; each track's ptr lives
        // in its own slot of the multiplex output. For cross-attn we concat them and
        // the network's multiplex handling picks out the relevant slot. Simple
        // approximation: use track-0's bank as the bucket-shared ptr stream.
        let ptrs = tracks.first?.objPtrBank ?? []

        var fullBank = memBank
        var fullPosBank = memPosBank
        var numKExclude = 0
        if !ptrs.isEmpty {
            let ptrTokens = concatenated(ptrs, axis: 1)
            fullBank = [ptrTokens] + memBank
            let ptrPos = buildObjPtrTemporalPos(numPtrs: ptrs.count, dim: ptrTokens.dim(-1),
                                                dtype: ptrTokens.dtype)
            fullPosBank = [ptrPos] + memPosBank
            numKExclude = ptrTokens.dim(1)
        }

        let outputs = model.trackerModel.trackStep(
            currentFeatures: features,
            memoryBank: fullBank,
            memoryPosBank: fullPosBank,
            numKExcludeRope: numKExclude,
            promptPoints: nil,
            promptBoxes: nil,
            promptMasks: nil,
            multimaskOutput: true,                         // multimask + best-IoU pick
            highResFeatures: highRes,
            keepMultiplex: true,                           // 5D output for demux
            numActiveSlots: tracks.count                   // bucket: N tracks in slots 0..N-1
        )
        MLX.eval(outputs.values.map { $0 })

        guard let predMasks5D = outputs["pred_masks"],     // (1, multiplex, num_masks, H, W)
              let iouScores4D = outputs["iou_scores"] else {     // (1, multiplex, num_masks)
            for i in tracks.indices { tracks[i].framesSinceSeen += 1 }
            return
        }
        // Per-slot obj_ptr (B=1, M, D) — slot i is track i's obj_ptr.
        let perSlotPtr = outputs["object_pointer"]          // shape (1, M, D)

        // Demux per track: track at index i lives in multiplex slot i.
        for i in tracks.indices {
            let trackMasks = predMasks5D[0, i]              // (num_masks, H, W)
            let trackIou = iouScores4D[0, i]                // (num_masks,)
            let trackPtr = perSlotPtr.map { $0[0, i] }      // (D,) → reshape (1, D) below
            applyPropagationOutputs(trackIndex: i, predMasks: trackMasks[.newAxis, .ellipsis],
                                    iouScores: trackIou[.newAxis, .ellipsis],
                                    objectPointer: trackPtr.map { $0[.newAxis, 0...] },
                                    origH: origH, origW: origW)
        }

        // Encode the new bucket memory entry (all tracks' new masks together).
        let perTrackMasks: [MLXArray] = tracks.map { track in
            // Use the just-stored lastMask if available; else fall back to a zero mask.
            // We need the low-res logits; pull from the demuxed predMasks5D.
            // Convention: same slot index = track index.
            return MLXArray(0.0)   // placeholder, overwritten below via index
        }
        var maskLogitsList: [MLXArray] = []
        var condList: [Bool] = []
        for i in tracks.indices {
            let trackIou = iouScores4D[0, i].asType(.float32).asArray(Float.self)
            var bestK = 0
            for k in 1..<trackIou.count where trackIou[k] > trackIou[bestK] { bestK = k }
            let lowRes = predMasks5D[0, i, bestK]
            maskLogitsList.append(lowRes[.newAxis, .ellipsis])
            // python multiplex always treats initial-detection tracks as conditioning
            condList.append(true)
        }
        _ = perTrackMasks
        if !maskLogitsList.isEmpty {
            let bucketEntry = model.encodeMultiplexMasksToMemory(
                features: features,
                maskLogitsList: maskLogitsList,
                isConditioningList: condList,
                alreadySigmoid: false                         // raw logits from decoder
            )
            MLX.eval(bucketEntry)
            sharedMemoryBank.append(bucketEntry)
            if sharedMemoryBank.count > maxMemoryBank {
                sharedMemoryBank.removeFirst(sharedMemoryBank.count - maxMemoryBank)
            }
        }
    }

    /// Per-track propagation path: each track has its own memoryBank (single-slot encoding).
    /// Falls back to box-prompt for tracks that haven't been re-detected yet.
    private func propagatePerTrack(features: MLXArray, highRes: [MLXArray]?, origH: Int, origW: Int) {
        for i in tracks.indices {
            // Bank = [conditioning anchor] + propagation FIFO. The anchor is pinned
            // (never evicted) — matches python/sam3's cond_frame_outputs.
            var bank: [MLXArray] = []
            if let anchor = tracks[i].condAnchor { bank.append(anchor) }
            bank.append(contentsOf: tracks[i].memoryBank)
            let outputs: [String: MLXArray]
            if !bank.isEmpty {
                let memPosBank = buildTemporalPosBank(bank: bank, hasConditioningHead: tracks[i].condAnchor != nil)
                let ptrs = tracks[i].objPtrBank
                var fullBank = bank
                var fullPosBank = memPosBank
                var numKExclude = 0
                if !ptrs.isEmpty {
                    let ptrTokens = concatenated(ptrs, axis: 1)
                    fullBank = [ptrTokens] + bank
                    let ptrPos = buildObjPtrTemporalPos(numPtrs: ptrs.count, dim: ptrTokens.dim(-1),
                                                        dtype: ptrTokens.dtype)
                    fullPosBank = [ptrPos] + memPosBank
                    numKExclude = ptrTokens.dim(1)
                }
                // Image memory bank — past frames' raw flattened image features.
                // Same length/order as `bank` (one image-mem entry per past frame).
                // We don't have an "anchor" for image memory (no separate cond version);
                // we just use the last `bank.count` entries of `tracks[i].imageMemoryBank`.
                let imgBank: [MLXArray]? = tracks[i].imageMemoryBank.isEmpty
                    ? nil
                    : Array(tracks[i].imageMemoryBank.suffix(bank.count))
                outputs = model.trackerModel.trackStep(
                    currentFeatures: features,
                    memoryBank: fullBank,
                    memoryPosBank: fullPosBank,
                    memoryImageBank: imgBank,
                    numKExcludeRope: numKExclude,
                    promptPoints: nil, promptBoxes: nil, promptMasks: nil,
                    multimaskOutput: true,
                    highResFeatures: highRes,
                    numActiveSlots: 1                        // per-track path: only slot 0 is the object
                )
            } else {
                // Track has no memory yet — box-prompt fallback.
                outputs = model.trackerModel.trackStep(
                    currentFeatures: features,
                    memoryBank: nil,
                    promptPoints: nil,
                    promptBoxes: tracks[i].lastBoxNormalized,
                    promptMasks: nil,
                    multimaskOutput: true,
                    highResFeatures: highRes
                )
            }
            MLX.eval(outputs.values.map { $0 })
            guard let predMasks = outputs["pred_masks"],
                  let iouScores = outputs["iou_scores"] else {
                tracks[i].framesSinceSeen += 1
                continue
            }
            // Encode the predicted mask back into this track's bank.
            let iouHost = iouScores.asType(.float32).asArray(Float.self)
            var bestK = 0
            for k in 1..<iouHost.count where iouHost[k] > iouHost[bestK] { bestK = k }
            let maskLowRes = predMasks[0, bestK]
            // python/sam3 multiplex passes `conditioning_objects = current_out["conditioning_objects"]`
            // on EVERY frame — for tracks established via initial text-prompt detection,
            // they remain "conditioning objects" across all subsequent frames. So the
            // conditioning channel is 1.0 throughout, not just on the re-detect frame.
            let memEntry = model.encodeMaskToMemory(
                features: features,
                maskLogits: maskLowRes[.newAxis, .ellipsis],
                isConditioning: true,
                alreadySigmoid: false
            )
            MLX.eval(memEntry)
            tracks[i].memoryBank.append(memEntry)
            // Image-memory bank: this frame's flattened image features.
            // Used by image_cross_attn_k_proj on the NEXT frame.
            let imgFlat = features.reshaped([1, -1, features.dim(3)])
            tracks[i].imageMemoryBank.append(imgFlat)
            // FIFO over propagation entries only — anchor is pinned, so cap at maxMemoryBank-1.
            let propCap = max(1, maxMemoryBank - 1)
            if tracks[i].memoryBank.count > propCap {
                tracks[i].memoryBank.removeFirst(tracks[i].memoryBank.count - propCap)
            }
            // Image bank: keep total bank size (anchor + propCap = maxMemoryBank).
            if tracks[i].imageMemoryBank.count > maxMemoryBank {
                tracks[i].imageMemoryBank.removeFirst(tracks[i].imageMemoryBank.count - maxMemoryBank)
            }
            applyPropagationOutputs(trackIndex: i, predMasks: predMasks,
                                    iouScores: iouScores,
                                    objectPointer: outputs["object_pointer"],
                                    origH: origH, origW: origW)
        }
    }

    /// Apply per-track propagation outputs (mask, box update, obj_ptr append).
    private func applyPropagationOutputs(
        trackIndex i: Int,
        predMasks: MLXArray,            // (1, num_masks, H, W)
        iouScores: MLXArray?,           // (1, num_masks) or nil
        objectPointer: MLXArray?,       // (1, D) or nil
        origH: Int, origW: Int
    ) {
        let iouHost = iouScores?.asType(.float32).asArray(Float.self) ?? [0]
        var bestK = 0
        for k in 1..<iouHost.count where iouHost[k] > iouHost[bestK] { bestK = k }
        let maskLowRes = predMasks[0, bestK]
        if let ptr = objectPointer {
            let ptrTok = ptr.reshaped([1, 1, ptr.dim(-1)])
            tracks[i].objPtrBank.append(ptrTok)
            if tracks[i].objPtrBank.count > 16 {
                tracks[i].objPtrBank.removeFirst(tracks[i].objPtrBank.count - 16)
            }
        }
        let (upMask, boxPx) = upsampleAndExtractBox(
            maskLowRes: maskLowRes, origH: origH, origW: origW)
        guard let boxPx = boxPx else {
            tracks[i].framesSinceSeen += 1
            return
        }
        let scale: [Float] = [Float(origW), Float(origH), Float(origW), Float(origH)]
        let normBox = MLXArray(zip(boxPx, scale).map { $0 / $1 }).reshaped([1, 4])
        tracks[i].lastBoxNormalized = normBox
        tracks[i].lastBoxPixels = boxPx
        tracks[i].lastMask = upMask
        tracks[i].framesSinceSeen = 0
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
        var maskBool = maskLogits .> MLXArray(0.5)
        // Keep only the largest connected component. Without this, a single
        // track's mask can grow to cover two nearby people (SAM's decoder is
        // happy to segment any salient blob inside the prompt region) and the
        // axis-aligned box from `boxFromMask` would then wrap both.
        if let lcc = largestConnectedComponent(maskBool) {
            maskBool = lcc
        }
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

    /// Return a new (H, W) bool MLXArray containing only the largest 4-connected
    /// component of `mask`. Returns nil if the mask is empty.
    /// Host-side iterative flood-fill — O(H*W). For 720×1280 that's <1ms in
    /// practice, dwarfed by the propagation forward pass.
    private func largestConnectedComponent(_ mask: MLXArray) -> MLXArray? {
        let h = mask.dim(0), w = mask.dim(1)
        let flat = mask.asArray(Bool.self)
        var visited = [Bool](repeating: false, count: flat.count)
        var bestSize = 0
        var bestPixels: [Int] = []
        var stack: [Int] = []
        stack.reserveCapacity(1024)

        for start in 0..<flat.count where flat[start] && !visited[start] {
            stack.removeAll(keepingCapacity: true)
            stack.append(start)
            visited[start] = true
            var component: [Int] = [start]

            while let p = stack.popLast() {
                let y = p / w, x = p % w
                // 4-connected neighbours
                if x > 0 {
                    let q = p - 1
                    if flat[q] && !visited[q] { visited[q] = true; stack.append(q); component.append(q) }
                }
                if x < w - 1 {
                    let q = p + 1
                    if flat[q] && !visited[q] { visited[q] = true; stack.append(q); component.append(q) }
                }
                if y > 0 {
                    let q = p - w
                    if flat[q] && !visited[q] { visited[q] = true; stack.append(q); component.append(q) }
                }
                if y < h - 1 {
                    let q = p + w
                    if flat[q] && !visited[q] { visited[q] = true; stack.append(q); component.append(q) }
                }
            }

            if component.count > bestSize {
                bestSize = component.count
                bestPixels = component
            }
        }

        guard bestSize > 0 else { return nil }
        var out = [Bool](repeating: false, count: flat.count)
        for p in bestPixels { out[p] = true }
        return MLXArray(out, [h, w])
    }

    /// Mask IoU between two bool masks of the same shape (H, W).
    static func maskIoU(_ a: MLXArray, _ b: MLXArray) -> Float {
        let ai = a.asType(.int32)
        let bi = b.asType(.int32)
        let inter = MLX.sum(ai * bi).item(Int32.self)
        let union = MLX.sum(MLX.maximum(ai, bi)).item(Int32.self)
        return union > 0 ? Float(inter) / Float(union) : 0
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
