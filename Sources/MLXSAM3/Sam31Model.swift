// Sam31Model.swift — SAM 3.1 main model.
// Python source: sam3_1/sam3_1.py
// Weight keys: detector_model.* / tracker_model.*

import Foundation
import MLX
import MLXNN

// MARK: - Sam31GeometryEncoder

/// SAM 3.1 geometry encoder — extends GeometryEncoder with point prompt projections.
public class Sam31GeometryEncoder: GeometryEncoder {
    @ModuleInfo(key: "points_direct_project") var pointsDirectProject: Linear
    @ModuleInfo(key: "points_pool_project") var pointsPoolProject: Linear
    @ModuleInfo(key: "points_pos_enc_project") var pointsPosEncProject: Linear

    override public init(_ config: GeometryEncoderConfig) {
        let d = config.hiddenSize
        self._pointsDirectProject = ModuleInfo(wrappedValue: Linear(2, d), key: "points_direct_project")
        self._pointsPoolProject = ModuleInfo(wrappedValue: Linear(d, d), key: "points_pool_project")
        self._pointsPosEncProject = ModuleInfo(wrappedValue: Linear(d, d), key: "points_pos_enc_project")
        super.init(config)
    }
}

// MARK: - Sam31DetectorModel

public class Sam31DetectorModel: Module {
    @ModuleInfo(key: "vision_encoder") var visionEncoder: Sam31VisionEncoder
    @ModuleInfo(key: "text_encoder") var textEncoder: TextEncoder
    @ModuleInfo(key: "text_projection") var textProjection: Linear
    @ModuleInfo(key: "detr_encoder") var detrEncoder: DETREncoder
    @ModuleInfo(key: "detr_decoder") var detrDecoder: DETRDecoder
    @ModuleInfo(key: "geometry_encoder") var geometryEncoder: Sam31GeometryEncoder
    @ModuleInfo(key: "mask_decoder") var maskDecoder: MaskDecoder
    @ModuleInfo(key: "dot_product_scoring") var dotProductScoring: DotProductScoring

    let posEnc: PositionEmbeddingSine

    public init(_ config: Sam31ModelConfig) {
        let det = config.detectorConfig
        let d = det.detrEncoderConfig.hiddenSize
        self._visionEncoder = ModuleInfo(wrappedValue: Sam31VisionEncoder(det.visionConfig), key: "vision_encoder")
        self._textEncoder = ModuleInfo(wrappedValue: TextEncoder(det.textConfig, dModel: d), key: "text_encoder")
        self._textProjection = ModuleInfo(wrappedValue: Linear(det.textConfig.hiddenSize, d), key: "text_projection")
        self._detrEncoder = ModuleInfo(wrappedValue: DETREncoder(det.detrEncoderConfig), key: "detr_encoder")
        self._detrDecoder = ModuleInfo(wrappedValue: DETRDecoder(det.detrDecoderConfig), key: "detr_decoder")
        self._geometryEncoder = ModuleInfo(wrappedValue: Sam31GeometryEncoder(det.geometryEncoderConfig), key: "geometry_encoder")
        self._maskDecoder = ModuleInfo(wrappedValue: MaskDecoder(det.maskDecoderConfig), key: "mask_decoder")
        self._dotProductScoring = ModuleInfo(wrappedValue: DotProductScoring(hiddenSize: d), key: "dot_product_scoring")
        self.posEnc = PositionEmbeddingSine(numPosFeats: d / 2)
        super.init()
    }

    public func getInputEmbeddings(_ inputIds: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        textProjection(textEncoder(inputIds, attentionMask: attentionMask))
    }

    public func callAsFunction(
        pixelValues: MLXArray,
        inputIds: MLXArray? = nil,
        attentionMask: MLXArray? = nil,
        inputsEmbeds: MLXArray? = nil
    ) -> [String: MLXArray] {
        let (detFeatures, _, _) = visionEncoder(pixelValues, needDet: true, needInteractive: false, needPropagation: false)

        let fpnPos = detFeatures.map { posEnc($0) }
        let encFeat = detFeatures.last!
        let encPos = fpnPos.last!
        let b = encFeat.dim(0), h = encFeat.dim(1), w = encFeat.dim(2), d = encFeat.dim(3)
        let src = encFeat.reshaped([b, h * w, d])
        let posFlat = encPos.reshaped([b, h * w, d])

        let embeds: MLXArray
        if let ie = inputsEmbeds {
            embeds = ie
        } else {
            embeds = getInputEmbeddings(inputIds!, attentionMask: attentionMask)
        }

        let encoded = detrEncoder(src, pos: posFlat, textMemory: embeds, textMask: attentionMask)
        let (hs, refBoxes, presenceLogits) = detrDecoder(
            visionFeatures: encoded, inputsEmbeds: embeds, visionPosEncoding: posFlat,
            textMask: attentionMask, spatialShape: (h, w)
        )

        let predBoxesCxcywh = refBoxes[refBoxes.dim(0) - 1]
        let cx = predBoxesCxcywh[.ellipsis, 0], cy = predBoxesCxcywh[.ellipsis, 1]
        let bw = predBoxesCxcywh[.ellipsis, 2], bh = predBoxesCxcywh[.ellipsis, 3]
        let predBoxesXYXY = stacked([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis: -1)

        let allPredLogits = dotProductScoring(hs, inputsEmbeds: embeds, textMask: attentionMask)
        let predLogits = allPredLogits[allPredLogits.dim(0) - 1]
        let presence = presenceLogits[presenceLogits.dim(0) - 1]

        let lastHs = hs[hs.dim(0) - 1]
        let (predMasks, semanticSeg) = maskDecoder(
            objQueries: lastHs,
            backboneFeatures: detFeatures,
            encoderHiddenStates: encoded,
            promptFeatures: embeds,
            promptMask: attentionMask
        )

        return [
            "pred_logits": predLogits.squeezed(axes: [-1]),
            "pred_boxes": predBoxesXYXY,
            "pred_masks": predMasks,
            "presence_logits": presence,
            "semantic_seg": semanticSeg,
            "intermediate_hidden_states": hs,
            "intermediate_boxes": refBoxes,
            "encoder_hidden_states": encoded,
        ]
    }
}

// MARK: - Sam31Model

public class Sam31Model: Module {
    @ModuleInfo(key: "detector_model") var detectorModel: Sam31DetectorModel
    @ModuleInfo(key: "tracker_model") var trackerModel: MultiplexTrackerModel

    public init(_ config: Sam31ModelConfig) {
        self._detectorModel = ModuleInfo(wrappedValue: Sam31DetectorModel(config), key: "detector_model")
        self._trackerModel = ModuleInfo(wrappedValue: MultiplexTrackerModel(config.trackerConfig), key: "tracker_model")
        super.init()
    }

    public func detect(
        pixelValues: MLXArray,
        inputIds: MLXArray? = nil,
        attentionMask: MLXArray? = nil,
        inputsEmbeds: MLXArray? = nil
    ) -> [String: MLXArray] {
        detectorModel(pixelValues: pixelValues, inputIds: inputIds, attentionMask: attentionMask, inputsEmbeds: inputsEmbeds)
    }

    public func detectProfiled(
        pixelValues: MLXArray,
        inputIds: MLXArray? = nil,
        attentionMask: MLXArray? = nil,
        inputsEmbeds: MLXArray? = nil
    ) -> ([String: MLXArray], [(String, Double)]) {
        var timings: [(String, Double)] = []

        func measure<T>(_ name: String, _ body: () -> T, eval: (T) -> Void) -> T {
            let start = Date().timeIntervalSinceReferenceDate
            let value = body()
            eval(value)
            timings.append((name, (Date().timeIntervalSinceReferenceDate - start) * 1000))
            return value
        }

        let det = detectorModel

        let backboneFeatures = measure("vision.backbone", {
            det.visionEncoder.backbone(pixelValues)
        }, eval: { MLX.eval($0) })

        let detFeatures = measure("vision.neck.det", {
            det.visionEncoder.neck(
                backboneFeatures,
                needDet: true,
                needInteractive: false,
                needPropagation: false
            ).0
        }, eval: { MLX.eval($0) })

        let (src, posFlat, h, w) = measure("posenc_flatten", {
            let fpnPos = detFeatures.map { det.posEnc($0) }
            let encFeat = detFeatures.last!
            let encPos = fpnPos.last!
            let b = encFeat.dim(0), h = encFeat.dim(1), w = encFeat.dim(2), d = encFeat.dim(3)
            return (
                encFeat.reshaped([b, h * w, d]),
                encPos.reshaped([b, h * w, d]),
                h,
                w
            )
        }, eval: { MLX.eval($0.0, $0.1) })

        let embeds = measure("text_encoder", {
            inputsEmbeds ?? det.getInputEmbeddings(inputIds!, attentionMask: attentionMask)
        }, eval: { MLX.eval($0) })

        let encoded = measure("detr_encoder", {
            det.detrEncoder(src, pos: posFlat, textMemory: embeds, textMask: attentionMask)
        }, eval: { MLX.eval($0) })

        let (hs, refBoxes, presenceLogits) = measure("detr_decoder", {
            det.detrDecoder(
                visionFeatures: encoded,
                inputsEmbeds: embeds,
                visionPosEncoding: posFlat,
                textMask: attentionMask,
                spatialShape: (h, w)
            )
        }, eval: { MLX.eval($0.0, $0.1, $0.2) })

        let (predLogits, predBoxesXYXY, presence) = measure("scoring_boxes", {
            let predBoxesCxcywh = refBoxes[refBoxes.dim(0) - 1]
            let cx = predBoxesCxcywh[.ellipsis, 0], cy = predBoxesCxcywh[.ellipsis, 1]
            let bw = predBoxesCxcywh[.ellipsis, 2], bh = predBoxesCxcywh[.ellipsis, 3]
            let predBoxesXYXY = stacked([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis: -1)
            let allPredLogits = det.dotProductScoring(hs, inputsEmbeds: embeds, textMask: attentionMask)
            return (
                allPredLogits[allPredLogits.dim(0) - 1],
                predBoxesXYXY,
                presenceLogits[presenceLogits.dim(0) - 1]
            )
        }, eval: { MLX.eval($0.0, $0.1, $0.2) })

        let (predMasks, semanticSeg) = measure("mask_decoder", {
            det.maskDecoder(
                objQueries: hs[hs.dim(0) - 1],
                backboneFeatures: detFeatures,
                encoderHiddenStates: encoded,
                promptFeatures: embeds,
                promptMask: attentionMask
            )
        }, eval: { MLX.eval($0.0, $0.1) })

        return ([
            "pred_logits": predLogits.squeezed(axes: [-1]),
            "pred_boxes": predBoxesXYXY,
            "pred_masks": predMasks,
            "presence_logits": presence,
            "semantic_seg": semanticSeg,
            "intermediate_hidden_states": hs,
            "intermediate_boxes": refBoxes,
            "encoder_hidden_states": encoded,
        ], timings)
    }

    public func getInputEmbeddings(_ inputIds: MLXArray, attentionMask: MLXArray? = nil) -> (MLXArray, MLXArray?) {
        let emb = detectorModel.getInputEmbeddings(inputIds, attentionMask: attentionMask)
        MLX.eval(emb)
        return (emb, attentionMask)
    }

    func getTrackerFeatures(_ backboneFeatures: MLXArray) -> [MLXArray] {
        let (_, _, prop) = detectorModel.visionEncoder.neck(backboneFeatures,
            needDet: false, needInteractive: false, needPropagation: true)
        return prop
    }

    /// Vision-only forward producing just the tracker propagation pyramid.
    /// Use on non-redetect frames to skip DETR/text cost.
    public func propagationFeatures(pixelValues: MLXArray) -> [MLXArray] {
        let backbone = detectorModel.visionEncoder.backbone(pixelValues)
        return getTrackerFeatures(backbone)
    }

    /// Propagate a single object using pre-computed tracker features.
    /// Slim v2: memory bank left nil; identity is maintained via box prompt only.
    public func propagateStep(
        propagationFeatures prop: [MLXArray],
        promptBoxNormalized box: MLXArray
    ) -> [String: MLXArray] {
        let features = prop[2]
        // Order matches python: hrf[0]=mid-spatial (144x144) for upscale_conv1+, hrf[1]=large (288x288) for upscale_conv2+.
        let highRes = prop.count > 2 ? [prop[1], prop[0]] : nil
        return trackerModel.trackStep(
            currentFeatures: features,
            memoryBank: nil,
            promptPoints: nil,
            promptBoxes: box,
            promptMasks: nil,
            multimaskOutput: false,
            highResFeatures: highRes
        )
    }

    /// Encode a single object's predicted mask + propagation features into a multiplex
    /// memory entry suitable for the memory bank.
    ///
    /// Mirrors python/sam3's multiplex `_encode_new_memory` for the single-object case:
    ///   mask_for_mem = sigmoid(logits) * 2.0 + (-1.0)        // SAM 3.1 defaults
    ///   mux into multiplex slot 0 (channels 1..15 zeroed)
    ///   append conditioning-flag channel (1.0 on prompt/init frame, 0.0 otherwise),
    ///   also muxed → channels 17..31 zeroed
    ///   final mask input: (1, H, W, 32)
    ///
    /// - Parameters:
    ///   - features: (1, H_f, W_f, 256) propagation FPN level [2] (1× scale).
    ///   - maskLogits: (1, H_m, W_m) raw mask logits (NOT sigmoid). Will be upsampled
    ///     to mask_downsampler input resolution = features × 16 along each spatial dim.
    ///   - isConditioning: true if this frame had an explicit prompt (e.g. detection
    ///     match or user click); affects the conditioning channel.
    /// - Returns: encoded memory at (1, H_f * W_f, 256) — track_step bank layout.
    public func encodeMaskToMemory(
        features: MLXArray,
        maskLogits: MLXArray,
        isConditioning: Bool,
        alreadySigmoid: Bool = false
    ) -> MLXArray {
        // SAM 3.1 multiplex defaults (HF config); could be lifted from Sam31TrackerConfig
        // if Sam31Model stored the config — currently it doesn't.
        let multiplex = 16                                   // multiplex_count
        let totalCh = multiplex * 2                          // 32
        let hf = features.dim(1), wf = features.dim(2), d = features.dim(3)
        let downsampleStride = 16                            // mask_downsampler_total_stride
        let mh = hf * downsampleStride, mw = wf * downsampleStride

        // 1. Upsample mask logits (1, H_m, W_m) → (1, mh, mw, 1) via bilinear.
        let logits4 = maskLogits[.newAxis, .ellipsis, .newAxis] // (1, 1, H_m, W_m, 1) → drop one
        // Actually maskLogits is already (1, H_m, W_m). Add channel dim for Upsample.
        let logits = maskLogits.reshaped([1, maskLogits.dim(1), maskLogits.dim(2), 1])
        let scaleH = Float(mh) / Float(logits.dim(1))
        let scaleW = Float(mw) / Float(logits.dim(2))
        let up = Upsample(scaleFactor: .array([scaleH, scaleW]), mode: .linear(alignCorners: false))(logits)
        // 2. mask_for_mem = sigmoid(up) * 2.0 - 1.0 (SAM 3.1 defaults from model_builder.py).
        //    If caller already applied sigmoid, just rescale.
        let probs = alreadySigmoid ? up : sigmoid(up)
        let maskForMem = probs * MLXArray(Float(2.0)) - MLXArray(Float(1.0))

        // 3. Build 32-channel input. Slot 0 = mask, slots 1..15 = 0; chan 16 = cond flag, 17..31 = 0.
        let zerosMask = MLXArray.zeros([1, mh, mw, multiplex - 1], dtype: maskForMem.dtype)
        let condValue: Float = isConditioning ? 1.0 : 0.0
        let condFlag = MLXArray.full([1, mh, mw, 1], values: MLXArray(condValue))
            .asType(maskForMem.dtype)
        let zerosCond = MLXArray.zeros([1, mh, mw, multiplex - 1], dtype: maskForMem.dtype)
        let stacked = concatenated([maskForMem, zerosMask, condFlag, zerosCond], axis: -1)
        // Sanity: should be (1, mh, mw, 32).
        assert(stacked.dim(-1) == totalCh)

        let memory = trackerModel.memoryEncoder(features, stacked)
        // NOTE: python applies a no_obj_embed_spatial correction here:
        //   maskmem += sum_over_inactive_slots(no_obj_embed_spatial[i]) broadcast over HW.
        // Adding it in Swift regressed mean IoU 0.312 → 0.167 on the 10-frame fixture.
        // Mathematically equivalent to python's `((1-is_obj_appearing) * noObj).sum(dim=1)`
        // but somehow worsens parity. Possible causes: (a) python's effective values for
        // is_obj_appearing differ for active slots in our setup; (b) downstream consumers
        // see DOUBLE correction once it's already on memory_attention values. Disabled
        // until properly diagnosed (see Phase 11.5c memory).
        return memory.reshaped([1, -1, d])
    }

    /// Multi-track variant of `encodeMaskToMemory`: encodes up to multiplex_count=16
    /// tracks' masks into ONE shared 32-channel memory entry (the proper multiplex
    /// encoding python/sam3 uses). Slots beyond `maskLogitsList.count` are zeroed.
    ///
    /// - Parameters:
    ///   - features: (1, H_f, W_f, 256) propagation FPN level [2].
    ///   - maskLogitsList: per-slot mask logits, each (1, H, W). Up to 16 slots.
    ///   - isConditioningList: per-slot conditioning flag.
    ///   - alreadySigmoid: pass true if logits are already in [0,1] (post-sigmoid).
    public func encodeMultiplexMasksToMemory(
        features: MLXArray,
        maskLogitsList: [MLXArray],
        isConditioningList: [Bool],
        alreadySigmoid: Bool = false
    ) -> MLXArray {
        let multiplex = 16
        let downsampleStride = 16
        let n = maskLogitsList.count
        precondition(n == isConditioningList.count, "mismatched slot counts")
        precondition(n > 0 && n <= multiplex, "1..16 slots required")
        let hf = features.dim(1), wf = features.dim(2), d = features.dim(3)
        let mh = hf * downsampleStride, mw = wf * downsampleStride

        // Per-slot upsampled mask values: (1, mh, mw, 1) each.
        var slotMasks: [MLXArray] = []
        slotMasks.reserveCapacity(multiplex)
        for logit in maskLogitsList {
            let logit3 = logit.reshaped([1, logit.dim(1), logit.dim(2), 1])
            let scaleH = Float(mh) / Float(logit3.dim(1))
            let scaleW = Float(mw) / Float(logit3.dim(2))
            let up = Upsample(scaleFactor: .array([scaleH, scaleW]), mode: .linear(alignCorners: false))(logit3)
            let probs = alreadySigmoid ? up : sigmoid(up)
            let slot = probs * MLXArray(Float(2.0)) - MLXArray(Float(1.0))
            slotMasks.append(slot)
        }
        // Pad with zeros up to multiplex slots.
        let padCount = multiplex - n
        if padCount > 0 {
            let zeroSlot = MLXArray.zeros([1, mh, mw, 1], dtype: slotMasks[0].dtype)
            for _ in 0..<padCount { slotMasks.append(zeroSlot) }
        }

        // Per-slot conditioning flags: each (1, mh, mw, 1) filled with 1.0 or 0.0.
        var slotConds: [MLXArray] = []
        slotConds.reserveCapacity(multiplex)
        for cond in isConditioningList {
            let v: Float = cond ? 1.0 : 0.0
            slotConds.append(MLXArray.full([1, mh, mw, 1], values: MLXArray(v)).asType(slotMasks[0].dtype))
        }
        if padCount > 0 {
            let zeroCond = MLXArray.zeros([1, mh, mw, 1], dtype: slotMasks[0].dtype)
            for _ in 0..<padCount { slotConds.append(zeroCond) }
        }

        // Concatenate: 16 mask channels followed by 16 condition channels = 32.
        let stacked = concatenated(slotMasks + slotConds, axis: -1)
        precondition(stacked.dim(-1) == multiplex * 2)

        let mem = trackerModel.memoryEncoder(features, stacked)
        // NOTE: no_obj_embed_spatial correction disabled (regresses on 10-frame test).
        // See `encodeMaskToMemory` for context.
        return mem.reshaped([1, -1, d])
    }

    @available(*, deprecated, message: "Buggy single-channel encoding; use encodeMaskToMemory.")
    public func trackInit(
        backboneFeatures: MLXArray,
        detectionMasks: MLXArray
    ) -> (memory: MLXArray, features: MLXArray) {
        let prop = getTrackerFeatures(backboneFeatures)
        let features = prop[2]
        let b = features.dim(0), d = features.dim(3)
        let maskInput = detectionMasks[0..., ..<1].transposed(0, 2, 3, 1)
        let memory = trackerModel.memoryEncoder(features, maskInput)
        let flat = memory.reshaped([b, -1, d])
        return (memory: flat, features: features)
    }

    public func trackStep(
        backboneFeatures: MLXArray,
        memoryBank: [MLXArray]? = nil,
        promptPoints: (MLXArray, MLXArray)? = nil,
        promptBoxes: MLXArray? = nil,
        promptMasks: MLXArray? = nil,
        multimaskOutput: Bool = false
    ) -> [String: MLXArray] {
        let prop = getTrackerFeatures(backboneFeatures)
        let features = prop[2]
        let highRes = prop.count > 1 ? [prop[1], prop[0]] : nil
        return trackerModel.trackStep(
            currentFeatures: features,
            memoryBank: memoryBank,
            promptPoints: promptPoints,
            promptBoxes: promptBoxes,
            promptMasks: promptMasks,
            multimaskOutput: multimaskOutput,
            highResFeatures: highRes
        )
    }

    // MARK: - Sanitize

    /// Remap checkpoint keys to match Swift module parameter paths.
    ///
    /// The mlx-community checkpoint is already in MLX weight format (OHWI for Conv,
    /// OHWI for ConvTranspose), so no tensor transpositions are needed here.
    public static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        let maskEmbedRemap: [String: String] = [
            "mask_embed.0.": "mask_embed.conv1.",
            "mask_embed.1.": "mask_embed.layer_norm1.",
            "mask_embed.3.": "mask_embed.conv2.",
            "mask_embed.4.": "mask_embed.layer_norm2.",
            "mask_embed.6.": "mask_embed.conv3.",
        ]

        var out: [String: MLXArray] = [:]
        out.reserveCapacity(weights.count)

        for (rawKey, value) in weights {
            var key = rawKey.replacingOccurrences(of: "scale_layers.2.", with: "scale_layers.1.")

            for (old, new) in maskEmbedRemap where key.contains(old) {
                key = key.replacingOccurrences(of: old, with: new)
                break
            }

            if key.contains("memory_fuser") && key.contains(".norm.") {
                key = key.replacingOccurrences(of: ".norm.", with: ".layer_norm.")
            }

            if key.contains("mask_downsampler.layers.4.conv.") {
                key = key.replacingOccurrences(of: "mask_downsampler.layers.4.conv.", with: "mask_downsampler.final_conv.")
            }

            out[key] = value
        }
        return out
    }
}

/// Returns true if the module at `path` should be quantized (SAM 3.1).
public func sam31QuantPredicate(path: String, module: Module) -> Bool {
    let skipPatterns = [
        "conv", "depthwise", "mask_downsample", "pixel_decoder",
        "instance_projection", "semantic_projection", "fpn_layers", "patch_embeddings",
        "query_embed", "reference_points", "presence_token", "label_embed", "cls_embed",
        "point_embed", "not_a_point", "no_mask_embed", "no_memory", "no_object",
        "iou_token", "mask_tokens", "obj_score_token", "shared_embedding",
        "shared_image_embedding", "occlusion_spatial", "memory_temporal", "position_embedding",
        "output_valid_embed", "output_invalid_embed", "no_obj_embed_spatial",
        "image_pe_layer", "interactivity_no_mem",
    ]
    if skipPatterns.contains(where: { path.contains($0) }) { return false }
    return true
}
