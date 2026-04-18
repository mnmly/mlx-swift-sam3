// Sam3Model.swift — SAM 3 main model (detector + tracker).
// Python source: sam3/sam3.py
// Weight keys: detector_model.* / tracker_model.* / tracker_neck.*

import MLX
import MLXNN

// MARK: - Detector

public class DetectorModel: Module {
    @ModuleInfo(key: "vision_encoder") var visionEncoder: VisionEncoder
    @ModuleInfo(key: "text_encoder") var textEncoder: TextEncoder
    @ModuleInfo(key: "text_projection") var textProjection: Linear
    @ModuleInfo(key: "detr_encoder") var detrEncoder: DETREncoder
    @ModuleInfo(key: "detr_decoder") var detrDecoder: DETRDecoder
    @ModuleInfo(key: "geometry_encoder") var geometryEncoder: GeometryEncoder
    @ModuleInfo(key: "mask_decoder") var maskDecoder: MaskDecoder
    @ModuleInfo(key: "dot_product_scoring") var dotProductScoring: DotProductScoring

    let posEnc: PositionEmbeddingSine

    public init(_ config: ModelConfig) {
        let det = config.detectorConfig
        let d = det.detrEncoderConfig.hiddenSize
        self._visionEncoder = ModuleInfo(wrappedValue: VisionEncoder(det.visionConfig), key: "vision_encoder")
        self._textEncoder = ModuleInfo(wrappedValue: TextEncoder(det.textConfig, dModel: d), key: "text_encoder")
        self._textProjection = ModuleInfo(wrappedValue: Linear(det.textConfig.hiddenSize, d), key: "text_projection")
        self._detrEncoder = ModuleInfo(wrappedValue: DETREncoder(det.detrEncoderConfig), key: "detr_encoder")
        self._detrDecoder = ModuleInfo(wrappedValue: DETRDecoder(det.detrDecoderConfig), key: "detr_decoder")
        self._geometryEncoder = ModuleInfo(wrappedValue: GeometryEncoder(det.geometryEncoderConfig), key: "geometry_encoder")
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
        let b = pixelValues.dim(0)
        _ = b

        let fpnFeatures = visionEncoder(pixelValues)
        let fpnPos = fpnFeatures.map { posEnc($0) }

        // Drop coarsest level (0.5x): first 3 of 4 levels
        let fpnTrimmed = Array(fpnFeatures.prefix(3))
        let fpnPosTrimmed = Array(fpnPos.prefix(3))

        let embeds: MLXArray
        if let ie = inputsEmbeds {
            embeds = ie
        } else {
            embeds = getInputEmbeddings(inputIds!, attentionMask: attentionMask)
        }

        let encFeat = fpnTrimmed[fpnTrimmed.count - 1]  // coarsest trimmed = 72x72
        let encPos = fpnPosTrimmed[fpnPosTrimmed.count - 1]
        let bEnc = encFeat.dim(0), h = encFeat.dim(1), w = encFeat.dim(2), d = encFeat.dim(3)
        let src = encFeat.reshaped([bEnc, h * w, d])
        let posFlat = encPos.reshaped([bEnc, h * w, d])

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
            backboneFeatures: fpnTrimmed,
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
        ]
    }
}

// MARK: - Main Model

public class Sam3Model: Module {
    @ModuleInfo(key: "detector_model") var detectorModel: DetectorModel
    @ModuleInfo(key: "tracker_model") var trackerModel: TrackerModel
    @ModuleInfo(key: "tracker_neck") var trackerNeck: FPNNeck

    public init(_ config: ModelConfig) {
        self._detectorModel = ModuleInfo(wrappedValue: DetectorModel(config), key: "detector_model")
        self._trackerModel = ModuleInfo(wrappedValue: TrackerModel(config.trackerConfig), key: "tracker_model")
        self._trackerNeck = ModuleInfo(wrappedValue: FPNNeck(config.trackerConfig.visionConfig), key: "tracker_neck")
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

    public func getInputEmbeddings(_ inputIds: MLXArray, attentionMask: MLXArray? = nil) -> (MLXArray, MLXArray?) {
        let emb = detectorModel.getInputEmbeddings(inputIds, attentionMask: attentionMask)
        MLX.eval(emb)
        return (emb, attentionMask)
    }

    public func trackStep(
        backboneFeatures: MLXArray,
        memoryBank: [MLXArray],
        promptPoints: (MLXArray, MLXArray)? = nil,
        promptBoxes: MLXArray? = nil,
        promptMasks: MLXArray? = nil,
        multimaskOutput: Bool = false
    ) -> [String: MLXArray] {
        let fpn = trackerNeck(backboneFeatures)
        let features = fpn[2]
        let highRes = fpn.count > 1 ? [fpn[0], fpn[1]] : nil
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
        var out: [String: MLXArray] = [:]
        out.reserveCapacity(weights.count)

        for (rawKey, value) in weights {
            // scale_layers.2 is the second ConvTranspose (index 1 skipped by GELU placeholder)
            let key = rawKey.replacingOccurrences(of: "scale_layers.2.", with: "scale_layers.1.")
            out[key] = value
        }
        return out
    }
}

/// Returns true if the module at `path` should be quantized.
public func quantPredicate(path: String, module: Module) -> Bool {
    let skipPatterns = [
        "conv", "depthwise", "mask_downsample", "pixel_decoder",
        "instance_projection", "semantic_projection", "fpn_layers", "patch_embeddings",
        "query_embed", "reference_points", "presence_token", "label_embed", "cls_embed",
        "point_embed", "not_a_point", "no_mask_embed", "no_memory", "no_object",
        "iou_token", "mask_tokens", "obj_score_token", "shared_embedding",
        "shared_image_embedding", "occlusion_spatial", "memory_temporal", "position_embedding",
    ]
    if skipPatterns.contains(where: { path.contains($0) }) { return false }
    return true
}
