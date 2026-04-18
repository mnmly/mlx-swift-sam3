// Sam31Model.swift — SAM 3.1 main model.
// Python source: sam3_1/sam3_1.py
// Weight keys: detector_model.* / tracker_model.*

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
        let highRes = prop.count > 2 ? [prop[0], prop[1]] : nil
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
        let highRes = prop.count > 1 ? [prop[0], prop[1]] : nil
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
