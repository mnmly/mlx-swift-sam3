// Sam31Tracker.swift — SAM 3.1 multiplex tracker components.
// Python source: sam3_1/tracker.py
// Weight keys: tracker_model.*

import Foundation
import MLX
import MLXNN

// MARK: - MultiplexMaskDownSampler

class MultiplexMaskDownSampler: Module {
    @ModuleInfo(key: "layers") var layers: [DownsampleConvBlock]
    @ModuleInfo(key: "final_conv") var finalConv: Conv2d

    init(_ config: Sam31TrackerConfig) {
        let ks = config.maskDownsamplerKernelSize
        let st = config.maskDownsamplerStride
        let pd = config.maskDownsamplerPadding
        let embedDim = config.maskDownsamplerEmbedDim
        let first = config.maskDownsamplerFirstChannels  // 16
        let channels = [first, first * 4, first * 16, first * 64]
        var builtLayers: [DownsampleConvBlock] = []
        var inCh = first * 2  // 32 input channels (multiplex_count * 2)
        for outCh in channels {
            builtLayers.append(DownsampleConvBlock(inCh: inCh, outCh: outCh, kernelSize: ks, stride: st, padding: pd))
            inCh = outCh
        }
        self._layers = ModuleInfo(wrappedValue: builtLayers, key: "layers")
        self._finalConv = ModuleInfo(wrappedValue: Conv2d(inputChannels: channels.last!, outputChannels: embedDim, kernelSize: 1, bias: true), key: "final_conv")
        super.init()
    }

    func callAsFunction(_ masks: MLXArray) -> MLXArray {
        var x = masks
        for layer in layers { x = layer(x) }
        return finalConv(x)
    }
}

// MARK: - MultiplexMemoryEncoder

public class MultiplexMemoryEncoder: Module {
    @ModuleInfo(key: "mask_downsampler") var maskDownsampler: MultiplexMaskDownSampler
    @ModuleInfo(key: "memory_fuser") var memoryFuser: MemoryFuser
    @ModuleInfo(key: "feature_projection") var featureProjection: Conv2d

    public init(_ config: Sam31TrackerConfig) {
        let dim = config.memoryEncoderHiddenSize
        self._maskDownsampler = ModuleInfo(wrappedValue: MultiplexMaskDownSampler(config), key: "mask_downsampler")
        self._memoryFuser = ModuleInfo(wrappedValue: MemoryFuser(config), key: "memory_fuser")
        self._featureProjection = ModuleInfo(wrappedValue: Conv2d(inputChannels: dim, outputChannels: dim, kernelSize: 1, bias: true), key: "feature_projection")
        super.init()
    }

    public func callAsFunction(_ features: MLXArray, _ masks: MLXArray) -> MLXArray {
        let maskFeatures = maskDownsampler(masks)
        let fused = featureProjection(features) + maskFeatures
        return memoryFuser(fused)
    }
}

// MARK: - Sam31ObjectPointerMLP

class Sam31ObjectPointerMLP: Module {
    @ModuleInfo(key: "layers") var layers: [Linear]

    init(hiddenSize: Int) {
        self._layers = ModuleInfo(wrappedValue: [
            Linear(hiddenSize, hiddenSize),
            Linear(hiddenSize, hiddenSize),
            Linear(hiddenSize, hiddenSize),
        ], key: "layers")
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x
        for (i, layer) in layers.enumerated() {
            out = layer(out)
            if i < layers.count - 1 { out = relu(out) }
        }
        return out
    }
}

// MARK: - MultiplexTrackerModel

public class MultiplexTrackerModel: Module {
    @ModuleInfo(key: "interactive_sam_prompt_encoder") var interactiveSamPromptEncoder: SAMPromptEncoder
    @ModuleInfo(key: "interactive_sam_mask_decoder") var interactiveSamMaskDecoder: MultiplexMaskDecoder
    @ModuleInfo(key: "sam_mask_decoder") var samMaskDecoder: MultiplexMaskDecoder
    @ModuleInfo(key: "memory_attention") var memoryAttention: DecoupledMemoryAttention
    @ModuleInfo(key: "memory_encoder") var memoryEncoder: MultiplexMemoryEncoder
    @ModuleInfo(key: "obj_ptr_proj") var objPtrProj: Sam31ObjectPointerMLP
    @ModuleInfo(key: "interactive_obj_ptr_proj") var interactiveObjPtrProj: Sam31ObjectPointerMLP
    @ModuleInfo(key: "temporal_positional_encoding_projection_layer") var temporalPosEncProjLayer: Linear
    @ModuleInfo(key: "no_obj_ptr_linear") var noObjPtrLinear: Linear
    @ModuleInfo(key: "image_pe_layer") var imagePeLayer: PositionalEmbedding
    @ModuleInfo(key: "shared_image_embedding") var sharedImageEmbedding: PositionalEmbedding
    @ModuleInfo(key: "interactive_mask_downsample") var interactiveMaskDownsample: Conv2d

    @ParameterInfo(key: "memory_temporal_positional_encoding") var memoryTemporalPositionalEncoding: MLXArray
    @ParameterInfo(key: "output_valid_embed") var outputValidEmbed: MLXArray
    @ParameterInfo(key: "output_invalid_embed") var outputInvalidEmbed: MLXArray
    @ParameterInfo(key: "no_obj_embed_spatial") var noObjEmbedSpatial: MLXArray
    @ParameterInfo(key: "interactivity_no_mem_embed") var interactivityNoMemEmbed: MLXArray

    public init(_ config: Sam31TrackerConfig) {
        let d = config.memoryAttentionHiddenSize
        let M = config.multiplexCount

        self._interactiveSamPromptEncoder = ModuleInfo(wrappedValue: SAMPromptEncoder(config.promptEncoderConfig), key: "interactive_sam_prompt_encoder")

        var interactiveCfg = config.maskDecoderConfig
        interactiveCfg.multiplexCount = 1
        interactiveCfg.numMultimaskOutputs = 4
        self._interactiveSamMaskDecoder = ModuleInfo(wrappedValue: MultiplexMaskDecoder(interactiveCfg), key: "interactive_sam_mask_decoder")
        self._samMaskDecoder = ModuleInfo(wrappedValue: MultiplexMaskDecoder(config.maskDecoderConfig), key: "sam_mask_decoder")

        self._memoryAttention = ModuleInfo(wrappedValue: DecoupledMemoryAttention(config), key: "memory_attention")
        self._memoryEncoder = ModuleInfo(wrappedValue: MultiplexMemoryEncoder(config), key: "memory_encoder")
        self._objPtrProj = ModuleInfo(wrappedValue: Sam31ObjectPointerMLP(hiddenSize: d), key: "obj_ptr_proj")
        self._interactiveObjPtrProj = ModuleInfo(wrappedValue: Sam31ObjectPointerMLP(hiddenSize: d), key: "interactive_obj_ptr_proj")

        self._temporalPosEncProjLayer = ModuleInfo(wrappedValue: Linear(d, d), key: "temporal_positional_encoding_projection_layer")
        self._noObjPtrLinear = ModuleInfo(wrappedValue: Linear(d, d), key: "no_obj_ptr_linear")
        self._imagePeLayer = ModuleInfo(wrappedValue: PositionalEmbedding(numPosFeats: d / 2), key: "image_pe_layer")
        self._sharedImageEmbedding = ModuleInfo(wrappedValue: PositionalEmbedding(numPosFeats: d / 2), key: "shared_image_embedding")
        self._interactiveMaskDownsample = ModuleInfo(wrappedValue: Conv2d(inputChannels: 1, outputChannels: 1, kernelSize: .init(4), stride: .init(4), bias: true), key: "interactive_mask_downsample")

        self._memoryTemporalPositionalEncoding = ParameterInfo(wrappedValue: MLXArray.zeros([config.numMaskmem, 1, 1, d]), key: "memory_temporal_positional_encoding")
        self._outputValidEmbed = ParameterInfo(wrappedValue: MLXArray.zeros([M, d]), key: "output_valid_embed")
        self._outputInvalidEmbed = ParameterInfo(wrappedValue: MLXArray.zeros([M, d]), key: "output_invalid_embed")
        self._noObjEmbedSpatial = ParameterInfo(wrappedValue: MLXArray.zeros([M, d]), key: "no_obj_embed_spatial")
        self._interactivityNoMemEmbed = ParameterInfo(wrappedValue: MLXArray.zeros([1, 1, d]), key: "interactivity_no_mem_embed")
        super.init()
    }

    public func trackStep(
        currentFeatures: MLXArray,
        memoryBank: [MLXArray]? = nil,
        promptPoints: (MLXArray, MLXArray)? = nil,
        promptBoxes: MLXArray? = nil,
        promptMasks: MLXArray? = nil,
        multimaskOutput: Bool = false,
        highResFeatures: [MLXArray]? = nil
    ) -> [String: MLXArray] {
        let b = currentFeatures.dim(0)
        let h = currentFeatures.dim(1), w = currentFeatures.dim(2), d = currentFeatures.dim(3)
        var src = currentFeatures.reshaped([b, h * w, d])

        if let bank = memoryBank, !bank.isEmpty {
            let memory = concatenated(bank, axis: 1)
            src = memoryAttention(src, memory: memory)
        }

        let imagePE: MLXArray
        let basePE = interactiveSamPromptEncoder.getDensePE()
        let peLen = basePE.dim(1)
        if peLen != h * w {
            let peSide = Int(Double(peLen).squareRoot())
            let peSpatial = basePE.reshaped([1, peSide, peSide, d])
            let peCropped = peSpatial[0..., ..<h, ..<w, 0...]
            imagePE = broadcast(peCropped, to: [b, h, w, d]).reshaped([b, h * w, d])
        } else {
            imagePE = broadcast(basePE, to: [b, h * w, d])
        }

        let (sparseEmb, denseEmb) = interactiveSamPromptEncoder(points: promptPoints, boxes: promptBoxes, masks: promptMasks)

        let (masks, iouPred, samTokens, objScore) = samMaskDecoder(
            imageEmbeddings: src,
            imagePE: imagePE,
            sparsePromptEmbeddings: sparseEmb,
            densePromptEmbeddings: denseEmb,
            multimaskOutput: multimaskOutput,
            highResFeatures: highResFeatures
        )

        let objPtr = objPtrProj(samTokens[0..., 0])

        var outMasks = masks, outIou = iouPred, outObj = objScore
        if masks.ndim == 5 {
            outMasks = masks[0..., 0]
            outIou = iouPred[0..., 0]
            outObj = objScore[0..., 0]
        }

        return [
            "pred_masks": outMasks,
            "iou_scores": outIou,
            "obj_scores": outObj,
            "object_pointer": objPtr,
        ]
    }
}
