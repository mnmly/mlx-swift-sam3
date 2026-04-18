// Tracker.swift — SAM2-style memory tracker for SAM 3.
// Python source: sam3/tracker.py
// Weight keys: tracker_model.*

import Foundation
import MLX
import MLXNN

// MARK: - Mask downsampler

class DownsampleConvBlock: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm2d

    init(inCh: Int, outCh: Int, kernelSize: Int, stride: Int, padding: Int) {
        self._conv = ModuleInfo(wrappedValue: Conv2d(inputChannels: inCh, outputChannels: outCh, kernelSize: .init(kernelSize), stride: .init(stride), padding: .init(padding)), key: "conv")
        self._layerNorm = ModuleInfo(wrappedValue: LayerNorm2d(numChannels: outCh), key: "layer_norm")
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { gelu(layerNorm(conv(x))) }
}

class SimpleMaskDownSampler: Module {
    @ModuleInfo(key: "layers") var layers: [DownsampleConvBlock]
    @ModuleInfo(key: "final_conv") var finalConv: Conv2d

    init(_ config: TrackerConfig) {
        let ks = config.maskDownsamplerKernelSize
        let st = config.maskDownsamplerStride
        let pd = config.maskDownsamplerPadding
        let ed = config.maskDownsamplerEmbedDim
        let channels = [1, 4, 16, 64, ed]
        let blocks = (0..<4).map { i in
            DownsampleConvBlock(inCh: channels[i], outCh: channels[i + 1], kernelSize: ks, stride: st, padding: pd)
        }
        self._layers = ModuleInfo(wrappedValue: blocks, key: "layers")
        self._finalConv = ModuleInfo(wrappedValue: Conv2d(inputChannels: ed, outputChannels: ed, kernelSize: 1, bias: true), key: "final_conv")
        super.init()
    }

    func callAsFunction(_ masks: MLXArray) -> MLXArray {
        var x = masks
        for layer in layers { x = layer(x) }
        return finalConv(x)
    }
}

// MARK: - CXBlock (ConvNeXt-style)

class CXBlock: Module {
    @ModuleInfo(key: "depthwise_conv") var depthwiseConv: Conv2d
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm2d
    @ModuleInfo(key: "pointwise_conv1") var pointwiseConv1: Linear
    @ModuleInfo(key: "pointwise_conv2") var pointwiseConv2: Linear
    @ParameterInfo(key: "scale") var scale: MLXArray

    init(_ config: any MemoryFuserConfigProtocol) {
        let dim = config.memoryFuserEmbedDim
        let ks = config.memoryFuserKernelSize
        let pd = config.memoryFuserPadding
        let intDim = config.memoryFuserIntermediateDim
        self._depthwiseConv = ModuleInfo(wrappedValue: Conv2d(inputChannels: dim, outputChannels: dim, kernelSize: .init(ks), padding: .init(pd), groups: dim), key: "depthwise_conv")
        self._layerNorm = ModuleInfo(wrappedValue: LayerNorm2d(numChannels: dim), key: "layer_norm")
        self._pointwiseConv1 = ModuleInfo(wrappedValue: Linear(dim, intDim), key: "pointwise_conv1")
        self._pointwiseConv2 = ModuleInfo(wrappedValue: Linear(intDim, dim), key: "pointwise_conv2")
        self._scale = ParameterInfo(wrappedValue: MLXArray.ones([dim]) * config.memoryFuserLayerScaleInitValue, key: "scale")
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var out = layerNorm(depthwiseConv(x))
        let b = out.dim(0), h = out.dim(1), w = out.dim(2), c = out.dim(3)
        out = out.reshaped([b * h * w, c])
        out = gelu(pointwiseConv1(out))
        out = pointwiseConv2(out)
        out = out.reshaped([b, h, w, c])
        return residual + scale * out
    }
}

class MemoryFuser: Module {
    @ModuleInfo(key: "layers") var layers: [CXBlock]

    init(_ config: any MemoryFuserConfigProtocol) {
        self._layers = ModuleInfo(wrappedValue: (0..<config.memoryFuserNumLayers).map { _ in CXBlock(config) }, key: "layers")
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x
        for layer in layers { out = layer(out) }
        return out
    }
}

public class MemoryEncoder: Module {
    @ModuleInfo(key: "mask_downsampler") var maskDownsampler: SimpleMaskDownSampler
    @ModuleInfo(key: "memory_fuser") var memoryFuser: MemoryFuser
    @ModuleInfo(key: "feature_projection") var featureProjection: Conv2d
    @ModuleInfo(key: "projection") var projection: Conv2d

    public init(_ config: TrackerConfig) {
        let dim = config.memoryEncoderHiddenSize
        let outDim = config.memoryEncoderOutputChannels
        self._maskDownsampler = ModuleInfo(wrappedValue: SimpleMaskDownSampler(config), key: "mask_downsampler")
        self._memoryFuser = ModuleInfo(wrappedValue: MemoryFuser(config), key: "memory_fuser")
        self._featureProjection = ModuleInfo(wrappedValue: Conv2d(inputChannels: dim, outputChannels: dim, kernelSize: 1, bias: true), key: "feature_projection")
        self._projection = ModuleInfo(wrappedValue: Conv2d(inputChannels: dim, outputChannels: outDim, kernelSize: 1, bias: true), key: "projection")
        super.init()
    }

    public func callAsFunction(_ features: MLXArray, _ masks: MLXArray) -> MLXArray {
        let maskFeatures = maskDownsampler(masks)
        let fused = memoryFuser(featureProjection(features) + maskFeatures)
        return projection(fused)
    }
}

// MARK: - Memory Attention

class MemoryAttentionLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: RoPEAttention
    @ModuleInfo(key: "cross_attn_image") var crossAttnImage: RoPEAttention
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm
    @ModuleInfo(key: "layer_norm3") var layerNorm3: LayerNorm
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear

    init(_ config: TrackerConfig) {
        let d = config.memoryAttentionHiddenSize
        let nh = config.memoryAttentionNumAttentionHeads
        let dr = config.memoryAttentionDownsampleRate
        let fs = (config.memoryAttentionRopeFeatSizes[0], config.memoryAttentionRopeFeatSizes[1])
        let rt = config.memoryAttentionRopeTheta
        let memDim = config.memoryEncoderOutputChannels
        self._selfAttn = ModuleInfo(wrappedValue: RoPEAttention(hiddenSize: d, numHeads: nh, downsampleRate: dr, featSizes: fs, ropeTheta: rt), key: "self_attn")
        self._crossAttnImage = ModuleInfo(wrappedValue: RoPEAttention(hiddenSize: d, numHeads: nh, downsampleRate: dr, featSizes: fs, ropeTheta: rt, kvDim: memDim, ropeKRepeat: true), key: "cross_attn_image")
        self._layerNorm1 = ModuleInfo(wrappedValue: LayerNorm(dimensions: d), key: "layer_norm1")
        self._layerNorm2 = ModuleInfo(wrappedValue: LayerNorm(dimensions: d), key: "layer_norm2")
        self._layerNorm3 = ModuleInfo(wrappedValue: LayerNorm(dimensions: d), key: "layer_norm3")
        self._linear1 = ModuleInfo(wrappedValue: Linear(d, config.memoryAttentionFeedForwardHiddenSize), key: "linear1")
        self._linear2 = ModuleInfo(wrappedValue: Linear(config.memoryAttentionFeedForwardHiddenSize, d), key: "linear2")
        super.init()
    }

    func callAsFunction(_ src: MLXArray, memory: MLXArray) -> MLXArray {
        var out = layerNorm1(src + selfAttn(src, src, src))
        out = layerNorm2(out + crossAttnImage(out, memory, memory))
        out = layerNorm3(out + linear2(relu(linear1(out))))
        return out
    }
}

public class MemoryAttention: Module {
    @ModuleInfo(key: "layers") var layers: [MemoryAttentionLayer]
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm

    public init(_ config: TrackerConfig) {
        let d = config.memoryAttentionHiddenSize
        self._layers = ModuleInfo(wrappedValue: (0..<config.memoryAttentionNumLayers).map { _ in MemoryAttentionLayer(config) }, key: "layers")
        self._layerNorm = ModuleInfo(wrappedValue: LayerNorm(dimensions: d), key: "layer_norm")
        super.init()
    }

    public func callAsFunction(_ src: MLXArray, memory: MLXArray) -> MLXArray {
        var out = src
        for layer in layers { out = layer(out, memory: memory) }
        return layerNorm(out)
    }
}

// MARK: - Object Pointer (SAM 3: 2 layers)

class ObjectPointerMLP: Module {
    @ModuleInfo(key: "proj_in") var projIn: Linear
    @ModuleInfo(key: "layers") var layers: [Linear]
    @ModuleInfo(key: "proj_out") var projOut: Linear

    init(hiddenSize: Int) {
        self._projIn = ModuleInfo(wrappedValue: Linear(hiddenSize, hiddenSize), key: "proj_in")
        self._layers = ModuleInfo(wrappedValue: [Linear(hiddenSize, hiddenSize)], key: "layers")
        self._projOut = ModuleInfo(wrappedValue: Linear(hiddenSize, hiddenSize), key: "proj_out")
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = relu(projIn(x))
        for layer in layers { out = relu(layer(out)) }
        return projOut(out)
    }
}

// MARK: - Shared Image Embedding

class SharedImageEmbedding: Module {
    @ParameterInfo(key: "positional_embedding") var positionalEmbedding: MLXArray

    init(numPosFeats: Int = 128) {
        self._positionalEmbedding = ParameterInfo(
            wrappedValue: MLXArray.zeros([2, numPosFeats]),
            key: "positional_embedding"
        )
        super.init()
    }

    func callAsFunction(_ size: (Int, Int)) -> MLXArray {
        let (h, w) = size
        let gridY = MLXArray(0..<h).asType(.float32) / Float(h)
        let gridX = MLXArray(0..<w).asType(.float32) / Float(w)
        let gy = MLX.tiled(gridY.reshaped([h, 1]), repetitions: [1, w]).reshaped([-1])
        let gx = MLX.tiled(gridX.reshaped([1, w]), repetitions: [h, 1]).reshaped([-1])
        let coords = stacked([gx, gy], axis: -1)  // (H*W, 2)
        let scaled = (coords * 2 - 1).matmul(positionalEmbedding) * Float(2.0 * Double.pi)
        return concatenated([MLX.sin(scaled), MLX.cos(scaled)], axis: -1)
    }
}

// MARK: - Tracker Model

public class TrackerModel: Module {
    @ModuleInfo(key: "prompt_encoder") var promptEncoder: SAMPromptEncoder
    @ModuleInfo(key: "mask_decoder") var maskDecoder: SAMMaskDecoder
    @ModuleInfo(key: "memory_attention") var memoryAttention: MemoryAttention
    @ModuleInfo(key: "memory_encoder") var memoryEncoder: MemoryEncoder
    @ModuleInfo(key: "object_pointer_proj") var objectPointerProj: ObjectPointerMLP
    @ModuleInfo(key: "mask_downsample") var maskDownsample: Conv2d
    @ModuleInfo(key: "temporal_positional_encoding_projection_layer") var temporalPosEncProj: Linear
    @ModuleInfo(key: "shared_image_embedding") var sharedImageEmbedding: SharedImageEmbedding

    @ParameterInfo(key: "no_memory_embedding") var noMemoryEmbedding: MLXArray
    @ParameterInfo(key: "no_memory_positional_encoding") var noMemoryPositionalEncoding: MLXArray
    @ParameterInfo(key: "no_object_pointer") var noObjectPointer: MLXArray
    @ParameterInfo(key: "memory_temporal_positional_encoding") var memoryTemporalPositionalEncoding: MLXArray
    @ParameterInfo(key: "occlusion_spatial_embedding_parameter") var occlusionSpatialEmbeddingParameter: MLXArray

    let config: TrackerConfig

    public init(_ config: TrackerConfig) {
        self.config = config
        let d = config.memoryAttentionHiddenSize
        let memDim = config.memoryEncoderOutputChannels

        self._promptEncoder = ModuleInfo(wrappedValue: SAMPromptEncoder(config.promptEncoderConfig), key: "prompt_encoder")
        self._maskDecoder = ModuleInfo(wrappedValue: SAMMaskDecoder(config.maskDecoderConfig), key: "mask_decoder")
        self._memoryAttention = ModuleInfo(wrappedValue: MemoryAttention(config), key: "memory_attention")
        self._memoryEncoder = ModuleInfo(wrappedValue: MemoryEncoder(config), key: "memory_encoder")
        self._objectPointerProj = ModuleInfo(wrappedValue: ObjectPointerMLP(hiddenSize: d), key: "object_pointer_proj")
        self._maskDownsample = ModuleInfo(wrappedValue: Conv2d(inputChannels: 1, outputChannels: 1, kernelSize: 4, stride: 4, bias: true), key: "mask_downsample")
        self._temporalPosEncProj = ModuleInfo(wrappedValue: Linear(d, memDim), key: "temporal_positional_encoding_projection_layer")
        self._sharedImageEmbedding = ModuleInfo(wrappedValue: SharedImageEmbedding(numPosFeats: d / 2), key: "shared_image_embedding")

        self._noMemoryEmbedding = ParameterInfo(wrappedValue: MLXArray.zeros([1, 1, d]), key: "no_memory_embedding")
        self._noMemoryPositionalEncoding = ParameterInfo(wrappedValue: MLXArray.zeros([1, 1, d]), key: "no_memory_positional_encoding")
        self._noObjectPointer = ParameterInfo(wrappedValue: MLXArray.zeros([1, d]), key: "no_object_pointer")
        self._memoryTemporalPositionalEncoding = ParameterInfo(wrappedValue: MLXArray.zeros([config.numMaskmem, 1, 1, memDim]), key: "memory_temporal_positional_encoding")
        self._occlusionSpatialEmbeddingParameter = ParameterInfo(wrappedValue: MLXArray.zeros([1, memDim]), key: "occlusion_spatial_embedding_parameter")
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
        let b = currentFeatures.dim(0), h = currentFeatures.dim(1), w = currentFeatures.dim(2), d = currentFeatures.dim(3)
        var src = currentFeatures.reshaped([b, h * w, d])

        if let mb = memoryBank, !mb.isEmpty {
            let memory = concatenated(mb, axis: 1)
            src = memoryAttention(src, memory: memory)
        }

        let imagePE = broadcast(promptEncoder.getDensePE(), to: [b, h * w, d])
        let (sparseEmb, denseEmb) = promptEncoder(points: promptPoints, boxes: promptBoxes, masks: promptMasks)
        let (masks, iouPred, samTokens, objScore) = maskDecoder(
            imageEmbeddings: src, imagePE: imagePE,
            sparsePromptEmbeddings: sparseEmb, densePromptEmbeddings: denseEmb,
            multimaskOutput: multimaskOutput, highResFeatures: highResFeatures
        )

        let objPtr = objectPointerProj(samTokens[0..., 0, 0...])

        var maskForMem = masks[0..., 0...0, 0..., 0...].transposed(0, 2, 3, 1)  // (B, H, W, 1)
        let target = 1152
        if maskForMem.dim(1) != target {
            let sh = Float(target) / Float(maskForMem.dim(1))
            let sw = Float(target) / Float(maskForMem.dim(2))
            maskForMem = Upsample(scaleFactor: .array([sh, sw]), mode: .nearest)(maskForMem)
        }

        let mem = memoryEncoder(currentFeatures, maskForMem)
        let bm = mem.dim(0), hm = mem.dim(1), wm = mem.dim(2), cm = mem.dim(3)
        let memFlat = mem.reshaped([bm, hm * wm, cm])

        return [
            "pred_masks": masks,
            "iou_scores": iouPred,
            "obj_scores": objScore,
            "object_pointer": objPtr,
            "memory": memFlat,
        ]
    }
}
