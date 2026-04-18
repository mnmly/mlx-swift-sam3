// Segmentation.swift — pixel decoder, mask decoder, and text scoring.
// Python source: sam3/segmentation.py
// Weight keys: detector_model.mask_decoder.* / dot_product_scoring.*

import MLX
import MLXNN

class PixelDecoder: Module {
    @ModuleInfo(key: "conv_layers") var convLayers: [Conv2d]
    @ModuleInfo(key: "norms") var norms: [GroupNorm]

    init(hiddenSize: Int, numUpsamplingStages: Int = 3) {
        self._convLayers = ModuleInfo(
            wrappedValue: (0..<numUpsamplingStages).map { _ in
                Conv2d(inputChannels: hiddenSize, outputChannels: hiddenSize, kernelSize: 3, padding: 1)
            },
            key: "conv_layers"
        )
        self._norms = ModuleInfo(
            wrappedValue: (0..<numUpsamplingStages).map { _ in
                GroupNorm(groupCount: 8, dimensions: hiddenSize)
            },
            key: "norms"
        )
        super.init()
    }

    func callAsFunction(_ features: [MLXArray]) -> MLXArray {
        var x = features[features.count - 1]
        let finerFeatures = features.dropLast().reversed()
        for (i, feat) in finerFeatures.enumerated() {
            let th = feat.dim(1), tw = feat.dim(2)
            x = interpolateNearest(x, targetH: th, targetW: tw)
            x = x + feat
            x = convLayers[i](x)
            x = norms[i](x)
            x = relu(x)
        }
        return x
    }

    private func interpolateNearest(_ x: MLXArray, targetH: Int, targetW: Int) -> MLXArray {
        let h = x.dim(1), w = x.dim(2)
        if h == targetH && w == targetW { return x }
        let sh = Float(targetH) / Float(h)
        let sw = Float(targetW) / Float(w)
        return Upsample(scaleFactor: .array([sh, sw]), mode: .nearest)(x)
    }
}

class MaskEmbedder: Module {
    @ModuleInfo(key: "layers") var layers: [Linear]

    init(hiddenSize: Int) {
        self._layers = ModuleInfo(
            wrappedValue: (0..<3).map { _ in Linear(hiddenSize, hiddenSize) },
            key: "layers"
        )
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

public class MaskDecoder: Module {
    @ModuleInfo(key: "pixel_decoder") var pixelDecoder: PixelDecoder
    @ModuleInfo(key: "mask_embedder") var maskEmbedder: MaskEmbedder
    @ModuleInfo(key: "prompt_cross_attn") var promptCrossAttn: MultiheadAttention
    @ModuleInfo(key: "prompt_cross_attn_norm") var promptCrossAttnNorm: LayerNorm
    @ModuleInfo(key: "semantic_projection") var semanticProjection: Conv2d
    @ModuleInfo(key: "instance_projection") var instanceProjection: Conv2d

    public init(_ config: DetectorMaskDecoderConfig) {
        let d = config.hiddenSize
        self._pixelDecoder = ModuleInfo(wrappedValue: PixelDecoder(hiddenSize: d, numUpsamplingStages: config.numUpsamplingStages), key: "pixel_decoder")
        self._maskEmbedder = ModuleInfo(wrappedValue: MaskEmbedder(hiddenSize: d), key: "mask_embedder")
        self._promptCrossAttn = ModuleInfo(wrappedValue: MultiheadAttention(hiddenSize: d, numHeads: config.numAttentionHeads), key: "prompt_cross_attn")
        self._promptCrossAttnNorm = ModuleInfo(wrappedValue: LayerNorm(dimensions: d, eps: config.layerNormEps), key: "prompt_cross_attn_norm")
        self._semanticProjection = ModuleInfo(wrappedValue: Conv2d(inputChannels: d, outputChannels: 1, kernelSize: 1), key: "semantic_projection")
        self._instanceProjection = ModuleInfo(wrappedValue: Conv2d(inputChannels: d, outputChannels: d, kernelSize: 1), key: "instance_projection")
        super.init()
    }

    public func callAsFunction(
        objQueries: MLXArray,
        backboneFeatures: [MLXArray],
        encoderHiddenStates: MLXArray? = nil,
        promptFeatures: MLXArray? = nil,
        promptMask: MLXArray? = nil
    ) -> (predMasks: MLXArray, semanticSeg: MLXArray) {
        var encStates = encoderHiddenStates
        if let pf = promptFeatures, var es = encStates {
            var crossMask: MLXArray? = nil
            if let pm = promptMask {
                crossMask = (1 - pm[0..., .newAxis, .newAxis, 0...].asType(.float32)) * (-1e9 as Float)
            }
            let normed = promptCrossAttnNorm(es)
            es = es + promptCrossAttn(normed, pf, pf, mask: crossMask)
            encStates = es
        }

        var featsForFPN = backboneFeatures
        if let es = encStates {
            let finest = featsForFPN[featsForFPN.count - 1]
            let b = finest.dim(0), h = finest.dim(1), w = finest.dim(2), d = finest.dim(3)
            let spatialDim = h * w
            let encVisual = es[0..., ..<spatialDim, 0...].reshaped([b, h, w, d])
            featsForFPN[featsForFPN.count - 1] = encVisual
        }

        let pixelEmbed = pixelDecoder(featsForFPN)
        let instanceEmbed = instanceProjection(pixelEmbed)

        let maskEmbeddings = maskEmbedder(objQueries)  // (B, Q, D)
        let b = instanceEmbed.dim(0), h = instanceEmbed.dim(1), w = instanceEmbed.dim(2), d = instanceEmbed.dim(3)
        let instFlat = instanceEmbed.reshaped([b, h * w, d]).transposed(0, 2, 1)  // (B, D, HW)
        let predMasks = matmul(maskEmbeddings, instFlat).reshaped([b, -1, h, w])

        var semanticSeg = semanticProjection(pixelEmbed)  // (B, H, W, 1)
        semanticSeg = semanticSeg.transposed(0, 3, 1, 2)  // (B, 1, H, W)

        return (predMasks, semanticSeg)
    }
}

class TextScoringMLP: Module {
    @ModuleInfo(key: "layer1") var layer1: Linear
    @ModuleInfo(key: "layer2") var layer2: Linear

    init(hiddenSize: Int) {
        self._layer1 = ModuleInfo(wrappedValue: Linear(hiddenSize, hiddenSize * 8), key: "layer1")
        self._layer2 = ModuleInfo(wrappedValue: Linear(hiddenSize * 8, hiddenSize), key: "layer2")
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { layer2(relu(layer1(x))) }
}

public class DotProductScoring: Module {
    @ModuleInfo(key: "query_proj") var queryProj: Linear
    @ModuleInfo(key: "text_proj") var textProj: Linear
    @ModuleInfo(key: "text_mlp") var textMlp: TextScoringMLP
    @ModuleInfo(key: "text_mlp_out_norm") var textMlpOutNorm: LayerNorm

    let scale: Float
    let clampMaxVal: Float = 12.0

    public init(hiddenSize: Int) {
        self.scale = 1.0 / Float(hiddenSize).squareRoot()
        self._queryProj = ModuleInfo(wrappedValue: Linear(hiddenSize, hiddenSize), key: "query_proj")
        self._textProj = ModuleInfo(wrappedValue: Linear(hiddenSize, hiddenSize), key: "text_proj")
        self._textMlp = ModuleInfo(wrappedValue: TextScoringMLP(hiddenSize: hiddenSize), key: "text_mlp")
        self._textMlpOutNorm = ModuleInfo(wrappedValue: LayerNorm(dimensions: hiddenSize), key: "text_mlp_out_norm")
        super.init()
    }

    /// - Parameters:
    ///   - hs: (L, B, Q, D)
    ///   - inputsEmbeds: (B, T, D)
    ///   - textMask: (B, T) 1=valid
    /// - Returns: (L, B, Q, 1)
    public func callAsFunction(_ hs: MLXArray, inputsEmbeds: MLXArray, textMask: MLXArray? = nil) -> MLXArray {
        let textProcessed = textMlpOutNorm(textMlp(inputsEmbeds) + inputsEmbeds)

        let pooledText: MLXArray
        if let tm = textMask {
            let isValid = tm[.ellipsis, .newAxis].asType(.float32)
            let numValid = MLX.maximum(isValid.sum(axes: [1]), MLXArray(1.0))
            pooledText = (textProcessed * isValid).sum(axes: [1]) / numValid
        } else {
            pooledText = textProcessed.mean(axes: [1])
        }

        let projText = textProj(pooledText)          // (B, D)
        let projQueries = queryProj(hs)              // (L, B, Q, D)
        let scores = matmul(projQueries, projText[.newAxis, 0..., 0..., .newAxis]) * scale
        return MLX.clip(scores, min: -clampMaxVal, max: clampMaxVal)
    }
}
