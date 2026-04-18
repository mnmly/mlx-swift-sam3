// TextEncoder.swift — CLIP text encoder for SAM 3.
// Python source: sam3/text_encoder.py
// Weight keys: detector_model.text_encoder.text_model.* / text_projection.*

import MLX
import MLXFast
import MLXNN

class CLIPAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ config: TextEncoderConfig) {
        self.numHeads = config.numAttentionHeads
        self.headDim = config.hiddenSize / config.numAttentionHeads
        self.scale = 1.0 / Float(config.hiddenSize / config.numAttentionHeads).squareRoot()
        self._qProj = ModuleInfo(wrappedValue: Linear(config.hiddenSize, config.hiddenSize), key: "q_proj")
        self._kProj = ModuleInfo(wrappedValue: Linear(config.hiddenSize, config.hiddenSize), key: "k_proj")
        self._vProj = ModuleInfo(wrappedValue: Linear(config.hiddenSize, config.hiddenSize), key: "v_proj")
        self._outProj = ModuleInfo(wrappedValue: Linear(config.hiddenSize, config.hiddenSize), key: "out_proj")
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let b = x.dim(0), n = x.dim(1), c = x.dim(2)
        let q = qProj(x).reshaped([b, n, numHeads, headDim]).transposed(0, 2, 1, 3)
        let k = kProj(x).reshaped([b, n, numHeads, headDim]).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped([b, n, numHeads, headDim]).transposed(0, 2, 1, 3)
        var out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: mask)
        out = out.transposed(0, 2, 1, 3).reshaped([b, n, c])
        return outProj(out)
    }
}

class CLIPMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(_ config: TextEncoderConfig) {
        self._fc1 = ModuleInfo(wrappedValue: Linear(config.hiddenSize, config.intermediateSize), key: "fc1")
        self._fc2 = ModuleInfo(wrappedValue: Linear(config.intermediateSize, config.hiddenSize), key: "fc2")
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(gelu(fc1(x))) }
}

class CLIPEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: CLIPAttention
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: CLIPMLP
    @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm

    init(_ config: TextEncoderConfig) {
        self._selfAttn = ModuleInfo(wrappedValue: CLIPAttention(config), key: "self_attn")
        self._layerNorm1 = ModuleInfo(wrappedValue: LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps), key: "layer_norm1")
        self._mlp = ModuleInfo(wrappedValue: CLIPMLP(config), key: "mlp")
        self._layerNorm2 = ModuleInfo(wrappedValue: LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps), key: "layer_norm2")
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var out = x + selfAttn(layerNorm1(x), mask: mask)
        out = out + mlp(layerNorm2(out))
        return out
    }
}

class CLIPEncoder: Module {
    @ModuleInfo(key: "layers") var layers: [CLIPEncoderLayer]

    init(_ config: TextEncoderConfig) {
        self._layers = ModuleInfo(
            wrappedValue: (0..<config.numHiddenLayers).map { _ in CLIPEncoderLayer(config) },
            key: "layers"
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var out = x
        for layer in layers { out = layer(out, mask: mask) }
        return out
    }
}

class CLIPTextEmbeddings: Module {
    @ModuleInfo(key: "token_embedding") var tokenEmbedding: Embedding
    @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

    init(_ config: TextEncoderConfig) {
        self._tokenEmbedding = ModuleInfo(wrappedValue: Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize), key: "token_embedding")
        self._positionEmbedding = ModuleInfo(wrappedValue: Embedding(embeddingCount: config.maxPositionEmbeddings, dimensions: config.hiddenSize), key: "position_embedding")
        super.init()
    }

    func callAsFunction(_ inputIds: MLXArray) -> MLXArray {
        let seqLen = inputIds.dim(1)
        let positionIds = MLXArray(0..<seqLen)
        return tokenEmbedding(inputIds) + positionEmbedding(positionIds)
    }
}

class CLIPTextModel: Module {
    @ModuleInfo(key: "embeddings") var embeddings: CLIPTextEmbeddings
    @ModuleInfo(key: "encoder") var encoder: CLIPEncoder
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    let config: TextEncoderConfig

    init(_ config: TextEncoderConfig) {
        self.config = config
        self._embeddings = ModuleInfo(wrappedValue: CLIPTextEmbeddings(config), key: "embeddings")
        self._encoder = ModuleInfo(wrappedValue: CLIPEncoder(config), key: "encoder")
        self._finalLayerNorm = ModuleInfo(wrappedValue: LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps), key: "final_layer_norm")
        super.init()
    }

    /// - Parameter inputIds: (B, seqLen)
    /// - Returns: (B, seqLen, hiddenSize)
    func callAsFunction(_ inputIds: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        var x = embeddings(inputIds)
        let seqLen = inputIds.dim(1)
        var causalMask = MultiHeadAttention.createAdditiveCausalMask(seqLen).asType(x.dtype)
        if let mask = attentionMask {
            let padMask = (1 - mask[0..., .newAxis, .newAxis, 0...].asType(x.dtype)) * (-1e9 as Float)
            causalMask = causalMask + padMask
        }
        x = encoder(x, mask: causalMask)
        return finalLayerNorm(x)
    }
}

public class TextEncoder: Module {
    @ModuleInfo(key: "text_model") var textModel: CLIPTextModel
    @ModuleInfo(key: "text_projection") var textProjection: Linear

    public init(_ config: TextEncoderConfig, dModel: Int = 256) {
        self._textModel = ModuleInfo(wrappedValue: CLIPTextModel(config), key: "text_model")
        self._textProjection = ModuleInfo(wrappedValue: Linear(config.hiddenSize, config.projectionDim, bias: false), key: "text_projection")
        super.init()
    }

    /// - Parameter inputIds: (B, seqLen)
    /// - Returns: (B, seqLen, hiddenSize)
    public func callAsFunction(_ inputIds: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        textModel(inputIds, attentionMask: attentionMask)
    }
}
