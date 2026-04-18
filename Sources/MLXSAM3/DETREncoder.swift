// DETREncoder.swift — DETR encoder with text cross-attention + MLP helper.
// Python source: sam3/encoder.py
// Weight keys: detector_model.detr_encoder.layers.*
// MLP and MultiheadAttention are also used by DETRDecoder and GeometryEncoder.

import MLX
import MLXFast
import MLXNN

// MARK: - Shared primitives

/// Standard multi-head attention (query may differ from key/value source).
class MultiheadAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(hiddenSize: Int, numHeads: Int, dropout: Float = 0.0, kvDim: Int? = nil) {
        self.numHeads = numHeads
        self.headDim = hiddenSize / numHeads
        self.scale = 1.0 / Float(hiddenSize / numHeads).squareRoot()
        let kd = kvDim ?? hiddenSize
        self._qProj = ModuleInfo(wrappedValue: Linear(hiddenSize, hiddenSize), key: "q_proj")
        self._kProj = ModuleInfo(wrappedValue: Linear(kd, hiddenSize), key: "k_proj")
        self._vProj = ModuleInfo(wrappedValue: Linear(kd, hiddenSize), key: "v_proj")
        self._oProj = ModuleInfo(wrappedValue: Linear(hiddenSize, hiddenSize), key: "o_proj")
        super.init()
    }

    func callAsFunction(_ query: MLXArray, _ key: MLXArray, _ value: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let b = query.dim(0), nq = query.dim(1), nk = key.dim(1)
        let q = qProj(query).reshaped([b, nq, numHeads, headDim]).transposed(0, 2, 1, 3)
        let k = kProj(key).reshaped([b, nk, numHeads, headDim]).transposed(0, 2, 1, 3)
        let v = vProj(value).reshaped([b, nk, numHeads, headDim]).transposed(0, 2, 1, 3)
        var out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: mask)
        out = out.transposed(0, 2, 1, 3).reshaped([b, nq, -1])
        return oProj(out)
    }
}

/// Two-layer MLP with configurable activation (relu or gelu).
class MLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    let act: String

    init(hiddenSize: Int, intermediateSize: Int, act: String = "relu") {
        self.act = act
        self._fc1 = ModuleInfo(wrappedValue: Linear(hiddenSize, intermediateSize), key: "fc1")
        self._fc2 = ModuleInfo(wrappedValue: Linear(intermediateSize, hiddenSize), key: "fc2")
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let hidden = act == "relu" ? relu(fc1(x)) : gelu(fc1(x))
        return fc2(hidden)
    }
}

// MARK: - DETR Encoder

class DETREncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: MultiheadAttention
    @ModuleInfo(key: "cross_attn") var crossAttn: MultiheadAttention
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm
    @ModuleInfo(key: "layer_norm3") var layerNorm3: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: MLP

    init(_ config: DETREncoderConfig) {
        let d = config.hiddenSize
        self._selfAttn = ModuleInfo(wrappedValue: MultiheadAttention(hiddenSize: d, numHeads: config.numAttentionHeads, dropout: config.dropout), key: "self_attn")
        self._crossAttn = ModuleInfo(wrappedValue: MultiheadAttention(hiddenSize: d, numHeads: config.numAttentionHeads, dropout: config.dropout), key: "cross_attn")
        self._layerNorm1 = ModuleInfo(wrappedValue: LayerNorm(dimensions: d, eps: config.layerNormEps), key: "layer_norm1")
        self._layerNorm2 = ModuleInfo(wrappedValue: LayerNorm(dimensions: d, eps: config.layerNormEps), key: "layer_norm2")
        self._layerNorm3 = ModuleInfo(wrappedValue: LayerNorm(dimensions: d, eps: config.layerNormEps), key: "layer_norm3")
        self._mlp = ModuleInfo(wrappedValue: MLP(hiddenSize: d, intermediateSize: config.intermediateSize, act: config.hiddenAct), key: "mlp")
        super.init()
    }

    func callAsFunction(_ src: MLXArray, pos: MLXArray, textMemory: MLXArray, textMask: MLXArray? = nil) -> MLXArray {
        // 1. Self-attention (pre-norm, pos added to q/k)
        let h1 = layerNorm1(src)
        let hPos = h1 + pos
        var out = src + selfAttn(hPos, hPos, h1)

        // 2. Cross-attention to text (pre-norm)
        var crossMask: MLXArray? = nil
        if let tm = textMask {
            crossMask = (1 - tm[0..., .newAxis, .newAxis, 0...].asType(out.dtype)) * (-1e9 as Float)
        }
        let h2 = layerNorm2(out)
        out = out + crossAttn(h2, textMemory, textMemory, mask: crossMask)

        // 3. FFN (pre-norm)
        out = out + mlp(layerNorm3(out))
        return out
    }
}

public class DETREncoder: Module {
    @ModuleInfo(key: "layers") var layers: [DETREncoderLayer]

    public init(_ config: DETREncoderConfig) {
        self._layers = ModuleInfo(
            wrappedValue: (0..<config.numLayers).map { _ in DETREncoderLayer(config) },
            key: "layers"
        )
        super.init()
    }

    /// - Parameters:
    ///   - src: (B, HW, D)
    ///   - pos: (B, HW, D)
    ///   - textMemory: (B, T, D)
    ///   - textMask: (B, T) optional padding mask
    /// - Returns: (B, HW, D)
    public func callAsFunction(_ src: MLXArray, pos: MLXArray, textMemory: MLXArray, textMask: MLXArray? = nil) -> MLXArray {
        var out = src
        for layer in layers {
            out = layer(out, pos: pos, textMemory: textMemory, textMask: textMask)
            MLX.eval(out)
        }
        return out
    }
}
