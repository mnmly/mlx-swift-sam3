// Vision.swift — ViT backbone + FPN neck for SAM 3.
// Python source: sam3/vision.py
// Weight key prefix: detector_model.vision_encoder.backbone.* / neck.*

import MLX
import MLXFast
import MLXNN

// MARK: - ViT Components

class PatchProjection: Module {
    @ModuleInfo(key: "projection") var projection: Conv2d

    init(_ config: ViTConfig) {
        self._projection = ModuleInfo(
            wrappedValue: Conv2d(
                inputChannels: config.numChannels,
                outputChannels: config.hiddenSize,
                kernelSize: .init(config.patchSize),
                stride: .init(config.patchSize),
                bias: false
            ),
            key: "projection"
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { projection(x) }
}

class PatchEmbeddings: Module {
    @ModuleInfo(key: "patch_embeddings") var patchEmbeddings: PatchProjection
    @ParameterInfo(key: "position_embeddings") var positionEmbeddings: MLXArray

    init(_ config: ViTConfig) {
        let numPatches = (config.pretrainImageSize / config.patchSize) * (config.pretrainImageSize / config.patchSize)
        self._patchEmbeddings = ModuleInfo(wrappedValue: PatchProjection(config), key: "patch_embeddings")
        self._positionEmbeddings = ParameterInfo(
            wrappedValue: MLXArray.zeros([1, numPatches, config.hiddenSize]),
            key: "position_embeddings"
        )
        super.init()
    }

    /// - Parameter x: (B, H, W, C)
    /// - Returns: (B, N, hiddenSize)
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let out = patchEmbeddings(x)  // (B, H', W', C)
        let b = out.dim(0), h = out.dim(1), w = out.dim(2), c = out.dim(3)
        return out.reshaped([b, h * w, c])
    }
}

class VitAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ config: ViTConfig) {
        self.numHeads = config.numAttentionHeads
        self.headDim = config.hiddenSize / config.numAttentionHeads
        self.scale = 1.0 / Float(config.hiddenSize / config.numAttentionHeads).squareRoot()
        self._qProj = ModuleInfo(wrappedValue: Linear(config.hiddenSize, config.hiddenSize, bias: config.qkvBias), key: "q_proj")
        self._kProj = ModuleInfo(wrappedValue: Linear(config.hiddenSize, config.hiddenSize, bias: config.qkvBias), key: "k_proj")
        self._vProj = ModuleInfo(wrappedValue: Linear(config.hiddenSize, config.hiddenSize, bias: config.qkvBias), key: "v_proj")
        self._oProj = ModuleInfo(wrappedValue: Linear(config.hiddenSize, config.hiddenSize, bias: true), key: "o_proj")
        super.init()
    }

    func callAsFunction(_ x: MLXArray, ropeCos: MLXArray? = nil, ropeSin: MLXArray? = nil) -> MLXArray {
        let inputShape = x.shape
        let b: Int
        let n: Int
        let c: Int
        var xFlat = x
        if x.ndim == 4 {
            b = x.dim(0); let h = x.dim(1); let w = x.dim(2); c = x.dim(3)
            n = h * w
            xFlat = x.reshaped([b, n, c])
        } else {
            b = x.dim(0); n = x.dim(1); c = x.dim(2)
        }

        var q = qProj(xFlat).reshaped([b, n, numHeads, headDim]).transposed(0, 2, 1, 3)
        var k = kProj(xFlat).reshaped([b, n, numHeads, headDim]).transposed(0, 2, 1, 3)
        let v = vProj(xFlat).reshaped([b, n, numHeads, headDim]).transposed(0, 2, 1, 3)

        if let cos = ropeCos, let sin = ropeSin {
            (q, k) = applyRotaryEnc(xq: q, xk: k, cos: cos, sin: sin)
        }

        var out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: nil)
        out = out.transposed(0, 2, 1, 3).reshaped([b, n, c])
        out = oProj(out)

        if inputShape.count == 4 {
            out = out.reshaped(inputShape)
        }
        return out
    }
}

class VitMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(_ config: ViTConfig) {
        self._fc1 = ModuleInfo(wrappedValue: Linear(config.hiddenSize, config.intermediateSize), key: "fc1")
        self._fc2 = ModuleInfo(wrappedValue: Linear(config.intermediateSize, config.hiddenSize), key: "fc2")
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(gelu(fc1(x))) }
}

class VitBlock: Module {
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    @ModuleInfo(key: "attention") var attention: VitAttention
    @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: VitMLP

    let windowSize: Int
    let isGlobal: Bool

    init(_ config: ViTConfig, isGlobal: Bool = false) {
        self.isGlobal = isGlobal
        self.windowSize = isGlobal ? 0 : config.windowSize
        self._layerNorm1 = ModuleInfo(wrappedValue: LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps), key: "layer_norm1")
        self._attention = ModuleInfo(wrappedValue: VitAttention(config), key: "attention")
        self._layerNorm2 = ModuleInfo(wrappedValue: LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps), key: "layer_norm2")
        self._mlp = ModuleInfo(wrappedValue: VitMLP(config), key: "mlp")
        super.init()
    }

    /// - Parameter x: (B, H, W, C)
    func callAsFunction(_ x: MLXArray, ropeCos: MLXArray? = nil, ropeSin: MLXArray? = nil) -> MLXArray {
        var out = x
        var normed = layerNorm1(out)

        if windowSize > 0 {
            let h = normed.dim(1), w = normed.dim(2)
            let (windowed, padHW) = windowPartition(normed, windowSize: windowSize)
            let attnOut = attention(windowed, ropeCos: ropeCos, ropeSin: ropeSin)
            normed = windowUnpartition(attnOut, windowSize: windowSize, padHW: padHW, originalHW: (h, w))
        } else {
            normed = attention(normed, ropeCos: ropeCos, ropeSin: ropeSin)
        }
        out = out + normed
        out = out + mlp(layerNorm2(out))
        return out
    }
}

// MARK: - Window partition helpers

func windowPartition(_ x: MLXArray, windowSize ws: Int) -> (MLXArray, (Int, Int)) {
    let b = x.dim(0), h = x.dim(1), w = x.dim(2), c = x.dim(3)
    let padH = (ws - h % ws) % ws
    let padW = (ws - w % ws) % ws
    var xp = x
    if padH > 0 || padW > 0 {
        xp = padded(xp, widths: [.init(0), .init((0, padH)), .init((0, padW)), .init(0)])
    }
    let hp = h + padH, wp = w + padW
    let nH = hp / ws, nW = wp / ws
    xp = xp.reshaped([b, nH, ws, nW, ws, c])
    xp = xp.transposed(0, 1, 3, 2, 4, 5)
    xp = xp.reshaped([b * nH * nW, ws, ws, c])
    return (xp, (hp, wp))
}

func windowUnpartition(_ x: MLXArray, windowSize ws: Int, padHW: (Int, Int), originalHW: (Int, Int)) -> MLXArray {
    let (hp, wp) = padHW, (origH, origW) = originalHW
    let nH = hp / ws, nW = wp / ws
    let b = x.dim(0) / (nH * nW)
    let c = x.dim(3)
    var out = x.reshaped([b, nH, nW, ws, ws, c])
    out = out.transposed(0, 1, 3, 2, 4, 5)
    out = out.reshaped([b, hp, wp, c])
    if hp > origH || wp > origW {
        out = out[0..., ..<origH, ..<origW, 0...]
    }
    return out
}

// MARK: - ViT Backbone

public class ViTBackbone: Module {
    @ModuleInfo(key: "embeddings") var embeddings: PatchEmbeddings
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "layers") var layers: [VitBlock]

    let config: ViTConfig
    let featSize: Int

    // Precomputed RoPE — not weights, recomputed each init
    var ropeWindowCos: MLXArray
    var ropeWindowSin: MLXArray
    var ropeGlobalCos: MLXArray
    var ropeGlobalSin: MLXArray

    public init(_ config: ViTConfig) {
        self.config = config
        let featSize = config.imageSize / config.patchSize
        self.featSize = featSize

        let globalSet = Set(config.globalAttnIndexes)
        let blocks = (0..<config.numHiddenLayers).map { i in
            VitBlock(config, isGlobal: globalSet.contains(i))
        }

        let headDim = config.hiddenSize / config.numAttentionHeads
        let (wc, ws) = computeAxialCis(dim: headDim, endX: config.windowSize, endY: config.windowSize, theta: config.ropeTheta)
        let (gc, gs) = computeAxialCis(dim: headDim, endX: featSize, endY: featSize, theta: config.ropeTheta)

        self._embeddings = ModuleInfo(wrappedValue: PatchEmbeddings(config), key: "embeddings")
        self._layerNorm = ModuleInfo(wrappedValue: LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps), key: "layer_norm")
        self._layers = ModuleInfo(wrappedValue: blocks, key: "layers")
        self.ropeWindowCos = wc; self.ropeWindowSin = ws
        self.ropeGlobalCos = gc; self.ropeGlobalSin = gs
        super.init()
    }

    /// - Parameter x: (B, H, W, C)
    /// - Returns: (B, featH, featW, hiddenSize)
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0)
        let inputH = x.dim(1), inputW = x.dim(2)
        let h = inputH / config.patchSize
        let w = inputW / config.patchSize

        var out = embeddings(x)  // (B, N, C)
        let pos = tilePositionEmbed(embeddings.positionEmbeddings, targetH: h, targetW: w)
        out = out + pos
        out = out.reshaped([b, h, w, -1])
        out = layerNorm(out)

        let (globalCos, globalSin): (MLXArray, MLXArray)
        if h != featSize || w != featSize {
            let headDim = config.hiddenSize / config.numAttentionHeads
            (globalCos, globalSin) = computeAxialCis(dim: headDim, endX: h, endY: w, theta: config.ropeTheta)
        } else {
            (globalCos, globalSin) = (ropeGlobalCos, ropeGlobalSin)
        }

        for layer in layers {
            if layer.isGlobal {
                out = layer(out, ropeCos: globalCos, ropeSin: globalSin)
            } else {
                out = layer(out, ropeCos: ropeWindowCos, ropeSin: ropeWindowSin)
            }
        }
        return out
    }

    func tilePositionEmbedPublic(_ pos: MLXArray, targetH: Int, targetW: Int) -> MLXArray {
        tilePositionEmbed(pos, targetH: targetH, targetW: targetW)
    }

    private func tilePositionEmbed(_ pos: MLXArray, targetH: Int, targetW: Int) -> MLXArray {
        let n = pos.dim(1)
        let pretrainSize = Int(Double(n).squareRoot().rounded())
        let hiddenSize = pos.dim(2)

        if pretrainSize == targetH && pretrainSize == targetW { return pos }

        var p = pos.reshaped([1, pretrainSize, pretrainSize, hiddenSize])
        let repH = targetH / pretrainSize + 1
        let repW = targetW / pretrainSize + 1
        p = MLX.tiled(p, repetitions: [1, repH, repW, 1])
        p = p[0..., ..<targetH, ..<targetW, 0...]
        return p.reshaped([1, targetH * targetW, hiddenSize])
    }
}

// MARK: - FPN Neck

class FPNLayer: Module {
    // Only the learnable ConvTransposed2d layers (sanitize remaps scale_layers.2.* -> scale_layers.1.*)
    @ModuleInfo(key: "scale_layers") var scaleLayers: [ConvTransposed2d]
    @ModuleInfo(key: "proj1") var proj1: Conv2d
    @ModuleInfo(key: "proj2") var proj2: Conv2d

    let numUpscale: Int
    let isDownsample: Bool

    init(inChannels: Int, outChannels: Int, scaleFactor: Float, fpnKernelSize: Int = 2, fpnStride: Int = 2) {
        self.isDownsample = scaleFactor <= 0.5
        var currentChannels = inChannels
        var convLayers: [ConvTransposed2d] = []
        var upscale = 0

        if scaleFactor >= 4.0 {
            let mid = currentChannels / 2
            let mid2 = mid / 2
            convLayers = [
                ConvTransposed2d(inputChannels: currentChannels, outputChannels: mid, kernelSize: .init(fpnKernelSize), stride: .init(fpnStride)),
                ConvTransposed2d(inputChannels: mid, outputChannels: mid2, kernelSize: .init(fpnKernelSize), stride: .init(fpnStride)),
            ]
            currentChannels = mid2
            upscale = 2
        } else if scaleFactor >= 2.0 {
            let mid = currentChannels / 2
            convLayers = [
                ConvTransposed2d(inputChannels: currentChannels, outputChannels: mid, kernelSize: .init(fpnKernelSize), stride: .init(fpnStride)),
            ]
            currentChannels = mid
            upscale = 1
        }
        self.numUpscale = upscale

        self._scaleLayers = ModuleInfo(wrappedValue: convLayers, key: "scale_layers")
        self._proj1 = ModuleInfo(wrappedValue: Conv2d(inputChannels: currentChannels, outputChannels: outChannels, kernelSize: 1, bias: true), key: "proj1")
        self._proj2 = ModuleInfo(wrappedValue: Conv2d(inputChannels: outChannels, outputChannels: outChannels, kernelSize: 3, padding: 1, bias: true), key: "proj2")
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x
        if numUpscale > 0 {
            for (i, conv) in scaleLayers.enumerated() {
                out = conv(out)
                if i < scaleLayers.count - 1 {
                    out = gelu(out)
                }
            }
        } else if isDownsample {
            let b = out.dim(0), h = out.dim(1), w = out.dim(2), c = out.dim(3)
            out = out.reshaped([b, h / 2, 2, w / 2, 2, c])
            out = out.max(axes: [2, 4])
        }
        out = proj1(out)
        out = proj2(out)
        return out
    }
}

class FPNNeck: Module {
    @ModuleInfo(key: "fpn_layers") var fpnLayers: [FPNLayer]

    init(_ config: VisionEncoderConfig) {
        let inChannels = config.backboneConfig.hiddenSize
        let layers = config.scaleFactors.map { sf in
            FPNLayer(inChannels: inChannels, outChannels: config.fpnHiddenSize,
                     scaleFactor: sf, fpnKernelSize: config.fpnKernelSize, fpnStride: config.fpnStride)
        }
        self._fpnLayers = ModuleInfo(wrappedValue: layers, key: "fpn_layers")
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> [MLXArray] {
        fpnLayers.map { $0(x) }
    }
}

// MARK: - Full Vision Encoder

public class VisionEncoder: Module {
    @ModuleInfo(key: "backbone") var backbone: ViTBackbone
    @ModuleInfo(key: "neck") var neck: FPNNeck

    public init(_ config: VisionEncoderConfig) {
        self._backbone = ModuleInfo(wrappedValue: ViTBackbone(config.backboneConfig), key: "backbone")
        self._neck = ModuleInfo(wrappedValue: FPNNeck(config), key: "neck")
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> [MLXArray] {
        let features = backbone(x)
        return neck(features)
    }
}
