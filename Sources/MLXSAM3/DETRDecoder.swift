// DETRDecoder.swift — DETR transformer decoder for SAM 3.
// Python source: sam3/decoder.py
// Weight keys: detector_model.detr_decoder.*

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Free functions

func inverseSigmoid(_ x: MLXArray, eps: Float = 1e-5) -> MLXArray {
    let xc = MLX.clip(x, min: eps, max: 1 - eps)
    return MLX.log(xc / (1 - xc))
}

// MARK: - Sub-modules

class BoxHead: Module {
    @ModuleInfo(key: "layer1") var layer1: Linear
    @ModuleInfo(key: "layer2") var layer2: Linear
    @ModuleInfo(key: "layer3") var layer3: Linear

    init(hiddenSize: Int) {
        self._layer1 = ModuleInfo(wrappedValue: Linear(hiddenSize, hiddenSize), key: "layer1")
        self._layer2 = ModuleInfo(wrappedValue: Linear(hiddenSize, hiddenSize), key: "layer2")
        self._layer3 = ModuleInfo(wrappedValue: Linear(hiddenSize, 4), key: "layer3")
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        layer3(relu(layer2(relu(layer1(x)))))
    }
}

class PresenceHead: Module {
    @ModuleInfo(key: "layer1") var layer1: Linear
    @ModuleInfo(key: "layer2") var layer2: Linear
    @ModuleInfo(key: "layer3") var layer3: Linear

    init(hiddenSize: Int) {
        self._layer1 = ModuleInfo(wrappedValue: Linear(hiddenSize, hiddenSize), key: "layer1")
        self._layer2 = ModuleInfo(wrappedValue: Linear(hiddenSize, hiddenSize), key: "layer2")
        self._layer3 = ModuleInfo(wrappedValue: Linear(hiddenSize, 1), key: "layer3")
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        layer3(relu(layer2(relu(layer1(x)))))
    }
}

class RefPointHead: Module {
    @ModuleInfo(key: "layer1") var layer1: Linear
    @ModuleInfo(key: "layer2") var layer2: Linear

    init(hiddenSize: Int) {
        self._layer1 = ModuleInfo(wrappedValue: Linear(hiddenSize * 2, hiddenSize), key: "layer1")
        self._layer2 = ModuleInfo(wrappedValue: Linear(hiddenSize, hiddenSize), key: "layer2")
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        relu(layer2(relu(layer1(x))))
    }
}

class BoxRPBEmbed: Module {
    @ModuleInfo(key: "layer1") var layer1: Linear
    @ModuleInfo(key: "layer2") var layer2: Linear

    init(numHeads: Int, hiddenSize: Int) {
        self._layer1 = ModuleInfo(wrappedValue: Linear(2, hiddenSize), key: "layer1")
        self._layer2 = ModuleInfo(wrappedValue: Linear(hiddenSize, numHeads), key: "layer2")
        super.init()
    }

    func callAsFunction(_ deltas: MLXArray) -> MLXArray {
        layer2(relu(layer1(deltas)))
    }
}

/// Sine position encoding for 4D box coordinates (not a Module — no weights).
struct SinePositionEmbeddingForBoxes {
    let numPosFeats: Int
    let temperature: Float
    let scale: Float

    init(numPosFeats: Int = 128, temperature: Float = 10000.0) {
        self.numPosFeats = numPosFeats
        self.temperature = temperature
        self.scale = Float(2.0 * Double.pi)
    }

    /// - Parameter boxes: (B, Q, 4) box coordinates
    /// - Returns: (B, Q, numPosFeats*4)
    func encodeBoxes(_ boxes: MLXArray) -> MLXArray {
        let d = numPosFeats
        let dimIdx = MLXArray(0..<d).asType(.float32)
        let halfDimIdx = MLX.floor(dimIdx / 2)
        let logTemp = Foundation.log(temperature)
        let dimT = MLX.exp(logTemp * (2.0 * halfDimIdx / Float(d)))

        var encodings: [MLXArray] = []
        // Python order: y, x, w, h (indices 1, 0, 2, 3)
        for i in [1, 0, 2, 3] {
            let coord = boxes[.ellipsis, i...i] * scale  // (B, Q, 1)
            let pos = coord / dimT                        // (B, Q, D)
            let pairs = pos.reshaped(Array(pos.shape.dropLast()) + [d / 2, 2])
            let enc = stacked([MLX.sin(pairs[.ellipsis, 0]), MLX.cos(pairs[.ellipsis, 1])], axis: -1)
                .reshaped(Array(pos.shape.dropLast()) + [d])
            encodings.append(enc)
        }
        return concatenated(encodings, axis: -1)
    }
}

// MARK: - Decoder Layer

class DETRDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: MultiheadAttention
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttnLayerNorm: LayerNorm
    @ModuleInfo(key: "text_cross_attn") var textCrossAttn: MultiheadAttention
    @ModuleInfo(key: "text_cross_attn_layer_norm") var textCrossAttnLayerNorm: LayerNorm
    @ModuleInfo(key: "vision_cross_attn") var visionCrossAttn: MultiheadAttention
    @ModuleInfo(key: "vision_cross_attn_layer_norm") var visionCrossAttnLayerNorm: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: MLP
    @ModuleInfo(key: "mlp_layer_norm") var mlpLayerNorm: LayerNorm

    init(_ config: DETRDecoderConfig) {
        let d = config.hiddenSize
        self._selfAttn = ModuleInfo(wrappedValue: MultiheadAttention(hiddenSize: d, numHeads: config.numAttentionHeads), key: "self_attn")
        self._selfAttnLayerNorm = ModuleInfo(wrappedValue: LayerNorm(dimensions: d, eps: config.layerNormEps), key: "self_attn_layer_norm")
        self._textCrossAttn = ModuleInfo(wrappedValue: MultiheadAttention(hiddenSize: d, numHeads: config.numAttentionHeads), key: "text_cross_attn")
        self._textCrossAttnLayerNorm = ModuleInfo(wrappedValue: LayerNorm(dimensions: d, eps: config.layerNormEps), key: "text_cross_attn_layer_norm")
        self._visionCrossAttn = ModuleInfo(wrappedValue: MultiheadAttention(hiddenSize: d, numHeads: config.numAttentionHeads), key: "vision_cross_attn")
        self._visionCrossAttnLayerNorm = ModuleInfo(wrappedValue: LayerNorm(dimensions: d, eps: config.layerNormEps), key: "vision_cross_attn_layer_norm")
        self._mlp = ModuleInfo(wrappedValue: MLP(hiddenSize: d, intermediateSize: config.intermediateSize, act: config.hiddenAct), key: "mlp")
        self._mlpLayerNorm = ModuleInfo(wrappedValue: LayerNorm(dimensions: d, eps: config.layerNormEps), key: "mlp_layer_norm")
        super.init()
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        queryPos: MLXArray,
        inputsEmbeds: MLXArray,
        visionFeatures: MLXArray,
        visionPosEncoding: MLXArray,
        textCrossAttnMask: MLXArray? = nil,
        visionCrossAttnMask: MLXArray? = nil
    ) -> MLXArray {
        // 1. Self-attention (post-norm)
        let qk1 = hiddenStates + queryPos
        var hs = selfAttnLayerNorm(hiddenStates + selfAttn(qk1, qk1, hiddenStates))

        // 2. Text cross-attention
        let qk2 = hs + queryPos
        hs = textCrossAttnLayerNorm(hs + textCrossAttn(qk2, inputsEmbeds, inputsEmbeds, mask: textCrossAttnMask))

        // 3. Vision cross-attention
        let qk3 = hs + queryPos
        let kVis = visionFeatures + visionPosEncoding
        hs = visionCrossAttnLayerNorm(hs + visionCrossAttn(qk3, kVis, visionFeatures, mask: visionCrossAttnMask))

        // 4. MLP
        hs = mlpLayerNorm(hs + mlp(hs))
        return hs
    }
}

// MARK: - DETR Decoder

public class DETRDecoder: Module {
    @ModuleInfo(key: "layers") var layers: [DETRDecoderLayer]
    @ModuleInfo(key: "output_layer_norm") var outputLayerNorm: LayerNorm
    @ModuleInfo(key: "query_embed") var queryEmbed: Embedding
    @ModuleInfo(key: "reference_points") var referencePoints: Embedding
    @ModuleInfo(key: "presence_token") var presenceToken: Embedding
    @ModuleInfo(key: "presence_head") var presenceHead: PresenceHead
    @ModuleInfo(key: "presence_layer_norm") var presenceLayerNorm: LayerNorm
    @ModuleInfo(key: "box_head") var boxHead: BoxHead
    @ModuleInfo(key: "ref_point_head") var refPointHead: RefPointHead
    @ModuleInfo(key: "box_rpb_embed_x") var boxRpbEmbedX: BoxRPBEmbed
    @ModuleInfo(key: "box_rpb_embed_y") var boxRpbEmbedY: BoxRPBEmbed

    let config: DETRDecoderConfig
    let clampPresenceLogitMaxVal: Float = 10.0
    let posEnc: SinePositionEmbeddingForBoxes

    public init(_ config: DETRDecoderConfig) {
        self.config = config
        let d = config.hiddenSize
        let numQ = config.numQueries
        let numH = config.numAttentionHeads
        self._layers = ModuleInfo(wrappedValue: (0..<config.numLayers).map { _ in DETRDecoderLayer(config) }, key: "layers")
        self._outputLayerNorm = ModuleInfo(wrappedValue: LayerNorm(dimensions: d, eps: config.layerNormEps), key: "output_layer_norm")
        self._queryEmbed = ModuleInfo(wrappedValue: Embedding(embeddingCount: numQ, dimensions: d), key: "query_embed")
        self._referencePoints = ModuleInfo(wrappedValue: Embedding(embeddingCount: numQ, dimensions: 4), key: "reference_points")
        self._presenceToken = ModuleInfo(wrappedValue: Embedding(embeddingCount: 1, dimensions: d), key: "presence_token")
        self._presenceHead = ModuleInfo(wrappedValue: PresenceHead(hiddenSize: d), key: "presence_head")
        self._presenceLayerNorm = ModuleInfo(wrappedValue: LayerNorm(dimensions: d, eps: config.layerNormEps), key: "presence_layer_norm")
        self._boxHead = ModuleInfo(wrappedValue: BoxHead(hiddenSize: d), key: "box_head")
        self._refPointHead = ModuleInfo(wrappedValue: RefPointHead(hiddenSize: d), key: "ref_point_head")
        self._boxRpbEmbedX = ModuleInfo(wrappedValue: BoxRPBEmbed(numHeads: numH, hiddenSize: d), key: "box_rpb_embed_x")
        self._boxRpbEmbedY = ModuleInfo(wrappedValue: BoxRPBEmbed(numHeads: numH, hiddenSize: d), key: "box_rpb_embed_y")
        self.posEnc = SinePositionEmbeddingForBoxes(numPosFeats: d / 2)
        super.init()
    }

    /// - Returns: (intermediateHs: (L,B,Q,D), intermediateBoxes: (L,B,Q,4), presenceLogits: (L,B,1))
    public func callAsFunction(
        visionFeatures: MLXArray,
        inputsEmbeds: MLXArray,
        visionPosEncoding: MLXArray,
        textMask: MLXArray? = nil,
        spatialShape: (Int, Int)? = nil
    ) -> (MLXArray, MLXArray, MLXArray) {
        let b = visionFeatures.dim(0)
        let numQ = config.numQueries
        let d = config.hiddenSize

        let queryEmbeds = broadcast(queryEmbed.weight[.newAxis, 0...], to: [b, numQ, d])
        var refBoxes = MLX.sigmoid(broadcast(referencePoints.weight[.newAxis, 0...], to: [b, numQ, 4]))
        let presToken = broadcast(presenceToken.weight[.newAxis, 0...], to: [b, 1, d])
        var hs = concatenated([presToken, queryEmbeds], axis: 1)  // (B, Q+1, D)

        let textCrossMask: MLXArray?
        if let tm = textMask {
            textCrossMask = (1 - tm[0..., .newAxis, .newAxis, 0...].asType(.float32)) * (-1e9 as Float)
        } else {
            textCrossMask = nil
        }

        var allHs: [MLXArray] = []
        var allBoxes: [MLXArray] = []
        var allPresence: [MLXArray] = []

        for layer in layers {
            let querySineEmbed = posEnc.encodeBoxes(refBoxes)  // (B, Q, 2D)
            let queryPos = refPointHead(querySineEmbed)         // (B, Q, D)
            let queryPosPadded = concatenated([MLXArray.zeros([b, 1, d]), queryPos], axis: 1)

            var visionCrossMask: MLXArray? = nil
            if let (height, width) = spatialShape {
                let rpb = computeRPB(refBoxes: refBoxes, height: height, width: width)
                let rpbPadded = concatenated(
                    [MLXArray.zeros([b, rpb.dim(1), 1, rpb.dim(3)]), rpb], axis: 2
                )
                visionCrossMask = rpbPadded
            }

            hs = layer(hs, queryPos: queryPosPadded, inputsEmbeds: inputsEmbeds,
                       visionFeatures: visionFeatures, visionPosEncoding: visionPosEncoding,
                       textCrossAttnMask: textCrossMask, visionCrossAttnMask: visionCrossMask)

            let queryHs = outputLayerNorm(hs[0..., 1..., 0...])
            let delta = boxHead(queryHs)
            let newRef = MLX.sigmoid(inverseSigmoid(refBoxes) + delta)
            refBoxes = MLX.stopGradient(newRef)

            allHs.append(queryHs)
            allBoxes.append(newRef)

            let presHidden = hs[0..., ..<1, 0...]
            var presLogit = presenceHead(presenceLayerNorm(presHidden)).squeezed(axes: [-1])
            presLogit = MLX.clip(presLogit, min: -clampPresenceLogitMaxVal, max: clampPresenceLogitMaxVal)
            allPresence.append(presLogit)

            MLX.eval(hs, refBoxes)
        }

        return (stacked(allHs), stacked(allBoxes), stacked(allPresence))
    }

    private func computeRPB(refBoxes: MLXArray, height: Int, width: Int) -> MLXArray {
        let b = refBoxes.dim(0), q = refBoxes.dim(1)
        let cx = refBoxes[.ellipsis, 0], cy = refBoxes[.ellipsis, 1]
        let w = refBoxes[.ellipsis, 2], h = refBoxes[.ellipsis, 3]
        let x1 = cx - w / 2, y1 = cy - h / 2
        let x2 = cx + w / 2, y2 = cy + h / 2

        let coordsH = (MLXArray(0..<height).asType(.float32) + 0.5) / Float(height)
        let coordsW = (MLXArray(0..<width).asType(.float32) + 0.5) / Float(width)

        let log2of8 = Foundation.log2(8.0 as Float)

        let yBounds = stacked([y1, y2], axis: -1).reshaped([b * q, 1, 2])
        let deltasY = (coordsH.reshaped([1, height, 1]) - yBounds).reshaped([b, q, height, 2])

        let xBounds = stacked([x1, x2], axis: -1).reshaped([b * q, 1, 2])
        let deltasX = (coordsW.reshaped([1, width, 1]) - xBounds).reshaped([b, q, width, 2])


        func logEncode(_ d: MLXArray) -> MLXArray {
            let d8 = d * 8.0
            return MLX.sign(d8) * MLX.log2(MLX.abs(d8) + 1.0) / log2of8
        }

        let rpbX = boxRpbEmbedX(logEncode(deltasX))  // (B, Q, W, numHeads)
        let rpbY = boxRpbEmbedY(logEncode(deltasY))  // (B, Q, H, numHeads)

        // Combine spatial dimensions: (B, Q, H, W, numHeads)
        let rpb = rpbY[0..., 0..., 0..., .newAxis, 0...] + rpbX[0..., 0..., .newAxis, 0..., 0...]
        return rpb.reshaped([b, q, height * width, -1]).transposed(0, 3, 1, 2)
    }
}
