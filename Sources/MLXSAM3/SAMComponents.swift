// SAMComponents.swift — SAM-style building blocks for the tracker.
// Python source: sam3/sam_components.py
// Weight keys: tracker_model.prompt_encoder.* / mask_decoder.*

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Basic blocks

class MLPBlock: Module {
    @ModuleInfo(key: "proj_in") var projIn: Linear
    @ModuleInfo(key: "proj_out") var projOut: Linear
    let act: String

    init(inputDim: Int, hiddenDim: Int, act: String = "relu") {
        self.act = act
        self._projIn = ModuleInfo(wrappedValue: Linear(inputDim, hiddenDim), key: "proj_in")
        self._projOut = ModuleInfo(wrappedValue: Linear(hiddenDim, inputDim), key: "proj_out")
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = act == "gelu" ? gelu(projIn(x)) : relu(projIn(x))
        return projOut(h)
    }
}

class LayerNorm2d: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray
    let eps: Float

    init(numChannels: Int, eps: Float = 1e-6) {
        self.eps = eps
        self._weight = ParameterInfo(wrappedValue: MLXArray.ones([numChannels]), key: "weight")
        self._bias = ParameterInfo(wrappedValue: MLXArray.zeros([numChannels]), key: "bias")
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let mean = x.mean(axes: [-1], keepDims: true)
        let variance = ((x - mean) * (x - mean)).mean(axes: [-1], keepDims: true)
        let normalized = (x - mean) / MLX.sqrt(variance + eps)
        return normalized * weight + bias
    }
}

// MARK: - Attention

class SAMAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(hiddenSize: Int, numHeads: Int, downsampleRate: Int = 1) {
        self.numHeads = numHeads
        let internalDim = hiddenSize / downsampleRate
        self.headDim = internalDim / numHeads
        self.scale = 1.0 / Float(internalDim / numHeads).squareRoot()
        self._qProj = ModuleInfo(wrappedValue: Linear(hiddenSize, internalDim), key: "q_proj")
        self._kProj = ModuleInfo(wrappedValue: Linear(hiddenSize, internalDim), key: "k_proj")
        self._vProj = ModuleInfo(wrappedValue: Linear(hiddenSize, internalDim), key: "v_proj")
        self._oProj = ModuleInfo(wrappedValue: Linear(internalDim, hiddenSize), key: "o_proj")
        super.init()
    }

    func callAsFunction(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray) -> MLXArray {
        let b = q.dim(0), nq = q.dim(1), nk = k.dim(1)
        let qp = qProj(q).reshaped([b, nq, numHeads, headDim]).transposed(0, 2, 1, 3)
        let kp = kProj(k).reshaped([b, nk, numHeads, headDim]).transposed(0, 2, 1, 3)
        let vp = vProj(v).reshaped([b, nk, numHeads, headDim]).transposed(0, 2, 1, 3)
        var out = MLXFast.scaledDotProductAttention(queries: qp, keys: kp, values: vp, scale: scale, mask: nil)
        out = out.transposed(0, 2, 1, 3).reshaped([b, nq, -1])
        return oProj(out)
    }
}

class RoPEAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float
    let ropeKRepeat: Bool

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    // Precomputed — not weights
    var freqsCos: MLXArray
    var freqsSin: MLXArray

    init(
        hiddenSize: Int,
        numHeads: Int,
        downsampleRate: Int = 1,
        featSizes: (Int, Int) = (72, 72),
        ropeTheta: Float = 10000.0,
        kvDim: Int? = nil,
        ropeKRepeat: Bool = false
    ) {
        self.numHeads = numHeads
        let internalDim = hiddenSize / downsampleRate
        self.headDim = internalDim / numHeads
        self.scale = 1.0 / Float(internalDim / numHeads).squareRoot()
        self.ropeKRepeat = ropeKRepeat
        let kd = kvDim ?? hiddenSize
        self._qProj = ModuleInfo(wrappedValue: Linear(hiddenSize, internalDim), key: "q_proj")
        self._kProj = ModuleInfo(wrappedValue: Linear(kd, internalDim), key: "k_proj")
        self._vProj = ModuleInfo(wrappedValue: Linear(kd, internalDim), key: "v_proj")
        self._oProj = ModuleInfo(wrappedValue: Linear(internalDim, hiddenSize), key: "o_proj")
        let (fc, fs) = init2dFreqs(dim: internalDim, featH: featSizes.0, featW: featSizes.1, theta: ropeTheta)
        self.freqsCos = fc
        self.freqsSin = fs
        super.init()
    }

    func callAsFunction(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray) -> MLXArray {
        let b = q.dim(0), nq = q.dim(1), nk = k.dim(1)
        var qp = qProj(q).reshaped([b, nq, numHeads, headDim])
        var kp = kProj(k).reshaped([b, nk, numHeads, headDim])
        let vp = vProj(v).reshaped([b, nk, numHeads, headDim])
        (qp, kp) = applyRotaryEnc1d(xq: qp, xk: kp, freqsCos: freqsCos, freqsSin: freqsSin, repeatFreqsK: ropeKRepeat)
        let qt = qp.transposed(0, 2, 1, 3)
        let kt = kp.transposed(0, 2, 1, 3)
        let vt = vp.transposed(0, 2, 1, 3)
        var out = MLXFast.scaledDotProductAttention(queries: qt, keys: kt, values: vt, scale: scale, mask: nil)
        out = out.transposed(0, 2, 1, 3).reshaped([b, nq, -1])
        return oProj(out)
    }
}

// MARK: - TwoWayTransformer

class TwoWayAttentionBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: SAMAttention
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    @ModuleInfo(key: "cross_attn_token_to_image") var crossAttnTokenToImage: SAMAttention
    @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: MLPBlock
    @ModuleInfo(key: "layer_norm3") var layerNorm3: LayerNorm
    @ModuleInfo(key: "cross_attn_image_to_token") var crossAttnImageToToken: SAMAttention
    @ModuleInfo(key: "layer_norm4") var layerNorm4: LayerNorm

    init(hiddenSize: Int, numHeads: Int, mlpDim: Int = 2048, attentionDownsampleRate: Int = 2) {
        self._selfAttn = ModuleInfo(wrappedValue: SAMAttention(hiddenSize: hiddenSize, numHeads: numHeads), key: "self_attn")
        self._layerNorm1 = ModuleInfo(wrappedValue: LayerNorm(dimensions: hiddenSize), key: "layer_norm1")
        self._crossAttnTokenToImage = ModuleInfo(wrappedValue: SAMAttention(hiddenSize: hiddenSize, numHeads: numHeads, downsampleRate: attentionDownsampleRate), key: "cross_attn_token_to_image")
        self._layerNorm2 = ModuleInfo(wrappedValue: LayerNorm(dimensions: hiddenSize), key: "layer_norm2")
        self._mlp = ModuleInfo(wrappedValue: MLPBlock(inputDim: hiddenSize, hiddenDim: mlpDim, act: "relu"), key: "mlp")
        self._layerNorm3 = ModuleInfo(wrappedValue: LayerNorm(dimensions: hiddenSize), key: "layer_norm3")
        self._crossAttnImageToToken = ModuleInfo(wrappedValue: SAMAttention(hiddenSize: hiddenSize, numHeads: numHeads, downsampleRate: attentionDownsampleRate), key: "cross_attn_image_to_token")
        self._layerNorm4 = ModuleInfo(wrappedValue: LayerNorm(dimensions: hiddenSize), key: "layer_norm4")
        super.init()
    }

    func callAsFunction(_ queries: MLXArray, _ keys: MLXArray, queryPE: MLXArray, keyPE: MLXArray) -> (MLXArray, MLXArray) {
        var q = queries, k = keys
        // Self-attention
        let qSelf = q + queryPE
        q = layerNorm1(q + selfAttn(qSelf, qSelf, q))
        // Token → image cross-attention
        q = layerNorm2(q + crossAttnTokenToImage(q + queryPE, k + keyPE, k))
        // MLP
        q = layerNorm3(q + mlp(q))
        // Image → token cross-attention
        k = layerNorm4(k + crossAttnImageToToken(k + keyPE, q + queryPE, q))
        return (q, k)
    }
}

class TwoWayTransformer: Module {
    @ModuleInfo(key: "layers") var layers: [TwoWayAttentionBlock]
    @ModuleInfo(key: "final_attn_token_to_image") var finalAttnTokenToImage: SAMAttention
    @ModuleInfo(key: "layer_norm_final_attn") var layerNormFinalAttn: LayerNorm

    init(hiddenSize: Int = 256, numHeads: Int = 8, numLayers: Int = 2, mlpDim: Int = 2048, attentionDownsampleRate: Int = 2) {
        self._layers = ModuleInfo(
            wrappedValue: (0..<numLayers).map { _ in TwoWayAttentionBlock(hiddenSize: hiddenSize, numHeads: numHeads, mlpDim: mlpDim, attentionDownsampleRate: attentionDownsampleRate) },
            key: "layers"
        )
        self._finalAttnTokenToImage = ModuleInfo(wrappedValue: SAMAttention(hiddenSize: hiddenSize, numHeads: numHeads, downsampleRate: attentionDownsampleRate), key: "final_attn_token_to_image")
        self._layerNormFinalAttn = ModuleInfo(wrappedValue: LayerNorm(dimensions: hiddenSize), key: "layer_norm_final_attn")
        super.init()
    }

    func callAsFunction(imageEmbedding: MLXArray, imagePE: MLXArray, pointEmbedding: MLXArray) -> (MLXArray, MLXArray) {
        var queries = pointEmbedding
        var keys = imageEmbedding
        for layer in layers {
            (queries, keys) = layer(queries, keys, queryPE: pointEmbedding, keyPE: imagePE)
        }
        queries = layerNormFinalAttn(queries + finalAttnTokenToImage(queries + pointEmbedding, keys + imagePE, keys))
        return (queries, keys)
    }
}

// MARK: - Prompt Encoder

class MaskEmbedConvs: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "conv3") var conv3: Conv2d
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm2d
    @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm2d

    init(embedDim: Int, maskInChans: Int) {
        self._conv1 = ModuleInfo(wrappedValue: Conv2d(inputChannels: 1, outputChannels: maskInChans / 4, kernelSize: 2, stride: 2), key: "conv1")
        self._conv2 = ModuleInfo(wrappedValue: Conv2d(inputChannels: maskInChans / 4, outputChannels: maskInChans, kernelSize: 2, stride: 2), key: "conv2")
        self._conv3 = ModuleInfo(wrappedValue: Conv2d(inputChannels: maskInChans, outputChannels: embedDim, kernelSize: 1), key: "conv3")
        self._layerNorm1 = ModuleInfo(wrappedValue: LayerNorm2d(numChannels: maskInChans / 4), key: "layer_norm1")
        self._layerNorm2 = ModuleInfo(wrappedValue: LayerNorm2d(numChannels: maskInChans), key: "layer_norm2")
        super.init()
    }

    /// - Parameter masks: (B, H, W, 1)
    /// - Returns: (B, H'*W', embedDim)
    func callAsFunction(_ masks: MLXArray) -> MLXArray {
        var x = gelu(layerNorm1(conv1(masks)))
        x = gelu(layerNorm2(conv2(x)))
        x = conv3(x)
        let b = x.dim(0), h = x.dim(1), w = x.dim(2), c = x.dim(3)
        return x.reshaped([b, h * w, c])
    }
}

class PositionalEmbedding: Module {
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
        let coords = stacked([gx, gy], axis: -1).expandedDimensions(axis: 0)  // (1, H*W, 2)
        return forwardWithCoords(coords)[0]  // (H*W, D)
    }

    func forwardWithCoords(_ coords: MLXArray) -> MLXArray {
        let c = (coords * 2 - 1).matmul(positionalEmbedding)  // (B, N, D/2)
        let scaled = c * Float(2.0 * Double.pi)
        return concatenated([MLX.sin(scaled), MLX.cos(scaled)], axis: -1)
    }
}

public class SAMPromptEncoder: Module {
    @ModuleInfo(key: "point_embed") var pointEmbed: Embedding
    @ModuleInfo(key: "not_a_point_embed") var notAPointEmbed: Embedding
    @ModuleInfo(key: "mask_embed") var maskEmbed: MaskEmbedConvs
    @ModuleInfo(key: "no_mask_embed") var noMaskEmbed: Embedding
    @ModuleInfo(key: "shared_embedding") var sharedEmbedding: PositionalEmbedding

    let embedDim: Int
    let imageEmbeddingSize: (Int, Int)

    public init(_ config: PromptEncoderConfig) {
        let d = config.hiddenSize
        self.embedDim = d
        self.imageEmbeddingSize = (config.imageSize / config.patchSize, config.imageSize / config.patchSize)
        self._pointEmbed = ModuleInfo(wrappedValue: Embedding(embeddingCount: config.numPointEmbeddings, dimensions: d), key: "point_embed")
        self._notAPointEmbed = ModuleInfo(wrappedValue: Embedding(embeddingCount: 1, dimensions: d), key: "not_a_point_embed")
        self._maskEmbed = ModuleInfo(wrappedValue: MaskEmbedConvs(embedDim: d, maskInChans: config.maskInputChannels), key: "mask_embed")
        self._noMaskEmbed = ModuleInfo(wrappedValue: Embedding(embeddingCount: 1, dimensions: d), key: "no_mask_embed")
        self._sharedEmbedding = ModuleInfo(wrappedValue: PositionalEmbedding(numPosFeats: d / 2), key: "shared_embedding")
        super.init()
    }

    public func getDensePE() -> MLXArray {
        let (h, w) = imageEmbeddingSize
        return sharedEmbedding((h, w)).expandedDimensions(axis: 0)  // (1, H*W, D)
    }

    /// - Returns: (sparseEmbeddings: (B, N, D), denseEmbeddings: (B, HW, D))
    public func callAsFunction(
        points: (MLXArray, MLXArray)? = nil,
        boxes: MLXArray? = nil,
        masks: MLXArray? = nil
    ) -> (MLXArray, MLXArray) {
        var b = 1
        var sparse = MLXArray.zeros([b, 0, embedDim])

        if let (coords, labels) = points {
            b = coords.dim(0)
            let pointEmb = embedPoints(coords: coords, labels: labels)
            sparse = concatenated([broadcast(sparse, to: [b, 0, embedDim]), pointEmb], axis: 1)
        }
        if let bx = boxes {
            b = bx.dim(0)
            sparse = concatenated([sparse, embedBoxes(bx)], axis: 1)
        }

        let dense: MLXArray
        if let m = masks {
            b = m.dim(0)
            dense = maskEmbed(m)
        } else {
            let (h, w) = imageEmbeddingSize
            dense = broadcast(noMaskEmbed.weight.reshaped([1, 1, embedDim]), to: [b, h * w, embedDim])
        }

        return (sparse, dense)
    }

    private func embedPoints(coords: MLXArray, labels: MLXArray) -> MLXArray {
        let shifted = (coords + 0.5) / MLXArray([Float(imageEmbeddingSize.1), Float(imageEmbeddingSize.0)])
        var emb = sharedEmbedding.forwardWithCoords(shifted)
        let n = labels.dim(1)
        for i in 0..<n {
            let lbl = labels[0..., i...i].asType(.int32)
            emb = emb.at[0..., i...i].add(pointEmbed(lbl))
        }
        let paddingMask = labels .== -1
        if paddingMask.any().item(Bool.self) {
            emb = MLX.where(paddingMask[.ellipsis, .newAxis], notAPointEmbed.weight, emb)
        }
        return emb
    }

    private func embedBoxes(_ boxes: MLXArray) -> MLXArray {
        let b = boxes.dim(0)
        let coords = boxes.reshaped([b, -1, 2, 2])
        var emb = sharedEmbedding.forwardWithCoords(coords.reshaped([b, -1, 2]))
        emb = emb.reshaped([b, -1, 2, embedDim])
        emb = emb.at[0..., 0..., 0...1].add(pointEmbed(MLXArray([2]).reshaped([1, 1])))
        emb = emb.at[0..., 0..., 1...2].add(pointEmbed(MLXArray([3]).reshaped([1, 1])))
        return emb.reshaped([b, -1, embedDim])
    }
}

// MARK: - OutputMLP (shared by SAMMaskDecoder and MultiplexMaskDecoder)

public class OutputMLP: Module {
    @ModuleInfo(key: "proj_in") var projIn: Linear
    @ModuleInfo(key: "layers") var layers: [Linear]
    @ModuleInfo(key: "proj_out") var projOut: Linear

    public init(inputDim: Int, hiddenDim: Int, outputDim: Int) {
        self._projIn = ModuleInfo(wrappedValue: Linear(inputDim, hiddenDim), key: "proj_in")
        self._layers = ModuleInfo(wrappedValue: [Linear(hiddenDim, hiddenDim)], key: "layers")
        self._projOut = ModuleInfo(wrappedValue: Linear(hiddenDim, outputDim), key: "proj_out")
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = relu(projIn(x))
        for layer in layers { out = relu(layer(out)) }
        return projOut(out)
    }
}

// MARK: - SAM Mask Decoder

public class SAMMaskDecoder: Module {
    @ModuleInfo(key: "transformer") var transformer: TwoWayTransformer
    @ModuleInfo(key: "iou_token") var iouToken: Embedding
    @ModuleInfo(key: "mask_tokens") var maskTokens: Embedding
    @ModuleInfo(key: "obj_score_token") var objScoreToken: Embedding
    @ModuleInfo(key: "output_hypernetworks_mlps") var outputHypernetworksMlps: [OutputMLP]
    @ModuleInfo(key: "iou_prediction_head") var iouPredictionHead: OutputMLP
    @ModuleInfo(key: "pred_obj_score_head") var predObjScoreHead: OutputMLP
    @ModuleInfo(key: "upscale_conv1") var upscaleConv1: ConvTransposed2d
    @ModuleInfo(key: "upscale_conv2") var upscaleConv2: ConvTransposed2d
    @ModuleInfo(key: "upscale_layer_norm") var upscaleLayerNorm: LayerNorm2d
    @ModuleInfo(key: "conv_s0") var convS0: Conv2d
    @ModuleInfo(key: "conv_s1") var convS1: Conv2d

    let numMaskTokens: Int
    let numMultimaskOutputs: Int
    let dynamicMultimaskViaStability: Bool
    let dynamicMultimaskStabilityDelta: Float
    let dynamicMultimaskStabilityThresh: Float

    public init(_ config: TrackerMaskDecoderConfig) {
        let d = config.hiddenSize
        let nmt = config.numMultimaskOutputs + 1
        self.numMaskTokens = nmt
        self.numMultimaskOutputs = config.numMultimaskOutputs
        self.dynamicMultimaskViaStability = config.dynamicMultimaskViaStability
        self.dynamicMultimaskStabilityDelta = config.dynamicMultimaskStabilityDelta
        self.dynamicMultimaskStabilityThresh = config.dynamicMultimaskStabilityThresh

        self._transformer = ModuleInfo(wrappedValue: TwoWayTransformer(hiddenSize: d, numHeads: config.numAttentionHeads, numLayers: config.numHiddenLayers, mlpDim: config.mlpDim, attentionDownsampleRate: config.attentionDownsampleRate), key: "transformer")
        self._iouToken = ModuleInfo(wrappedValue: Embedding(embeddingCount: 1, dimensions: d), key: "iou_token")
        self._maskTokens = ModuleInfo(wrappedValue: Embedding(embeddingCount: nmt, dimensions: d), key: "mask_tokens")
        self._objScoreToken = ModuleInfo(wrappedValue: Embedding(embeddingCount: 1, dimensions: d), key: "obj_score_token")
        self._outputHypernetworksMlps = ModuleInfo(wrappedValue: (0..<nmt).map { _ in OutputMLP(inputDim: d, hiddenDim: d, outputDim: d / 8) }, key: "output_hypernetworks_mlps")
        self._iouPredictionHead = ModuleInfo(wrappedValue: OutputMLP(inputDim: d, hiddenDim: d, outputDim: nmt), key: "iou_prediction_head")
        self._predObjScoreHead = ModuleInfo(wrappedValue: OutputMLP(inputDim: d, hiddenDim: d, outputDim: 1), key: "pred_obj_score_head")
        self._upscaleConv1 = ModuleInfo(wrappedValue: ConvTransposed2d(inputChannels: d, outputChannels: d / 4, kernelSize: .init(2), stride: .init(2)), key: "upscale_conv1")
        self._upscaleConv2 = ModuleInfo(wrappedValue: ConvTransposed2d(inputChannels: d / 4, outputChannels: d / 8, kernelSize: .init(2), stride: .init(2)), key: "upscale_conv2")
        self._upscaleLayerNorm = ModuleInfo(wrappedValue: LayerNorm2d(numChannels: d / 4), key: "upscale_layer_norm")
        self._convS0 = ModuleInfo(wrappedValue: Conv2d(inputChannels: d, outputChannels: d / 8, kernelSize: 1, bias: true), key: "conv_s0")
        self._convS1 = ModuleInfo(wrappedValue: Conv2d(inputChannels: d, outputChannels: d / 4, kernelSize: 1, bias: true), key: "conv_s1")
        super.init()
    }

    /// - Returns: (masks: (B, N, H, W), iouPred: (B, N), samTokens: (B, T, D), objScore: (B, 1))
    public func callAsFunction(
        imageEmbeddings: MLXArray,
        imagePE: MLXArray,
        sparsePromptEmbeddings: MLXArray,
        densePromptEmbeddings: MLXArray,
        multimaskOutput: Bool = true,
        highResFeatures: [MLXArray]? = nil
    ) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
        let b = imageEmbeddings.dim(0)
        let d = imageEmbeddings.dim(-1)

        let tokens = concatenated([
            broadcast(iouToken.weight[.newAxis, 0...], to: [b, 1, d]),
            broadcast(maskTokens.weight[.newAxis, 0...], to: [b, numMaskTokens, d]),
            broadcast(objScoreToken.weight[.newAxis, 0...], to: [b, 1, d]),
            sparsePromptEmbeddings,
        ], axis: 1)

        let src = imageEmbeddings + densePromptEmbeddings
        let (hs, srcOut) = transformer(imageEmbedding: src, imagePE: imagePE, pointEmbedding: tokens)

        let iouTokenOut = hs[0..., 0...0, 0...]
        let maskTokensOut = hs[0..., 1...(numMaskTokens), 0...]
        let objScoreTokenOut = hs[0..., (numMaskTokens + 1)...(numMaskTokens + 1), 0...]

        let hw = srcOut.dim(1)
        let side = Int(Double(hw).squareRoot().rounded())
        var up = srcOut.reshaped([b, side, side, d])
        up = gelu(upscaleLayerNorm(upscaleConv1(up)))

        if let hrf = highResFeatures, !hrf.isEmpty {
            let s1 = convS1(hrf[0])
            if s1.dim(1) == up.dim(1) { up = up + s1 }
        }
        up = gelu(upscaleConv2(up))
        if let hrf = highResFeatures, hrf.count >= 2 {
            let s0 = convS0(hrf[1])
            if s0.dim(1) == up.dim(1) { up = up + s0 }
        }

        let hUp = up.dim(1), wUp = up.dim(2), cUp = up.dim(3)
        let upFlat = up.reshaped([b, hUp * wUp, cUp])

        var masks: [MLXArray] = []
        for i in 0..<numMaskTokens {
            let hyper = outputHypernetworksMlps[i](maskTokensOut[0..., i, 0...])
            let mask = (upFlat * hyper[0..., .newAxis, 0...]).sum(axes: [-1])
            masks.append(mask.reshaped([b, 1, hUp, wUp]))
        }
        let allMasks = concatenated(masks, axis: 1)

        let iouPred = iouPredictionHead(iouTokenOut.squeezed(axes: [1]))
        let objScore = predObjScoreHead(objScoreTokenOut.squeezed(axes: [1]))

        if multimaskOutput {
            return (allMasks[0..., 1..., 0..., 0...], iouPred[0..., 1...], hs, objScore)
        } else {
            return (allMasks[0..., 0...0, 0..., 0...], iouPred[0..., 0...0], hs, objScore)
        }
    }
}
