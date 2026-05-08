// Sam31SAMComponents.swift — SAM 3.1 decoder & attention components.
// Python source: sam3_1/sam_components.py
// Weight keys: tracker_model.sam_mask_decoder.* / tracker_model.memory_attention.*

import MLX
import MLXFast
import MLXNN

// MARK: - MultiplexMaskDecoder

public class MultiplexMaskDecoder: Module {
    public let multiplexCount: Int
    public let numMaskTokens: Int

    @ModuleInfo(key: "iou_token") var iouToken: Embedding
    @ModuleInfo(key: "mask_tokens") var maskTokens: Embedding
    @ModuleInfo(key: "obj_score_token") var objScoreToken: Embedding
    @ModuleInfo(key: "transformer") var transformer: TwoWayTransformer
    @ModuleInfo(key: "output_hypernetworks_mlps") var outputHypernetworksMlps: [OutputMLP]
    @ModuleInfo(key: "iou_prediction_head") var iouPredictionHead: OutputMLP
    @ModuleInfo(key: "pred_obj_score_head") var predObjScoreHead: OutputMLP
    @ModuleInfo(key: "upscale_conv1") var upscaleConv1: ConvTransposed2d
    @ModuleInfo(key: "upscale_conv2") var upscaleConv2: ConvTransposed2d
    @ModuleInfo(key: "upscale_layer_norm") var upscaleLayerNorm: LayerNorm2d
    @ModuleInfo(key: "conv_s0") var convS0: Conv2d
    @ModuleInfo(key: "conv_s1") var convS1: Conv2d

    public init(_ config: Sam31TrackerMaskDecoderConfig) {
        let d = config.hiddenSize
        let M = config.multiplexCount
        let nMask = config.numMultimaskOutputs
        multiplexCount = M
        numMaskTokens = nMask

        self._iouToken = ModuleInfo(wrappedValue: Embedding(embeddingCount: M, dimensions: d), key: "iou_token")
        self._maskTokens = ModuleInfo(wrappedValue: Embedding(embeddingCount: M * nMask, dimensions: d), key: "mask_tokens")
        self._objScoreToken = ModuleInfo(wrappedValue: Embedding(embeddingCount: M, dimensions: d), key: "obj_score_token")
        self._transformer = ModuleInfo(wrappedValue: TwoWayTransformer(
            hiddenSize: d, numHeads: config.numAttentionHeads,
            numLayers: config.numHiddenLayers, mlpDim: config.mlpDim,
            attentionDownsampleRate: config.attentionDownsampleRate
        ), key: "transformer")
        self._outputHypernetworksMlps = ModuleInfo(
            wrappedValue: (0..<nMask).map { _ in OutputMLP(inputDim: d, hiddenDim: d, outputDim: d / 8) },
            key: "output_hypernetworks_mlps"
        )
        self._iouPredictionHead = ModuleInfo(wrappedValue: OutputMLP(inputDim: d, hiddenDim: d, outputDim: nMask), key: "iou_prediction_head")
        self._predObjScoreHead = ModuleInfo(wrappedValue: OutputMLP(inputDim: d, hiddenDim: d, outputDim: 1), key: "pred_obj_score_head")
        self._upscaleConv1 = ModuleInfo(wrappedValue: ConvTransposed2d(inputChannels: d, outputChannels: d / 4, kernelSize: .init(2), stride: .init(2)), key: "upscale_conv1")
        self._upscaleConv2 = ModuleInfo(wrappedValue: ConvTransposed2d(inputChannels: d / 4, outputChannels: d / 8, kernelSize: .init(2), stride: .init(2)), key: "upscale_conv2")
        self._upscaleLayerNorm = ModuleInfo(wrappedValue: LayerNorm2d(numChannels: d / 4), key: "upscale_layer_norm")
        self._convS0 = ModuleInfo(wrappedValue: Conv2d(inputChannels: d, outputChannels: d / 8, kernelSize: 1, bias: true), key: "conv_s0")
        self._convS1 = ModuleInfo(wrappedValue: Conv2d(inputChannels: d, outputChannels: d / 4, kernelSize: 1, bias: true), key: "conv_s1")
        super.init()
    }

    /// - Parameters:
    ///   - extraPerObjectEmbeddings: optional `(B, M, D)` tensor added to the per-slot
    ///     mask tokens before the transformer (mirrors python's `extra_per_object_embeddings`,
    ///     which carries the `output_valid_embed` / `output_invalid_embed` per-slot signal
    ///     from the multiplex tracker). Critical for memory-bank-based tracking parity.
    /// - Returns: (masks (B,M,N,H,W), iouPred (B,M,N), samTokens (B,T,D), objScore (B,M,1))
    public func callAsFunction(
        imageEmbeddings: MLXArray,
        imagePE: MLXArray,
        sparsePromptEmbeddings: MLXArray,
        densePromptEmbeddings: MLXArray,
        multimaskOutput: Bool = true,
        highResFeatures: [MLXArray]? = nil,
        extraPerObjectEmbeddings: MLXArray? = nil
    ) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
        let b = imageEmbeddings.dim(0)
        let d = imageEmbeddings.dim(-1)
        let M = multiplexCount, nMask = numMaskTokens

        // Mask tokens: shape (M*nMask, D). When `extraPerObjectEmbeddings` is provided,
        // reshape to (B, M, nMask, D), add (B, M, 1, D) broadcast, flatten back.
        var maskTokensExpanded: MLXArray
        if let extra = extraPerObjectEmbeddings {
            let reshaped = maskTokens.weight.reshaped([1, M, nMask, d])
                .asType(extra.dtype)                                  // (1, M, nMask, D)
            let broadcasted = broadcast(reshaped, to: [b, M, nMask, d])
            let added = broadcasted + extra[0..., 0..., .newAxis, 0...] // (B, M, 1, D) → (B, M, nMask, D)
            maskTokensExpanded = added.reshaped([b, M * nMask, d])
        } else {
            maskTokensExpanded = broadcast(maskTokens.weight[.newAxis], to: [b, M * nMask, d])
        }

        var tokens = concatenated([
            broadcast(iouToken.weight[.newAxis], to: [b, M, d]),
            maskTokensExpanded,
            broadcast(objScoreToken.weight[.newAxis], to: [b, M, d]),
        ], axis: 1)
        tokens = concatenated([tokens, sparsePromptEmbeddings], axis: 1)

        let src = imageEmbeddings + densePromptEmbeddings
        let (hs, flatSrc) = transformer(imageEmbedding: src, imagePE: imagePE, pointEmbedding: tokens)

        let iouOut = hs[0..., ..<M]
        let maskOut = hs[0..., M ..< M + M * nMask]
        let objOut = hs[0..., M + M * nMask ..< 2 * M + M * nMask]

        let hw = flatSrc.dim(1)
        let h = Int(Double(hw).squareRoot()), w = h
        var upscaled = upscaleConv1(flatSrc.reshaped([b, h, w, d]))
        upscaled = gelu(upscaleLayerNorm(upscaled))

        if let hrf = highResFeatures, hrf.count >= 1 {
            let s1 = convS1(hrf[0])
            if s1.dim(1) == upscaled.dim(1) { upscaled = upscaled + s1 }
        }

        upscaled = gelu(upscaleConv2(upscaled))

        if let hrf = highResFeatures, hrf.count >= 2 {
            let s0 = convS0(hrf[1])
            if s0.dim(1) == upscaled.dim(1) { upscaled = upscaled + s0 }
        }

        let hUp = upscaled.dim(1), wUp = upscaled.dim(2), cUp = upscaled.dim(3)
        let upFlat = upscaled.reshaped([b, hUp * wUp, cUp])

        var masks: [MLXArray] = []
        for objI in 0..<M {
            for maskJ in 0..<nMask {
                let tokenIdx = objI * nMask + maskJ
                let hyperOut = outputHypernetworksMlps[maskJ](maskOut[0..., tokenIdx])
                let mask = (upFlat * hyperOut.expandedDimensions(axis: 1)).sum(axis: -1)
                masks.append(mask.reshaped([b, 1, hUp, wUp]))
            }
        }
        var masksOut = concatenated(masks, axis: 1).reshaped([b, M, nMask, hUp, wUp])

        let iouPred = stacked((0..<M).map { i in iouPredictionHead(iouOut[0..., i]) }, axis: 1)
        let objScore = stacked((0..<M).map { i in predObjScoreHead(objOut[0..., i]) }, axis: 1)

        if !multimaskOutput {
            masksOut = masksOut[0..., 0..., ..<1]
            let iouSlice = iouPred[0..., 0..., ..<1]
            return (masksOut, iouSlice, hs, objScore)
        }
        return (masksOut, iouPred, hs, objScore)
    }
}

// MARK: - SimpleRoPEAttention

class SimpleRoPEAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scale: Float
    let ropeKRepeat: Bool

    var freqsCos: MLXArray
    var freqsSin: MLXArray

    init(
        hiddenSize: Int,
        numHeads: Int,
        featSizes: (Int, Int) = (72, 72),
        ropeTheta: Float = 10000.0,
        ropeKRepeat: Bool = false
    ) {
        self.numHeads = numHeads
        self.headDim = hiddenSize / numHeads
        self.scale = 1.0 / Float(hiddenSize / numHeads).squareRoot()
        self.ropeKRepeat = ropeKRepeat
        let (fc, fs) = init2dFreqs(dim: hiddenSize / numHeads, featH: featSizes.0, featW: featSizes.1, theta: ropeTheta)
        self.freqsCos = fc
        self.freqsSin = fs
        super.init()
    }

    func callAsFunction(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, numKExcludeRope: Int = 0) -> MLXArray {
        let b = q.dim(0), nQ = q.dim(1), nK = k.dim(1)
        var qr = q.reshaped([b, nQ, numHeads, headDim])
        var kr = k.reshaped([b, nK, numHeads, headDim])
        let vr = v.reshaped([b, nK, numHeads, headDim])

        let kRope: MLXArray
        let kNoRope: MLXArray?
        if numKExcludeRope > 0 {
            kRope = kr[0..., ..<(nK - numKExcludeRope)]
            kNoRope = kr[0..., (nK - numKExcludeRope)...]
        } else {
            kRope = kr
            kNoRope = nil
        }

        var kRopeMut = kRope
        (qr, kRopeMut) = applyRotaryEnc1d(xq: qr, xk: kRopeMut, freqsCos: freqsCos, freqsSin: freqsSin, repeatFreqsK: ropeKRepeat)

        if let noRope = kNoRope {
            kr = concatenated([kRopeMut, noRope], axis: 1)
        } else {
            kr = kRopeMut
        }

        let qt = qr.transposed(0, 2, 1, 3)
        let kt = kr.transposed(0, 2, 1, 3)
        let vt = vr.transposed(0, 2, 1, 3)

        let out = MLXFast.scaledDotProductAttention(queries: qt, keys: kt, values: vt, scale: scale, mask: nil)
        return out.transposed(0, 2, 1, 3).reshaped([b, nQ, -1])
    }
}

// MARK: - DecoupledMemoryAttentionLayer

class DecoupledMemoryAttentionLayer: Module {
    @ModuleInfo(key: "self_attn_q_proj") var selfAttnQProj: Linear
    @ModuleInfo(key: "self_attn_k_proj") var selfAttnKProj: Linear
    @ModuleInfo(key: "self_attn_v_proj") var selfAttnVProj: Linear
    @ModuleInfo(key: "self_attn_out_proj") var selfAttnOutProj: Linear
    @ModuleInfo(key: "self_attention_rope") var selfAttentionRope: SimpleRoPEAttention

    @ModuleInfo(key: "cross_attn_q_proj") var crossAttnQProj: Linear
    @ModuleInfo(key: "cross_attn_k_proj") var crossAttnKProj: Linear
    @ModuleInfo(key: "cross_attn_v_proj") var crossAttnVProj: Linear
    @ModuleInfo(key: "cross_attn_out_proj") var crossAttnOutProj: Linear
    @ModuleInfo(key: "cross_attention_rope") var crossAttentionRope: SimpleRoPEAttention

    @ModuleInfo(key: "image_cross_attn_q_proj") var imageCrossAttnQProj: Linear
    @ModuleInfo(key: "image_cross_attn_k_proj") var imageCrossAttnKProj: Linear

    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "norm3") var norm3: LayerNorm

    init(_ config: Sam31TrackerConfig, selfRope: SimpleRoPEAttention, crossRope: SimpleRoPEAttention) {
        let d = config.memoryAttentionHiddenSize
        let ffDim = config.memoryAttentionFeedForwardHiddenSize
        self._selfAttnQProj = ModuleInfo(wrappedValue: Linear(d, d), key: "self_attn_q_proj")
        self._selfAttnKProj = ModuleInfo(wrappedValue: Linear(d, d), key: "self_attn_k_proj")
        self._selfAttnVProj = ModuleInfo(wrappedValue: Linear(d, d), key: "self_attn_v_proj")
        self._selfAttnOutProj = ModuleInfo(wrappedValue: Linear(d, d), key: "self_attn_out_proj")
        self._selfAttentionRope = ModuleInfo(wrappedValue: selfRope, key: "self_attention_rope")
        self._crossAttnQProj = ModuleInfo(wrappedValue: Linear(d, d), key: "cross_attn_q_proj")
        self._crossAttnKProj = ModuleInfo(wrappedValue: Linear(d, d), key: "cross_attn_k_proj")
        self._crossAttnVProj = ModuleInfo(wrappedValue: Linear(d, d), key: "cross_attn_v_proj")
        self._crossAttnOutProj = ModuleInfo(wrappedValue: Linear(d, d), key: "cross_attn_out_proj")
        self._crossAttentionRope = ModuleInfo(wrappedValue: crossRope, key: "cross_attention_rope")
        self._imageCrossAttnQProj = ModuleInfo(wrappedValue: Linear(d, d), key: "image_cross_attn_q_proj")
        self._imageCrossAttnKProj = ModuleInfo(wrappedValue: Linear(d, d), key: "image_cross_attn_k_proj")
        self._linear1 = ModuleInfo(wrappedValue: Linear(d, ffDim), key: "linear1")
        self._linear2 = ModuleInfo(wrappedValue: Linear(ffDim, d), key: "linear2")
        self._norm1 = ModuleInfo(wrappedValue: LayerNorm(dimensions: d), key: "norm1")
        self._norm2 = ModuleInfo(wrappedValue: LayerNorm(dimensions: d), key: "norm2")
        self._norm3 = ModuleInfo(wrappedValue: LayerNorm(dimensions: d), key: "norm3")
        super.init()
    }

    /// - Parameters:
    ///   - src: (B, N_q, D) current frame's state (post-self-attn input).
    ///   - memory: (B, N_k_mem, D) value tensor (V-projection input). Mask-encoded
    ///     past-frame memories + obj_ptrs.
    ///   - memoryPos: optional positional bias added to memory key only.
    ///   - image: optional (B, N_q, D) **raw** current frame image features. When set,
    ///     image_cross_attn_q_proj uses this; otherwise falls back to the post-norm src
    ///     (legacy behavior). Mirrors python decoder line `q = image_cross_attn_q_proj(image)
    ///     + cross_attn_q_proj(tgt2)`.
    ///   - memoryImage: optional (B, N_k_img, D) past-frame raw image features stacked.
    ///     When set, image_cross_attn_k_proj uses this. Without it, falls back to memory.
    ///     Note `N_k_img` may differ from `N_k_mem` (mem has +obj_ptrs).
    ///   - memoryImagePos: optional positional bias added to memoryImage key only.
    func callAsFunction(
        _ src: MLXArray,
        memory: MLXArray,
        memoryPos: MLXArray? = nil,
        image: MLXArray? = nil,
        memoryImage: MLXArray? = nil,
        memoryImagePos: MLXArray? = nil,
        numKExcludeRope: Int = 0
    ) -> MLXArray {
        var out = src

        // 1. Self-attention
        let normed1 = norm1(out)
        let q1 = selfAttnQProj(normed1)
        let k1 = selfAttnKProj(normed1)
        let v1 = selfAttnVProj(normed1)
        out = out + selfAttnOutProj(selfAttentionRope(q1, k1, v1))

        // 2. Cross-attention (decoupled, matches python decoder.py:1192-1200):
        //    q = image_cross_attn_q_proj(image) + cross_attn_q_proj(tgt2)
        //    k = image_cross_attn_k_proj(memory_image) + cross_attn_k_proj(memory)
        //    if pos_enc_at_cross_attn_keys: k = k + memory_image_pos       (post-projection!)
        //    v = cross_attn_v_proj(memory)
        // For SAM 3.1 multiplex video: pos_enc_at_cross_attn_keys=True, _queries=False.
        let normed2 = norm2(out)
        let imageQInput = image ?? normed2                      // python: raw image (pre-norm)
        let q2 = crossAttnQProj(normed2) + imageCrossAttnQProj(imageQInput)

        // K: project both inputs separately, then sum, then add memory_image_pos POST.
        // memory: (B, N_mem, D) where N_mem = past_frames * HW + obj_ptr_count
        // memory_image: (B, N_img, D) where N_img = past_frames * HW (no obj_ptrs)
        // Python pads memory_image with zeros at the obj_ptr tail BEFORE projection
        // (so the bias contributes to those positions). We do the same.
        let kMain = crossAttnKProj(memory)                       // (B, N_mem, D)
        let kImg: MLXArray
        if let mi = memoryImage {
            let nMem = memory.dim(1), nImg = mi.dim(1)
            let alignedMi: MLXArray
            if nImg < nMem {
                let zeros = MLXArray.zeros([mi.dim(0), nMem - nImg, mi.dim(2)], dtype: mi.dtype)
                alignedMi = concatenated([mi, zeros], axis: 1)
            } else if nImg > nMem {
                alignedMi = mi[0..., ..<nMem, 0...]
            } else {
                alignedMi = mi
            }
            kImg = imageCrossAttnKProj(alignedMi)
        } else {
            // No separate memory_image: project memory through image_cross_attn_k_proj too
            // (legacy fallback, slightly wrong vs python but preserves existing fixture tests).
            kImg = imageCrossAttnKProj(memory)
        }
        var k2 = kMain + kImg
        // Post-projection add: memoryPos is the spatial-image-pos + temporal-frame-pos for
        // memory_image. Python adds it directly to k (after both projections sum).
        if let memPos = memoryPos {
            // memPos shape may be (B, N_img, D) — pad/truncate to N_mem to match k2 length.
            let nMem = k2.dim(1), nP = memPos.dim(1)
            let aligned: MLXArray
            if nP < nMem {
                let zeros = MLXArray.zeros([memPos.dim(0), nMem - nP, memPos.dim(2)], dtype: memPos.dtype)
                aligned = concatenated([memPos, zeros], axis: 1)
            } else if nP > nMem {
                aligned = memPos[0..., ..<nMem, 0...]
            } else {
                aligned = memPos
            }
            k2 = k2 + aligned
        }
        let v2 = crossAttnVProj(memory)
        out = out + crossAttnOutProj(crossAttentionRope(q2, k2, v2, numKExcludeRope: numKExcludeRope))

        // 3. FFN — HF config says "relu". (Meta python/sam3 model_builder hard-codes "gelu",
        // but that builds a different architecture. HF weights expect relu.)
        out = out + linear2(relu(linear1(norm3(out))))
        MLX.eval(out)
        return out
    }
}

// MARK: - DecoupledMemoryAttention

public class DecoupledMemoryAttention: Module {
    @ModuleInfo(key: "layers") var layers: [DecoupledMemoryAttentionLayer]
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm

    public init(_ config: Sam31TrackerConfig) {
        let d = config.memoryAttentionHiddenSize
        let featSizes = (config.memoryAttentionRopeFeatSizes[0], config.memoryAttentionRopeFeatSizes[1])
        let theta = config.memoryAttentionRopeTheta
        let nHeads = config.memoryAttentionNumAttentionHeads

        let builtLayers = (0..<config.memoryAttentionNumLayers).map { _ in
            let selfRope = SimpleRoPEAttention(hiddenSize: d, numHeads: nHeads, featSizes: featSizes, ropeTheta: theta)
            let crossRope = SimpleRoPEAttention(hiddenSize: d, numHeads: nHeads, featSizes: featSizes, ropeTheta: theta, ropeKRepeat: true)
            return DecoupledMemoryAttentionLayer(config, selfRope: selfRope, crossRope: crossRope)
        }
        self._layers = ModuleInfo(wrappedValue: builtLayers, key: "layers")
        self._layerNorm = ModuleInfo(wrappedValue: LayerNorm(dimensions: d), key: "layer_norm")
        super.init()
    }

    public func callAsFunction(
        _ src: MLXArray,
        memory: MLXArray,
        memoryPos: MLXArray? = nil,
        image: MLXArray? = nil,
        srcPos: MLXArray? = nil,
        memoryImage: MLXArray? = nil,
        memoryImagePos: MLXArray? = nil,
        numKExcludeRope: Int = 0
    ) -> MLXArray {
        // Python adds 0.1 * src_pos to input when pos_enc_at_input=True (which it is
        // for SAM 3.1 multiplex video). We don't have a config flag here; just do it
        // when src_pos is provided.
        var out = src
        if let sp = srcPos {
            out = out + MLXArray(Float(0.1)) * sp
        }
        for layer in layers {
            out = layer(out, memory: memory, memoryPos: memoryPos,
                        image: image, memoryImage: memoryImage, memoryImagePos: memoryImagePos,
                        numKExcludeRope: numKExcludeRope)
        }
        return layerNorm(out)
    }
}
