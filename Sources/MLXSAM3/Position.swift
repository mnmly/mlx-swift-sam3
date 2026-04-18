// Position.swift — sinusoidal 2D and rotary position encodings.
// Python source: sam3/position.py

import Foundation
import MLX
import MLXNN

/// Sinusoidal 2D position embedding (used in DETR encoder/decoder and memory encoder).
public class PositionEmbeddingSine: Module {
    public let numPosFeats: Int
    public let temperature: Float
    public let normalize: Bool
    public let scale: Float

    public init(
        numPosFeats: Int = 256,
        temperature: Float = 10000.0,
        normalize: Bool = true,
        scale: Float? = nil
    ) {
        self.numPosFeats = numPosFeats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale ?? Float(2.0 * Double.pi)
        super.init()
    }

    /// - Parameter x: (B, H, W, C) feature map
    /// - Returns: (B, H, W, numPosFeats*2) position encoding
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0), h = x.dim(1), w = x.dim(2)

        var yEmbed = broadcast(
            (MLXArray(0..<h).asType(.float32) + 1).reshaped([1, h, 1]),
            to: [b, h, w]
        )
        var xEmbed = broadcast(
            (MLXArray(0..<w).asType(.float32) + 1).reshaped([1, 1, w]),
            to: [b, h, w]
        )

        if normalize {
            let eps: Float = 1e-6
            // y_embed / (y_embed[:, -1:, :] + eps) * scale
            yEmbed = yEmbed / (yEmbed[0..., (-1)..., 0...] + eps) * scale
            // x_embed / (x_embed[:, :, -1:] + eps) * scale
            xEmbed = xEmbed / (xEmbed[0..., 0..., (-1)...] + eps) * scale
        }

        // dim_t = temperature ** (2 * (arange // 2) / numPosFeats)
        let dimIdx = MLXArray(0..<numPosFeats).asType(.float32)
        let halfDimIdx = MLX.floor(dimIdx / 2)
        let logTemp = Foundation.log(temperature)
        let dimT = MLX.exp(logTemp * (2.0 * halfDimIdx / Float(numPosFeats)))

        var posX = xEmbed[.ellipsis, .newAxis] / dimT  // (B, H, W, D)
        var posY = yEmbed[.ellipsis, .newAxis] / dimT

        // Stack sin(even) and cos(odd) then flatten: [sin0, cos0, sin1, cos1, ...]
        let d = numPosFeats
        posX = stacked([
            MLX.sin(posX.reshaped([b, h, w, d / 2, 2])[.ellipsis, 0]),
            MLX.cos(posX.reshaped([b, h, w, d / 2, 2])[.ellipsis, 1])
        ], axis: -1).reshaped([b, h, w, d])
        posY = stacked([
            MLX.sin(posY.reshaped([b, h, w, d / 2, 2])[.ellipsis, 0]),
            MLX.cos(posY.reshaped([b, h, w, d / 2, 2])[.ellipsis, 1])
        ], axis: -1).reshaped([b, h, w, d])

        return concatenated([posY, posX], axis: -1)  // (B, H, W, 2*D)
    }

    /// Encode 1D normalized coordinates (used by SequenceGeometryEncoder).
    /// - x, y: shape (N,), normalized to [0, 1].
    /// - Returns (posX, posY) each of shape (N, numPosFeats), sin/cos interleaved.
    public func encodeXY(x: MLXArray, y: MLXArray) -> (MLXArray, MLXArray) {
        let xEmbed = x * scale
        let yEmbed = y * scale

        let dimIdx = MLXArray(0..<numPosFeats).asType(.float32)
        let halfDimIdx = MLX.floor(dimIdx / 2)
        let logTemp = Foundation.log(temperature)
        let dimT = MLX.exp(logTemp * (2.0 * halfDimIdx / Float(numPosFeats)))

        let posX = xEmbed[0..., .newAxis] / dimT  // (N, D)
        let posY = yEmbed[0..., .newAxis] / dimT
        let d = numPosFeats
        let n = x.dim(0)

        let encX = stacked([
            MLX.sin(posX.reshaped([n, d / 2, 2])[0..., 0..., 0]),
            MLX.cos(posX.reshaped([n, d / 2, 2])[0..., 0..., 1])
        ], axis: -1).reshaped([n, d])
        let encY = stacked([
            MLX.sin(posY.reshaped([n, d / 2, 2])[0..., 0..., 0]),
            MLX.cos(posY.reshaped([n, d / 2, 2])[0..., 0..., 1])
        ], axis: -1).reshaped([n, d])
        return (encX, encY)
    }

    /// Encode a batch of boxes in normalized (cx, cy, w, h) to (N, 2*numPosFeats + 2).
    /// Returns [posY, posX, h, w] concatenated along last dim (matches Python `encode_boxes`).
    public func encodeBoxes(x: MLXArray, y: MLXArray, w: MLXArray, h: MLXArray) -> MLXArray {
        let (encX, encY) = encodeXY(x: x, y: y)
        return concatenated([encY, encX, h[0..., .newAxis], w[0..., .newAxis]], axis: -1)
    }
}

/// Compute 2D axial rotary position embeddings for the ViT backbone.
/// Returns (cos, sin) each of shape (end_x*end_y, dim).
public func computeAxialCis(dim: Int, endX: Int, endY: Int, theta: Float = 10000.0) -> (MLXArray, MLXArray) {
    let n = dim / 4
    let indices = MLXArray(0..<n).asType(.float32) * 4.0
    let logTheta = Foundation.log(theta)
    let freqs = MLX.exp(-logTheta * (indices / Float(dim)))

    let total = endX * endY
    let flatIdx = MLXArray(0..<total)
    let xPositions = (flatIdx % endX).asType(.float32)
    
    // FIX: Use floorDivide to prevent true division from generating fractional Y-coordinates
    let yPositions = MLX.floorDivide(flatIdx, endX).asType(.float32)

    let freqsX = outer(xPositions, freqs)  // (N, dim//4)
    let freqsY = outer(yPositions, freqs)  // (N, dim//4)

    // Concatenate x and y: (N, dim//2), then repeat_interleave(2) → (N, dim)
    var invFreq = concatenated([freqsX, freqsY], axis: -1)  // (N, dim//2)
    invFreq = stacked([invFreq, invFreq], axis: -1).reshaped([total, dim])

    return (MLX.cos(invFreq), MLX.sin(invFreq))
}

/// Pairwise rotation: (x0,x1,x2,x3,...) -> (-x1,x0,-x3,x2,...)
public func rotatePairwise(_ x: MLXArray) -> MLXArray {
    let shape = x.shape
    let d = shape[shape.count - 1]
    let reshaped = x.reshaped(Array(shape.dropLast()) + [d / 2, 2])
    let x1 = reshaped[.ellipsis, 0]
    let x2 = reshaped[.ellipsis, 1]
    return stacked([-x2, x1], axis: -1).reshaped(shape)
}

/// Apply 2D rotary position encoding (ViT backbone, VitAttention).
/// - xq/xk: (B, numHeads, N, D); cos/sin: (N, D)
public func applyRotaryEnc(
    xq: MLXArray, xk: MLXArray,
    cos: MLXArray, sin: MLXArray
) -> (MLXArray, MLXArray) {
    return (
        xq * cos + rotatePairwise(xq) * sin,
        xk * cos + rotatePairwise(xk) * sin
    )
}

/// Apply 1D RoPE for tracker memory attention.
/// - xq: (B, N_q, H, D); xk: (B, N_k, H, D)
/// - freqsCos/freqsSin: (N, D//2)
/// - repeatFreqsK: tile freqs to match key length (cross-attn)
public func applyRotaryEnc1d(
    xq: MLXArray, xk: MLXArray,
    freqsCos: MLXArray, freqsSin: MLXArray,
    repeatFreqsK: Bool = false
) -> (MLXArray, MLXArray) {
    let cosQ = freqsCos[.newAxis, ..<xq.dim(1), .newAxis, 0...]
    let sinQ = freqsSin[.newAxis, ..<xq.dim(1), .newAxis, 0...]

    let cosK: MLXArray
    let sinK: MLXArray
    if repeatFreqsK {
        let nk = xk.dim(1)
        let nf = freqsCos.dim(0)
        let repeats = (nk + nf - 1) / nf
        cosK = MLX.tiled(freqsCos, repetitions: [repeats, 1])[.newAxis, ..<nk, .newAxis, 0...]
        sinK = MLX.tiled(freqsSin, repetitions: [repeats, 1])[.newAxis, ..<nk, .newAxis, 0...]
    } else {
        cosK = freqsCos[.newAxis, ..<xk.dim(1), .newAxis, 0...]
        sinK = freqsSin[.newAxis, ..<xk.dim(1), .newAxis, 0...]
    }

    // Split into real (even) and imaginary (odd) parts via reshape
    let dq = xq.dim(-1), dk = xk.dim(-1)
    let qPairs = xq.reshaped(Array(xq.shape.dropLast()) + [dq / 2, 2])
    let kPairs = xk.reshaped(Array(xk.shape.dropLast()) + [dk / 2, 2])
    let qR = qPairs[.ellipsis, 0], qI = qPairs[.ellipsis, 1]
    let kR = kPairs[.ellipsis, 0], kI = kPairs[.ellipsis, 1]

    let qOutR = qR * cosQ - qI * sinQ
    let qOutI = qR * sinQ + qI * cosQ
    let kOutR = kR * cosK - kI * sinK
    let kOutI = kR * sinK + kI * cosK

    let xqOut = stacked([qOutR, qOutI], axis: -1).reshaped(xq.shape)
    let xkOut = stacked([kOutR, kOutI], axis: -1).reshaped(xk.shape)
    return (xqOut, xkOut)
}

/// Initialize 2D RoPE frequencies for memory attention.
/// Returns (cos, sin) each of shape (feat_h*feat_w, dim//2).
public func init2dFreqs(dim: Int, featH: Int, featW: Int, theta: Float = 10000.0) -> (MLXArray, MLXArray) {
    let half = dim / 2
    // arange(0, half, 2): shape (half//2,)
    let n = half / 2
    let indices = MLXArray(0..<n).asType(.float32) * 2.0
    let logTheta = Foundation.log(theta)
    let freqs = MLX.exp(-logTheta * (indices / Float(half)))

    // meshgrid(ij): grid_y[i,j]=i, grid_x[i,j]=j → flatten
    let tY = MLXArray(0..<featH).asType(.float32)
    let tX = MLXArray(0..<featW).asType(.float32)
    let gridY = MLX.tiled(tY.reshaped([featH, 1]), repetitions: [1, featW]).reshaped([-1])
    let gridX = MLX.tiled(tX.reshaped([1, featW]), repetitions: [featH, 1]).reshaped([-1])

    let freqsY = outer(gridY, freqs)  // (H*W, half//2)
    let freqsX = outer(gridX, freqs)

    let freqsAll = concatenated([freqsY, freqsX], axis: -1)  // (H*W, half)
    return (MLX.cos(freqsAll), MLX.sin(freqsAll))
}
