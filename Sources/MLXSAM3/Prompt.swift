// Prompt.swift — geometric prompt container + right-padded concatenation.
// Python source: sam3/model/geometry_encoders.py (class Prompt, concat_padded_sequences).
//
// Tensor layout mirrors Python: sequence-first, batch-second.
//   box_embeddings:   (N_box,  B, 4)
//   point_embeddings: (N_pt,   B, 2)
//   mask_embeddings:  (N_mask, B, 1, H, W)  — resolution arbitrary, carried verbatim
//   *_mask:           (B, N)  bool, true for padded slots
//   *_labels:         (N, B)  long, 0/1 (negatives/positives), +2/+4 for TL/BR corners
//                              when encode_boxes_as_points is on.

import Foundation
import MLX

public struct Prompt {
    public var boxEmbeddings: MLXArray?
    public var boxMask: MLXArray?
    public var boxLabels: MLXArray?

    public var pointEmbeddings: MLXArray?
    public var pointMask: MLXArray?
    public var pointLabels: MLXArray?

    public var maskEmbeddings: MLXArray?
    public var maskMask: MLXArray?
    public var maskLabels: MLXArray?

    public var isEmpty: Bool {
        boxEmbeddings == nil && pointEmbeddings == nil && maskEmbeddings == nil
    }

    public init(
        boxEmbeddings: MLXArray? = nil,
        pointEmbeddings: MLXArray? = nil,
        maskEmbeddings: MLXArray? = nil,
        boxLabels: MLXArray? = nil,
        pointLabels: MLXArray? = nil,
        maskLabels: MLXArray? = nil,
        boxMask: MLXArray? = nil,
        pointMask: MLXArray? = nil,
        maskMask: MLXArray? = nil
    ) {
        if boxEmbeddings == nil && pointEmbeddings == nil && maskEmbeddings == nil {
            return
        }

        // Resolve batch size and seq lengths.
        var bs = 0
        var boxSeq = 0, pointSeq = 0, maskSeq = 0
        if let b = boxEmbeddings {
            boxSeq = b.dim(0); bs = b.dim(1)
        }
        if let p = pointEmbeddings {
            pointSeq = p.dim(0)
            if bs == 0 { bs = p.dim(1) }
        }
        if let m = maskEmbeddings {
            maskSeq = m.dim(0)
            if bs == 0 { bs = m.dim(1) }
        }

        self.boxEmbeddings = boxEmbeddings ?? MLXArray.zeros([boxSeq, bs, 4])
        self.boxLabels = boxLabels ?? MLXArray.ones([boxSeq, bs], dtype: .int64)
        self.boxMask = boxMask ?? MLXArray.zeros([bs, boxSeq], dtype: .bool)

        self.pointEmbeddings = pointEmbeddings ?? MLXArray.zeros([pointSeq, bs, 2])
        self.pointLabels = pointLabels ?? MLXArray.ones([pointSeq, bs], dtype: .int64)
        self.pointMask = pointMask ?? MLXArray.zeros([bs, pointSeq], dtype: .bool)

        self.maskEmbeddings = maskEmbeddings
        self.maskLabels = maskLabels ?? MLXArray.ones([maskSeq, bs], dtype: .int64)
        self.maskMask = maskMask ?? MLXArray.zeros([bs, maskSeq], dtype: .bool)
    }
}

/// Concatenate two right-padded sequences so the result is contiguous and right-padded.
/// - seq1: (L1, B, H), mask1: (B, L1)
/// - seq2: (L2, B, H), mask2: (B, L2)
/// - Returns (seq, mask) with shapes ((L1+L2, B, H), (B, L1+L2)).
///
/// Mirrors Python `concat_padded_sequences`: seq1 is placed at the front; seq2 is scattered
/// starting at each row's actual seq1 length.
public func concatPaddedSequences(
    seq1: MLXArray, mask1: MLXArray,
    seq2: MLXArray, mask2: MLXArray
) -> (MLXArray, MLXArray) {
    let l1 = seq1.dim(0), bs = seq1.dim(1), h = seq1.dim(2)
    let l2 = seq2.dim(0)
    let maxLen = l1 + l2

    let actualL1 = MLX.sum((.!mask1).asType(.int32), axis: -1)  // (B,)
    let actualL2 = MLX.sum((.!mask2).asType(.int32), axis: -1)  // (B,)
    let finalLengths = actualL1 + actualL2

    let positions = MLXArray(0..<maxLen)[.newAxis, 0...]  // (1, maxLen)
    let concatMask = positions .>= finalLengths[0..., .newAxis]

    // Place seq1 in the first L1 rows; scatter seq2 after each row's actualL1.
    // We build index (maxLen, B) that picks source positions, with -1 for padding.
    // For simplicity, broadcast assignment via scatter emulated with where().
    //
    // Strategy: build two candidate tensors —
    //   a: seq1 padded to maxLen with zeros
    //   b: seq2 shifted right by actualL1 per-row
    // The boolean mask decides which one to take.
    let pad = MLXArray.zeros([l2, bs, h], dtype: seq1.dtype)
    let seq1Padded = concatenated([seq1, pad], axis: 0)  // (maxLen, B, H)

    // Build row indices for seq2 shift: for each (t in 0..<maxLen, b), pick seq2[t - actualL1[b]]
    // where that index is in [0, L2). Otherwise, zero.
    let tIdx = MLXArray(0..<maxLen)[0..., .newAxis]  // (maxLen, 1)
    let shifted = tIdx - actualL1[.newAxis, 0...]     // (maxLen, B)
    let validShift = MLX.logicalAnd(
        MLX.greaterEqual(shifted, MLXArray(Int32(0))),
        MLX.less(shifted, MLXArray(Int32(l2)))
    )
    let clamped = MLX.clip(shifted, min: 0, max: Int32(l2 - 1))

    // Gather seq2 along axis 0 using `clamped` as per-row indices.
    // Broadcast indices to (maxLen, B, H) and use takeAlong on axis 0.
    let clampedExpanded = MLX.broadcast(
        clamped.expandedDimensions(axis: -1),  // (maxLen, B, 1)
        to: [maxLen, bs, h]
    )
    let seq2Gathered = MLX.takeAlong(seq2, clampedExpanded, axis: 0)  // (maxLen, B, H)

    let useSeq2 = validShift.expandedDimensions(axis: -1)  // (T, B, 1)
    let combined = MLX.where(useSeq2, seq2Gathered, seq1Padded)
    return (combined, concatMask)
}
