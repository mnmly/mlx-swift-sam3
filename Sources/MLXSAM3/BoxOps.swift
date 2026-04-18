// BoxOps.swift — bounding-box utilities. Python source: sam3/model/box_ops.py

import Foundation
import MLX

public enum BoxOps {

    public static func boxCxcywhToXyxy(_ x: MLXArray) -> MLXArray {
        let xc = x[.ellipsis, 0]
        let yc = x[.ellipsis, 1]
        let w = x[.ellipsis, 2]
        let h = x[.ellipsis, 3]
        return stacked([xc - 0.5 * w, yc - 0.5 * h, xc + 0.5 * w, yc + 0.5 * h], axis: -1)
    }

    public static func boxXyxyToCxcywh(_ x: MLXArray) -> MLXArray {
        let x0 = x[.ellipsis, 0]
        let y0 = x[.ellipsis, 1]
        let x1 = x[.ellipsis, 2]
        let y1 = x[.ellipsis, 3]
        return stacked([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], axis: -1)
    }

    public static func boxCxcywhToXywh(_ x: MLXArray) -> MLXArray {
        let xc = x[.ellipsis, 0]
        let yc = x[.ellipsis, 1]
        let w = x[.ellipsis, 2]
        let h = x[.ellipsis, 3]
        return stacked([xc - 0.5 * w, yc - 0.5 * h, w, h], axis: -1)
    }

    public static func boxXywhToXyxy(_ x: MLXArray) -> MLXArray {
        let x0 = x[.ellipsis, 0]
        let y0 = x[.ellipsis, 1]
        let w = x[.ellipsis, 2]
        let h = x[.ellipsis, 3]
        return stacked([x0, y0, x0 + w, y0 + h], axis: -1)
    }

    public static func boxXywhToCxcywh(_ x: MLXArray) -> MLXArray {
        let x0 = x[.ellipsis, 0]
        let y0 = x[.ellipsis, 1]
        let w = x[.ellipsis, 2]
        let h = x[.ellipsis, 3]
        return stacked([x0 + 0.5 * w, y0 + 0.5 * h, w, h], axis: -1)
    }

    public static func boxXyxyToXywh(_ x: MLXArray) -> MLXArray {
        let x0 = x[.ellipsis, 0]
        let y0 = x[.ellipsis, 1]
        let x1 = x[.ellipsis, 2]
        let y1 = x[.ellipsis, 3]
        return stacked([x0, y0, x1 - x0, y1 - y0], axis: -1)
    }

    /// Area of boxes in xyxy format; input shape (..., 4) → (...).
    public static func boxArea(_ boxes: MLXArray) -> MLXArray {
        let x0 = boxes[.ellipsis, 0]
        let y0 = boxes[.ellipsis, 1]
        let x1 = boxes[.ellipsis, 2]
        let y1 = boxes[.ellipsis, 3]
        return (x1 - x0) * (y1 - y0)
    }

    /// Pairwise IoU for boxes in xyxy format.
    /// - boxes1: (..., N, 4), boxes2: (..., M, 4) → (iou, union) each (..., N, M)
    public static func boxIoU(_ boxes1: MLXArray, _ boxes2: MLXArray) -> (MLXArray, MLXArray) {
        let area1 = boxArea(boxes1)
        let area2 = boxArea(boxes2)

        let b1Lt = boxes1[.ellipsis, 0..., .newAxis, ..<2]
        let b2Lt = boxes2[.ellipsis, .newAxis, 0..., ..<2]
        let b1Rb = boxes1[.ellipsis, 0..., .newAxis, 2...]
        let b2Rb = boxes2[.ellipsis, .newAxis, 0..., 2...]

        let lt = MLX.maximum(b1Lt, b2Lt)
        let rb = MLX.minimum(b1Rb, b2Rb)
        let wh = MLX.clip(rb - lt, min: 0)
        let inter = wh[.ellipsis, 0] * wh[.ellipsis, 1]

        let union = area1[.ellipsis, .newAxis] + area2[.ellipsis, .newAxis, 0...] - inter
        return (inter / union, union)
    }

    /// Generalized IoU (https://giou.stanford.edu/), boxes in xyxy.
    public static func generalizedBoxIoU(_ boxes1: MLXArray, _ boxes2: MLXArray) -> MLXArray {
        let (iou, union) = boxIoU(boxes1, boxes2)

        let b1Lt = boxes1[.ellipsis, 0..., .newAxis, ..<2]
        let b2Lt = boxes2[.ellipsis, .newAxis, 0..., ..<2]
        let b1Rb = boxes1[.ellipsis, 0..., .newAxis, 2...]
        let b2Rb = boxes2[.ellipsis, .newAxis, 0..., 2...]

        let lt = MLX.minimum(b1Lt, b2Lt)
        let rb = MLX.maximum(b1Rb, b2Rb)
        let wh = MLX.clip(rb - lt, min: 0)
        let area = wh[.ellipsis, 0] * wh[.ellipsis, 1]

        return iou - (area - union) / area
    }
}
