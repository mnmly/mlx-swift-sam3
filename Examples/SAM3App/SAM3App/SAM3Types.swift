import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXSAM3

extension SAM3Predictor: @unchecked @retroactive Sendable {}

struct SAM3DisplayResult: Sendable {
    var detections: [SAM3DisplayDetection]
    var imageWidth: Int
    var imageHeight: Int

    var count: Int { detections.count }
}

struct SAM3DisplayDetection: Identifiable, Sendable {
    var id: Int
    var box: [Float]
    var score: Float
    var mask: [UInt8]
}

struct SAM3ProcessedFrame: Sendable {
    let image: CGImage
    let result: SAM3DisplayResult
}

enum SAM3AppError: LocalizedError {
    case modelNotLoaded
    case noImage
    case emptyPrompt

    var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            "Model not loaded. Select a SAM 3.1 model directory first."
        case .noImage:
            "No image selected."
        case .emptyPrompt:
            "Enter a text prompt before running inference."
        }
    }
}

nonisolated func makeDisplayResult(from prediction: SAM3ImagePrediction, width: Int, height: Int) -> SAM3DisplayResult {
    let n = prediction.scores.dim(0)
    guard n > 0 else {
        return SAM3DisplayResult(detections: [], imageWidth: width, imageHeight: height)
    }

    let boxes = prediction.boxes.asType(.float32).asArray(Float.self)
    let scores = prediction.scores.asType(.float32).asArray(Float.self)
    let masks = prediction.masks.asType(.uint8).asArray(UInt8.self)
    let maskStride = width * height

    var detections: [SAM3DisplayDetection] = []
    detections.reserveCapacity(n)
    for i in 0..<n {
        let box = Array(boxes[(i * 4)..<(i * 4 + 4)])
        let start = i * maskStride
        let end = min(start + maskStride, masks.count)
        detections.append(SAM3DisplayDetection(
            id: i,
            box: box,
            score: scores[i],
            mask: Array(masks[start..<end])
        ))
    }

    return SAM3DisplayResult(detections: detections, imageWidth: width, imageHeight: height)
}

nonisolated func loadCGImage(from url: URL) -> CGImage? {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
    return CGImageSourceCreateImageAtIndex(source, 0, nil)
}
