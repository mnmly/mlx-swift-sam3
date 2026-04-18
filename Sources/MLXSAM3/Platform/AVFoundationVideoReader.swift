// AVFoundationVideoReader.swift — AVAssetReader → MLXArray frame iterator.
//
// Decodes an asset frame-by-frame as `(1, H, W, 3)` MLXArrays normalized via
// `ImagePreprocess.pixelValues`. Inference-only; no seeking, no audio.

import AVFoundation
import CoreGraphics
import CoreImage
import CoreVideo
import Foundation
import MLX

public enum VideoReaderError: Error, CustomStringConvertible {
    case noVideoTrack
    case readerStartFailed(String)
    case pixelBufferUnavailable
    case cgImageCreationFailed

    public var description: String {
        switch self {
        case .noVideoTrack:             return "Asset contains no video track."
        case .readerStartFailed(let m): return "AVAssetReader failed to start: \(m)"
        case .pixelBufferUnavailable:   return "Sample buffer had no pixel buffer."
        case .cgImageCreationFailed:    return "Could not create CGImage from pixel buffer."
        }
    }
}

public struct VideoFrame {
    public let image: CGImage
    public let presentationTime: CMTime
    public let width: Int
    public let height: Int
}

public final class AVFoundationVideoReader {
    public let asset: AVAsset
    public let nominalFrameRate: Float
    public let naturalSize: CGSize

    private let reader: AVAssetReader
    private let output: AVAssetReaderTrackOutput
    private let ciContext = CIContext(options: nil)

    public init(url: URL) async throws {
        let asset = AVURLAsset(url: url)
        self.asset = asset

        let tracks = try await asset.loadTracks(withMediaType: .video)
        guard let track = tracks.first else { throw VideoReaderError.noVideoTrack }

        let (rate, size) = try await track.load(.nominalFrameRate, .naturalSize)
        self.nominalFrameRate = rate
        self.naturalSize = size

        let reader = try AVAssetReader(asset: asset)
        let settings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA),
        ]
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: settings)
        output.alwaysCopiesSampleData = false
        reader.add(output)
        guard reader.startReading() else {
            throw VideoReaderError.readerStartFailed(reader.error?.localizedDescription ?? "unknown")
        }
        self.reader = reader
        self.output = output
    }

    /// Pulls the next decoded frame. Returns nil at end-of-stream.
    public func nextFrame() throws -> VideoFrame? {
        guard let sample = output.copyNextSampleBuffer() else {
            if reader.status == .failed {
                throw VideoReaderError.readerStartFailed(reader.error?.localizedDescription ?? "unknown")
            }
            return nil
        }
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sample) else {
            throw VideoReaderError.pixelBufferUnavailable
        }
        let pts = CMSampleBufferGetPresentationTimeStamp(sample)
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let w = CVPixelBufferGetWidth(pixelBuffer)
        let h = CVPixelBufferGetHeight(pixelBuffer)
        guard let cg = ciContext.createCGImage(ciImage, from: ciImage.extent) else {
            throw VideoReaderError.cgImageCreationFailed
        }
        return VideoFrame(image: cg, presentationTime: pts, width: w, height: h)
    }

    /// Convenience: decode next frame straight into a preprocessed tensor.
    public func nextPixelValues(
        size: Int = 1008,
        mean: Float = 0.5,
        std: Float = 0.5,
        dtype: DType = .float16
    ) throws -> (MLXArray, VideoFrame)? {
        guard let frame = try nextFrame() else { return nil }
        let arr = try ImagePreprocess.pixelValues(
            from: frame.image, size: size, mean: mean, std: std, dtype: dtype)
        return (arr, frame)
    }

    public func cancel() {
        reader.cancelReading()
    }
}
