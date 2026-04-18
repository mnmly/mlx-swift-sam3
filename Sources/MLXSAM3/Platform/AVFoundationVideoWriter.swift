// AVFoundationVideoWriter.swift — `CGImage` overlay frames → H.264 mp4.
//
// Thin wrapper over `AVAssetWriter`. Caller pushes `CGImage`s with monotonic
// presentation times; writer encodes into an mp4 on disk.

import AVFoundation
import CoreGraphics
import CoreVideo
import Foundation

public enum VideoWriterError: Error, CustomStringConvertible {
    case writerCreationFailed(String)
    case writerStartFailed(String)
    case pixelBufferPoolUnavailable
    case pixelBufferAllocFailed
    case appendFailed(String)
    case finishFailed(String)

    public var description: String {
        switch self {
        case .writerCreationFailed(let m): return "AVAssetWriter init failed: \(m)"
        case .writerStartFailed(let m):    return "AVAssetWriter start failed: \(m)"
        case .pixelBufferPoolUnavailable:  return "Pixel buffer pool was unavailable."
        case .pixelBufferAllocFailed:      return "CVPixelBuffer allocation failed."
        case .appendFailed(let m):         return "appendPixelBuffer failed: \(m)"
        case .finishFailed(let m):         return "finishWriting failed: \(m)"
        }
    }
}

public final class AVFoundationVideoWriter {
    private let writer: AVAssetWriter
    private let input: AVAssetWriterInput
    private let adaptor: AVAssetWriterInputPixelBufferAdaptor
    public let width: Int
    public let height: Int
    private var started = false

    public init(url: URL, width: Int, height: Int, frameRate: Int32 = 30) throws {
        try? FileManager.default.removeItem(at: url)
        do {
            self.writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        } catch {
            throw VideoWriterError.writerCreationFailed(error.localizedDescription)
        }
        self.width = width
        self.height = height

        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
        ]
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.expectsMediaDataInRealTime = false
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA),
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
        ]
        self.adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input, sourcePixelBufferAttributes: attrs)
        if writer.canAdd(input) { writer.add(input) }
        self.input = input
    }

    public func append(image: CGImage, at time: CMTime) throws {
        if !started {
            guard writer.startWriting() else {
                throw VideoWriterError.writerStartFailed(writer.error?.localizedDescription ?? "unknown")
            }
            writer.startSession(atSourceTime: time)
            started = true
        }
        guard let pool = adaptor.pixelBufferPool else {
            throw VideoWriterError.pixelBufferPoolUnavailable
        }
        var pb: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pb) == kCVReturnSuccess,
              let buffer = pb else {
            throw VideoWriterError.pixelBufferAllocFailed
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        let base = CVPixelBufferGetBaseAddress(buffer)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo: UInt32 =
            CGImageAlphaInfo.premultipliedFirst.rawValue
            | CGBitmapInfo.byteOrder32Little.rawValue
        guard let ctx = CGContext(
            data: base, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: bytesPerRow,
            space: colorSpace, bitmapInfo: bitmapInfo)
        else { throw VideoWriterError.pixelBufferAllocFailed }
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        // Block until the input is ready — we're not real-time.
        while !input.isReadyForMoreMediaData {
            Thread.sleep(forTimeInterval: 0.005)
        }
        if !adaptor.append(buffer, withPresentationTime: time) {
            throw VideoWriterError.appendFailed(writer.error?.localizedDescription ?? "unknown")
        }
    }

    public func finish() async throws {
        input.markAsFinished()
        await writer.finishWriting()
        if writer.status == .failed {
            throw VideoWriterError.finishFailed(writer.error?.localizedDescription ?? "unknown")
        }
    }
}
