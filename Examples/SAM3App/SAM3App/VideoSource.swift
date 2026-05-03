import AVFoundation
import CoreGraphics
import CoreMedia
import Foundation

final class VideoSource {
    private let generator: AVAssetImageGenerator
    private let timestamps: [CMTime]
    private var index = 0

    let totalFrames: Int
    let nominalFPS: Double

    init(url: URL, frameStride: Int = 1) async throws {
        let asset = AVURLAsset(url: url)
        let tracks = try await asset.loadTracks(withMediaType: .video)
        guard let track = tracks.first else {
            throw NSError(
                domain: "SAM3App.VideoSource",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "No video track in \(url.lastPathComponent)."]
            )
        }

        let duration = try await asset.load(.duration)
        let fps = try await track.load(.nominalFrameRate)
        nominalFPS = Double(max(fps, 1))

        let durationSec = CMTimeGetSeconds(duration)
        let totalApprox = max(1, Int(durationSec * nominalFPS))
        let stride = max(1, frameStride)

        var times: [CMTime] = []
        var i = 0
        while i < totalApprox {
            times.append(CMTime(seconds: Double(i) / nominalFPS, preferredTimescale: 600))
            i += stride
        }
        timestamps = times
        totalFrames = times.count

        let gen = AVAssetImageGenerator(asset: asset)
        gen.appliesPreferredTrackTransform = true
        gen.requestedTimeToleranceBefore = .zero
        gen.requestedTimeToleranceAfter = CMTime(seconds: 1.0 / nominalFPS, preferredTimescale: 600)
        generator = gen
    }

    func nextFrame() -> CGImage? {
        guard index < timestamps.count else { return nil }
        let timestamp = timestamps[index]
        index += 1
        return try? generator.copyCGImage(at: timestamp, actualTime: nil)
    }

    func release() {
        generator.cancelAllCGImageGeneration()
    }
}
