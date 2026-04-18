import AVFoundation
import CoreGraphics
import CoreVideo
import XCTest
@testable import MLXSAM3

final class VideoIOTests: XCTestCase {

    /// Make a solid-color CGImage with a small marker square in one corner so
    /// frames are distinguishable by index.
    private func makeFrame(width: Int, height: Int, index: Int) -> CGImage {
        let cs = CGColorSpaceCreateDeviceRGB()
        let ctx = CGContext(
            data: nil, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: width * 4, space: cs,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
                | CGBitmapInfo.byteOrder32Big.rawValue)!
        let r = CGFloat((index * 37) % 256) / 255.0
        let g = CGFloat((index * 73) % 256) / 255.0
        let b = CGFloat((index * 131) % 256) / 255.0
        ctx.setFillColor(red: r, green: g, blue: b, alpha: 1)
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        ctx.setFillColor(red: 1, green: 1, blue: 1, alpha: 1)
        let mk = min(width, height) / 8
        ctx.fill(CGRect(x: 0, y: 0, width: mk, height: mk))
        return ctx.makeImage()!
    }

    func testVideoRoundTrip() async throws {
        let width = 128
        let height = 128
        let frameCount = 6
        let fps: Int32 = 30
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlxsam3_roundtrip_\(UUID().uuidString).mp4")

        // 1. Write a known sequence.
        let writer = try AVFoundationVideoWriter(
            url: tmp, width: width, height: height, frameRate: fps)
        for i in 0..<frameCount {
            let img = makeFrame(width: width, height: height, index: i)
            let pts = CMTime(value: CMTimeValue(i), timescale: fps)
            try writer.append(image: img, at: pts)
        }
        try await writer.finish()
        XCTAssertTrue(FileManager.default.fileExists(atPath: tmp.path))

        // 2. Read it back.
        let reader = try await AVFoundationVideoReader(url: tmp)
        XCTAssertEqual(Int(reader.naturalSize.width), width)
        XCTAssertEqual(Int(reader.naturalSize.height), height)

        var decoded = 0
        while let frame = try reader.nextFrame() {
            XCTAssertEqual(frame.width, width)
            XCTAssertEqual(frame.height, height)
            decoded += 1
        }
        XCTAssertEqual(decoded, frameCount)

        try? FileManager.default.removeItem(at: tmp)
    }

    func testReaderNextPixelValuesShape() async throws {
        let width = 64
        let height = 64
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlxsam3_px_\(UUID().uuidString).mp4")
        let writer = try AVFoundationVideoWriter(
            url: tmp, width: width, height: height, frameRate: 30)
        try writer.append(image: makeFrame(width: width, height: height, index: 0),
                          at: CMTime(value: 0, timescale: 30))
        try await writer.finish()

        let reader = try await AVFoundationVideoReader(url: tmp)
        guard let (px, _) = try reader.nextPixelValues(size: 48) else {
            XCTFail("no frame decoded")
            return
        }
        XCTAssertEqual(px.shape, [1, 48, 48, 3])
        try? FileManager.default.removeItem(at: tmp)
    }
}
