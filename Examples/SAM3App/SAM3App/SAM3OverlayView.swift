import AppKit
import CoreGraphics
import Foundation
import SwiftUI

struct SAM3OverlayView: View {
    let cgImage: CGImage
    let result: SAM3DisplayResult?

    static let palette: [SIMD4<Float>] = [
        SIMD4(0.0, 0.58, 1.0, 1),
        SIMD4(1.0, 0.36, 0.20, 1),
        SIMD4(0.18, 0.78, 0.42, 1),
        SIMD4(0.95, 0.72, 0.10, 1),
        SIMD4(0.72, 0.30, 0.88, 1),
        SIMD4(0.95, 0.22, 0.48, 1),
        SIMD4(0.22, 0.70, 0.72, 1),
        SIMD4(0.58, 0.58, 0.62, 1),
    ]

    var body: some View {
        let imgW = CGFloat(cgImage.width)
        let imgH = CGFloat(cgImage.height)

        GeometryReader { geo in
            let scale = min(geo.size.width / imgW, geo.size.height / imgH)
            let drawW = imgW * scale
            let drawH = imgH * scale
            let offsetX = (geo.size.width - drawW) / 2
            let offsetY = (geo.size.height - drawH) / 2

            ZStack {
                Image(cgImage, scale: 1.0, label: Text("Source"))
                    .resizable()
                    .aspectRatio(contentMode: .fit)

                if let result, let masksImage = maskOverlayImage(result: result) {
                    Image(masksImage, scale: 1.0, label: Text("Masks"))
                        .resizable()
                        .frame(width: drawW, height: drawH)
                        .position(x: geo.size.width / 2, y: geo.size.height / 2)
                        .opacity(0.55)
                        .allowsHitTesting(false)
                }

                if let result {
                    Canvas { context, _ in
                        drawBoxes(in: context, result: result, scale: scale, offset: CGPoint(x: offsetX, y: offsetY))
                    }
                    .allowsHitTesting(false)
                }
            }
        }
    }

    private func drawBoxes(in context: GraphicsContext, result: SAM3DisplayResult, scale: CGFloat, offset: CGPoint) {
        for (index, detection) in result.detections.enumerated() {
            let b = detection.box
            guard b.count == 4 else { continue }
            let rect = CGRect(
                x: CGFloat(b[0]) * scale + offset.x,
                y: CGFloat(b[1]) * scale + offset.y,
                width: CGFloat(b[2] - b[0]) * scale,
                height: CGFloat(b[3] - b[1]) * scale
            )
            // Persistent track id (when present) gives a stable color across
            // frames; per-frame predictions fall back to iteration index.
            let colorIndex = detection.trackId ?? index
            let color = colorAt(colorIndex)
            context.stroke(Path(rect), with: .color(color), lineWidth: 2)
            let tag = detection.trackId.map { String(format: "id %d  %.2f", $0, detection.score) }
                ?? String(format: "#%d %.2f", index + 1, detection.score)
            drawTag(text: tag, at: rect.origin, color: color, in: context)
        }
    }

    private func maskOverlayImage(result: SAM3DisplayResult) -> CGImage? {
        let w = result.imageWidth
        let h = result.imageHeight
        guard w > 0, h > 0, !result.detections.isEmpty else { return nil }

        var pixels = [UInt8](repeating: 0, count: h * w * 4)
        for (i, detection) in result.detections.enumerated() {
            let colorIndex = detection.trackId ?? i
            let color = Self.palette[colorIndex % Self.palette.count]
            let r = UInt8(color.x * 255)
            let g = UInt8(color.y * 255)
            let b = UInt8(color.z * 255)
            let limit = min(detection.mask.count, h * w)
            for px in 0..<limit where detection.mask[px] > 0 {
                let pi = px * 4
                pixels[pi] = r
                pixels[pi + 1] = g
                pixels[pi + 2] = b
                pixels[pi + 3] = 170
            }
        }

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let provider = CGDataProvider(data: Data(pixels) as CFData) else { return nil }
        return CGImage(
            width: w,
            height: h,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: w * 4,
            space: colorSpace,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        )
    }

    private func colorAt(_ index: Int) -> Color {
        let v = Self.palette[index % Self.palette.count]
        return Color(red: Double(v.x), green: Double(v.y), blue: Double(v.z), opacity: Double(v.w))
    }

    private func drawTag(text: String, at origin: CGPoint, color: Color, in context: GraphicsContext) {
        let fontSize: CGFloat = 11
        let font = NSFont.monospacedSystemFont(ofSize: fontSize, weight: .semibold)
        let textWidth = text.size(withAttributes: [.font: font]).width + 8
        let textHeight: CGFloat = fontSize + 6
        let tagY = max(0, origin.y - textHeight)
        let tag = CGRect(x: origin.x, y: tagY, width: textWidth, height: textHeight)
        context.fill(Path(tag), with: .color(color))
        context.draw(
            Text(text).font(.system(size: fontSize, weight: .semibold)).foregroundStyle(.white),
            at: CGPoint(x: tag.midX, y: tag.midY)
        )
    }
}

struct SAM3ResultsSummary: View {
    let result: SAM3DisplayResult

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Results").font(.headline)
            Text("Detections: \(result.count)").font(.caption)
            if let best = result.detections.map(\.score).max() {
                Text(String(format: "Best score: %.2f", best)).font(.caption)
            }
        }
    }
}
