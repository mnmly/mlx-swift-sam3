// ImageUtils.swift — cross-platform image → MLXArray preprocessing.
//
// Converts NSImage / UIImage / CGImage into the (1, H, W, 3) Float16
// tensor SAM 3 / 3.1 expects. Values are normalized to [0, 1].

import CoreGraphics
import Foundation
import MLX

#if canImport(AppKit)
import AppKit
public typealias PlatformImage = NSImage
#elseif canImport(UIKit)
import UIKit
public typealias PlatformImage = UIImage
#endif

public enum ImagePreprocessError: Error, CustomStringConvertible {
    case cgImageUnavailable
    case contextCreationFailed
    case pixelDataUnavailable

    public var description: String {
        switch self {
        case .cgImageUnavailable:   return "Could not obtain a CGImage from the source image."
        case .contextCreationFailed: return "Could not create a CGBitmapContext for resizing."
        case .pixelDataUnavailable:  return "Bitmap context returned no pixel data."
        }
    }
}

public enum ImagePreprocess {

    /// Resize a `CGImage` to `size × size` and return an RGB uint8 buffer
    /// in row-major layout `[H, W, 3]`.
    public static func resizedRGBBytes(_ image: CGImage, size: Int) throws -> [UInt8] {
        let bytesPerRow = size * 4
        var buffer = [UInt8](repeating: 0, count: size * size * 4)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo: UInt32 =
            CGImageAlphaInfo.premultipliedLast.rawValue
            | CGBitmapInfo.byteOrder32Big.rawValue

        guard
            let context = buffer.withUnsafeMutableBytes({ raw -> CGContext? in
                CGContext(
                    data: raw.baseAddress,
                    width: size,
                    height: size,
                    bitsPerComponent: 8,
                    bytesPerRow: bytesPerRow,
                    space: colorSpace,
                    bitmapInfo: bitmapInfo)
            })
        else { throw ImagePreprocessError.contextCreationFailed }

        context.interpolationQuality = .high
        context.draw(image, in: CGRect(x: 0, y: 0, width: size, height: size))

        // Strip alpha — keep RGB only.
        var rgb = [UInt8](repeating: 0, count: size * size * 3)
        for i in 0..<(size * size) {
            rgb[i * 3 + 0] = buffer[i * 4 + 0]
            rgb[i * 3 + 1] = buffer[i * 4 + 1]
            rgb[i * 3 + 2] = buffer[i * 4 + 2]
        }
        return rgb
    }

    /// Convert a `CGImage` to a normalized `(1, size, size, 3)` tensor.
    /// Default `mean=0.5, std=0.5` matches `sam3_image_processor.py` (output in [-1, 1]).
    /// Pass `mean=0, std=1` to keep the legacy [0, 1] range.
    public static func pixelValues(
        from image: CGImage,
        size: Int = 1008,
        mean: Float = 0.5,
        std: Float = 0.5,
        dtype: DType = .float16
    ) throws -> MLXArray {
        let rgb = try resizedRGBBytes(image, size: size)
        let inv = 1.0 / (255.0 * std)
        let offset = mean / std
        let floats = rgb.map { Float($0) * inv - offset }
        let arr = MLXArray(floats, [1, size, size, 3])
        return arr.asType(dtype)
    }

    #if canImport(AppKit) || canImport(UIKit)
    /// Convert an `NSImage` (macOS) or `UIImage` (iOS/tvOS/visionOS) to a
    /// normalized `(1, size, size, 3)` float16 tensor.
    public static func pixelValues(
        from image: PlatformImage,
        size: Int = 1008,
        mean: Float = 0.5,
        std: Float = 0.5,
        dtype: DType = .float16
    ) throws -> MLXArray {
        #if canImport(AppKit)
        var rect = CGRect(x: 0, y: 0, width: image.size.width, height: image.size.height)
        guard let cg = image.cgImage(forProposedRect: &rect, context: nil, hints: nil) else {
            throw ImagePreprocessError.cgImageUnavailable
        }
        #else
        guard let cg = image.cgImage else { throw ImagePreprocessError.cgImageUnavailable }
        #endif
        return try pixelValues(from: cg, size: size, mean: mean, std: std, dtype: dtype)
    }
    #endif
}
