// Sam31VideoTrackFull — CLI: SAM 3.1 video tracking with persistent IDs.
//
// IMPORTANT: build via xcodebuild, NOT `swift build` / `swift run`. MLX needs
// a Metal-capable toolchain at runtime.
//
// Build:
//     xcodebuild -scheme Sam31VideoTrackFull -destination 'platform=macOS' \
//         -configuration release -derivedDataPath .xcdd build
//
// Run:
//     .xcdd/Build/Products/Release/Sam31VideoTrackFull \
//         --input  path/to/video.mp4 \
//         --prompt "a cat" --tokenizer <snapshot-dir> \
//         --weights <snapshot-dir> \
//         --output path/to/overlay.mp4 \
//         [--redetect-every 15] [--threshold 0.5]
//
// Uses Sam31VideoTracker (slim v2): re-detect every N frames + trackStep-driven
// propagation in between. Per-track stable colors + ID labels. See
// Sources/MLXSAM3/Sam31VideoTracker.swift for known gaps vs python parity.

import AVFoundation
import CoreGraphics
import CoreMedia
import Foundation
import ImageIO
import MLX
import MLXSAM3
import UniformTypeIdentifiers

struct S31VTFArgs {
    var input: URL
    var prompt: String?
    var tokenizer: URL?
    var tokens: URL?
    var weights: URL
    var output: URL
    var resolution: Int = 1008
    var threshold: Float = 0.5
    var redetectEvery: Int = 15
    var frameRate: Int32 = 30
    var dtype: DType = .bfloat16
    var maxFrames: Int?
}

func parse31VTF() -> S31VTFArgs {
    var dict: [String: String] = [:]
    var it = CommandLine.arguments.dropFirst().makeIterator()
    while let k = it.next() {
        guard k.hasPrefix("--"), let v = it.next() else { continue }
        dict[String(k.dropFirst(2))] = v
    }
    guard let input = dict["input"],
          let weights = dict["weights"],
          let output = dict["output"] else {
        FileHandle.standardError.write(Data("""
        usage: Sam31VideoTrackFull --input <mp4> --weights <dir> --output <mp4> \
        (--prompt "a cat" --tokenizer <snapshot-dir> | --tokens <safetensors>) \
        [--resolution 1008] [--threshold 0.5] [--redetect-every 15] [--fps 30] [--max-frames N]
        """.utf8))
        exit(2)
    }
    var a = S31VTFArgs(
        input:   URL(fileURLWithPath: input),
        weights: URL(fileURLWithPath: weights),
        output:  URL(fileURLWithPath: output))
    a.prompt = dict["prompt"]
    if let t = dict["tokenizer"] { a.tokenizer = URL(fileURLWithPath: t) }
    if let t = dict["tokens"] { a.tokens = URL(fileURLWithPath: t) }
    if a.tokens == nil && (a.prompt == nil || a.tokenizer == nil) {
        FileHandle.standardError.write(Data(
            "must pass either --tokens <safetensors> or (--prompt <text> --tokenizer <dir>)\n".utf8))
        exit(2)
    }
    if let s = dict["resolution"], let v = Int(s) { a.resolution = v }
    if let s = dict["threshold"], let v = Float(s) { a.threshold = v }
    if let s = dict["redetect-every"], let v = Int(s) { a.redetectEvery = v }
    if let s = dict["fps"], let v = Int32(s) { a.frameRate = v }
    if let s = dict["max-frames"], let v = Int(s) { a.maxFrames = v }
    if let s = dict["dtype"] {
        switch s {
        case "float16", "fp16", "f16": a.dtype = .float16
        case "bfloat16", "bf16": a.dtype = .bfloat16
        case "float32", "fp32", "f32": a.dtype = .float32
        default: FileHandle.standardError.write(Data("unknown --dtype \(s)\n".utf8)); exit(2)
        }
    }
    return a
}

// Stable hue-from-ID palette.
func colorForID(_ id: Int) -> CGColor {
    let hue = CGFloat((id &* 2654435761) % 360) / 360.0
    return CGColor(colorSpace: CGColorSpaceCreateDeviceRGB(),
                   components: hsvToRGBA(h: hue, s: 0.8, v: 0.95, a: 1))!
}

func hsvToRGBA(h: CGFloat, s: CGFloat, v: CGFloat, a: CGFloat) -> [CGFloat] {
    let i = Int(h * 6), f = h * 6 - CGFloat(i)
    let p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s)
    switch i % 6 {
    case 0: return [v, t, p, a]
    case 1: return [q, v, p, a]
    case 2: return [p, v, t, a]
    case 3: return [p, q, v, a]
    case 4: return [t, p, v, a]
    default: return [v, p, q, a]
    }
}

func renderOverlayFull(image: CGImage, tracks: [TrackedObject]) -> CGImage {
    let w = image.width, h = image.height
    let ctx = CGContext(
        data: nil, width: w, height: h,
        bitsPerComponent: 8, bytesPerRow: w * 4,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            | CGBitmapInfo.byteOrder32Big.rawValue)!
    ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))

    for t in tracks {
        let rgba = hsvToRGBA(
            h: CGFloat((t.id &* 2654435761) % 360) / 360.0, s: 0.8, v: 0.95, a: 1)
        let fill = CGColor(colorSpace: CGColorSpaceCreateDeviceRGB(),
                           components: [rgba[0], rgba[1], rgba[2], 0.35])!
        let stroke = CGColor(colorSpace: CGColorSpaceCreateDeviceRGB(),
                             components: [rgba[0], rgba[1], rgba[2], 1.0])!

        if let mask = t.lastMask {
            let mh = mask.dim(0), mw = mask.dim(1)
            let arr = mask.asType(.int32).asArray(Int32.self)
            ctx.setFillColor(fill)
            for y in 0..<mh {
                for x in 0..<mw where arr[y * mw + x] != 0 {
                    ctx.fill(CGRect(x: x, y: h - 1 - y, width: 1, height: 1))
                }
            }
        }
        ctx.setStrokeColor(stroke)
        ctx.setLineWidth(3)
        let b = t.lastBoxPixels
        let x0 = CGFloat(b[0]), y0 = CGFloat(b[1])
        let x1 = CGFloat(b[2]), y1 = CGFloat(b[3])
        ctx.stroke(CGRect(x: x0, y: CGFloat(h) - y1, width: x1 - x0, height: y1 - y0))
    }
    return ctx.makeImage()!
}

@main
struct Sam31VideoTrackFullMain {
    static func main() async throws {
        let args = parse31VTF()

        let inputIds: MLXArray
        let attentionMask: MLXArray?
        if let tokensURL = args.tokens {
            let d = try MLX.loadArrays(url: tokensURL)
            guard let ids = d["input_ids"] else {
                FileHandle.standardError.write(Data("tokens file missing `input_ids`\n".utf8))
                exit(1)
            }
            inputIds = ids
            attentionMask = d["attention_mask"]
        } else {
            let tok = try await SAM3Tokenizer.fromSnapshotOrOpenAIClip(args.tokenizer!, maxLength: 32)
            let out = tok.encode(args.prompt!)
            inputIds = out.inputIds
            attentionMask = out.attentionMask
            print("tokenized '\(args.prompt!)' → \(inputIds.dim(-1)) ids")
        }

        print("loading SAM 3.1 weights from \(args.weights.path)…")
        let model = Sam31Model(Sam31ModelConfig())
        try ModelLoader.load(sam31: model, from: args.weights, dtype: args.dtype)

        let tracker = Sam31VideoTracker(
            model: model,
            resolution: args.resolution,
            dtype: args.dtype,
            detectThreshold: args.threshold,
            redetectEvery: args.redetectEvery)

        let reader = try await AVFoundationVideoReader(url: args.input)
        let outW = Int(reader.naturalSize.width)
        let outH = Int(reader.naturalSize.height)
        let writer = try AVFoundationVideoWriter(
            url: args.output, width: outW, height: outH, frameRate: args.frameRate)

        var frameIndex: Int64 = 0
        while let frame = try reader.nextFrame() {
            if let maxFrames = args.maxFrames, frameIndex >= maxFrames { break }
            let pixel = try ImagePreprocess.pixelValues(
                from: frame.image, size: args.resolution, dtype: args.dtype)
            let live = tracker.step(
                pixelValues: pixel,
                inputIds: inputIds,
                attentionMask: attentionMask,
                originalHeight: frame.image.height,
                originalWidth: frame.image.width)
            let overlay = renderOverlayFull(image: frame.image, tracks: live)
            try writer.append(
                image: overlay,
                at: CMTime(value: frameIndex, timescale: args.frameRate))
            frameIndex += 1
            if frameIndex % 10 == 0 {
                print("  frame \(frameIndex): \(live.count) live tracks (total \(tracker.tracks.count))")
            }
            MLX.Memory.clearCache()
        }
        try await writer.finish()
        print("wrote \(args.output.path) — \(frameIndex) frames, \(tracker.tracks.count) tracks seen")
    }
}
