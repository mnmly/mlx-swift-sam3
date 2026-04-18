// VideoTrack — CLI: per-frame SAM3 inference on a video; writes overlay mp4.
//
// Build:   swift build --product VideoTrack
// Run:     swift run VideoTrack \
//              --input  path/to/video.mp4 \
//              --tokens path/to/tokens.safetensors \
//              --weights weights/sam3 \
//              --output path/to/overlay.mp4
//
// Slim demo: runs `detect()` on each frame independently (no memory-bank
// tracking — that path is the deferred Phase 9 video orchestration).

import AVFoundation
import CoreGraphics
import CoreMedia
import Foundation
import ImageIO
import MLX
import MLXSAM3
import UniformTypeIdentifiers

struct VTArgs {
    var input: URL
    var tokens: URL?
    var prompt: String?
    var tokenizer: URL?
    var weights: URL
    var output: URL
    var resolution: Int = 1008
    var threshold: Float = 0.5
    var frameRate: Int32 = 30
    var dtype: DType = .bfloat16
}

func parse() -> VTArgs {
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
        usage: VideoTrack --input <mp4> --weights <dir> --output <mp4> \
        (--prompt "a cat" --tokenizer <snapshot-dir> | --tokens <safetensors>) \
        [--resolution 1008] [--threshold 0.5] [--fps 30]
        """.utf8))
        exit(2)
    }
    var a = VTArgs(
        input:   URL(fileURLWithPath: input),
        weights: URL(fileURLWithPath: weights),
        output:  URL(fileURLWithPath: output))
    if let t = dict["tokens"] { a.tokens = URL(fileURLWithPath: t) }
    a.prompt = dict["prompt"]
    if let t = dict["tokenizer"] { a.tokenizer = URL(fileURLWithPath: t) }
    if a.tokens == nil && (a.prompt == nil || a.tokenizer == nil) {
        FileHandle.standardError.write(Data(
            "must pass either --tokens <safetensors> or (--prompt <text> --tokenizer <dir>)\n".utf8))
        exit(2)
    }
    if let s = dict["resolution"], let v = Int(s) { a.resolution = v }
    if let s = dict["threshold"], let v = Float(s) { a.threshold = v }
    if let s = dict["fps"], let v = Int32(s) { a.frameRate = v }
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

func renderOverlay(image: CGImage, prediction: SAM3ImagePrediction) -> CGImage {
    let w = image.width, h = image.height
    let cs = CGColorSpaceCreateDeviceRGB()
    let ctx = CGContext(
        data: nil, width: w, height: h,
        bitsPerComponent: 8, bytesPerRow: w * 4, space: cs,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            | CGBitmapInfo.byteOrder32Big.rawValue)!
    ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))

    if prediction.masks.dim(0) > 0 {
        let masks = prediction.masks.asType(.float32).asArray(Float.self)
        let mh = prediction.masks.dim(1), mw = prediction.masks.dim(2)
        let stride = mh * mw
        ctx.setFillColor(CGColor(red: 1, green: 0.2, blue: 0.2, alpha: 0.35))
        for n in 0..<prediction.masks.dim(0) {
            for y in 0..<mh {
                for x in 0..<mw where masks[n * stride + y * mw + x] > 0.5 {
                    ctx.fill(CGRect(x: x, y: h - 1 - y, width: 1, height: 1))
                }
            }
        }
    }
    ctx.setStrokeColor(CGColor(red: 1, green: 0.2, blue: 0.2, alpha: 1))
    ctx.setLineWidth(3)
    if prediction.boxes.dim(0) > 0 {
        let boxes = prediction.boxes.asType(.float32).asArray(Float.self)
        for n in 0..<prediction.boxes.dim(0) {
            let x0 = CGFloat(boxes[n * 4 + 0])
            let y0 = CGFloat(boxes[n * 4 + 1])
            let x1 = CGFloat(boxes[n * 4 + 2])
            let y1 = CGFloat(boxes[n * 4 + 3])
            ctx.stroke(CGRect(x: x0, y: CGFloat(h) - y1, width: x1 - x0, height: y1 - y0))
        }
    }
    return ctx.makeImage()!
}

@main
struct VideoTrackMain {
    static func main() async throws {
        let args = parse()

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
            let tok = try await SAM3Tokenizer.fromSnapshot(args.tokenizer!)
            let out = tok.encode(args.prompt!)
            inputIds = out.inputIds
            attentionMask = out.attentionMask
            print("tokenized '\(args.prompt!)' → \(inputIds.dim(-1)) ids")
        }

        print("loading SAM3 weights from \(args.weights.path)…")
        let model = Sam3Model(ModelConfig())
        try ModelLoader.load(sam3: model, from: args.weights, dtype: args.dtype)
        let reader = try await AVFoundationVideoReader(url: args.input)
        let outW = Int(reader.naturalSize.width)
        let outH = Int(reader.naturalSize.height)
        let writer = try AVFoundationVideoWriter(
            url: args.output, width: outW, height: outH, frameRate: args.frameRate)

        var frameIndex: Int64 = 0
        while let frame = try reader.nextFrame() {
            let pixel = try ImagePreprocess.pixelValues(
                from: frame.image, size: args.resolution, dtype: args.dtype)
            let outputs = model.detect(
                pixelValues: pixel, inputIds: inputIds, attentionMask: attentionMask)
            MLX.eval(Array(outputs.values))
            if frameIndex < 3,
               let logits = outputs["pred_logits"], let pres = outputs["presence_logits"] {
                let cls = MLX.sigmoid(logits)[0]
                let pr = MLX.sigmoid(pres)[0].squeezed()
                let sc = (cls * pr).asType(.float32).asArray(Float.self)
                let top = sc.sorted(by: >).prefix(5)
                print("  frame \(frameIndex) scores: max=\(sc.max() ?? 0) top5=\(Array(top))")
                if frameIndex == 0 {
                    let dbg = args.output.deletingLastPathComponent()
                        .appendingPathComponent("videotrack_frame0.png")
                    if let dst = CGImageDestinationCreateWithURL(
                        dbg as CFURL, UTType.png.identifier as CFString, 1, nil) {
                        CGImageDestinationAddImage(dst, frame.image, nil)
                        _ = CGImageDestinationFinalize(dst)
                        print("  (wrote \(dbg.path) for sanity check)")
                    }
                }
            }
            let prediction = SAM3ImagePredictor.postprocess(
                outputs: outputs,
                originalHeight: frame.image.height,
                originalWidth: frame.image.width,
                threshold: args.threshold)
            let overlay = renderOverlay(image: frame.image, prediction: prediction)
            try writer.append(
                image: overlay,
                at: CMTime(value: frameIndex, timescale: args.frameRate))
            frameIndex += 1
            if frameIndex % 10 == 0 { print("  frame \(frameIndex) (\(prediction.boxes.dim(0)) dets)") }
            MLX.Memory.clearCache()
        }
        try await writer.finish()
        print("wrote \(args.output.path) — \(frameIndex) frames")
    }
}
