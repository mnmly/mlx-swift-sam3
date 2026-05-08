// Sam31VideoTrack — CLI: per-frame SAM 3.1 inference on a video; writes overlay mp4.
//
// IMPORTANT: build via xcodebuild, NOT `swift build` / `swift run`. MLX needs
// a Metal-capable toolchain at runtime.
//
// Build:
//     xcodebuild -scheme Sam31VideoTrack -destination 'platform=macOS' \
//         -configuration release -derivedDataPath .xcdd build
//
// Run:
//     .xcdd/Build/Products/Release/Sam31VideoTrack \
//         --input  path/to/video.mp4 \
//         --prompt "a cat" --tokenizer <snapshot-dir> \
//         --weights <snapshot-dir> \
//         --output path/to/overlay.mp4
//
// Slim demo: runs `detect()` on each frame independently (no memory-bank
// tracking — that is deferred video orchestration work).

import AVFoundation
import CoreGraphics
import CoreMedia
import Foundation
import ImageIO
import MLX
import MLXSAM3
import UniformTypeIdentifiers

struct S31VTArgs {
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
    var maxFrames: Int?
    var benchmark = false
}

func parse31VT() -> S31VTArgs {
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
        usage: Sam31VideoTrack --input <mp4> --weights <dir> --output <mp4> \
        (--prompt "a cat" --tokenizer <snapshot-dir> | --tokens <safetensors>) \
        [--resolution 1008] [--threshold 0.5] [--fps 30] [--max-frames N] [--benchmark true]
        """.utf8))
        exit(2)
    }
    var a = S31VTArgs(
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
    if let s = dict["max-frames"], let v = Int(s) { a.maxFrames = v }
    if dict["benchmark"] != nil { a.benchmark = true }
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

func now31VT() -> TimeInterval {
    Date().timeIntervalSinceReferenceDate
}

func printBench31VT(_ name: String, _ value: TimeInterval) {
    print("  [bench] \(name)=\(String(format: "%.3f", value * 1000)) ms")
}

func evalDetectionOutputs31VT(_ outputs: [String: MLXArray]) {
    MLX.eval([
        outputs["pred_logits"]!,
        outputs["pred_boxes"]!,
        outputs["pred_masks"]!,
        outputs["presence_logits"]!,
    ])
}

func renderOverlay31VT(image: CGImage, prediction: SAM3ImagePrediction) -> CGImage {
    let w = image.width, h = image.height
    let ctx = CGContext(
        data: nil, width: w, height: h,
        bitsPerComponent: 8, bytesPerRow: w * 4,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
            | CGBitmapInfo.byteOrder32Big.rawValue)!
    ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))

    if prediction.masks.dim(0) > 0 {
        let m = prediction.masks.asType(.float32).asArray(Float.self)
        let mh = prediction.masks.dim(1), mw = prediction.masks.dim(2)
        let stride = mh * mw
        ctx.setFillColor(CGColor(red: 0.2, green: 0.8, blue: 0.4, alpha: 0.35))
        for n in 0..<prediction.masks.dim(0) {
            for y in 0..<mh {
                for x in 0..<mw where m[n * stride + y * mw + x] > 0.5 {
                    ctx.fill(CGRect(x: x, y: h - 1 - y, width: 1, height: 1))
                }
            }
        }
    }
    ctx.setStrokeColor(CGColor(red: 0.2, green: 0.8, blue: 0.4, alpha: 1))
    ctx.setLineWidth(3)
    if prediction.boxes.dim(0) > 0 {
        let b = prediction.boxes.asType(.float32).asArray(Float.self)
        for n in 0..<prediction.boxes.dim(0) {
            let x0 = CGFloat(b[n*4+0]), y0 = CGFloat(b[n*4+1])
            let x1 = CGFloat(b[n*4+2]), y1 = CGFloat(b[n*4+3])
            ctx.stroke(CGRect(x: x0, y: CGFloat(h) - y1, width: x1 - x0, height: y1 - y0))
        }
    }
    return ctx.makeImage()!
}

@main
struct Sam31VideoTrackMain {
    static func main() async throws {
        let args = parse31VT()
        let tTotal0 = now31VT()

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
            let t0 = now31VT()
            let tok = try await SAM3Tokenizer.fromSnapshotOrOpenAIClip(args.tokenizer!, maxLength: 32)
            let out = tok.encode(args.prompt!)
            inputIds = out.inputIds
            attentionMask = out.attentionMask
            print("tokenized '\(args.prompt!)' → \(inputIds.dim(-1)) ids")
            if args.benchmark { printBench31VT("tokenize", now31VT() - t0) }
        }

        print("loading SAM 3.1 weights from \(args.weights.path)…")
        let tLoad0 = now31VT()
        let model = Sam31Model(Sam31ModelConfig())
        try ModelLoader.load(sam31: model, from: args.weights, dtype: args.dtype)
        if args.benchmark { printBench31VT("model.load", now31VT() - tLoad0) }

        let tText0 = now31VT()
        let (inputsEmbeds, promptMask) = model.getInputEmbeddings(inputIds, attentionMask: attentionMask)
        if args.benchmark { printBench31VT("text_encoder", now31VT() - tText0) }

        let tIO0 = now31VT()
        let reader = try await AVFoundationVideoReader(url: args.input)
        let outW = Int(reader.naturalSize.width)
        let outH = Int(reader.naturalSize.height)
        let writer = try AVFoundationVideoWriter(
            url: args.output, width: outW, height: outH, frameRate: args.frameRate)
        if args.benchmark { printBench31VT("video.open", now31VT() - tIO0) }

        var frameIndex: Int64 = 0
        var preTotal: TimeInterval = 0
        var forwardTotal: TimeInterval = 0
        var postTotal: TimeInterval = 0
        var overlayWriteTotal: TimeInterval = 0
        while let frame = try reader.nextFrame() {
            if let maxFrames = args.maxFrames, frameIndex >= maxFrames { break }
            let tPre0 = now31VT()
            let pixel = try ImagePreprocess.pixelValues(
                from: frame.image, size: args.resolution, dtype: args.dtype)
            MLX.eval(pixel)
            preTotal += now31VT() - tPre0
            let tForward0 = now31VT()
            let outputs: [String: MLXArray]
            if args.benchmark, frameIndex == 0 {
                let (profiledOutputs, timings) = model.detectProfiled(
                    pixelValues: pixel,
                    attentionMask: promptMask,
                    inputsEmbeds: inputsEmbeds)
                outputs = profiledOutputs
                for (name, ms) in timings {
                    print("  [forward frame0] \(name)=\(String(format: "%.3f", ms)) ms")
                }
            } else {
                outputs = model.detect(
                    pixelValues: pixel, attentionMask: promptMask, inputsEmbeds: inputsEmbeds)
            }
            evalDetectionOutputs31VT(outputs)
            forwardTotal += now31VT() - tForward0
            if !args.benchmark,
               frameIndex < 3,
               let logits = outputs["pred_logits"], let pres = outputs["presence_logits"] {
                let cls = MLX.sigmoid(logits)[0]
                let pr = MLX.sigmoid(pres)[0].squeezed()
                let sc = (cls * pr).asType(.float32).asArray(Float.self)
                let top = sc.sorted(by: >).prefix(5)
                print("  frame \(frameIndex) scores: max=\(sc.max() ?? 0) top5=\(Array(top))")
            }
            let tPost0 = now31VT()
            let prediction = SAM3ImagePredictor.postprocess(
                outputs: outputs,
                originalHeight: frame.image.height,
                originalWidth: frame.image.width,
                threshold: args.threshold)
            postTotal += now31VT() - tPost0
            let tOverlay0 = now31VT()
            let overlay = renderOverlay31VT(image: frame.image, prediction: prediction)
            try writer.append(
                image: overlay,
                at: CMTime(value: frameIndex, timescale: args.frameRate))
            overlayWriteTotal += now31VT() - tOverlay0
            frameIndex += 1
            if frameIndex % 10 == 0 { print("  frame \(frameIndex) (\(prediction.boxes.dim(0)) dets)") }
            MLX.Memory.clearCache()
        }
        try await writer.finish()
        if args.benchmark, frameIndex > 0 {
            let n = Double(frameIndex)
            printBench31VT("preprocess.avg", preTotal / n)
            printBench31VT("forward.avg", forwardTotal / n)
            printBench31VT("postprocess.avg", postTotal / n)
            printBench31VT("overlay.write.avg", overlayWriteTotal / n)
            let total = now31VT() - tTotal0
            printBench31VT("total", total)
            print("  [bench] end_to_end_fps=\(String(format: "%.2f", Double(frameIndex) / total))")
        }
        print("wrote \(args.output.path) — \(frameIndex) frames")
    }
}
