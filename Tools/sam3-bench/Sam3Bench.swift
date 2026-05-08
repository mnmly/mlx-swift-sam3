// Sam3Bench — protocol-matched benchmark for SAM 3.1 image inference (Swift / mlx-swift).
//
// Build:
//   xcodebuild -scheme sam3-bench -destination 'platform=macOS' \
//     -configuration release -derivedDataPath .xcdd build
//   .xcdd/Build/Products/release/sam3-bench --weights ... --image ... --prompt "a person"
//
// Companion Python script: Benchmarks/torch_sam3_bench.py — same flags, same protocol.

import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXSAM3

private struct BenchArgs {
    var weights: URL
    var tokenizer: URL
    var image: URL
    var prompt: String
    var dtype: DType = .float16
    var resolution: Int = 1008
    var threshold: Float = 0.3
    var iters: Int = 20
    var warmup: Int = 3
    var includeLoad: Bool = false
    var json: Bool = false
}

private func parse() -> BenchArgs {
    var dict: [String: String] = [:]
    var flags: Set<String> = []
    var it = CommandLine.arguments.dropFirst().makeIterator()
    while let k = it.next() {
        guard k.hasPrefix("--") else { continue }
        let key = String(k.dropFirst(2))
        if ["include-load", "json"].contains(key) { flags.insert(key); continue }
        guard let v = it.next() else { continue }
        dict[key] = v
    }
    guard let w = dict["weights"], let t = dict["tokenizer"] ?? dict["weights"] as String?,
          let i = dict["image"], let p = dict["prompt"] else {
        FileHandle.standardError.write(Data("""
        usage: sam3-bench --weights <dir> [--tokenizer <dir>] --image <path> --prompt "<text>"
                         [--iters 20] [--warmup 3] [--dtype fp16|bf16|fp32]
                         [--resolution 1008] [--threshold 0.3]
                         [--include-load] [--json]
        """.utf8))
        exit(2)
    }
    var a = BenchArgs(
        weights: URL(fileURLWithPath: w),
        tokenizer: URL(fileURLWithPath: t),
        image: URL(fileURLWithPath: i),
        prompt: p)
    if let s = dict["iters"], let v = Int(s) { a.iters = v }
    if let s = dict["warmup"], let v = Int(s) { a.warmup = v }
    if let s = dict["resolution"], let v = Int(s) { a.resolution = v }
    if let s = dict["threshold"], let v = Float(s) { a.threshold = v }
    if let s = dict["dtype"] {
        switch s {
        case "fp16", "float16", "f16": a.dtype = .float16
        case "bf16", "bfloat16": a.dtype = .bfloat16
        case "fp32", "float32", "f32": a.dtype = .float32
        default: FileHandle.standardError.write(Data("unknown --dtype \(s)\n".utf8)); exit(2)
        }
    }
    a.includeLoad = flags.contains("include-load")
    a.json = flags.contains("json")
    return a
}

private func now() -> TimeInterval { Date().timeIntervalSinceReferenceDate }

private func loadImage(_ url: URL) -> CGImage {
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
          let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
        FileHandle.standardError.write(Data("failed to load image: \(url.path)\n".utf8))
        exit(1)
    }
    return cg
}

private func runOnce(
    model: Sam31Model,
    image: CGImage,
    inputsEmbeds: MLXArray,
    promptMask: MLXArray?,
    args: BenchArgs,
    includeLoad: Bool
) -> [String: MLXArray] {
    var img = image
    if includeLoad {
        // Re-decode the JPEG/PNG from disk on every iter — measures end-to-end with I/O.
        img = loadImage(args.image)
    }
    let pixel = try! ImagePreprocess.pixelValues(from: img, size: args.resolution, dtype: args.dtype)
    let outputs = model.detect(pixelValues: pixel, attentionMask: promptMask, inputsEmbeds: inputsEmbeds)
    // Force materialization — without eval we'd time graph construction, not compute.
    MLX.eval(Array(outputs.values))
    let _ = SAM3ImagePredictor.postprocess(outputs: outputs, originalHeight: img.height,
                                           originalWidth: img.width, threshold: args.threshold)
    return outputs
}

private struct Stats {
    var mean: Double = 0; var median: Double = 0; var min: Double = 0; var max: Double = 0
    init(_ samples: [Double]) {
        guard !samples.isEmpty else { return }
        let sorted = samples.sorted()
        mean = samples.reduce(0, +) / Double(samples.count)
        median = sorted[sorted.count / 2]
        min = sorted.first!
        max = sorted.last!
    }
}

@main
struct Sam3Bench {
    static func main() async throws {
        let args = parse()

        // Cold model load
        let tLoad0 = now()
        let model = Sam31Model(Sam31ModelConfig())
        try ModelLoader.load(sam31: model, from: args.weights, dtype: args.dtype)
        let loadS = now() - tLoad0

        // Tokenize once
        let tok = try await SAM3Tokenizer.fromSnapshotOrOpenAIClip(args.tokenizer, maxLength: 32)
        let encoded = tok.encode(args.prompt)
        let (inputsEmbeds, promptMask) = model.getInputEmbeddings(encoded.inputIds, attentionMask: encoded.attentionMask)

        // Preload image once for inference-only path
        let image = loadImage(args.image)

        // Warmup
        for _ in 0..<args.warmup {
            _ = runOnce(model: model, image: image, inputsEmbeds: inputsEmbeds,
                        promptMask: promptMask, args: args, includeLoad: false)
        }

        // Inference-only
        var infOnly: [Double] = []
        for _ in 0..<args.iters {
            let t0 = now()
            _ = runOnce(model: model, image: image, inputsEmbeds: inputsEmbeds,
                        promptMask: promptMask, args: args, includeLoad: false)
            infOnly.append(now() - t0)
        }
        let sInf = Stats(infOnly)

        // End-to-end (includes image reload from disk + pixel-buffer decode)
        var e2e: [Double] = []
        if args.includeLoad {
            for _ in 0..<args.iters {
                let t0 = now()
                _ = runOnce(model: model, image: image, inputsEmbeds: inputsEmbeds,
                            promptMask: promptMask, args: args, includeLoad: true)
                e2e.append(now() - t0)
            }
        }
        let sE2e = Stats(e2e)

        if args.json {
            let dtype = "\(args.dtype)"
            print("""
            {"backend":"mlx-swift","dtype":"\(dtype)","iters":\(args.iters),"warmup":\(args.warmup),\
            "load_s":\(loadS),\
            "inference_only":{"mean_s":\(sInf.mean),"median_s":\(sInf.median),"min_s":\(sInf.min),"max_s":\(sInf.max)},\
            "end_to_end":{"mean_s":\(sE2e.mean),"median_s":\(sE2e.median),"min_s":\(sE2e.min),"max_s":\(sE2e.max)}}
            """)
        } else {
            print("backend       : mlx-swift")
            print("dtype         : \(args.dtype)")
            print("resolution    : \(args.resolution)")
            print("warmup/iters  : \(args.warmup)/\(args.iters)")
            print(String(format: "cold_load_s   : %.3f", loadS))
            print(String(format: "inference_only: mean=%.3fs median=%.3fs min=%.3fs max=%.3fs",
                         sInf.mean, sInf.median, sInf.min, sInf.max))
            if args.includeLoad {
                print(String(format: "end_to_end    : mean=%.3fs median=%.3fs min=%.3fs max=%.3fs",
                             sE2e.mean, sE2e.median, sE2e.min, sE2e.max))
            }
        }
    }
}
