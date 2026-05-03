// Sam31ImageInfer — CLI: SAM 3.1 image + text prompt → mask/box overlay PNG.
//
// Build:   swift build --product Sam31ImageInfer
// Run:     swift run Sam31ImageInfer \
//              --image  path/to/input.jpg \
//              --prompt "a cat" \
//              --tokenizer weights/sam3.1 \
//              --weights  weights/sam3.1 \
//              --output   path/to/overlay.png

import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXSAM3
import UniformTypeIdentifiers

struct S31Args {
    var image: URL
    var tokens: URL?
    var prompt: String?
    var tokenizer: URL?
    var weights: URL
    var output: URL
    var resolution: Int = 1008
    var threshold: Float = 0.5
    var dtype: DType = .bfloat16
    var benchmark = false
}

func parse31() -> S31Args {
    var dict: [String: String] = [:]
    var it = CommandLine.arguments.dropFirst().makeIterator()
    while let k = it.next() {
        guard k.hasPrefix("--"), let v = it.next() else { continue }
        dict[String(k.dropFirst(2))] = v
    }
    guard let image = dict["image"],
          let weights = dict["weights"],
          let output = dict["output"] else {
        FileHandle.standardError.write(Data("""
        usage: Sam31ImageInfer --image <path> --weights <dir> --output <path> \
        (--prompt "a cat" --tokenizer <snapshot-dir> | --tokens <safetensors>) \
        [--resolution 1008] [--threshold 0.5] [--benchmark true]
        """.utf8))
        exit(2)
    }
    var a = S31Args(
        image:   URL(fileURLWithPath: image),
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

func loadCG(_ url: URL) -> CGImage {
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
          let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
        FileHandle.standardError.write(Data("failed to load image: \(url.path)\n".utf8))
        exit(1)
    }
    return cg
}

func now31() -> TimeInterval {
    Date().timeIntervalSinceReferenceDate
}

func printBench31(_ name: String, _ start: TimeInterval, _ end: TimeInterval) {
    print("  [bench] \(name)=\(String(format: "%.3f", (end - start) * 1000)) ms")
}

func evalDetectionOutputs31(_ outputs: [String: MLXArray]) {
    MLX.eval([
        outputs["pred_logits"]!,
        outputs["pred_boxes"]!,
        outputs["pred_masks"]!,
        outputs["presence_logits"]!,
    ])
}

func savePNG31(_ image: CGImage, to url: URL) {
    guard let dst = CGImageDestinationCreateWithURL(
        url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
        FileHandle.standardError.write(Data("failed to create png destination\n".utf8))
        exit(1)
    }
    CGImageDestinationAddImage(dst, image, nil)
    _ = CGImageDestinationFinalize(dst)
}

func printProbe(_ name: String, _ array: MLXArray) {
    let values = array.asType(.float32).asArray(Float.self)
    let minValue = values.min() ?? .nan
    let maxValue = values.max() ?? .nan
    let hasNaN = values.contains { $0.isNaN }
    let hasInf = values.contains { $0.isInfinite }
    print("  [probe] \(name) shape=\(array.shape) min=\(minValue) max=\(maxValue) nan=\(hasNaN) inf=\(hasInf)")
}

func overlay31(image: CGImage, prediction: SAM3ImagePrediction) -> CGImage {
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
struct Sam31ImageInferMain {
    static func main() async throws {
        let args = parse31()
        let tTotal0 = now31()

        let inputIds: MLXArray
        let attentionMask: MLXArray?
        if let tokensURL = args.tokens {
            let t0 = now31()
            let d = try MLX.loadArrays(url: tokensURL)
            guard let ids = d["input_ids"] else {
                FileHandle.standardError.write(Data("tokens file missing `input_ids`\n".utf8))
                exit(1)
            }
            inputIds = ids
            attentionMask = d["attention_mask"]
            if args.benchmark { printBench31("tokens.load", t0, now31()) }
        } else {
            let t0 = now31()
            let tok = try await SAM3Tokenizer.fromSnapshotOrOpenAIClip(args.tokenizer!, maxLength: 32)
            let out = tok.encode(args.prompt!)
            inputIds = out.inputIds
            attentionMask = out.attentionMask
            print("tokenized '\(args.prompt!)' → \(inputIds.dim(-1)) ids")
            if !args.benchmark {
                print("  ids=\(inputIds.asArray(Int32.self))")
                print("  attention_mask=\(attentionMask!.asArray(Int32.self))")
            }
            if args.benchmark { printBench31("tokenize", t0, now31()) }
        }

        let tImage0 = now31()
        let image = loadCG(args.image)
        if args.benchmark { printBench31("image.load", tImage0, now31()) }

        print("loading SAM 3.1 weights from \(args.weights.path)…")
        let tLoad0 = now31()
        let model = Sam31Model(Sam31ModelConfig())
        try ModelLoader.load(sam31: model, from: args.weights, dtype: args.dtype)
        if args.benchmark { printBench31("model.load", tLoad0, now31()) }

        let tText0 = now31()
        let (inputsEmbeds, promptMask) = model.getInputEmbeddings(inputIds, attentionMask: attentionMask)
        if args.benchmark { printBench31("text_encoder", tText0, now31()) }

        print("running detection (image=\(image.width)×\(image.height))…")
        let tPre0 = now31()
        let pixel = try ImagePreprocess.pixelValues(
            from: image, size: args.resolution, dtype: args.dtype)
        MLX.eval(pixel)
        if args.benchmark { printBench31("preprocess", tPre0, now31()) }
        let tForward0 = now31()
        let outputs: [String: MLXArray]
        if args.benchmark {
            let (profiledOutputs, timings) = model.detectProfiled(
                pixelValues: pixel,
                attentionMask: promptMask,
                inputsEmbeds: inputsEmbeds)
            outputs = profiledOutputs
            for (name, ms) in timings {
                print("  [forward] \(name)=\(String(format: "%.3f", ms)) ms")
            }
        } else {
            outputs = model.detect(
                pixelValues: pixel,
                attentionMask: promptMask,
                inputsEmbeds: inputsEmbeds)
        }
        evalDetectionOutputs31(outputs)
        if args.benchmark { printBench31("forward", tForward0, now31()) }
        if !args.benchmark, let logits = outputs["pred_logits"], let pres = outputs["presence_logits"] {
            let cls = MLX.sigmoid(logits)[0]
            let pr = MLX.sigmoid(pres)[0].squeezed()
            let sc = (cls * pr).asType(.float32).asArray(Float.self)
            let top = sc.sorted(by: >).prefix(5)
            if let encoded = outputs["encoder_hidden_states"] {
                printProbe("detr.encoded", encoded)
            }
            if let hs = outputs["intermediate_hidden_states"] {
                printProbe("detr.hs", hs)
            }
            if let boxes = outputs["intermediate_boxes"] {
                printProbe("detr.refBoxes", boxes)
            }
            printProbe("detr.presenceLogits", pres)
            printProbe("dotProductScoring", logits)
            print("  score distribution: max=\(sc.max() ?? 0) top5=\(Array(top))")
        }

        let tPost0 = now31()
        let prediction = SAM3ImagePredictor.postprocess(
            outputs: outputs,
            originalHeight: image.height,
            originalWidth: image.width,
            threshold: args.threshold)
        if args.benchmark { printBench31("postprocess", tPost0, now31()) }
        print("→ \(prediction.boxes.dim(0)) detections above threshold=\(args.threshold)")

        let tOverlay0 = now31()
        let png = overlay31(image: image, prediction: prediction)
        savePNG31(png, to: args.output)
        if args.benchmark {
            printBench31("overlay.write", tOverlay0, now31())
            printBench31("total", tTotal0, now31())
        }
        print("wrote \(args.output.path)")
    }
}
