// ImageInfer — CLI: SAM3 image + text prompt → mask/box overlay PNG.
//

// build: xcodebuild -scheme ImageInfer -destination 'platform=macOS' build -derivedDataPath .build/xcode
// convert the pytorch checkpoint to mlx safetensors via python
// uv run scripts/convert_pytorch_to_mlx.py--checkpoint /path/to/pytorch/sam3_checkpoint.pth --output /path/to/mlx/sam3_checkpoint.safetensors
// SNAP: weights/mlx-sam3
// run .build/xcode/Build/Products/Debug/ImageInfer \
// --image ~/Downloads/default.jpeg \
// --prompt "animals" \
// --tokenizer "$SNAP" \
// --weights "$SNAP" \
// --output out.png \
// --threshold 0.5 \
// --dtype float3
//
// `tokens.safetensors` must contain an `input_ids` key of shape (1, T) Int32.
// Tokenizer is deferred — generate this offline (e.g. via the Python CLIP BPE).

import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXSAM3
import UniformTypeIdentifiers

#if canImport(AppKit)
import AppKit
#endif

struct Args {
    var image: URL
    var tokens: URL?       // pre-tokenized safetensors (optional)
    var prompt: String?    // raw text prompt (requires --tokenizer)
    var tokenizer: URL?    // HF snapshot dir containing tokenizer.json
    var weights: URL
    var output: URL
    var resolution: Int = 1008
    var threshold: Float = 0.5
    var dtype: DType = .bfloat16
}

func parseArgs() -> Args {
    var dict: [String: String] = [:]
    let argv = CommandLine.arguments.dropFirst()
    var it = argv.makeIterator()
    while let k = it.next() {
        guard k.hasPrefix("--"), let v = it.next() else { continue }
        dict[String(k.dropFirst(2))] = v
    }
    guard let image = dict["image"],
          let weights = dict["weights"],
          let output = dict["output"] else {
        FileHandle.standardError.write(Data("""
        usage: ImageInfer --image <path> --weights <dir> --output <path> \
        (--prompt "a cat" --tokenizer <snapshot-dir> | --tokens <safetensors>) \
        [--resolution 1008] [--threshold 0.5]
        """.utf8))
        exit(2)
    }
    var a = Args(
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

func loadCGImage(_ url: URL) -> CGImage {
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
          let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
        FileHandle.standardError.write(Data("failed to load image: \(url.path)\n".utf8))
        exit(1)
    }
    return cg
}

func savePNG(_ image: CGImage, to url: URL) {
    guard let dst = CGImageDestinationCreateWithURL(
        url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
        FileHandle.standardError.write(Data("failed to create png destination\n".utf8))
        exit(1)
    }
    CGImageDestinationAddImage(dst, image, nil)
    if !CGImageDestinationFinalize(dst) {
        FileHandle.standardError.write(Data("failed to write png\n".utf8))
        exit(1)
    }
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

    let mask = CGColor(red: 1, green: 0.2, blue: 0.2, alpha: 0.35)
    if prediction.masks.dim(0) > 0 {
        let masks = prediction.masks.asType(.float32).asArray(Float.self)
        let mh = prediction.masks.dim(1), mw = prediction.masks.dim(2)
        let stride = mh * mw
        ctx.setFillColor(mask)
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
struct ImageInferMain {
    static func main() async throws {
        let args = parseArgs()

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

        let image = loadCGImage(args.image)

        print("loading SAM3 weights from \(args.weights.path)…")
        let model = Sam3Model(ModelConfig())
        try ModelLoader.load(sam3: model, from: args.weights, dtype: args.dtype)

        let predictor = SAM3ImagePredictor(
            model: model, resolution: args.resolution, confidenceThreshold: args.threshold)

        print("running detection (image=\(image.width)×\(image.height))…")
        let pixel = try ImagePreprocess.pixelValues(
            from: image, size: args.resolution, dtype: args.dtype)
        let outputs = model.detect(
            pixelValues: pixel, inputIds: inputIds, attentionMask: attentionMask)
        MLX.eval(Array(outputs.values))
        if let logits = outputs["pred_logits"], let pres = outputs["presence_logits"] {
            let cls = MLX.sigmoid(logits)[0]
            let pr = MLX.sigmoid(pres)[0].squeezed()
            let sc = (cls * pr).asType(.float32).asArray(Float.self)
            let top = sc.sorted(by: >).prefix(5)
            print("  score distribution: max=\(sc.max() ?? 0) top5=\(Array(top))")
        }
        let prediction = SAM3ImagePredictor.postprocess(
            outputs: outputs,
            originalHeight: image.height,
            originalWidth: image.width,
            threshold: args.threshold)
        print("→ \(prediction.boxes.dim(0)) detections above threshold=\(args.threshold)")

        let overlay = renderOverlay(image: image, prediction: prediction)
        savePNG(overlay, to: args.output)
        print("wrote \(args.output.path)")
    }
}
