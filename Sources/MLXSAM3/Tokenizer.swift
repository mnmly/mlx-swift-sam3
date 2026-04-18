// Tokenizer.swift — thin wrapper over `huggingface/swift-transformers`.
//
// Replaces the deferred Python `tokenizer_ve.py` (OpenAI CLIP BPE). The HF
// `facebook/sam3` / `mlx-community/sam3-bf16` snapshots ship a `tokenizer.json`
// in HF fast-tokenizer format, which `AutoTokenizer.from(modelFolder:)` loads
// directly.

import Foundation
import MLX
import Tokenizers

public enum SAM3TokenizerError: Error, CustomStringConvertible {
    case encodeFailed(String)

    public var description: String {
        switch self {
        case .encodeFailed(let m): return "Tokenizer encode failed: \(m)"
        }
    }
}

public struct SAM3TokenizerOutput {
    public let inputIds: MLXArray       // (1, T) Int32
    public let attentionMask: MLXArray  // (1, T) Int32
}

public struct SAM3Tokenizer {
    public let tokenizer: Tokenizer
    public let maxLength: Int

    /// Load from an HF snapshot directory containing `tokenizer.json`
    /// (and optionally `tokenizer_config.json`).
    ///
    /// swift-transformers 1.3 doesn't register `CLIPTokenizer`, so if the
    /// config declares it we stage a temp dir with the class rewritten to
    /// `PreTrainedTokenizerFast` (the `tokenizer.json` carries everything
    /// needed to decode BPE).
    public static func fromSnapshot(_ directory: URL, maxLength: Int = 77) async throws -> SAM3Tokenizer {
        let staged = try stageIfNeeded(directory)
        let tok = try await AutoTokenizer.from(modelFolder: staged)
        return SAM3Tokenizer(tokenizer: tok, maxLength: maxLength)
    }

    private static func stageIfNeeded(_ directory: URL) throws -> URL {
        let cfgURL = directory.appendingPathComponent("tokenizer_config.json")
        guard FileManager.default.fileExists(atPath: cfgURL.path),
              let data = try? Data(contentsOf: cfgURL),
              var cfg = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let cls = cfg["tokenizer_class"] as? String,
              cls != "PreTrainedTokenizerFast"
        else { return directory }

        cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("sam3-tokenizer-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)

        let patched = try JSONSerialization.data(withJSONObject: cfg, options: [.prettyPrinted])
        try patched.write(to: tmp.appendingPathComponent("tokenizer_config.json"))

        for name in ["tokenizer.json", "vocab.json", "merges.txt", "special_tokens_map.json"] {
            let src = directory.appendingPathComponent(name).resolvingSymlinksInPath()
            if FileManager.default.fileExists(atPath: src.path) {
                try FileManager.default.copyItem(
                    at: src, to: tmp.appendingPathComponent(name))
            }
        }
        return tmp
    }

    /// Encode a single prompt to `(input_ids, attention_mask)` padded to `maxLength`.
    public func encode(_ prompt: String) -> SAM3TokenizerOutput {
        var ids = tokenizer.encode(text: prompt)
        if ids.count > maxLength {
            ids = Array(ids.prefix(maxLength))
        }
        let realLen = ids.count
        let padId = tokenizer.unknownTokenId ?? 0
        if ids.count < maxLength {
            ids.append(contentsOf: Array(repeating: padId, count: maxLength - ids.count))
        }
        let idsArr = MLXArray(ids.map { Int32($0) }, [1, maxLength])
        var maskVals = [Int32](repeating: 0, count: maxLength)
        for i in 0..<realLen { maskVals[i] = 1 }
        let maskArr = MLXArray(maskVals, [1, maxLength])
        return SAM3TokenizerOutput(inputIds: idsArr, attentionMask: maskArr)
    }
}
