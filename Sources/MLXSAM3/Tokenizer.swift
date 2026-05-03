// Tokenizer.swift — thin wrapper over `huggingface/swift-transformers`.
//
// Replaces the deferred Python `tokenizer_ve.py` (OpenAI CLIP BPE). The HF
// `facebook/sam3` / `mlx-community/sam3-bf16` snapshots ship a `tokenizer.json`
// in HF fast-tokenizer format, which `AutoTokenizer.from(modelFolder:)` loads
// directly.

import Foundation
import Hub
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

private struct CLIPBPETokenizer {
    let encoder: [String: Int]
    let bpeRanks: [Pair: Int]
    let byteEncoder: [UInt8: String]
    let bosTokenId: Int
    let eosTokenId: Int
    let padTokenId: Int

    init?(directory: URL) throws {
        let vocabURL = directory.appendingPathComponent("vocab.json").resolvingSymlinksInPath()
        let mergesURL = directory.appendingPathComponent("merges.txt").resolvingSymlinksInPath()
        guard FileManager.default.fileExists(atPath: vocabURL.path),
              FileManager.default.fileExists(atPath: mergesURL.path)
        else { return nil }

        let data = try Data(contentsOf: vocabURL)
        guard let vocab = try JSONSerialization.jsonObject(with: data) as? [String: Int] else {
            throw SAM3TokenizerError.encodeFailed("Could not parse CLIP vocab.json")
        }
        encoder = vocab
        bosTokenId = vocab["<|startoftext|>"] ?? 49406
        eosTokenId = vocab["<|endoftext|>"] ?? 49407
        padTokenId = eosTokenId
        byteEncoder = Self.bytesToUnicode()

        let mergeLines = try String(contentsOf: mergesURL, encoding: .utf8)
            .split(whereSeparator: \.isNewline)
            .map(String.init)
            .filter { !$0.hasPrefix("#") && !$0.isEmpty }

        var ranks: [Pair: Int] = [:]
        for (rank, line) in mergeLines.enumerated() {
            let parts = line.split(separator: " ", omittingEmptySubsequences: true).map(String.init)
            if parts.count == 2 {
                ranks[Pair(parts[0], parts[1])] = rank
            }
        }
        bpeRanks = ranks
    }

    func encode(_ text: String) -> [Int] {
        var ids = [bosTokenId]
        for token in regexTokens(text) {
            let mapped = token.utf8.map { byteEncoder[$0]! }.joined()
            for piece in bpe(mapped).split(separator: " ") {
                if let id = encoder[String(piece)] {
                    ids.append(id)
                }
            }
        }
        ids.append(eosTokenId)
        return ids
    }

    private func regexTokens(_ text: String) -> [String] {
        let pattern = #"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#
        let cleaned = text
            .replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return [cleaned] }
        let range = NSRange(cleaned.startIndex..<cleaned.endIndex, in: cleaned)
        return regex.matches(in: cleaned, range: range).compactMap { match in
            guard let matchRange = Range(match.range, in: cleaned) else { return nil }
            let token = String(cleaned[matchRange]).trimmingCharacters(in: .whitespacesAndNewlines)
            return token.isEmpty ? nil : token
        }
    }

    private func bpe(_ token: String) -> String {
        var word = token.map { String($0) }
        guard !word.isEmpty else { return token }
        word[word.count - 1] += "</w>"

        var pairs = getPairs(word)
        if pairs.isEmpty { return word.joined() }

        while true {
            let ranked = pairs.compactMap { pair -> (Pair, Int)? in
                guard let rank = bpeRanks[pair] else { return nil }
                return (pair, rank)
            }
            guard let bigram = ranked.min(by: { $0.1 < $1.1 })?.0 else { break }

            var newWord: [String] = []
            var i = 0
            while i < word.count {
                if let j = word[i...].firstIndex(of: bigram.first) {
                    newWord.append(contentsOf: word[i..<j])
                    i = j
                } else {
                    newWord.append(contentsOf: word[i...])
                    break
                }

                if i < word.count - 1, word[i] == bigram.first, word[i + 1] == bigram.second {
                    newWord.append(bigram.first + bigram.second)
                    i += 2
                } else {
                    newWord.append(word[i])
                    i += 1
                }
            }

            word = newWord
            if word.count == 1 { break }
            pairs = getPairs(word)
        }

        return word.joined(separator: " ")
    }

    private func getPairs(_ word: [String]) -> Set<Pair> {
        guard word.count > 1 else { return [] }
        var pairs = Set<Pair>()
        for i in 0..<(word.count - 1) {
            pairs.insert(Pair(word[i], word[i + 1]))
        }
        return pairs
    }

    private static func bytesToUnicode() -> [UInt8: String] {
        var bytes = Array(UInt8(33)...UInt8(126))
            + Array(UInt8(161)...UInt8(172))
            + Array(UInt8(174)...UInt8(255))
        var scalars = bytes.map(Int.init)
        var n = 0

        for byte in UInt8.min...UInt8.max where !bytes.contains(byte) {
            bytes.append(byte)
            scalars.append(256 + n)
            n += 1
        }

        var output: [UInt8: String] = [:]
        for (byte, scalar) in zip(bytes, scalars) {
            output[byte] = String(UnicodeScalar(scalar)!)
        }
        return output
    }

    struct Pair: Hashable {
        let first: String
        let second: String

        init(_ first: String, _ second: String) {
            self.first = first
            self.second = second
        }
    }
}

public struct SAM3TokenizerOutput {
    public let inputIds: MLXArray       // (1, T) Int32
    public let attentionMask: MLXArray  // (1, T) Int32
}

public struct SAM3Tokenizer {
    public let tokenizer: Tokenizer?
    public let maxLength: Int
    private let clipBPE: CLIPBPETokenizer?

    /// Load the CLIP tokenizer used by SAM 3 / SAM 3.1 text prompts.
    ///
    /// This mirrors the Python processor fallback:
    /// `CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")`.
    public static func openAIClip(maxLength: Int = 32) async throws -> SAM3Tokenizer {
        if let local = localOpenAIClipDirectory() {
            return try await fromSnapshot(local, maxLength: maxLength)
        }

        let snapshot = try await HubApi.shared.snapshot(
            from: "openai/clip-vit-base-patch32",
            matching: [
                "*.json",
                "vocab.json",
                "merges.txt",
            ]
        )
        if hasCLIPFiles(snapshot) || hasTokenizerJSON(snapshot) {
            return try await fromSnapshot(snapshot, maxLength: maxLength)
        }
        if let local = localOpenAIClipDirectory() {
            return try await fromSnapshot(local, maxLength: maxLength)
        }
        return try await fromSnapshot(snapshot, maxLength: maxLength)
    }

    /// Load a local tokenizer when present; otherwise fall back to OpenAI CLIP.
    public static func fromSnapshotOrOpenAIClip(
        _ directory: URL?,
        maxLength: Int = 32
    ) async throws -> SAM3Tokenizer {
        if let directory, hasTokenizerJSON(directory) || hasCLIPFiles(directory) {
            return try await fromSnapshot(directory, maxLength: maxLength)
        }
        return try await openAIClip(maxLength: maxLength)
    }

    /// Load from an HF snapshot directory containing `tokenizer.json`
    /// (and optionally `tokenizer_config.json`).
    ///
    /// swift-transformers 1.3 doesn't register `CLIPTokenizer`, so if the
    /// config declares it we stage a temp dir with the class rewritten to
    /// `PreTrainedTokenizerFast` (the `tokenizer.json` carries everything
    /// needed to decode BPE).
    public static func fromSnapshot(_ directory: URL, maxLength: Int = 77) async throws -> SAM3Tokenizer {
        let staged = try stageIfNeeded(directory)
        if let clipBPE = try CLIPBPETokenizer(directory: staged) {
            return SAM3Tokenizer(tokenizer: nil, maxLength: maxLength, clipBPE: clipBPE)
        }
        let tok = try await AutoTokenizer.from(modelFolder: staged, strict: false)
        return SAM3Tokenizer(tokenizer: tok, maxLength: maxLength, clipBPE: nil)
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

    private static func hasTokenizerJSON(_ directory: URL) -> Bool {
        FileManager.default.fileExists(
            atPath: directory.appendingPathComponent("tokenizer.json").path
        )
    }

    private static func hasCLIPFiles(_ directory: URL) -> Bool {
        FileManager.default.fileExists(atPath: directory.appendingPathComponent("vocab.json").path)
            && FileManager.default.fileExists(atPath: directory.appendingPathComponent("merges.txt").path)
    }

    private static func localOpenAIClipDirectory() -> URL? {
        var candidates: [URL] = []
        let cwd = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        candidates.append(cwd.appendingPathComponent("weights/mlx-sam3.1"))

        let home = FileManager.default.homeDirectoryForCurrentUser
        if !home.path.isEmpty {
            let repo = home
                .appendingPathComponent(".cache/huggingface/hub/models--openai--clip-vit-base-patch32")
            let ref = repo.appendingPathComponent("refs/main")
            if let revisionText = try? String(contentsOf: ref, encoding: .utf8) {
                let revision = revisionText.trimmingCharacters(in: .whitespacesAndNewlines)
                if !revision.isEmpty {
                    candidates.append(
                        repo.appendingPathComponent("snapshots").appendingPathComponent(revision)
                    )
                }
            }
        }

        return candidates.first { hasCLIPFiles($0) || hasTokenizerJSON($0) }
    }

    /// Encode a single prompt to `(input_ids, attention_mask)` padded to `maxLength`.
    public func encode(_ prompt: String) -> SAM3TokenizerOutput {
        var ids = clipBPE?.encode(prompt) ?? tokenizer!.encode(text: prompt)
        if ids.count > maxLength {
            ids = Array(ids.prefix(maxLength))
        }
        let realLen = ids.count
        let padId = clipBPE?.padTokenId ?? tokenizer!.unknownTokenId ?? 0
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
