// ModelLoader.swift — weight loading pipeline for SAM 3 and SAM 3.1.
//
// Handles both single-file checkpoints (model.safetensors) and
// HuggingFace-style sharded checkpoints (model.safetensors.index.json + shards).
//
// Typical usage:
//   let model = Sam31Model(Sam31ModelConfig())
//   try ModelLoader.load(sam31: model, from: URL(fileURLWithPath: "/path/to/weights"))
//   ModelLoader.quantize(model, predicate: sam31QuantPredicate)

import Foundation
import MLX
import MLXNN

public enum ModelLoader {

    // MARK: - Public entry points

    /// Load and sanitize weights into a SAM 3 model.
    public static func load(
        sam3 model: Sam3Model,
        from directory: URL,
        dtype: DType = .float16
    ) throws {
        let raw = try loadWeights(from: directory)
        let sanitized = Sam3Model.sanitize(raw)
        try apply(sanitized, to: model, dtype: dtype)
    }

    /// Load and sanitize weights into a SAM 3.1 model.
    public static func load(
        sam31 model: Sam31Model,
        from directory: URL,
        dtype: DType = .float16
    ) throws {
        let raw = try loadWeights(from: directory)
        let sanitized = Sam31Model.sanitize(raw)
        try apply(sanitized, to: model, dtype: dtype)
    }

    /// Quantize a model in-place using `MLXNN.quantize`.
    ///
    /// - Parameters:
    ///   - predicate: return `true` for modules that should be quantized.
    ///     Pass `quantPredicate` for SAM 3, `sam31QuantPredicate` for SAM 3.1.
    public static func quantize(
        _ model: Module,
        groupSize: Int = 64,
        bits: Int = 4,
        predicate: @escaping (String, Module) -> Bool = { _, _ in true }
    ) {
        MLXNN.quantize(model: model, groupSize: groupSize, bits: bits, filter: predicate)
    }

    // MARK: - Weight loading

    /// Load all weights from `directory`.
    ///
    /// Looks for, in order:
    /// 1. `model.safetensors.index.json` — sharded checkpoint; loads all unique shards.
    /// 2. `model.safetensors` — single-file checkpoint.
    ///
    /// Returns the merged raw `[key: array]` dict (unsanitized).
    public static func loadWeights(from directory: URL) throws -> [String: MLXArray] {
        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        if FileManager.default.fileExists(atPath: indexURL.path) {
            return try loadSharded(indexURL: indexURL, directory: directory)
        }
        let singleURL = directory.appendingPathComponent("model.safetensors")
        guard FileManager.default.fileExists(atPath: singleURL.path) else {
            throw LoadError.noWeightsFound(directory)
        }
        return try MLX.loadArrays(url: singleURL)
    }

    // MARK: - Private helpers

    private static func loadSharded(indexURL: URL, directory: URL) throws -> [String: MLXArray] {
        let data = try Data(contentsOf: indexURL)
        let index = try JSONDecoder().decode(SafetensorsIndex.self, from: data)
        let shardNames = Set(index.weightMap.values)
        var merged: [String: MLXArray] = [:]
        merged.reserveCapacity(index.weightMap.count)
        for name in shardNames {
            let shardURL = directory.appendingPathComponent(name)
            let shard = try MLX.loadArrays(url: shardURL)
            merged.merge(shard) { _, new in new }
        }
        return merged
    }

    private static func apply(
        _ weights: [String: MLXArray],
        to model: Module,
        dtype: DType
    ) throws {
        let remapped: [(String, MLXArray)] = weights.map { (key, value) in
            (key, value.asType(dtype))
        }
        try model.update(
            parameters: ModuleParameters.unflattened(remapped),
            verify: [.noUnusedKeys]
        )
        MLX.eval(model.parameters())
    }
}

// MARK: - Errors

public enum LoadError: Error, CustomStringConvertible {
    case noWeightsFound(URL)

    public var description: String {
        switch self {
        case .noWeightsFound(let dir):
            return "No model.safetensors or model.safetensors.index.json found in \(dir.path)"
        }
    }
}

// MARK: - JSON decoding

private struct SafetensorsIndex: Decodable {
    let weightMap: [String: String]

    enum CodingKeys: String, CodingKey {
        case weightMap = "weight_map"
    }
}
