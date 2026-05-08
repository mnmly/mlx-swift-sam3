// Sam31VideoPipeline.swift — high-level wrapper around `Sam31VideoTracker`.
//
// Mirrors the `SAM3Predictor` pattern: load weights + tokenizer once via
// `fromPretrained`, hand in a `CGImage` per frame via `processFrame`. The
// underlying `Sam31VideoTracker` is exposed for callers who need the raw
// orchestration knobs (re-detect cadence, association threshold, etc.).

import CoreGraphics
import Foundation
import MLX

public struct Sam31VideoFrameResult {
    public let frameIndex: Int
    public let tracks: [TrackedObject]   // live tracks (framesSinceSeen == 0)
}

public final class Sam31VideoPipeline {
    public let tracker: Sam31VideoTracker
    public let tokenizer: SAM3Tokenizer
    public let resolution: Int
    public let dtype: DType

    private var textCache: [String: (MLXArray, MLXArray?)] = [:]
    private var currentTextPrompt: String?
    private var currentInputIds: MLXArray?
    private var currentAttentionMask: MLXArray?
    private var frameCounter: Int = 0

    public init(
        tracker: Sam31VideoTracker,
        tokenizer: SAM3Tokenizer,
        resolution: Int = 1008,
        dtype: DType = .bfloat16
    ) {
        self.tracker = tracker
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.dtype = dtype
    }

    public static func fromPretrained(
        weightsDirectory: URL,
        tokenizerDirectory: URL? = nil,
        resolution: Int = 1008,
        dtype: DType = .bfloat16,
        detectThreshold: Float = 0.5,
        redetectEvery: Int = 15,
        assocIoUThreshold: Float = 0.3,
        maxMissingFrames: Int = 30,
        tokenizerMaxLength: Int = 32
    ) async throws -> Sam31VideoPipeline {
        let model = Sam31Model(Sam31ModelConfig())
        try ModelLoader.load(sam31: model, from: weightsDirectory, dtype: dtype)
        let tokenizer = try await SAM3Tokenizer.fromSnapshotOrOpenAIClip(
            tokenizerDirectory ?? weightsDirectory,
            maxLength: tokenizerMaxLength
        )
        let tracker = Sam31VideoTracker(
            model: model,
            resolution: resolution,
            dtype: dtype,
            detectThreshold: detectThreshold,
            redetectEvery: redetectEvery,
            assocIoUThreshold: assocIoUThreshold,
            maxMissingFrames: maxMissingFrames
        )
        return Sam31VideoPipeline(
            tracker: tracker, tokenizer: tokenizer,
            resolution: resolution, dtype: dtype
        )
    }

    /// Set the active text prompt. Tokenization is cached, so calling this with
    /// the same prompt across frames is free.
    public func setTextPrompt(_ prompt: String) {
        if currentTextPrompt == prompt { return }
        let cached = textCache[prompt] ?? {
            let enc = tokenizer.encode(prompt)
            textCache[prompt] = (enc.inputIds, enc.attentionMask)
            return (enc.inputIds, enc.attentionMask)
        }()
        currentTextPrompt = prompt
        currentInputIds = cached.0
        currentAttentionMask = cached.1
    }

    /// Run one video frame end-to-end (preprocess + tracker step). Requires
    /// `setTextPrompt` to have been called at least once.
    public func processFrame(_ image: CGImage) throws -> Sam31VideoFrameResult {
        guard let inputIds = currentInputIds else {
            throw SAM3PredictorError.tokenizerRequired
        }
        let pixel = try ImagePreprocess.pixelValues(
            from: image, size: resolution, dtype: dtype)
        let live = tracker.step(
            pixelValues: pixel,
            inputIds: inputIds,
            attentionMask: currentAttentionMask,
            originalHeight: image.height,
            originalWidth: image.width
        )
        let result = Sam31VideoFrameResult(frameIndex: frameCounter, tracks: live)
        frameCounter += 1
        return result
    }

    /// One-call convenience: tokenizes the prompt and runs a single frame.
    public func processFrame(_ image: CGImage, textPrompt: String) throws -> Sam31VideoFrameResult {
        setTextPrompt(textPrompt)
        return try processFrame(image)
    }

    /// Reset all tracker state (frame counter, live tracks, memory banks). Use
    /// before processing a new clip without re-instantiating the pipeline.
    public func reset() {
        frameCounter = 0
        tracker.reset()
    }
}

// MARK: - Top-level convenience

public enum MLXSAM3 {
    /// Image predictor convenience — see `SAM3Predictor.loadSAM31` for the full surface.
    public static func loadSAM31Image(
        weightsDirectory: URL,
        tokenizerDirectory: URL? = nil,
        scoreThreshold: Float = 0.5,
        dtype: DType = .bfloat16
    ) async throws -> SAM3Predictor {
        try await SAM3Predictor.loadSAM31(
            weightsDirectory: weightsDirectory,
            tokenizerDirectory: tokenizerDirectory,
            scoreThreshold: scoreThreshold,
            dtype: dtype
        )
    }

    /// Video tracker convenience — wraps `Sam31VideoPipeline.fromPretrained`.
    public static func loadSAM31Video(
        weightsDirectory: URL,
        tokenizerDirectory: URL? = nil,
        dtype: DType = .bfloat16
    ) async throws -> Sam31VideoPipeline {
        try await Sam31VideoPipeline.fromPretrained(
            weightsDirectory: weightsDirectory,
            tokenizerDirectory: tokenizerDirectory,
            dtype: dtype
        )
    }
}
