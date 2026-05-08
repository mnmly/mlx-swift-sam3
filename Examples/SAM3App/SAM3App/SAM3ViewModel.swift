import Combine
import CoreGraphics
import Foundation
import MLX
import MLXSAM3

@MainActor
final class SAM3ViewModel: ObservableObject {
    @Published var predictor: SAM3Predictor?
    @Published var videoPipeline: Sam31VideoPipeline?
    @Published var isLoaded = false
    @Published var isLoading = false
    @Published var errorMessage: String?
    @Published var modelPath: String = ""
    @Published var resolution: Int = 1008

    private var loadedURL: URL?

    func loadModel(from directory: URL) async {
        isLoading = true
        errorMessage = nil
        modelPath = directory.path
        defer { isLoading = false }

        do {
            let threshold: Float = 0.5
            let hasScopedAccess = directory.startAccessingSecurityScopedResource()
            defer {
                if hasScopedAccess {
                    directory.stopAccessingSecurityScopedResource()
                }
            }
            // If the user picked the HF cache repo root (`models--org--name/`),
            // resolve to the actual `snapshots/<rev>/` so MLX can mmap the
            // safetensors blobs through symlinks that stay inside the grant.
            let resolved = Self.resolveHuggingFaceSnapshot(directory) ?? directory
            modelPath = resolved.path
            let loaded = try await Task.detached(priority: .userInitiated) {
                () -> (SAM3Predictor, Sam31VideoPipeline) in
                let tokenizer = try await SAM3Tokenizer.fromSnapshotOrOpenAIClip(
                    resolved,
                    maxLength: 32
                )
                let model = Sam31Model(Sam31ModelConfig())
                try ModelLoader.load(sam31: model, from: resolved, dtype: .bfloat16)
                let imagePredictor = SAM3Predictor(
                    model: model,
                    tokenizer: tokenizer,
                    resolution: 1008,
                    scoreThreshold: threshold
                )
                // Share the same loaded `model` between the image predictor and the
                // video tracker — there's no need to load weights twice.
                let tracker = Sam31VideoTracker(
                    model: model,
                    resolution: 1008,
                    dtype: .bfloat16,
                    detectThreshold: threshold,
                    redetectEvery: 15
                )
                let videoPipeline = Sam31VideoPipeline(
                    tracker: tracker,
                    tokenizer: tokenizer,
                    resolution: 1008,
                    dtype: .bfloat16
                )
                return (imagePredictor, videoPipeline)
            }.value
            predictor = loaded.0
            videoPipeline = loaded.1
            resolution = loaded.0.resolution
            isLoaded = true
            loadedURL = directory
        } catch {
            errorMessage = Self.describeLoadError(error, directory: directory)
            predictor = nil
            videoPipeline = nil
            isLoaded = false
        }
    }

    func reloadIfNeeded() async {
        guard let loadedURL, !isLoading else { return }
        await loadModel(from: loadedURL)
    }

    nonisolated func predictAsync(
        _ cgImage: CGImage,
        prompt: String,
        scoreThreshold: Float
    ) async throws -> SAM3DisplayResult {
        let trimmedPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedPrompt.isEmpty else { throw SAM3AppError.emptyPrompt }
        guard let predictor = await self.predictor else { throw SAM3AppError.modelNotLoaded }

        return try await Task.detached(priority: .userInitiated) {
            predictor.scoreThreshold = scoreThreshold
            let prediction = try predictor.predict(image: cgImage, textPrompt: trimmedPrompt)
            return makeDisplayResult(
                from: prediction,
                width: cgImage.width,
                height: cgImage.height
            )
        }.value
    }

    /// Run one frame through the persistent-ID video tracker. Caller is responsible
    /// for calling `resetVideoTracker()` when starting a new clip / camera session.
    nonisolated func trackFrameAsync(
        _ cgImage: CGImage,
        prompt: String,
        scoreThreshold: Float
    ) async throws -> SAM3DisplayResult {
        let trimmedPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedPrompt.isEmpty else { throw SAM3AppError.emptyPrompt }
        guard let pipeline = await self.videoPipeline else {
            throw SAM3AppError.modelNotLoaded
        }

        return try await Task.detached(priority: .userInitiated) {
            pipeline.tracker.detectThreshold = scoreThreshold
            pipeline.setTextPrompt(trimmedPrompt)
            let frame = try pipeline.processFrame(cgImage)
            return makeDisplayResult(
                fromTracks: frame.tracks,
                width: cgImage.width,
                height: cgImage.height
            )
        }.value
    }

    func resetVideoTracker() {
        videoPipeline?.reset()
    }

    private static func describeLoadError(_ error: Error, directory: URL) -> String {
        let message = error.localizedDescription
        if message.contains("load_safetensors") {
            return """
            MLX could not read model.safetensors in \(directory.lastPathComponent). For a Hugging Face cache, pick the repo root (e.g. ~/.cache/huggingface/hub/models--<org>--<name>) so both snapshots/ and blobs/ are inside the sandbox grant — the app will resolve to the snapshot for you.
            """
        }
        return message
    }

    /// If `directory` is an HF cache repo root (`models--<org>--<name>/`),
    /// return the `snapshots/<rev>/` URL it points at. Returns nil otherwise.
    ///
    /// The user must pick the repo root rather than the snapshot itself so the
    /// security-scoped grant covers `blobs/` — the snapshot folder contains
    /// only relative symlinks (`../../blobs/<sha>`) which fail to mmap if the
    /// grant is scoped to the snapshot directory alone.
    private static func resolveHuggingFaceSnapshot(_ directory: URL) -> URL? {
        let fm = FileManager.default
        let snapshots = directory.appendingPathComponent("snapshots", isDirectory: true)
        let blobs = directory.appendingPathComponent("blobs", isDirectory: true)
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: snapshots.path, isDirectory: &isDir), isDir.boolValue,
              fm.fileExists(atPath: blobs.path, isDirectory: &isDir), isDir.boolValue
        else { return nil }

        // Prefer the revision pinned by refs/main, if present.
        let refsMain = directory.appendingPathComponent("refs/main")
        if let rev = try? String(contentsOf: refsMain, encoding: .utf8) {
            let trimmed = rev.trimmingCharacters(in: .whitespacesAndNewlines)
            let candidate = snapshots.appendingPathComponent(trimmed, isDirectory: true)
            if fm.fileExists(atPath: candidate.path) { return candidate }
        }

        // Fall back to the most recently modified snapshot directory.
        let entries = (try? fm.contentsOfDirectory(
            at: snapshots,
            includingPropertiesForKeys: [.contentModificationDateKey, .isDirectoryKey],
            options: [.skipsHiddenFiles]
        )) ?? []
        let dirs = entries.filter {
            (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
        }
        return dirs.max { a, b in
            let da = (try? a.resourceValues(forKeys: [.contentModificationDateKey])
                .contentModificationDate) ?? .distantPast
            let db = (try? b.resourceValues(forKeys: [.contentModificationDateKey])
                .contentModificationDate) ?? .distantPast
            return da < db
        }
    }
}
