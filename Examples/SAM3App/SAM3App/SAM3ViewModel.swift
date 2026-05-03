import Combine
import CoreGraphics
import Foundation
import MLX
import MLXSAM3

@MainActor
final class SAM3ViewModel: ObservableObject {
    @Published var predictor: SAM3Predictor?
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
            let loaded = try await Task.detached(priority: .userInitiated) {
                let tokenizer = try await SAM3Tokenizer.fromSnapshotOrOpenAIClip(
                    directory,
                    maxLength: 32
                )
                let model = Sam31Model(Sam31ModelConfig())
                try ModelLoader.load(sam31: model, from: directory, dtype: .bfloat16)
                return SAM3Predictor(
                    model: model,
                    tokenizer: tokenizer,
                    resolution: 1008,
                    scoreThreshold: threshold
                )
            }.value
            predictor = loaded
            resolution = loaded.resolution
            isLoaded = true
            loadedURL = directory
        } catch {
            errorMessage = Self.describeLoadError(error, directory: directory)
            predictor = nil
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

    private static func describeLoadError(_ error: Error, directory: URL) -> String {
        let message = error.localizedDescription
        if message.contains("load_safetensors") {
            return """
            MLX could not read model.safetensors in \(directory.lastPathComponent). If this is a Hugging Face cache snapshot, select the snapshot via the app's file picker and keep the app sandbox read permission. If it still fails, copy the snapshot with resolved symlinks to a normal folder.
            """
        }
        return message
    }
}
