import AppKit
import CoreGraphics
import SwiftUI
import UniformTypeIdentifiers

struct ImageDemoView: View {
    @ObservedObject var model: SAM3ViewModel

    @State private var sourceImage: CGImage?
    @State private var result: SAM3DisplayResult?
    @State private var prompt = "a person"
    @State private var scoreThreshold: Float = 0.3
    @State private var isRunning = false
    @State private var errorMessage: String?

    var body: some View {
        HStack(spacing: 0) {
            sidebar
            Divider()
            imageArea
        }
    }

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Image").font(.title2.bold())

            Button {
                selectImage()
            } label: {
                Label("Select Image", systemImage: "photo.badge.plus")
            }

            TextField("Prompt", text: $prompt)
                .textFieldStyle(.roundedBorder)

            Text("Score threshold: \(scoreThreshold, specifier: "%.2f")").font(.caption)
            Slider(value: $scoreThreshold, in: 0.01...1.0, step: 0.01)

            Button {
                run()
            } label: {
                Label("Run", systemImage: "play.fill")
            }
            .disabled(!model.isLoaded || sourceImage == nil || isRunning)

            if isRunning {
                ProgressView("Running")
            }

            if let result {
                Divider()
                SAM3ResultsSummary(result: result)
            }

            if let errorMessage {
                Text(errorMessage).font(.caption).foregroundStyle(.red)
            }

            Spacer()
        }
        .padding(12)
        .frame(width: 260)
        .background(Color(nsColor: .controlBackgroundColor))
    }

    @ViewBuilder
    private var imageArea: some View {
        if let sourceImage {
            SAM3OverlayView(cgImage: sourceImage, result: result)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            ContentUnavailableView(
                "No Image Selected",
                systemImage: "photo",
                description: Text("Select an image and enter a prompt")
            )
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    private func selectImage() {
        let panel = NSOpenPanel()
        panel.title = "Select Image"
        panel.allowedContentTypes = [.image]
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        guard let cg = loadCGImage(from: url) else {
            errorMessage = "Failed to load image."
            return
        }
        sourceImage = cg
        result = nil
        errorMessage = nil
    }

    private func run() {
        guard let sourceImage else { return }
        isRunning = true
        errorMessage = nil

        Task {
            do {
                let display = try await model.predictAsync(
                    sourceImage,
                    prompt: prompt,
                    scoreThreshold: scoreThreshold
                )
                result = display
                isRunning = false
            } catch {
                errorMessage = error.localizedDescription
                isRunning = false
            }
        }
    }
}
