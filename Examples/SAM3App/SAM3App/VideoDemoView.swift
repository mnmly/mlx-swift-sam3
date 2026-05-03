import AppKit
import CoreGraphics
import SwiftUI
import UniformTypeIdentifiers

struct VideoDemoView: View {
    @ObservedObject var model: SAM3ViewModel

    @State private var sourceURL: URL?
    @State private var frames: [SAM3ProcessedFrame] = []
    @State private var currentFrameIndex = 0
    @State private var prompt = "a person"
    @State private var scoreThreshold: Float = 0.3
    @State private var frameStride = 5
    @State private var playbackFPS = 10.0
    @State private var isProcessing = false
    @State private var isPlaying = false
    @State private var progress = 0.0
    @State private var totalFrames = 0
    @State private var errorMessage: String?
    @State private var processingTask: Task<Void, Never>?
    @State private var playTimer: Timer?

    var body: some View {
        HStack(spacing: 0) {
            sidebar
            Divider()
            videoArea
        }
        .onDisappear {
            stopPlayback()
            processingTask?.cancel()
        }
    }

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Video").font(.title2.bold())

            Button {
                selectVideo()
            } label: {
                Label("Select Video", systemImage: "film")
            }

            TextField("Prompt", text: $prompt)
                .textFieldStyle(.roundedBorder)

            Text("Score threshold: \(scoreThreshold, specifier: "%.2f")").font(.caption)
            Slider(value: $scoreThreshold, in: 0.01...1.0, step: 0.01)

            Picker("Frame stride", selection: $frameStride) {
                ForEach([1, 2, 3, 5, 10, 15], id: \.self) { stride in
                    Text(stride == 1 ? "every frame" : "every \(stride)").tag(stride)
                }
            }

            Picker("Playback FPS", selection: $playbackFPS) {
                ForEach([5.0, 10.0, 15.0, 24.0, 30.0], id: \.self) { fps in
                    Text("\(Int(fps)) fps").tag(fps)
                }
            }

            HStack {
                Button {
                    processVideo()
                } label: {
                    Label("Process", systemImage: "play.fill")
                }
                .disabled(!model.isLoaded || sourceURL == nil || isProcessing)

                if isProcessing {
                    Button(role: .destructive) {
                        processingTask?.cancel()
                    } label: {
                        Label("Cancel", systemImage: "xmark")
                    }
                }
            }

            if isProcessing {
                ProgressView(value: progress) { Text("Processing") }
                Text("\(Int(progress * 100))%  (\(frames.count)/\(totalFrames))")
                    .font(.caption2)
            }

            if !frames.isEmpty {
                Divider()
                HStack {
                    Button {
                        togglePlayback()
                    } label: {
                        Label(isPlaying ? "Pause" : "Play", systemImage: isPlaying ? "pause.fill" : "play.fill")
                    }

                    Button {
                        resetPlayback()
                    } label: {
                        Label("Reset", systemImage: "backward.end.fill")
                    }
                }

                if frames.count > 1 {
                    Slider(
                        value: Binding(
                            get: { Double(currentFrameIndex) },
                            set: { currentFrameIndex = Int($0) }
                        ),
                        in: 0...Double(frames.count - 1),
                        step: 1
                    )
                }

                Text("Frame \(currentFrameIndex + 1) of \(frames.count)").font(.caption)
                SAM3ResultsSummary(result: frames[currentFrameIndex].result)
            }

            if let errorMessage {
                Text(errorMessage).font(.caption).foregroundStyle(.red)
            }

            Spacer()
        }
        .padding(12)
        .frame(width: 280)
        .background(Color(nsColor: .controlBackgroundColor))
    }

    @ViewBuilder
    private var videoArea: some View {
        if !frames.isEmpty {
            let frame = frames[currentFrameIndex]
            SAM3OverlayView(cgImage: frame.image, result: frame.result)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            ContentUnavailableView(
                "No Processed Video",
                systemImage: "film.stack",
                description: Text("Select a video, prompt, and process frames")
            )
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    private func selectVideo() {
        let panel = NSOpenPanel()
        panel.title = "Select Video"
        panel.allowedContentTypes = [.movie, .video]
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        sourceURL = url
        frames = []
        currentFrameIndex = 0
        progress = 0
        errorMessage = nil
    }

    private func processVideo() {
        guard let sourceURL else { return }
        isProcessing = true
        frames = []
        currentFrameIndex = 0
        progress = 0
        errorMessage = nil
        stopPlayback()

        let prompt = prompt
        let threshold = scoreThreshold
        let stride = frameStride

        processingTask = Task {
            do {
                let source = try await VideoSource(url: sourceURL, frameStride: stride)
                totalFrames = source.totalFrames

                var collected: [SAM3ProcessedFrame] = []
                var index = 0

                while let frame = source.nextFrame() {
                    if Task.isCancelled { break }
                    let result = try await model.predictAsync(frame, prompt: prompt, scoreThreshold: threshold)
                    collected.append(SAM3ProcessedFrame(image: frame, result: result))
                    index += 1

                    if index == 1 || index % 2 == 0 {
                        frames = collected
                        currentFrameIndex = max(0, collected.count - 1)
                        progress = source.totalFrames > 0 ? Double(index) / Double(source.totalFrames) : 0
                    }
                }
                source.release()

                frames = collected
                currentFrameIndex = 0
                progress = Task.isCancelled ? progress : 1
                isProcessing = false
                processingTask = nil
            } catch {
                errorMessage = error.localizedDescription
                isProcessing = false
                processingTask = nil
            }
        }
    }

    private func togglePlayback() {
        if isPlaying {
            stopPlayback()
        } else {
            isPlaying = true
            let interval = playbackFPS > 0 ? 1.0 / playbackFPS : 0.1
            playTimer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { _ in
                Task { @MainActor in
                    if currentFrameIndex < frames.count - 1 {
                        currentFrameIndex += 1
                    } else {
                        stopPlayback()
                    }
                }
            }
        }
    }

    private func resetPlayback() {
        stopPlayback()
        currentFrameIndex = 0
    }

    private func stopPlayback() {
        playTimer?.invalidate()
        playTimer = nil
        isPlaying = false
    }
}
