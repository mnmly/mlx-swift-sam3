import AVFoundation
import CoreGraphics
import SwiftUI

struct CameraDemoView: View {
    @ObservedObject var model: SAM3ViewModel

    @State private var capture = CameraCapture()
    @State private var devices: [AVCaptureDevice] = []
    @State private var deviceID: String?
    @State private var currentFrame: CGImage?
    @State private var result: SAM3DisplayResult?
    @State private var prompt = "a person"
    @State private var scoreThreshold: Float = 0.3
    @State private var isRunning = false
    @State private var inferenceInFlight = false
    @State private var errorMessage: String?
    @State private var fps = 0.0
    @State private var lastFrameTime = Date()

    var body: some View {
        HStack(spacing: 0) {
            sidebar
            Divider()
            cameraArea
        }
        .onDisappear { stop() }
    }

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Camera").font(.title2.bold())

            Picker("Device", selection: Binding(
                get: { deviceID ?? "" },
                set: { newID in
                    deviceID = newID.isEmpty ? nil : newID
                    if isRunning { restart() }
                }
            )) {
                Text("Default").tag("")
                ForEach(devices, id: \.uniqueID) { device in
                    Text(device.localizedName).tag(device.uniqueID)
                }
            }

            TextField("Prompt", text: $prompt)
                .textFieldStyle(.roundedBorder)

            Text("Score threshold: \(scoreThreshold, specifier: "%.2f")").font(.caption)
            Slider(value: $scoreThreshold, in: 0.01...1.0, step: 0.01)

            Button {
                if isRunning {
                    stop()
                } else {
                    Task { await start() }
                }
            } label: {
                Label(isRunning ? "Stop" : "Start", systemImage: isRunning ? "stop.fill" : "camera.fill")
            }
            .disabled(!model.isLoaded && !isRunning)

            if isRunning {
                Text(String(format: "Camera FPS: %.1f", fps))
                    .font(.caption.monospacedDigit())
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
        .frame(width: 280)
        .background(Color(nsColor: .controlBackgroundColor))
        .onAppear { devices = capture.availableDevices }
    }

    @ViewBuilder
    private var cameraArea: some View {
        if let currentFrame {
            SAM3OverlayView(cgImage: currentFrame, result: result)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else if isRunning {
            ProgressView("Starting camera")
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            ContentUnavailableView(
                "Camera Off",
                systemImage: "camera",
                description: Text(model.isLoaded ? "Start the camera to run prompted inference" : "Load a model first")
            )
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    private func start() async {
        guard model.isLoaded else {
            errorMessage = SAM3AppError.modelNotLoaded.localizedDescription
            return
        }

        let granted = await capture.authorize()
        guard granted else {
            errorMessage = CameraCapture.CaptureError.denied.localizedDescription
            return
        }

        capture.onFrame = { cgImage in
            Task { @MainActor in
                handleFrame(cgImage)
            }
        }

        do {
            try capture.start(deviceID: deviceID)
            isRunning = true
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
            isRunning = false
        }
    }

    private func stop() {
        capture.stop()
        capture.onFrame = nil
        isRunning = false
        inferenceInFlight = false
    }

    private func restart() {
        stop()
        Task { await start() }
    }

    private func handleFrame(_ cgImage: CGImage) {
        currentFrame = cgImage

        let now = Date()
        let dt = now.timeIntervalSince(lastFrameTime)
        if dt > 0 {
            fps = 0.8 * fps + 0.2 * (1.0 / dt)
        }
        lastFrameTime = now

        if inferenceInFlight { return }
        inferenceInFlight = true

        let prompt = prompt
        let threshold = scoreThreshold
        Task {
            do {
                result = try await model.predictAsync(cgImage, prompt: prompt, scoreThreshold: threshold)
            } catch {
                errorMessage = error.localizedDescription
            }
            inferenceInFlight = false
        }
    }
}
