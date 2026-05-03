import AppKit
import SwiftUI

struct ContentView: View {
    @StateObject private var model = SAM3ViewModel()
    @State private var selectedTab: Tab = .image

    enum Tab: String, CaseIterable, Identifiable {
        case image, video, camera

        var id: String { rawValue }

        var label: String {
            switch self {
            case .image: "Image"
            case .video: "Video"
            case .camera: "Camera"
            }
        }

        var systemImage: String {
            switch self {
            case .image: "photo"
            case .video: "film"
            case .camera: "camera"
            }
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            modelBar
            Divider()
            TabView(selection: $selectedTab) {
                ImageDemoView(model: model)
                    .tabItem { Label(Tab.image.label, systemImage: Tab.image.systemImage) }
                    .tag(Tab.image)

                VideoDemoView(model: model)
                    .tabItem { Label(Tab.video.label, systemImage: Tab.video.systemImage) }
                    .tag(Tab.video)

                CameraDemoView(model: model)
                    .tabItem { Label(Tab.camera.label, systemImage: Tab.camera.systemImage) }
                    .tag(Tab.camera)
            }
        }
        .frame(minWidth: 980, minHeight: 680)
    }

    private var modelBar: some View {
        HStack(spacing: 12) {
            Text("SAM 3.1").font(.headline)

            Button {
                selectModelDirectory()
            } label: {
                Label("Select Model Directory", systemImage: "folder")
            }

            if model.isLoading {
                ProgressView().controlSize(.small)
            }

            if model.isLoaded {
                Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
                Text(URL(fileURLWithPath: model.modelPath).lastPathComponent)
                    .font(.caption)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text("res \(model.resolution)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if let err = model.errorMessage {
                Image(systemName: "xmark.circle.fill").foregroundStyle(.red)
                Text(err).font(.caption).foregroundStyle(.red).lineLimit(2)
            }

            Spacer()
        }
        .padding(8)
        .background(Color(nsColor: .controlBackgroundColor))
    }

    private func selectModelDirectory() {
        let panel = NSOpenPanel()
        panel.title = "Select SAM 3.1 Model Directory"
        panel.message = "Choose a directory containing model.safetensors and tokenizer.json"
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        Task { await model.loadModel(from: url) }
    }
}

#Preview {
    ContentView()
}
