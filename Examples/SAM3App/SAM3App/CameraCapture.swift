import AVFoundation
import CoreGraphics
import CoreImage
import Foundation

nonisolated final class CameraCapture: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {
    private let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "sam3app.camera.capture")
    private let ciContext = CIContext()
    private var device: AVCaptureDevice?
    private var inFlight = false
    private let inFlightLock = NSLock()

    var onFrame: ((CGImage) -> Void)?

    var availableDevices: [AVCaptureDevice] {
        AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInWideAngleCamera, .external, .continuityCamera],
            mediaType: .video,
            position: .unspecified
        ).devices
    }

    func authorize() async -> Bool {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            true
        case .notDetermined:
            await AVCaptureDevice.requestAccess(for: .video)
        default:
            false
        }
    }

    func start(deviceID: String? = nil) throws {
        session.beginConfiguration()
        session.sessionPreset = .high

        for input in session.inputs { session.removeInput(input) }
        for output in session.outputs { session.removeOutput(output) }

        let selectedDevice: AVCaptureDevice? = {
            if let deviceID, let device = AVCaptureDevice(uniqueID: deviceID) {
                return device
            }
            return AVCaptureDevice.default(for: .video)
        }()

        guard let selectedDevice else {
            session.commitConfiguration()
            throw CaptureError.noDevice
        }

        let input = try AVCaptureDeviceInput(device: selectedDevice)
        guard session.canAddInput(input) else {
            session.commitConfiguration()
            throw CaptureError.cannotAddInput
        }
        session.addInput(input)

        let output = AVCaptureVideoDataOutput()
        output.alwaysDiscardsLateVideoFrames = true
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        output.setSampleBufferDelegate(self, queue: queue)
        guard session.canAddOutput(output) else {
            session.commitConfiguration()
            throw CaptureError.cannotAddOutput
        }
        session.addOutput(output)

        device = selectedDevice
        session.commitConfiguration()

        if !session.isRunning {
            session.startRunning()
        }
    }

    func stop() {
        if session.isRunning {
            session.stopRunning()
        }
    }

    nonisolated func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        inFlightLock.lock()
        if inFlight {
            inFlightLock.unlock()
            return
        }
        inFlight = true
        inFlightLock.unlock()

        defer {
            inFlightLock.lock()
            inFlight = false
            inFlightLock.unlock()
        }

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else { return }
        onFrame?(cgImage)
    }

    enum CaptureError: LocalizedError {
        case noDevice
        case cannotAddInput
        case cannotAddOutput
        case denied

        var errorDescription: String? {
            switch self {
            case .noDevice:
                "No camera device available."
            case .cannotAddInput:
                "Capture session refused the camera input."
            case .cannotAddOutput:
                "Capture session refused the video output."
            case .denied:
                "Camera access denied. Enable it in System Settings."
            }
        }
    }
}
