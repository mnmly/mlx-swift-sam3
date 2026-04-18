// Sam31Vision.swift — SAM 3.1 vision encoder (TriViTDetNeck + ViTBackbone).
// Python source: sam3_1/vision.py
// Weight keys: detector_model.vision_encoder.*

import MLX
import MLXNN

// MARK: - TriViTDetNeck

public class TriViTDetNeck: Module {
    @ModuleInfo(key: "convs") var convs: [FPNLayer]
    @ModuleInfo(key: "interactive_convs") var interactiveConvs: [FPNLayer]
    @ModuleInfo(key: "propagation_convs") var propagationConvs: [FPNLayer]

    public init(_ config: Sam31VisionEncoderConfig) {
        let inCh = config.backboneConfig.hiddenSize
        func makeFPNs() -> [FPNLayer] {
            config.scaleFactors.map { sf in
                FPNLayer(inChannels: inCh, outChannels: config.fpnHiddenSize,
                         scaleFactor: sf,
                         fpnKernelSize: config.fpnKernelSize,
                         fpnStride: config.fpnStride)
            }
        }
        self._convs = ModuleInfo(wrappedValue: makeFPNs(), key: "convs")
        self._interactiveConvs = ModuleInfo(wrappedValue: makeFPNs(), key: "interactive_convs")
        self._propagationConvs = ModuleInfo(wrappedValue: makeFPNs(), key: "propagation_convs")
        super.init()
    }

    /// - Parameters:
    ///   - x: (B, H, W, C) ViT backbone output
    /// - Returns: (detFeatures, interactiveFeatures, propagationFeatures), each [scale] (B, H_i, W_i, D)
    public func callAsFunction(
        _ x: MLXArray,
        needDet: Bool = true,
        needInteractive: Bool = true,
        needPropagation: Bool = true
    ) -> ([MLXArray], [MLXArray], [MLXArray]) {
        let det = needDet ? convs.map { $0(x) } : []
        let inter = needInteractive ? interactiveConvs.map { $0(x) } : []
        let prop = needPropagation ? propagationConvs.map { $0(x) } : []
        return (det, inter, prop)
    }
}

// MARK: - Sam31VisionEncoder

public class Sam31VisionEncoder: Module {
    @ModuleInfo(key: "backbone") var backbone: ViTBackbone
    @ModuleInfo(key: "neck") var neck: TriViTDetNeck

    public init(_ config: Sam31VisionEncoderConfig) {
        self._backbone = ModuleInfo(wrappedValue: ViTBackbone(config.backboneConfig), key: "backbone")
        self._neck = ModuleInfo(wrappedValue: TriViTDetNeck(config), key: "neck")
        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        needDet: Bool = true,
        needInteractive: Bool = true,
        needPropagation: Bool = true
    ) -> ([MLXArray], [MLXArray], [MLXArray]) {
        let features = backbone(x)
        return neck(features, needDet: needDet, needInteractive: needInteractive, needPropagation: needPropagation)
    }
}
