// Configs.swift — all SAM 3 and SAM 3.1 configuration structs.
// Python sources: sam3/config.py, sam3_1/config.py

// MARK: - Shared protocols

public protocol MemoryFuserConfigProtocol {
    var memoryFuserEmbedDim: Int { get }
    var memoryFuserKernelSize: Int { get }
    var memoryFuserPadding: Int { get }
    var memoryFuserIntermediateDim: Int { get }
    var memoryFuserLayerScaleInitValue: Float { get }
    var memoryFuserNumLayers: Int { get }
}

// MARK: - Shared / SAM 3

public struct ViTConfig {
    public var hiddenSize: Int = 1024
    public var numHiddenLayers: Int = 32
    public var numAttentionHeads: Int = 16
    public var intermediateSize: Int = 4736
    public var hiddenAct: String = "gelu"
    public var imageSize: Int = 1008
    public var patchSize: Int = 14
    public var numChannels: Int = 3
    public var windowSize: Int = 24
    public var globalAttnIndexes: [Int] = [7, 15, 23, 31]
    public var qkvBias: Bool = true
    public var ropeTheta: Float = 10000.0
    public var pretrainImageSize: Int = 336
    public var layerNormEps: Float = 1e-6
    public var layerScaleInitValue: Float? = nil
    public var hiddenDropout: Float = 0.0
    public var attentionDropout: Float = 0.0
    public init() {}
}

/// SAM 3 vision encoder — ViTBackbone + FPNNeck with 4 scale factors.
public struct VisionEncoderConfig {
    public var backboneConfig: ViTConfig = ViTConfig()
    public var fpnHiddenSize: Int = 256
    public var fpnKernelSize: Int = 2
    public var fpnStride: Int = 2
    public var scaleFactors: [Float] = [4.0, 2.0, 1.0, 0.5]
    public var numFeatureLevels: Int = 3
    public var backboneFeatureSizes: [[Int]] = [[288, 288], [144, 144], [72, 72]]
    public var layerNormEps: Float = 1e-6
    public init() {}
}

public struct TextEncoderConfig {
    public var hiddenSize: Int = 1024
    public var numHiddenLayers: Int = 24
    public var numAttentionHeads: Int = 16
    public var intermediateSize: Int = 4096
    public var hiddenAct: String = "gelu"
    public var vocabSize: Int = 49408
    public var maxPositionEmbeddings: Int = 32
    public var projectionDim: Int = 512
    public var layerNormEps: Float = 1e-5
    public var attentionDropout: Float = 0.0
    public var bosTokenId: Int = 49406
    public var eosTokenId: Int = 49407
    public var padTokenId: Int = 1
    public init() {}
}

public struct DETREncoderConfig {
    public var hiddenSize: Int = 256
    public var numLayers: Int = 6
    public var numAttentionHeads: Int = 8
    public var intermediateSize: Int = 2048
    public var hiddenAct: String = "relu"
    public var dropout: Float = 0.1
    public var layerNormEps: Float = 1e-6
    public init() {}
}

public struct DETRDecoderConfig {
    public var hiddenSize: Int = 256
    public var numLayers: Int = 6
    public var numAttentionHeads: Int = 8
    public var numQueries: Int = 200
    public var intermediateSize: Int = 2048
    public var hiddenAct: String = "relu"
    public var dropout: Float = 0.1
    public var layerNormEps: Float = 1e-6
    public var boxRpbMode: String = "log"
    public var usePresenceToken: Bool = true
    public init() {}
}

public struct GeometryEncoderConfig {
    public var hiddenSize: Int = 256
    public var numLayers: Int = 3
    public var numAttentionHeads: Int = 8
    public var intermediateSize: Int = 2048
    public var hiddenAct: String = "relu"
    public var dropout: Float = 0.1
    public var roiSize: Int = 7
    public var layerNormEps: Float = 1e-6
    public init() {}
}

public struct DetectorMaskDecoderConfig {
    public var hiddenSize: Int = 256
    public var numAttentionHeads: Int = 8
    public var numUpsamplingStages: Int = 3
    public var dropout: Float = 0.0
    public var layerNormEps: Float = 1e-6
    public init() {}
}

/// SAM 3 detector (vision + text + DETR + geometry + mask decoder).
public struct DetectorConfig {
    public var visionConfig: VisionEncoderConfig = VisionEncoderConfig()
    public var textConfig: TextEncoderConfig = TextEncoderConfig()
    public var detrEncoderConfig: DETREncoderConfig = DETREncoderConfig()
    public var detrDecoderConfig: DETRDecoderConfig = DETRDecoderConfig()
    public var geometryEncoderConfig: GeometryEncoderConfig = GeometryEncoderConfig()
    public var maskDecoderConfig: DetectorMaskDecoderConfig = DetectorMaskDecoderConfig()
    public var initializerRange: Float = 0.02
    public init() {}
}

/// SAM 3 tracker mask decoder.
public struct TrackerMaskDecoderConfig {
    public var hiddenSize: Int = 256
    public var numHiddenLayers: Int = 2
    public var numAttentionHeads: Int = 8
    public var attentionDownsampleRate: Int = 2
    public var numMultimaskOutputs: Int = 3
    public var iouHeadDepth: Int = 3
    public var iouHeadHiddenDim: Int = 256
    public var mlpDim: Int = 2048
    public var hiddenAct: String = "gelu"
    public var dynamicMultimaskViaStability: Bool = true
    public var dynamicMultimaskStabilityDelta: Float = 0.05
    public var dynamicMultimaskStabilityThresh: Float = 0.98
    public init() {}
}

public struct PromptEncoderConfig {
    public var hiddenSize: Int = 256
    public var imageSize: Int = 1008
    public var patchSize: Int = 14
    public var maskInputChannels: Int = 16
    public var numPointEmbeddings: Int = 4
    public var hiddenAct: String = "gelu"
    public var scale: Int = 1
    public init() {}
}

/// SAM 3 tracker.
public struct TrackerConfig: MemoryFuserConfigProtocol {
    public var imageSize: Int = 1008
    public var visionConfig: VisionEncoderConfig = VisionEncoderConfig()
    public var maskDecoderConfig: TrackerMaskDecoderConfig = TrackerMaskDecoderConfig()
    public var promptEncoderConfig: PromptEncoderConfig = PromptEncoderConfig()

    // Memory attention
    public var memoryAttentionHiddenSize: Int = 256
    public var memoryAttentionNumLayers: Int = 4
    public var memoryAttentionNumAttentionHeads: Int = 1
    public var memoryAttentionFeedForwardHiddenSize: Int = 2048
    public var memoryAttentionFeedForwardHiddenAct: String = "relu"
    public var memoryAttentionDropout: Float = 0.1
    public var memoryAttentionRopeDropout: Float = 0.1
    public var memoryAttentionRopeTheta: Float = 10000.0
    public var memoryAttentionRopeFeatSizes: [Int] = [72, 72]
    public var memoryAttentionDownsampleRate: Int = 1

    // Memory encoder
    public var memoryEncoderHiddenSize: Int = 256
    public var memoryEncoderOutputChannels: Int = 64

    // Mask downsampler
    public var maskDownsamplerEmbedDim: Int = 256
    public var maskDownsamplerKernelSize: Int = 3
    public var maskDownsamplerStride: Int = 2
    public var maskDownsamplerPadding: Int = 1
    public var maskDownsamplerTotalStride: Int = 16
    public var maskDownsamplerHiddenAct: String = "gelu"

    // Memory fuser (CXBlock)
    public var memoryFuserEmbedDim: Int = 256
    public var memoryFuserKernelSize: Int = 7
    public var memoryFuserPadding: Int = 3
    public var memoryFuserNumLayers: Int = 2
    public var memoryFuserIntermediateDim: Int = 1024
    public var memoryFuserLayerScaleInitValue: Float = 1e-6
    public var memoryFuserHiddenAct: String = "gelu"

    // Tracker settings
    public var numMaskmem: Int = 7
    public var maxCondFrameNum: Int = 4
    public var maxObjectPointersInEncoder: Int = 16
    public var multimaskOutputInSam: Bool = true
    public var multimaskOutputForTracking: Bool = true
    public var multimaskMinPtNum: Int = 0
    public var multimaskMaxPtNum: Int = 1

    // Memory encoding params
    public var sigmoidBiasForMemEnc: Float = -10.0
    public var sigmoidScaleForMemEnc: Float = 20.0

    // Occlusion / temporal
    public var enableOcclusionSpatialEmbedding: Bool = true
    public var enableTemporalPosEncodingForObjectPointers: Bool = true

    public init() {}
}

/// SAM 3 top-level model config.
public struct ModelConfig {
    public var detectorConfig: DetectorConfig = DetectorConfig()
    public var trackerConfig: TrackerConfig = TrackerConfig()
    public var initializerRange: Float = 0.02
    public var lowResMaskSize: Int = 288

    // Tracking / association thresholds
    public var detNmsThresh: Float = 0.1
    public var assocIouThresh: Float = 0.1
    public var trkAssocIouThresh: Float = 0.5
    public var highConfThresh: Float = 0.8
    public var highIouThresh: Float = 0.8
    public var newDetThresh: Float = 0.7
    public var scoreThresholdDetection: Float = 0.5
    public var fillHoleArea: Int = 16
    public var maxNumObjects: Int = 10000

    // Tracker management
    public var initTrkKeepAlive: Int = 30
    public var maxTrkKeepAlive: Int = 30
    public var minTrkKeepAlive: Int = -1
    public var hotstartDelay: Int = 15
    public var hotstartDupThresh: Int = 8
    public var hotstartUnmatchThresh: Int = 8
    public var reconditionEveryNthFrame: Int = 16
    public var reconditionOnTrkMasks: Bool = false
    public var decreaseTrkKeepAliveForEmptyMasklets: Bool = false
    public var suppressUnmatchedOnlyWithinHotstart: Bool = true
    public var suppressOverlappingBasedOnRecentOcclusionThreshold: Float = 0.7

    public init() {}
}

// MARK: - SAM 3.1 additions

/// SAM 3.1 vision encoder — TriViTDetNeck with 3 scales (no 0.5x downsample).
public struct Sam31VisionEncoderConfig {
    public var backboneConfig: ViTConfig = ViTConfig()
    public var fpnHiddenSize: Int = 256
    public var fpnKernelSize: Int = 2
    public var fpnStride: Int = 2
    public var scaleFactors: [Float] = [4.0, 2.0, 1.0]
    public var numFeatureLevels: Int = 3
    public var backboneFeatureSizes: [[Int]] = [[288, 288], [144, 144], [72, 72]]
    public var layerNormEps: Float = 1e-6
    public init() {}
}

/// SAM 3.1 tracker mask decoder — adds multiplex support.
public struct Sam31TrackerMaskDecoderConfig {
    public var hiddenSize: Int = 256
    public var numHiddenLayers: Int = 2
    public var numAttentionHeads: Int = 8
    public var attentionDownsampleRate: Int = 2
    public var numMultimaskOutputs: Int = 3
    public var iouHeadDepth: Int = 3
    public var iouHeadHiddenDim: Int = 256
    public var mlpDim: Int = 2048
    public var hiddenAct: String = "gelu"
    public var dynamicMultimaskViaStability: Bool = true
    public var dynamicMultimaskStabilityDelta: Float = 0.05
    public var dynamicMultimaskStabilityThresh: Float = 0.98
    public var multiplexCount: Int = 16
    public init() {}
}

/// SAM 3.1 detector config.
public struct Sam31DetectorConfig {
    public var visionConfig: Sam31VisionEncoderConfig = Sam31VisionEncoderConfig()
    public var textConfig: TextEncoderConfig = TextEncoderConfig()
    public var detrEncoderConfig: DETREncoderConfig = DETREncoderConfig()
    public var detrDecoderConfig: DETRDecoderConfig = DETRDecoderConfig()
    public var geometryEncoderConfig: GeometryEncoderConfig = GeometryEncoderConfig()
    public var maskDecoderConfig: DetectorMaskDecoderConfig = DetectorMaskDecoderConfig()
    public var initializerRange: Float = 0.02
    public init() {}
}

/// SAM 3.1 tracker config — Object Multiplex.
public struct Sam31TrackerConfig: MemoryFuserConfigProtocol {
    public var imageSize: Int = 1008
    public var visionConfig: Sam31VisionEncoderConfig = Sam31VisionEncoderConfig()
    public var maskDecoderConfig: Sam31TrackerMaskDecoderConfig = Sam31TrackerMaskDecoderConfig()
    public var promptEncoderConfig: PromptEncoderConfig = PromptEncoderConfig()

    public var multiplexCount: Int = 16

    // Memory attention (decoupled)
    public var memoryAttentionHiddenSize: Int = 256
    public var memoryAttentionNumLayers: Int = 4
    // SAM 3.1 HF config: memory_attention_num_attention_heads=1 (single-head with head_dim=256).
    // Note: python/sam3 model_builder hard-codes num_heads=8 with the same checkpoint, so
    // the bisect test against python shows ~4% rel err that's INHERENT to the heads config —
    // not a Swift bug. HF transformers / mlx-community / facebook HF all agree on heads=1.
    public var memoryAttentionNumAttentionHeads: Int = 1
    public var memoryAttentionFeedForwardHiddenSize: Int = 2048
    public var memoryAttentionFeedForwardHiddenAct: String = "relu"
    public var memoryAttentionDropout: Float = 0.1
    public var memoryAttentionRopeDropout: Float = 0.1
    public var memoryAttentionRopeTheta: Float = 10000.0
    public var memoryAttentionRopeFeatSizes: [Int] = [72, 72]
    public var memoryAttentionDownsampleRate: Int = 1

    // Memory encoder — SAM 3.1 uses 256 output channels (SAM 3 used 64)
    public var memoryEncoderHiddenSize: Int = 256
    public var memoryEncoderOutputChannels: Int = 256

    // Mask downsampler
    public var maskDownsamplerEmbedDim: Int = 256
    public var maskDownsamplerKernelSize: Int = 3
    public var maskDownsamplerStride: Int = 2
    public var maskDownsamplerPadding: Int = 1
    public var maskDownsamplerTotalStride: Int = 16
    public var maskDownsamplerHiddenAct: String = "gelu"
    public var maskDownsamplerFirstChannels: Int = 16

    // Memory fuser (CXBlock)
    public var memoryFuserEmbedDim: Int = 256
    public var memoryFuserKernelSize: Int = 7
    public var memoryFuserPadding: Int = 3
    public var memoryFuserNumLayers: Int = 2
    public var memoryFuserIntermediateDim: Int = 1024
    public var memoryFuserLayerScaleInitValue: Float = 1e-6
    public var memoryFuserHiddenAct: String = "gelu"

    // Tracker settings
    public var numMaskmem: Int = 7
    public var maxCondFrameNum: Int = 4
    public var maxObjectPointersInEncoder: Int = 16
    public var multimaskOutputInSam: Bool = true
    public var multimaskOutputForTracking: Bool = true
    public var multimaskMinPtNum: Int = 0
    public var multimaskMaxPtNum: Int = 1

    // Memory encoding params
    public var sigmoidBiasForMemEnc: Float = -10.0
    public var sigmoidScaleForMemEnc: Float = 20.0

    // Occlusion / temporal
    public var enableOcclusionSpatialEmbedding: Bool = true
    public var enableTemporalPosEncodingForObjectPointers: Bool = true

    public init() {}
}

/// SAM 3.1 top-level model config.
public struct Sam31ModelConfig {
    public var detectorConfig: Sam31DetectorConfig = Sam31DetectorConfig()
    public var trackerConfig: Sam31TrackerConfig = Sam31TrackerConfig()
    public var initializerRange: Float = 0.02
    public var lowResMaskSize: Int = 288

    // Tracking / association thresholds (HF root-level)
    public var detNmsThresh: Float = 0.1
    public var assocIouThresh: Float = 0.1
    public var trkAssocIouThresh: Float = 0.5
    public var highConfThresh: Float = 0.8
    public var highIouThresh: Float = 0.8
    public var newDetThresh: Float = 0.7
    public var scoreThresholdDetection: Float = 0.5
    public var fillHoleArea: Int = 16
    public var maxNumObjects: Int = 10000

    // Tracker management (HF root-level — orchestration of multi-frame loop)
    public var initTrkKeepAlive: Int = 30
    public var maxTrkKeepAlive: Int = 30
    public var minTrkKeepAlive: Int = -1
    public var hotstartDelay: Int = 15
    public var hotstartDupThresh: Int = 8
    public var hotstartUnmatchThresh: Int = 8
    public var reconditionEveryNthFrame: Int = 16
    public var reconditionOnTrkMasks: Bool = false
    public var decreaseTrkKeepAliveForEmptyMasklets: Bool = false
    public var suppressUnmatchedOnlyWithinHotstart: Bool = true
    public var suppressOverlappingBasedOnRecentOcclusionThreshold: Float = 0.7

    public init() {}
}
