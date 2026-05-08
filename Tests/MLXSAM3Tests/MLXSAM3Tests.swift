import XCTest
import MLX
import MLXFast
import MLXNN
@testable import MLXSAM3
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

final class MLXSAM3Tests: XCTestCase {

    // MARK: - Fixture helpers

    static let fixturesURL: URL = {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures")
    }()

    static let modelURL: URL = URL(fileURLWithPath: NSHomeDirectory())
        .appendingPathComponent(".cache/huggingface/hub/models--mlx-community--sam3-bf16/snapshots/dfe573c3171dbcfda8399c650d9135afa7e94592")

    /// Load full checkpoint, sanitize, then strip `prefix` and return the sub-dict.
    func loadLayerWeights(prefix: String, sanitize: ([String: MLXArray]) -> [String: MLXArray]) throws -> [String: MLXArray] {
        let raw = try ModelLoader.loadWeights(from: Self.modelURL)
        let san = sanitize(raw)
        let stripped = san.reduce(into: [String: MLXArray]()) { dict, pair in
            if pair.key.hasPrefix(prefix) {
                dict[String(pair.key.dropFirst(prefix.count))] = pair.value
            }
        }
        return stripped
    }

    func loadFixture(_ name: String) throws -> [String: MLXArray] {
        let url = Self.fixturesURL.appendingPathComponent("\(name).safetensors")
        return try MLX.loadArrays(url: url)
    }

    /// Assert all elements are close within tolerance (atol + rtol).
    func assertAllClose(
        _ actual: MLXArray,
        _ expected: MLXArray,
        atol: Float = 1e-4,
        rtol: Float = 1e-4,
        _ message: String = "",
        file: StaticString = #file,
        line: UInt = #line
    ) {
        MLX.eval(actual, expected)
        let diff = MLX.abs(actual.asType(.float32) - expected.asType(.float32))
        let tol = atol + rtol * MLX.abs(expected.asType(.float32))
        let ok = MLX.all(diff .<= tol).item(Bool.self)
        if !ok {
            let maxErr = MLX.max(diff).item(Float.self)
            XCTFail("Not close (maxErr=\(maxErr)). \(message)", file: file, line: line)
        }
    }

    func testPackageBuilds() {
        XCTAssertTrue(true)
    }

    // MARK: - Config defaults

    func testConfigDefaults() {
        let cfg = Sam31ModelConfig()
        XCTAssertEqual(cfg.detectorConfig.detrEncoderConfig.hiddenSize, 256)
        XCTAssertEqual(cfg.trackerConfig.multiplexCount, 16)
        XCTAssertEqual(cfg.detectorConfig.visionConfig.scaleFactors.count, 3)
    }

    // MARK: - PositionEmbeddingSine shape

    func testPositionEmbeddingSineShape() {
        let enc = PositionEmbeddingSine(numPosFeats: 128)
        let x = MLXArray.zeros([1, 8, 8, 256])
        let out = enc(x)
        MLX.eval(out)
        XCTAssertEqual(out.shape, [1, 8, 8, 256])
    }

    func testPositionEmbeddingSineFixture() throws {
        let fix = try loadFixture("position_embedding_sine")
        let input = fix["input"]!
        let expected = fix["output"]!

        let enc = PositionEmbeddingSine(numPosFeats: 128)
        let actual = enc(input)
        MLX.eval(actual)

        XCTAssertEqual(actual.shape, expected.shape)
        assertAllClose(actual, expected, atol: 1e-5, rtol: 1e-5, "PositionEmbeddingSine")
    }

    func testViTBackboneFixture() throws {
        let fix = try loadFixture("vit_backbone")
        let input = fix["input"]!

        let weights = try loadLayerWeights(
            prefix: "detector_model.vision_encoder.backbone.",
            sanitize: Sam3Model.sanitize
        )

        let cfg = ViTConfig()
        let backbone = ViTBackbone(cfg)
        try backbone.update(parameters: ModuleParameters.unflattened(weights), verify: [.noUnusedKeys])
        MLX.eval(backbone.parameters())

        // --- Check each intermediate stage ---
        let b = input.dim(0), h = input.dim(1) / cfg.patchSize, w = input.dim(2) / cfg.patchSize

        let afterPatch = backbone.embeddings(input)
        MLX.eval(afterPatch)
        assertAllClose(afterPatch, fix["after_patch_embed"]!, atol: 1e-3, rtol: 1e-3, "after patch embed")

        let pos = backbone.tilePositionEmbedPublic(backbone.embeddings.positionEmbeddings, targetH: h, targetW: w)
        let afterPos = afterPatch + pos
        MLX.eval(afterPos)
        assertAllClose(afterPos, fix["after_pos_embed"]!, atol: 1e-3, rtol: 1e-3, "after pos embed")

        let afterReshape = afterPos.reshaped([b, h, w, -1])
        let afterLN = backbone.layerNorm(afterReshape)
        MLX.eval(afterLN)
        assertAllClose(afterLN, fix["after_layer_norm"]!, atol: 1e-3, rtol: 1e-3, "after layer norm")

        let headDim = cfg.hiddenSize / cfg.numAttentionHeads
        let (winCos, winSin) = computeAxialCis(dim: headDim, endX: cfg.windowSize, endY: cfg.windowSize, theta: cfg.ropeTheta)
        let (globCos, globSin) = computeAxialCis(dim: headDim, endX: h, endY: w, theta: cfg.ropeTheta)

        // --- Layer 0 internals ---
        let lyr0 = backbone.layers[0]
        let attn0 = lyr0.attention

        let l0AfterLN1 = lyr0.layerNorm1(afterLN)
        MLX.eval(l0AfterLN1)
        assertAllClose(l0AfterLN1, fix["l0_after_ln1"]!, atol: 1e-3, rtol: 1e-3, "l0 after ln1")

        // window partition (no-op for 24x24 with ws=24)
        let (windowed, padHW) = windowPartition(l0AfterLN1, windowSize: lyr0.windowSize)
        let b0 = windowed.dim(0), h0 = windowed.dim(1), w0 = windowed.dim(2), c0 = windowed.dim(3)
        let n0 = h0 * w0
        let flat0 = windowed.reshaped([b0, n0, c0])

        let l0q = attn0.qProj(flat0).reshaped([b0, n0, attn0.numHeads, attn0.headDim]).transposed(0, 2, 1, 3)
        let l0k = attn0.kProj(flat0).reshaped([b0, n0, attn0.numHeads, attn0.headDim]).transposed(0, 2, 1, 3)
        let l0v = attn0.vProj(flat0).reshaped([b0, n0, attn0.numHeads, attn0.headDim]).transposed(0, 2, 1, 3)
        MLX.eval(l0q, l0k, l0v)
        assertAllClose(l0q, fix["l0_q"]!, atol: 1e-2, rtol: 1e-3, "l0 q")
        assertAllClose(l0k, fix["l0_k"]!, atol: 1e-2, rtol: 1e-3, "l0 k")
        assertAllClose(l0v, fix["l0_v"]!, atol: 1e-2, rtol: 1e-3, "l0 v")

        MLX.eval(winCos, winSin)
        assertAllClose(winCos, fix["win_cos"]!, atol: 1e-5, rtol: 1e-5, "win_cos")
        assertAllClose(winSin, fix["win_sin"]!, atol: 1e-5, rtol: 1e-5, "win_sin")

        let (l0qRope, l0kRope) = applyRotaryEnc(xq: l0q, xk: l0k, cos: winCos, sin: winSin)
        MLX.eval(l0qRope, l0kRope)
        assertAllClose(l0qRope, fix["l0_q_rope"]!, atol: 1e-2, rtol: 1e-3, "l0 q_rope")
        assertAllClose(l0kRope, fix["l0_k_rope"]!, atol: 1e-2, rtol: 1e-3, "l0 k_rope")

        let l0sdpa = MLXFast.scaledDotProductAttention(queries: l0qRope, keys: l0kRope, values: l0v, scale: attn0.scale, mask: nil)
        MLX.eval(l0sdpa)
        assertAllClose(l0sdpa, fix["l0_sdpa"]!, atol: 1e-2, rtol: 1e-3, "l0 sdpa")

        let l0oproj = attn0.oProj(l0sdpa.transposed(0, 2, 1, 3).reshaped([b0, n0, c0])).reshaped([b0, h0, w0, c0])
        MLX.eval(l0oproj)
        assertAllClose(l0oproj, fix["l0_oproj"]!, atol: 1e-2, rtol: 1e-3, "l0 oproj")

        let l0unpart = windowUnpartition(l0oproj, windowSize: lyr0.windowSize, padHW: padHW, originalHW: (h, w))
        let l0AfterAttnResidual = afterLN + l0unpart
        MLX.eval(l0AfterAttnResidual)
        assertAllClose(l0AfterAttnResidual, fix["l0_after_attn_residual"]!, atol: 1e-2, rtol: 1e-3, "l0 after attn residual")
    }

    func testFPNLayer4xFixture() throws {
        let fix = try loadFixture("fpn_layer_4x")
        let input = fix["input"]!
        let expected = fix["output"]!

        let weights = try loadLayerWeights(
            prefix: "detector_model.vision_encoder.neck.fpn_layers.0.",
            sanitize: Sam3Model.sanitize
        )

        let layer = FPNLayer(inChannels: 1024, outChannels: 256, scaleFactor: 4.0)
        try layer.update(parameters: ModuleParameters.unflattened(weights), verify: [.noUnusedKeys])
        MLX.eval(layer.parameters())

        let actual = layer(input.asType(.float16)).asType(.float32)
        MLX.eval(actual)

        XCTAssertEqual(actual.shape, expected.shape)
        assertAllClose(actual, expected, atol: 5e-3, rtol: 1e-3, "FPNLayer 4x")
    }

    // MARK: - TriViTDetNeck shape

    func testTriViTDetNeckShape() {
        var vCfg = Sam31VisionEncoderConfig()
        vCfg.backboneConfig.hiddenSize = 64
        vCfg.fpnHiddenSize = 64
        vCfg.scaleFactors = [2.0, 1.0]

        let neck = TriViTDetNeck(vCfg)
        let x = MLXArray.zeros([1, 36, 36, 64])
        let (det, inter, prop) = neck(x)
        MLX.eval(det + inter + prop)
        XCTAssertEqual(det.count, 2)
        XCTAssertEqual(det[0].dim(1), 72)   // 2x scale
        XCTAssertEqual(det[1].dim(1), 36)   // 1x scale
        XCTAssertEqual(det[0].dim(3), 64)   // fpnHiddenSize
    }

    // MARK: - MultiplexMaskDecoder shape

    func testMultiplexMaskDecoderShape() {
        var cfg = Sam31TrackerMaskDecoderConfig()
        cfg.hiddenSize = 64
        cfg.multiplexCount = 2
        cfg.numMultimaskOutputs = 3
        cfg.mlpDim = 256
        cfg.numAttentionHeads = 4

        let decoder = MultiplexMaskDecoder(cfg)
        let B = 1, HW = 16, D = 64

        let imageEmb = MLXArray.zeros([B, HW, D])
        let imagePE = MLXArray.zeros([B, HW, D])
        let sparse = MLXArray.zeros([B, 0, D])
        let dense = MLXArray.zeros([B, HW, D])

        let (masks, iou, _, objScore) = decoder(
            imageEmbeddings: imageEmb, imagePE: imagePE,
            sparsePromptEmbeddings: sparse, densePromptEmbeddings: dense,
            multimaskOutput: true
        )
        MLX.eval(masks, iou, objScore)
        // masks: (B, M, N_mask, H_up, W_up)
        XCTAssertEqual(masks.dim(0), B)
        XCTAssertEqual(masks.dim(1), 2)   // M
        XCTAssertEqual(masks.dim(2), 3)   // N_mask
        // iou: (B, M, N_mask)
        XCTAssertEqual(iou.dim(1), 2)
        // objScore: (B, M, 1)
        XCTAssertEqual(objScore.dim(2), 1)
    }

    // MARK: - Sam31ObjectPointerMLP shape

    func testSam31ObjectPointerMLPShape() {
        let mlp = Sam31ObjectPointerMLP(hiddenSize: 64)
        let x = MLXArray.zeros([2, 64])
        let out = mlp(x)
        MLX.eval(out)
        XCTAssertEqual(out.shape, [2, 64])
    }

    // MARK: - DecoupledMemoryAttention shape

    func testDecoupledMemoryAttentionShape() {
        var cfg = Sam31TrackerConfig()
        cfg.memoryAttentionHiddenSize = 64
        cfg.memoryAttentionNumLayers = 1
        cfg.memoryAttentionNumAttentionHeads = 4
        cfg.memoryAttentionFeedForwardHiddenSize = 128
        cfg.memoryAttentionRopeFeatSizes = [8, 8]

        let attn = DecoupledMemoryAttention(cfg)
        let src = MLXArray.zeros([1, 64, 64])
        let mem = MLXArray.zeros([1, 7, 64])
        let out = attn(src, memory: mem)
        MLX.eval(out)
        XCTAssertEqual(out.shape, [1, 64, 64])
    }


// MARK: - TextEncoder Fixture

    func testTextEncoderFixture() throws {
        let fix = try loadFixture("text_encoder")
        let inputIds = fix["input_ids"]!
        let expected = fix["output"]!

        let cfg = TextEncoderConfig()
        let encoder = TextEncoder(cfg)
        let weights = try loadLayerWeights(prefix: "detector_model.text_encoder.", sanitize: Sam3Model.sanitize)
        try encoder.update(parameters: ModuleParameters.unflattened(weights), verify: [.noUnusedKeys])

        let actual = encoder(inputIds.asType(.int32))
        MLX.eval(actual)
        // bf16 accumulated drift across 24 layers — relative tolerance ~1%
        assertAllClose(actual, expected, atol: 5e-1, rtol: 1e-2, "TextEncoder")
    }

    // MARK: - VisionEncoder Fixture

    func testVisionEncoderFixture() throws {
        let fix = try loadFixture("vision_encoder")
        let input = fix["input"]!

        let cfg = VisionEncoderConfig()
        let encoder = VisionEncoder(cfg)
        let weights = try loadLayerWeights(prefix: "detector_model.vision_encoder.", sanitize: Sam3Model.sanitize)
        try encoder.update(parameters: ModuleParameters.unflattened(weights), verify: [.noUnusedKeys])

        let outputs = encoder(input)
        MLX.eval(outputs)
        assertAllClose(outputs[0], fix["output_0"]!, atol: 5e-2, rtol: 1e-2, "VisionEncoder output_0")
        assertAllClose(outputs[1], fix["output_1"]!, atol: 5e-1, rtol: 1e-2, "VisionEncoder output_1")
        assertAllClose(outputs[2], fix["output_2"]!, atol: 5e-1, rtol: 1e-2, "VisionEncoder output_2")
        assertAllClose(outputs[3], fix["output_3"]!, atol: 5e-2, rtol: 1e-2, "VisionEncoder output_3")
    }

    // MARK: - DETREncoder Fixture
    
    func testDETREncoderFixture() throws {
        let fix = try loadFixture("detr_encoder")
        let src = fix["src"]!
        let pos = fix["pos"]!
        let textHidden = fix["text_hidden"]!
        let textPadMask = fix["text_pad_mask"]!
        let expected = fix["output"]!

        let cfg = DETREncoderConfig()
        let encoder = DETREncoder(cfg)
        let weights = try loadLayerWeights(prefix: "detector_model.detr_encoder.", sanitize: Sam3Model.sanitize)
        try encoder.update(parameters: ModuleParameters.unflattened(weights), verify: [.noUnusedKeys])

        let actual = encoder(src, pos: pos, textMemory: textHidden, textMask: textPadMask)
        MLX.eval(actual)
        assertAllClose(actual, expected, atol: 1e-2, rtol: 1e-2, "DETREncoder")
    }

    // MARK: - DETRDecoder Fixture
    
    func testDETRDecoderFixture() throws {
        let fix = try loadFixture("detr_decoder")
        let memory = fix["memory"]!
        let queryEmbed = fix["query_embed"]!
        let pos = fix["pos"]!
        
        let cfg = DETRDecoderConfig()
        let decoder = DETRDecoder(cfg)
        let weights = try loadLayerWeights(prefix: "detector_model.detr_decoder.", sanitize: Sam3Model.sanitize)
        try decoder.update(parameters: ModuleParameters.unflattened(weights), verify: [.noUnusedKeys])

        let (outHs, outBoxes, outPresence) = decoder(visionFeatures: memory, inputsEmbeds: queryEmbed, visionPosEncoding: pos)
        MLX.eval(outHs, outBoxes, outPresence)
        assertAllClose(outHs, fix["out_hs"]!, atol: 5e-2, rtol: 1e-2, "DETRDecoder out_hs")
        assertAllClose(outBoxes, fix["out_boxes"]!, atol: 5e-2, rtol: 1e-2, "DETRDecoder out_boxes")
        assertAllClose(outPresence, fix["out_presence"]!, atol: 5e-2, rtol: 1e-2, "DETRDecoder out_presence")
    }

    // MARK: - MaskDecoder Fixture
    
    func testMaskDecoderFixture() throws {
        let fix = try loadFixture("mask_decoder")

        let cfg = DetectorMaskDecoderConfig()
        let decoder = MaskDecoder(cfg)
        let weights = try loadLayerWeights(prefix: "detector_model.mask_decoder.", sanitize: Sam3Model.sanitize)
        try decoder.update(parameters: ModuleParameters.unflattened(weights), verify: [.noUnusedKeys])

        let (predMasks, semanticSeg) = decoder(
            objQueries: fix["obj_queries"]!,
            backboneFeatures: [fix["backbone_feat_0"]!, fix["backbone_feat_1"]!, fix["backbone_feat_2"]!],
            encoderHiddenStates: fix["encoder_hidden_states"]!,
            promptFeatures: fix["prompt_features"]!,
            promptMask: fix["prompt_mask"]!
        )
        MLX.eval(predMasks, semanticSeg)
        assertAllClose(predMasks, fix["pred_masks"]!, atol: 1e-1, rtol: 1e-2, "MaskDecoder pred_masks")
        assertAllClose(semanticSeg, fix["semantic_seg"]!, atol: 1e-1, rtol: 1e-2, "MaskDecoder semantic_seg")
    }

    // MARK: - MemoryEncoder Fixture

    func testMemoryEncoderFixture() throws {
        let fix = try loadFixture("memory_encoder")
        let features = fix["features"]!
        let mask = fix["mask"]!
        let expected = fix["output"]!

        let cfg = TrackerConfig()
        let encoder = MemoryEncoder(cfg)

        let weights = try loadLayerWeights(prefix: "tracker_model.memory_encoder.", sanitize: Sam3Model.sanitize)
        try encoder.update(parameters: ModuleParameters.unflattened(weights), verify: [.noUnusedKeys])

        let actual = encoder(features, mask)
        MLX.eval(actual)
        assertAllClose(actual, expected, atol: 1e-3, rtol: 1e-3, "MemoryEncoder")
    }

    // MARK: - DecoupledMemoryAttention Fixture

    func testDecoupledMemoryAttentionFixture() throws {
        let fix = try loadFixture("decoupled_memory_attention")
        let src = fix["src"]!
        let memory = fix["memory"]!
        let expected = fix["output"]!

        // Fixture was generated with a small custom config (hidden=64, 1 layer) and random weights.
        // Model weights are stored inside the fixture under all keys except src/memory/output.
        var cfg = Sam31TrackerConfig()
        cfg.memoryAttentionHiddenSize = 64
        cfg.memoryAttentionNumLayers = 1
        cfg.memoryAttentionNumAttentionHeads = 4
        cfg.memoryAttentionFeedForwardHiddenSize = 128
        cfg.memoryAttentionRopeFeatSizes = [8, 8]
        let attn = DecoupledMemoryAttention(cfg)

        let ioKeys: Set<String> = ["src", "memory", "output"]
        let weights = fix.filter { !ioKeys.contains($0.key) }
        try attn.update(parameters: ModuleParameters.unflattened(weights), verify: [.noUnusedKeys])

        let actual = attn(src, memory: memory)
        MLX.eval(actual)
        assertAllClose(actual, expected, atol: 1e-3, rtol: 1e-3, "DecoupledMemoryAttention")
    }

    // MARK: - MultiplexMaskDecoder Fixture

    func testMultiplexMaskDecoderFixture() throws {
        let fix = try loadFixture("multiplex_mask_decoder")

        // Fixture was generated with a small custom config (hidden=64, M=2) and random weights.
        // Model weights are stored inside the fixture under all keys except the I/O tensors.
        var cfg = Sam31TrackerMaskDecoderConfig()
        cfg.hiddenSize = fix["image_embeddings"]!.dim(-1)
        cfg.multiplexCount = fix["masks"]!.dim(1)
        cfg.numMultimaskOutputs = 3
        let decoder = MultiplexMaskDecoder(cfg)

        let ioKeys: Set<String> = ["image_embeddings", "image_pe", "sparse_prompt", "dense_prompt", "masks", "iou", "objScore"]
        let weights = fix.filter { !ioKeys.contains($0.key) }
        try decoder.update(parameters: ModuleParameters.unflattened(weights), verify: [.noUnusedKeys])

        let (masks, iou, _, objScore) = decoder(
            imageEmbeddings: fix["image_embeddings"]!,
            imagePE: fix["image_pe"]!,
            sparsePromptEmbeddings: fix["sparse_prompt"]!,
            densePromptEmbeddings: fix["dense_prompt"]!,
            multimaskOutput: true
        )

        MLX.eval(masks, iou, objScore)
        assertAllClose(masks, fix["masks"]!, atol: 1e-3, rtol: 1e-3, "MultiplexMaskDecoder masks")
        assertAllClose(iou, fix["iou"]!, atol: 1e-3, rtol: 1e-3, "MultiplexMaskDecoder iou")
        assertAllClose(objScore, fix["objScore"]!, atol: 1e-3, rtol: 1e-3, "MultiplexMaskDecoder objScore")
    }

    // MARK: - Phase 1: torch-exported fixtures

    func testPositionEncodingSineTorchFixture() throws {
        // Torch fixture is NCHW: input (B, C, H, W), output (B, 2*D, H, W).
        // Swift impl is NHWC, so we transpose both ends.
        let fix = try loadFixture("position_encoding_sine")
        let xNCHW = fix["input/x"]!
        let expectedNCHW = fix["output/pos"]!

        let xNHWC = xNCHW.transposed(0, 2, 3, 1)
        // Python `num_pos_feats=256` means total output channels=256 (it stores num_pos_feats//2 internally).
        // Swift `numPosFeats` is per-direction and outputs 2*numPosFeats, so pass 128.
        let enc = PositionEmbeddingSine(numPosFeats: 128, normalize: true)
        let outNHWC = enc(xNHWC)
        let actualNCHW = outNHWC.transposed(0, 3, 1, 2)

        MLX.eval(actualNCHW)
        XCTAssertEqual(actualNCHW.shape, expectedNCHW.shape)
        assertAllClose(actualNCHW, expectedNCHW, atol: 1e-5, rtol: 1e-5, "PositionEmbeddingSine (torch)")
    }

    func testBoxOpsFixture() throws {
        let fix = try loadFixture("box_ops")

        let xyxy = BoxOps.boxCxcywhToXyxy(fix["input/cxcywh"]!)
        assertAllClose(xyxy, fix["output/xyxy"]!, atol: 1e-5, rtol: 1e-5, "boxCxcywhToXyxy")

        let area = BoxOps.boxArea(fix["output/xyxy"]!)
        assertAllClose(area, fix["output/area"]!, atol: 1e-5, rtol: 1e-5, "boxArea")

        let (iou, union) = BoxOps.boxIoU(fix["input/boxes1_xyxy"]!, fix["input/boxes2_xyxy"]!)
        assertAllClose(iou, fix["output/iou"]!, atol: 1e-5, rtol: 1e-5, "boxIoU.iou")
        assertAllClose(union, fix["output/union"]!, atol: 1e-5, rtol: 1e-5, "boxIoU.union")

        let giou = BoxOps.generalizedBoxIoU(fix["input/boxes1_xyxy"]!, fix["input/boxes2_xyxy"]!)
        assertAllClose(giou, fix["output/giou"]!, atol: 1e-5, rtol: 1e-5, "generalizedBoxIoU")
    }

    func testGeometryHelpersFixture() throws {
        let fix = try loadFixture("geometry_helpers")

        // encode_xy parity. Python num_pos_feats=256 → internal 128; Swift per-direction = 128.
        let enc = PositionEmbeddingSine(numPosFeats: 128, normalize: true)
        let (posX, posY) = enc.encodeXY(x: fix["input/x"]!, y: fix["input/y"]!)
        assertAllClose(posX, fix["output/pos_x"]!, atol: 1e-5, rtol: 1e-5, "encodeXY.pos_x")
        assertAllClose(posY, fix["output/pos_y"]!, atol: 1e-5, rtol: 1e-5, "encodeXY.pos_y")

        let boxPos = enc.encodeBoxes(
            x: fix["input/box_cx"]!, y: fix["input/box_cy"]!,
            w: fix["input/box_w"]!, h: fix["input/box_h"]!
        )
        assertAllClose(boxPos, fix["output/box_pos"]!, atol: 1e-5, rtol: 1e-5, "encodeBoxes")

        // concat_padded_sequences: masks are uint8 in the fixture; cast to bool for Swift.
        let mask1 = fix["input/mask1"]!.asType(.bool)
        let mask2 = fix["input/mask2"]!.asType(.bool)
        let expMask = fix["output/concat_mask"]!.asType(.bool)
        let (seq, mask) = concatPaddedSequences(
            seq1: fix["input/seq1"]!, mask1: mask1,
            seq2: fix["input/seq2"]!, mask2: mask2
        )
        assertAllClose(seq, fix["output/concat_seq"]!, atol: 1e-5, rtol: 1e-5, "concatPaddedSequences.seq")
        MLX.eval(mask)
        XCTAssertEqual(mask.shape, expMask.shape)
        XCTAssertTrue(MLX.all(MLX.equal(mask, expMask)).item(Bool.self))
    }

    // MARK: - SAM 3.1 image end-to-end (Phase 7 parity)

    static let sam31ModelURL: URL = URL(fileURLWithPath: NSHomeDirectory())
        .appendingPathComponent(".cache/huggingface/hub/models--mlx-community--sam3.1-bf16/snapshots/a992e302ea9b0f03f41dfd93414a4fd0e818f65b")

    /// Report max abs error and a relative-error summary; doesn't fail the test.
    private func reportDiff(_ name: String, _ actual: MLXArray, _ expected: MLXArray) {
        MLX.eval(actual, expected)
        let a = actual.asType(.float32)
        let e = expected.asType(.float32)
        let diff = MLX.abs(a - e)
        let maxAbs = MLX.max(diff).item(Float.self)
        let meanAbs = MLX.mean(diff).item(Float.self)
        let expMax = MLX.max(MLX.abs(e)).item(Float.self)
        print("  [diag] \(name): shape=\(actual.shape) maxErr=\(maxAbs) meanErr=\(meanAbs) expMaxAbs=\(expMax)")
    }

    /// Assert mean(|actual - expected|) / max(|expected|) < threshold.
    /// Standard parity criterion for cascaded networks where individual outlier
    /// elements drift on saturating activations but the bulk match.
    private func assertMeanRelClose(
        _ name: String,
        _ actual: MLXArray,
        _ expected: MLXArray,
        threshold: Float,
        file: StaticString = #file,
        line: UInt = #line
    ) {
        MLX.eval(actual, expected)
        let a = actual.asType(.float32)
        let e = expected.asType(.float32)
        let meanAbs = MLX.mean(MLX.abs(a - e)).item(Float.self)
        let expMax = MLX.max(MLX.abs(e)).item(Float.self)
        let rel = expMax > 0 ? meanAbs / expMax : meanAbs
        if rel >= threshold {
            XCTFail("\(name): meanRelErr=\(rel) ≥ \(threshold)", file: file, line: line)
        }
    }

    func testSam31ImageE2EFixture() throws {
        let fix = try loadFixture("sam3_1_image_e2e")
        let pixelValues = fix["input/pixel_values"]!.asType(.float32)
        let inputIds = fix["input/input_ids"]!.asType(.int32)
        let attentionMask = fix["input/attention_mask"]!.asType(.int32)

        let model = Sam31Model(Sam31ModelConfig())
        try ModelLoader.load(sam31: model, from: Self.sam31ModelURL, dtype: .float32)

        let out = model.detect(
            pixelValues: pixelValues,
            inputIds: inputIds,
            attentionMask: attentionMask
        )

        // Diagnostic — print max errs at each stage so divergence can be localized.
        if let enc = out["encoder_hidden_states"], let exp = fix["output/encoder_hidden_states"] {
            reportDiff("encoder_hidden_states", enc, exp)
        }
        if let hs = out["intermediate_hidden_states"], let exp = fix["output/intermediate_hidden_states"] {
            reportDiff("intermediate_hidden_states", hs, exp)
        }
        if let v = out["pred_logits"]    { reportDiff("pred_logits",     v, fix["output/pred_logits"]!) }
        if let v = out["pred_boxes"]     { reportDiff("pred_boxes",      v, fix["output/pred_boxes"]!) }
        if let v = out["pred_masks"]     { reportDiff("pred_masks",      v, fix["output/pred_masks"]!) }
        if let v = out["presence_logits"]{ reportDiff("presence_logits", v, fix["output/presence_logits"]!) }

        let expectedLogits = fix["output/pred_logits"]!
        let expectedBoxes = fix["output/pred_boxes"]!
        let expectedMasks = fix["output/pred_masks"]!
        let expectedPresence = fix["output/presence_logits"]!

        guard let predLogits = out["pred_logits"],
              let predBoxes = out["pred_boxes"],
              let predMasks = out["pred_masks"],
              let presence = out["presence_logits"]
        else {
            XCTFail("missing output keys: \(Array(out.keys))")
            return
        }

        XCTAssertEqual(predLogits.shape, expectedLogits.shape)
        XCTAssertEqual(predBoxes.shape, expectedBoxes.shape)
        XCTAssertEqual(predMasks.shape, expectedMasks.shape)
        XCTAssertEqual(presence.shape, expectedPresence.shape)

        // fp32 reference: mean-relative-error bar (mean|err| / max|expected|).
        // 1% is the standard parity bar — fp32 accumulation order can drift
        // individual outliers but the bulk should match within fp32 precision.
        assertMeanRelClose("pred_logits",     predLogits, expectedLogits, threshold: 0.01)
        assertMeanRelClose("pred_boxes",      predBoxes, expectedBoxes,   threshold: 0.01)
        assertMeanRelClose("pred_masks",      predMasks, expectedMasks,   threshold: 0.01)
        assertMeanRelClose("presence_logits", presence, expectedPresence, threshold: 0.01)
    }

    // MARK: - SAM 3.1 video tracker components (Phase 11.5b parity)

    /// Phase 11.5b parity for the new tracker code paths beyond image-detect:
    ///   A) memory_encoder(features, 32-ch mask) → memory  (multiplex mask path)
    ///   B) memory_attention(src, memory)                 (memory-bank cross-attn)
    ///
    /// Skipped: track_init (upstream mlx-vlm bug, 1-ch vs 32-ch mismatch) and
    /// track_step (prompt encoder dense_emb is pinned to 72×72 production grid).
    /// Mask decoder integration is covered by testMultiplexMaskDecoderFixture.
    func testSam31VideoTrackFixture() throws {
        let fix = try loadFixture("sam3_1_video_track")
        let features0 = fix["input/features_0"]!.asType(.float32)
        let maskIn = fix["input/mask_in"]!.asType(.float32)
        let src = fix["input/src"]!.asType(.float32)

        let model = Sam31Model(Sam31ModelConfig())
        try ModelLoader.load(sam31: model, from: Self.sam31ModelURL, dtype: .float32)

        // A) memory_encoder
        let memFeats = model.trackerModel.memoryEncoder(features0, maskIn)
        let b = memFeats.dim(0), d = memFeats.dim(3)
        let memory0 = memFeats.reshaped([b, -1, d])

        let expectedMemFeats = fix["output/memory_feats"]!
        let expectedMem0 = fix["output/memory_0"]!
        XCTAssertEqual(memFeats.shape, expectedMemFeats.shape)
        XCTAssertEqual(memory0.shape, expectedMem0.shape)
        reportDiff("memory_feats", memFeats, expectedMemFeats)
        reportDiff("memory_0",     memory0,  expectedMem0)
        assertMeanRelClose("memory_feats", memFeats, expectedMemFeats, threshold: 0.01)
        assertMeanRelClose("memory_0",     memory0,  expectedMem0,     threshold: 0.01)

        // B) memory_attention — manual layer-0 stepping for sub-stage diagnostics.
        let ma = model.trackerModel.memoryAttention
        let layer0 = ma.layers[0]

        // 1. Self-attn — granular dumps
        var residual = src
        let normed1 = layer0.norm1(src)
        let q1 = layer0.selfAttnQProj(normed1)
        let k1 = layer0.selfAttnKProj(normed1)
        let v1 = layer0.selfAttnVProj(normed1)
        if let exp = fix["output/l0_sa_q"] { reportDiff("l0_sa_q", q1, exp) }
        if let exp = fix["output/l0_sa_k"] { reportDiff("l0_sa_k", k1, exp) }
        if let exp = fix["output/l0_sa_v"] { reportDiff("l0_sa_v", v1, exp) }
        if let exp = fix["output/l0_sa_freqs_cos"] {
            reportDiff("l0_sa_freqs_cos", layer0.selfAttentionRope.freqsCos, exp)
        }
        if let exp = fix["output/l0_sa_freqs_sin"] {
            reportDiff("l0_sa_freqs_sin", layer0.selfAttentionRope.freqsSin, exp)
        }
        let saPre = layer0.selfAttentionRope(q1, k1, v1)
        if let exp = fix["output/l0_sa_pre"] { reportDiff("l0_sa_pre", saPre, exp) }
        let afterSA = residual + layer0.selfAttnOutProj(saPre)
        if let exp = fix["output/l0_after_sa"] { reportDiff("l0_after_sa", afterSA, exp) }

        // 2. Cross-attn
        residual = afterSA
        let normed2 = layer0.norm2(afterSA)
        let q2 = layer0.crossAttnQProj(normed2) + layer0.imageCrossAttnQProj(normed2)
        let k2 = layer0.crossAttnKProj(memory0) + layer0.imageCrossAttnKProj(memory0)
        let v2 = layer0.crossAttnVProj(memory0)
        let caPre = layer0.crossAttentionRope(q2, k2, v2)
        let afterCA = residual + layer0.crossAttnOutProj(caPre)
        if let exp = fix["output/l0_after_ca"] { reportDiff("l0_after_ca", afterCA, exp) }

        // 3. FFN
        let afterFFN = afterCA + layer0.linear2(relu(layer0.linear1(layer0.norm3(afterCA))))
        if let exp = fix["output/ma_layer_0"] { reportDiff("ma_layer_0", afterFFN, exp) }

        // Continue with remaining layers
        var cur = afterFFN
        for i in 1..<ma.layers.count {
            cur = ma.layers[i](cur, memory: memory0)
            if let exp = fix["output/ma_layer_\(i)"] { reportDiff("ma_layer_\(i)", cur, exp) }
        }
        let attended = ma.layerNorm(cur)
        let expectedAttended = fix["output/attended"]!
        XCTAssertEqual(attended.shape, expectedAttended.shape)
        reportDiff("attended", attended, expectedAttended)
        assertMeanRelClose("attended", attended, expectedAttended, threshold: 0.01)
    }

    /// Phase 11.5b parity: full `MultiplexTrackerModel.trackStep` at production
    /// resolution (1008² → features 72×72), where the prompt encoder's
    /// dense_emb spatial size matches the features so the mask decoder integrates
    /// cleanly. Synthetic features/memory (vision backbone is covered elsewhere).
    func testSam31TrackStepFixture() throws {
        let fix = try loadFixture("sam3_1_track_step")
        let features = fix["input/features"]!.asType(.float32)
        let highRes0 = fix["input/high_res_0"]!.asType(.float32)
        let highRes1 = fix["input/high_res_1"]!.asType(.float32)
        let memory0 = fix["input/memory_0"]!.asType(.float32)

        let model = Sam31Model(Sam31ModelConfig())
        try ModelLoader.load(sam31: model, from: Self.sam31ModelURL, dtype: .float32)

        let out = model.trackerModel.trackStep(
            currentFeatures: features,
            memoryBank: [memory0],
            promptPoints: nil,
            promptBoxes: nil,
            promptMasks: nil,
            multimaskOutput: false,
            highResFeatures: [highRes0, highRes1]
        )

        let expectedMasks = fix["output/pred_masks"]!
        let expectedIou = fix["output/iou_scores"]!
        let expectedObj = fix["output/obj_scores"]!
        let expectedPtr = fix["output/object_pointer"]!

        guard let predMasks = out["pred_masks"],
              let iou = out["iou_scores"],
              let obj = out["obj_scores"],
              let ptr = out["object_pointer"]
        else {
            XCTFail("missing trackStep output keys: \(Array(out.keys))")
            return
        }

        XCTAssertEqual(predMasks.shape, expectedMasks.shape)
        XCTAssertEqual(iou.shape, expectedIou.shape)
        XCTAssertEqual(obj.shape, expectedObj.shape)
        XCTAssertEqual(ptr.shape, expectedPtr.shape)

        reportDiff("pred_masks",     predMasks, expectedMasks)
        reportDiff("iou_scores",     iou, expectedIou)
        reportDiff("obj_scores",     obj, expectedObj)
        reportDiff("object_pointer", ptr, expectedPtr)

        assertMeanRelClose("pred_masks",     predMasks, expectedMasks, threshold: 0.01)
        assertMeanRelClose("iou_scores",     iou, expectedIou, threshold: 0.01)
        assertMeanRelClose("obj_scores",     obj, expectedObj, threshold: 0.01)
        assertMeanRelClose("object_pointer", ptr, expectedPtr, threshold: 0.01)
    }

    /// Bisect the memory attention layer against python's captured I/O.
    /// Feeds python's exact src + memory + memory_image + memory_image_pos through Swift's
    /// DecoupledMemoryAttention. Localizes whether the layer itself matches python.
    func testMemoryAttentionRealInputParity() throws {
        let fix = try loadFixture("sam3_1_memattn_capture")
        let model = Sam31Model(Sam31ModelConfig())
        try ModelLoader.load(sam31: model, from: Self.sam31ModelURL, dtype: .float32)

        for callIdx in [0, 1] {
            // Python tensors stored as (HW, B, D); Swift wants (B, N, D).
            func bnd(_ key: String) -> MLXArray? {
                fix["enc_\(callIdx)/\(key)"]?.transposed(1, 0, 2).asType(.float32)
            }
            guard let src = bnd("in_src"),
                  let memory = bnd("in_memory"),
                  let memoryImage = bnd("in_memory_image"),
                  let memoryImagePos = bnd("in_memory_image_pos"),
                  let outMem = bnd("out_memory")
            else { XCTFail("missing capture \(callIdx)"); continue }

            let nObjPtr = Int(fix["enc_\(callIdx)/num_obj_ptr_tokens"]!.asArray(Float.self)[0])
            // Pad memory_image_pos to memory length using memory_pos[obj_ptr_tail] —
            // matches python decoder.py:1340-1346 (NOT zeros).
            let nMem = memory.dim(1), nMip = memoryImagePos.dim(1)
            let memPos: MLXArray
            if nMip < nMem, let memoryPos = bnd("in_memory_pos"), nObjPtr > 0 {
                let tail = memoryPos[0..<1, (nMem - nObjPtr)..., 0...]
                memPos = concatenated([memoryImagePos, tail], axis: 1)
            } else {
                memPos = memoryImagePos
            }

            let srcPos = bnd("in_src_pos")
            let actual = model.trackerModel.memoryAttention(
                src,
                memory: memory,
                memoryPos: memPos,
                image: src,
                srcPos: srcPos,
                memoryImage: memoryImage,
                memoryImagePos: nil,
                numKExcludeRope: nObjPtr
            )
            MLX.eval(actual)
            reportDiff("call \(callIdx) memory_attention output", actual, outMem)
        }
    }

    /// Per-layer bisect of memory_attention. Feeds python's exact layer-0 inputs
    /// through Swift's layer 0 and compares output vs python's captured layer-0 tgt_out.
    /// Then chains: layer N input = python's layer (N-1) tgt_out, see if Swift's layer N
    /// can reproduce python's layer N output.
    func testMemoryAttentionPerLayerParity() throws {
        let fix = try loadFixture("sam3_1_memattn_perlayer_capture")
        let model = Sam31Model(Sam31ModelConfig())
        try ModelLoader.load(sam31: model, from: Self.sam31ModelURL, dtype: .float32)
        let memAttn = model.trackerModel.memoryAttention

        let callIdx = 0
        // Layer captures are taken AFTER python's batch_first transpose, so they're
        // already (B, N, D). Use directly without transposing.
        func bnd(_ key: String) -> MLXArray? {
            fix["enc_\(callIdx)/\(key)"]?.asType(.float32)
        }

        guard let imageIn = bnd("layer_0_in_image"),
              let tgtIn = bnd("layer_0_in_tgt"),
              let memory = bnd("layer_0_in_memory"),
              let memoryImage = bnd("layer_0_in_memory_image"),
              let memoryImagePos = bnd("layer_0_in_memory_image_pos")
        else { XCTFail("missing per-layer inputs"); return }
        let nObjPtr = Int(fix["enc_\(callIdx)/layer_0_in_num_k_exclude_rope"]!.asArray(Float.self)[0])

        // Python pre-pads memory_image inside encoder.forward (BEFORE layers see it).
        // Swift layer expects memory_image to ALREADY be padded to memory length, so
        // do the padding here (mirrors python encoder.forward lines 1316-1346).
        let nMem = memory.dim(1), nImg = memoryImage.dim(1)
        let paddedMemoryImage: MLXArray
        let paddedMemoryImagePos: MLXArray
        if nImg < nMem, nObjPtr > 0 {
            let zeros = MLXArray.zeros([memoryImage.dim(0), nMem - nImg, memoryImage.dim(2)],
                                       dtype: memoryImage.dtype)
            paddedMemoryImage = concatenated([memoryImage, zeros], axis: 1)
            // memory_image_pos padded with memory_pos[obj_ptr_tail]
            if let memoryPos = bnd("layer_0_in_memory_pos") {
                let tail = memoryPos[0..<1, (nMem - nObjPtr)..., 0...]
                paddedMemoryImagePos = concatenated([memoryImagePos, tail], axis: 1)
            } else {
                let zeros2 = MLXArray.zeros([1, nMem - nImg, memoryImagePos.dim(2)],
                                            dtype: memoryImagePos.dtype)
                paddedMemoryImagePos = concatenated([memoryImagePos, zeros2], axis: 1)
            }
        } else {
            paddedMemoryImage = memoryImage
            paddedMemoryImagePos = memoryImagePos
        }

        // Run each Swift layer with python's input for that layer; compare against
        // python's captured tgt_out for that layer. Each layer is independent — measures
        // *that layer's* error, not cumulative.
        for li in 0..<memAttn.layers.count {
            let inKey = li == 0 ? "layer_0_in_tgt" : "layer_\(li - 1)_tgt_out"
            guard let layerIn = bnd(inKey),
                  let layerOutExpected = bnd("layer_\(li)_tgt_out") else {
                XCTFail("missing layer \(li) capture"); continue
            }
            let actual = memAttn.layers[li](
                layerIn,
                memory: memory,
                memoryPos: paddedMemoryImagePos,
                image: imageIn,
                memoryImage: paddedMemoryImage,
                memoryImagePos: nil,
                numKExcludeRope: nObjPtr
            )
            MLX.eval(actual)
            reportDiff("layer \(li) tgt_out (py-input)", actual, layerOutExpected)
        }
        // Skip cross-checking against tgtIn at this point — already used as layer-0 input.
        _ = tgtIn
    }

    // MARK: - Sam31VideoTracker (slim) parity vs python/sam3 reference

    /// Phase 11.5c parity diagnostic — does NOT enforce a strict pass.
    /// Runs the slim Sam31VideoTracker (no memory bank, box-prompt-only propagation)
    /// on the same 3-frame clip python/sam3 was run on, and reports per-frame
    /// best-match mask IoU vs the captured reference. Quantifies the gap between
    /// the slim baseline and python/sam3's full multiplex tracker; informs how
    /// much memory-bank wiring (Phase 11.5c proper) actually buys us.
    func testSam31VideoTrackerSlimVsPySam3() async throws {
        try await runVideoTrackerParity(numFrames: 3, fixtureName: "sam3_1_video_track_3frame_ref",
                                        framesDir: "sam3_1_video_track_3frame",
                                        meanIouFloor: 0.50)
    }

    /// Shape-level smoke test — does NOT measure parity. Asserts:
    ///   - Tracker spawns ≥1 track on frame 0 (real prompt, real frames).
    ///   - Track IDs are stable across propagation frames (no spurious re-IDing).
    ///   - Per-track memory bank length grows monotonically and stays under cap.
    /// Catches regressions where tracker state machine breaks (e.g., banks reset every
    /// frame, IDs churn, no live tracks survive frame 1).
    func testSam31VideoTrackerSmoke() async throws {
        let framesDir = URL(fileURLWithPath: "/tmp/sam3_demo_frames_10")
        let frameCount = 5
        let fm = FileManager.default
        var cgFrames: [CGImage] = []
        for i in 0..<frameCount {
            let url = framesDir.appendingPathComponent("\(i).jpg")
            guard fm.fileExists(atPath: url.path) else {
                throw XCTSkip("smoke test needs /tmp/sam3_demo_frames_10/N.jpg frames")
            }
            guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
                  let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
                throw XCTSkip("failed to decode frame \(i)")
            }
            cgFrames.append(cg)
        }

        let model = Sam31Model(Sam31ModelConfig())
        try ModelLoader.load(sam31: model, from: Self.sam31ModelURL, dtype: .bfloat16)
        let tok = try await SAM3Tokenizer.fromSnapshotOrOpenAIClip(Self.sam31ModelURL, maxLength: 32)
        let encoded = tok.encode("person")

        let tracker = Sam31VideoTracker(
            model: model, resolution: 1008, dtype: .bfloat16,
            detectThreshold: 0.5, redetectEvery: 15
        )

        var seenIds: Set<Int> = []
        var lastBankLen: [Int: Int] = [:]   // id -> last memoryBank.count
        for (i, cg) in cgFrames.enumerated() {
            let pix = try ImagePreprocess.pixelValues(from: cg, size: 1008, dtype: .bfloat16)
            let live = tracker.step(
                pixelValues: pix,
                inputIds: encoded.inputIds,
                attentionMask: encoded.attentionMask,
                originalHeight: cg.height, originalWidth: cg.width
            )
            if i == 0 {
                XCTAssertGreaterThan(live.count, 0, "frame 0 detect produced no tracks")
                seenIds = Set(live.map(\.id))
            } else {
                // Propagation frames: every live track must have an ID seen on frame 0
                // (or a spawned-this-frame ID, but our smoke clip is short and stable
                // enough that we shouldn't see new IDs).
                for t in live {
                    XCTAssertTrue(seenIds.contains(t.id),
                                  "frame \(i) introduced unexpected new track id \(t.id)")
                    let prev = lastBankLen[t.id] ?? 0
                    XCTAssertGreaterThanOrEqual(t.memoryBank.count, prev,
                        "frame \(i) track \(t.id) memoryBank shrank \(prev) → \(t.memoryBank.count)")
                    XCTAssertLessThanOrEqual(t.memoryBank.count, tracker.maxMemoryBank,
                        "frame \(i) track \(t.id) memoryBank exceeded cap")
                }
            }
            for t in live { lastBankLen[t.id] = t.memoryBank.count }
            print("  [smoke] frame \(i): live=\(live.count) ids=\(live.map(\.id).sorted()) " +
                  "banks=\(live.map { $0.memoryBank.count }.sorted())")
        }
    }

    /// Same parity test on a 10-frame clip — exercises the architectural improvements
    /// (memory bank growth, temporal PE, obj_ptr stream, bucket sharing) that 3 static
    /// frames don't stress.
    func testSam31VideoTrackerSlimVsPySam3_10frame() async throws {
        // 2026-05-08: high_res features now properly added (was silently skipped). Frame-1
        // mean IoU jumped 0.74 → 0.78 but cumulative drift across 10 frames worsened
        // (each frame's sharper masks compound differently in memory bank). 3-frame
        // mean IoU is now 0.76 (was 0.58). 10-frame trades quality at frame 1 for drift
        // by frame 9. Architecture matches python; drift is a separate compounding issue.
        //
        // 2026-05-08 (later): LCC filter on tracker output trimmed mean from 0.297 to
        // ~0.258 — expected. python/sam3 reference doesn't apply LCC, so trimming
        // multi-blob masks makes Swift slightly smaller. Floor lowered to 0.25.
        // The visible behavior is more correct (single track no longer wraps two
        // people when the mask leaks); reference parity slips marginally.
        try await runVideoTrackerParity(numFrames: 10, fixtureName: "sam3_1_video_track_10frame_ref",
                                        framesDir: "sam3_1_video_track_10frame",
                                        meanIouFloor: 0.25)
    }

    /// Phase 11.5c diagnostic: feed python/sam3's exact `pix_feat` and `mux_mask_for_mem`
    /// (captured from a real 10-frame run) into Swift's `memoryEncoder` and compare
    /// the output `vision_features` byte-for-byte. This isolates whether the encoder
    /// itself diverges, or whether the bug is upstream (mask construction) / downstream
    /// (memory_attention) of it.
    /// Phase 11.5c diagnostic: bisect Swift's MultiplexMemoryEncoder against python's
    /// SimpleMaskEncoder step-by-step using captured intermediate state.
    func testMemoryEncoderStepByStepParity() throws {
        let fix = try loadFixture("sam3_1_encoding_capture")
        let model = Sam31Model(Sam31ModelConfig())
        try ModelLoader.load(sam31: model, from: Self.sam31ModelURL, dtype: .float32)
        let me = model.trackerModel.memoryEncoder
        for callIdx in [0, 2] {
            print("=== call_\(callIdx) ===")
            // Helper: load python NCHW tensor → Swift NHWC
            func nhwc(_ key: String) -> MLXArray {
                fix["call_\(callIdx)/\(key)"]!.transposed(0, 2, 3, 1).asType(.float32)
            }
            let pixFeat = nhwc("pix_feat")
            let muxMask = nhwc("mux_mask_for_mem")
            let pyPixProj = nhwc("pix_feat_proj_out")
            let pyMaskDownOut = nhwc("mask_downsampler_out")
            let pyAfterAdd = nhwc("after_add")
            let pyFuserOut = nhwc("fuser_out")
            let pyVisionFeats = nhwc("vision_features")

            // Step 1: feature_projection (1x1 conv)
            let swiftPixProj = me.featureProjection(pixFeat)
            reportDiff("step1 pix_feat_proj", swiftPixProj, pyPixProj)

            // Step 2: mask_downsampler — needs upsampled mask if < 1152
            var maskInput = muxMask
            if maskInput.dim(1) != 1152 || maskInput.dim(2) != 1152 {
                let scaleH = Float(1152) / Float(maskInput.dim(1))
                let scaleW = Float(1152) / Float(maskInput.dim(2))
                maskInput = Upsample(scaleFactor: .array([scaleH, scaleW]),
                                     mode: .linear(alignCorners: false))(maskInput)
            }
            let swiftMaskDown = me.maskDownsampler(maskInput)
            reportDiff("step2 mask_downsampler", swiftMaskDown, pyMaskDownOut)

            // Step 3: addition
            let swiftAfterAdd = swiftPixProj + swiftMaskDown
            reportDiff("step3 after_add", swiftAfterAdd, pyAfterAdd)

            // Step 4: memory_fuser (== python's fuser)
            let swiftFuserOut = me.memoryFuser(pyAfterAdd)   // feed PYTHON's after_add to isolate
            reportDiff("step4 fuser(py_after_add)", swiftFuserOut, pyFuserOut)
            let swiftFuserOutOurInput = me.memoryFuser(swiftAfterAdd)
            reportDiff("step4 fuser(our_after_add)", swiftFuserOutOurInput, pyFuserOut)

            // Step 5: out_proj is Identity → vision_features should equal fuser_out
            reportDiff("step5 final = fuser_out == vision_features?", pyFuserOut, pyVisionFeats)
        }
    }

    /// Compare what python feeds as mux_mask_for_mem (32-ch input) vs what my Swift
    /// `encodeMultiplexMasksToMemory` produces given the same source detection masks.
    /// Localizes whether the mask-construction step is the divergence.
    func testMaskConstructionParity() throws {
        let fix = try loadFixture("sam3_1_encoding_capture")
        // Inspect the per-channel mean/min/max of python's mux_mask_for_mem (1, 32, H, W).
        // Tells us which slots are active (non-zero) and what conditioning value is set.
        for callIdx in [0, 2] {
            let pyMux = fix["call_\(callIdx)/mux_mask_for_mem"]!.asType(.float32)  // NCHW
            print("=== call_\(callIdx) mux_mask channel inspection (shape \(pyMux.shape)) ===")
            for c in 0..<32 {
                let ch = pyMux[0, c]
                let chMean = MLX.mean(ch).item(Float.self)
                let chMin = MLX.min(ch).item(Float.self)
                let chMax = MLX.max(ch).item(Float.self)
                let ann = c < 16 ? "mask\(c)" : "cond\(c-16)"
                print("  ch[\(c)] (\(ann)): mean=\(String(format: "%.4f", chMean)) min=\(String(format: "%.4f", chMin)) max=\(String(format: "%.4f", chMax))")
            }
        }
    }

    func testMemoryEncoderRealInputParity() throws {
        let fix = try loadFixture("sam3_1_encoding_capture")

        // Python tensors are NCHW; Swift wants NHWC. Permute on load.
        // pix_feat (1, 256, 72, 72) NCHW → (1, 72, 72, 256) NHWC
        // mux_mask (1, 32, H, W)   NCHW → (1, H, W, 32) NHWC
        let model = Sam31Model(Sam31ModelConfig())
        try ModelLoader.load(sam31: model, from: Self.sam31ModelURL, dtype: .float32)

        for callIdx in [0, 2] {
            guard let pyPixFeat = fix["call_\(callIdx)/pix_feat"],
                  let pyMuxMask = fix["call_\(callIdx)/mux_mask_for_mem"],
                  let pyOut = fix["call_\(callIdx)/vision_features"]
            else {
                XCTFail("missing capture \(callIdx)")
                continue
            }
            let pixFeat = pyPixFeat.transposed(0, 2, 3, 1).asType(.float32)   // NHWC
            let muxMask = pyMuxMask.transposed(0, 2, 3, 1).asType(.float32)   // NHWC
            let expected = pyOut.transposed(0, 2, 3, 1).asType(.float32)      // NHWC

            // Swift's MultiplexMaskDownSampler does NOT interpolate internally. python's
            // SimpleMaskEncoder DOES interpolate to interpol_size=[1152,1152] before convs.
            // For call 0 (mux_mask 1008×1008), python interpolates up to 1152 inside the
            // encoder; for call 2 (1152×1152) no interpolation. Match python by upsampling
            // here when needed.
            var muxMaskUpsampled = muxMask
            if muxMask.dim(1) != 1152 || muxMask.dim(2) != 1152 {
                let scaleH = Float(1152) / Float(muxMask.dim(1))
                let scaleW = Float(1152) / Float(muxMask.dim(2))
                muxMaskUpsampled = Upsample(scaleFactor: .array([scaleH, scaleW]),
                                            mode: .linear(alignCorners: false))(muxMask)
            }

            let actual = model.trackerModel.memoryEncoder(pixFeat, muxMaskUpsampled)
            MLX.eval(actual)
            reportDiff("call_\(callIdx) memoryEncoder output", actual, expected)
        }
    }

    private func runVideoTrackerParity(
        numFrames: Int, fixtureName: String, framesDir: String, meanIouFloor: Float
    ) async throws {
        let fix = try loadFixture(fixtureName)
        let framesDirURL = Self.fixturesURL.appendingPathComponent(framesDir)

        let cgFrames: [CGImage] = (0..<numFrames).map { i in
            let url = framesDirURL.appendingPathComponent("\(i).jpg")
            guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
                  let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
                XCTFail("missing frame \(i).jpg at \(url.path)"); return CGImage.init(width: 1, height: 1, bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: 1, space: CGColorSpaceCreateDeviceGray(), bitmapInfo: [], provider: CGDataProvider(data: Data([0]) as CFData)!, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
            }
            return cg
        }
        let origH = cgFrames[0].height
        let origW = cgFrames[0].width
        XCTAssertEqual(origH, 720)
        XCTAssertEqual(origW, 1280)

        let model = Sam31Model(Sam31ModelConfig())
        try ModelLoader.load(sam31: model, from: Self.sam31ModelURL, dtype: .bfloat16)

        // Tokenize "person" — matches the prompt used in run_3frame_track.py.
        let tok = try await SAM3Tokenizer.fromSnapshotOrOpenAIClip(Self.sam31ModelURL, maxLength: 32)
        let encoded = tok.encode("person")

        let tracker = Sam31VideoTracker(
            model: model, resolution: 1008, dtype: .bfloat16,
            detectThreshold: 0.5, redetectEvery: 15,
            assocIoUThreshold: 0.3, maxMissingFrames: 30
        )
        // Default: per-track memory bank (better on this fixture). Toggle bucket via
        //   tracker.usePerTrackBank = false

        var perFrameIous: [[Float]] = []
        for (i, cg) in cgFrames.enumerated() {
            let pix = try ImagePreprocess.pixelValues(from: cg, size: 1008, dtype: .bfloat16)
            let live = tracker.step(
                pixelValues: pix,
                inputIds: encoded.inputIds,
                attentionMask: encoded.attentionMask,
                originalHeight: origH, originalWidth: origW
            )

            // Reference masks for this frame: (4, 720, 1280) {0,1} float32.
            let refMasks = fix["frame_\(i)/out_binary_masks"]!.asType(.float32)
            let nRef = refMasks.dim(0)
            // Greedy best-IoU match per ref mask against tracker's masks.
            var ious: [Float] = []
            for ri in 0..<nRef {
                let refMask = refMasks[ri]   // (H, W)
                var best: Float = 0
                for t in live {
                    guard let tm = t.lastMask else { continue }
                    let iou = boolMaskIoU(predicted: tm, reference: refMask)
                    if iou > best { best = iou }
                }
                ious.append(best)
            }
            perFrameIous.append(ious)
            print("  [py-sam3 parity] frame \(i): swift_tracks=\(live.count) ref_objs=\(nRef) per-obj IoU=\(ious.map { String(format: "%.3f", $0) })")
        }

        let allIous = perFrameIous.flatMap { $0 }
        let meanIou = allIous.reduce(0, +) / Float(max(allIous.count, 1))
        print("  [py-sam3 parity] mean IoU across all objects/frames: \(String(format: "%.3f", meanIou))")

        let frame0Mean = perFrameIous[0].reduce(0, +) / Float(perFrameIous[0].count)
        XCTAssertGreaterThanOrEqual(frame0Mean, 0.90,
            "frame 0 detection IoU regressed: \(frame0Mean) (was 0.985)")
        XCTAssertGreaterThanOrEqual(meanIou, meanIouFloor,
            "overall mean IoU regressed: \(meanIou) (floor: \(meanIouFloor))")
    }

    /// Mask IoU between a Swift-side bool mask (HxW) and a {0,1} float ref mask (HxW).
    private func boolMaskIoU(predicted: MLXArray, reference: MLXArray) -> Float {
        // Predicted may be shape (H, W) bool; reference is (H, W) float32 in {0,1}.
        // If sizes differ (e.g. mask at orig res), nearest-neighbor downsample reference to predicted size.
        var p = predicted
        if p.dtype != .bool { p = p .> MLXArray(0.5) }
        var r = reference .> MLXArray(0.5)
        if p.shape != r.shape {
            // Resize reference to predicted shape via integer index sampling.
            let ph = p.dim(0), pw = p.dim(1)
            let rh = r.dim(0), rw = r.dim(1)
            let yIdx = MLXArray((0..<ph).map { Int32(Double($0) * Double(rh) / Double(ph)) })
            let xIdx = MLXArray((0..<pw).map { Int32(Double($0) * Double(rw) / Double(pw)) })
            r = r[yIdx][0..., xIdx]
        }
        let pi = p.asType(.int32)
        let ri = r.asType(.int32)
        let inter = MLX.sum(pi * ri).item(Int32.self)
        let union = MLX.sum(MLX.maximum(pi, ri)).item(Int32.self)
        return union > 0 ? Float(inter) / Float(union) : 0
    }
}

