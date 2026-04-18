import XCTest
import MLX
import MLXFast
import MLXNN
@testable import MLXSAM3

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

}
