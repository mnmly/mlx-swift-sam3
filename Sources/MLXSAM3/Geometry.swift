// Geometry.swift — geometry encoder for box/point prompts.
// Python source: sam3/geometry.py
// Weight keys: detector_model.geometry_encoder.*

import MLX
import MLXNN

class GeometryEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: MultiheadAttention
    @ModuleInfo(key: "cross_attn") var crossAttn: MultiheadAttention
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm
    @ModuleInfo(key: "layer_norm3") var layerNorm3: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: MLP

    init(_ config: GeometryEncoderConfig) {
        let d = config.hiddenSize
        self._selfAttn = ModuleInfo(wrappedValue: MultiheadAttention(hiddenSize: d, numHeads: config.numAttentionHeads, dropout: config.dropout), key: "self_attn")
        self._crossAttn = ModuleInfo(wrappedValue: MultiheadAttention(hiddenSize: d, numHeads: config.numAttentionHeads, dropout: config.dropout), key: "cross_attn")
        self._layerNorm1 = ModuleInfo(wrappedValue: LayerNorm(dimensions: d, eps: config.layerNormEps), key: "layer_norm1")
        self._layerNorm2 = ModuleInfo(wrappedValue: LayerNorm(dimensions: d, eps: config.layerNormEps), key: "layer_norm2")
        self._layerNorm3 = ModuleInfo(wrappedValue: LayerNorm(dimensions: d, eps: config.layerNormEps), key: "layer_norm3")
        self._mlp = ModuleInfo(wrappedValue: MLP(hiddenSize: d, intermediateSize: config.intermediateSize, act: config.hiddenAct), key: "mlp")
        super.init()
    }

    func callAsFunction(_ src: MLXArray, memory: MLXArray, memoryPos: MLXArray? = nil) -> MLXArray {
        var out = src + selfAttn(src, src, src)
        out = layerNorm1(out)
        let key = memoryPos != nil ? memory + memoryPos! : memory
        out = out + crossAttn(out, key, memory)
        out = layerNorm2(out)
        out = out + mlp(out)
        out = layerNorm3(out)
        return out
    }
}

public class GeometryEncoder: Module {
    @ModuleInfo(key: "boxes_direct_project") var boxesDirectProject: Linear
    @ModuleInfo(key: "boxes_pool_project") var boxesPoolProject: Conv2d
    @ModuleInfo(key: "boxes_pos_enc_project") var boxesPosEncProject: Linear
    @ModuleInfo(key: "label_embed") var labelEmbed: Embedding
    @ModuleInfo(key: "cls_embed") var clsEmbed: Embedding
    @ModuleInfo(key: "final_proj") var finalProj: Linear
    @ModuleInfo(key: "layers") var layers: [GeometryEncoderLayer]
    @ModuleInfo(key: "output_layer_norm") var outputLayerNorm: LayerNorm
    @ModuleInfo(key: "prompt_layer_norm") var promptLayerNorm: LayerNorm
    @ModuleInfo(key: "vision_layer_norm") var visionLayerNorm: LayerNorm

    public init(_ config: GeometryEncoderConfig) {
        let d = config.hiddenSize
        self._boxesDirectProject = ModuleInfo(wrappedValue: Linear(4, d), key: "boxes_direct_project")
        self._boxesPoolProject = ModuleInfo(wrappedValue: Conv2d(inputChannels: d, outputChannels: d, kernelSize: .init(config.roiSize), bias: true), key: "boxes_pool_project")
        self._boxesPosEncProject = ModuleInfo(wrappedValue: Linear(d + 2, d), key: "boxes_pos_enc_project")
        self._labelEmbed = ModuleInfo(wrappedValue: Embedding(embeddingCount: 2, dimensions: d), key: "label_embed")
        self._clsEmbed = ModuleInfo(wrappedValue: Embedding(embeddingCount: 1, dimensions: d), key: "cls_embed")
        self._finalProj = ModuleInfo(wrappedValue: Linear(d, d), key: "final_proj")
        self._layers = ModuleInfo(wrappedValue: (0..<config.numLayers).map { _ in GeometryEncoderLayer(config) }, key: "layers")
        self._outputLayerNorm = ModuleInfo(wrappedValue: LayerNorm(dimensions: d, eps: config.layerNormEps), key: "output_layer_norm")
        self._promptLayerNorm = ModuleInfo(wrappedValue: LayerNorm(dimensions: d, eps: config.layerNormEps), key: "prompt_layer_norm")
        self._visionLayerNorm = ModuleInfo(wrappedValue: LayerNorm(dimensions: d, eps: config.layerNormEps), key: "vision_layer_norm")
        super.init()
    }

    func encodeBoxes(_ boxes: MLXArray) -> MLXArray {
        boxesDirectProject(boxes)
    }

    /// - Parameters:
    ///   - boxes: (B, N_box, 4) optional
    ///   - memory: (B, HW, D) image features
    ///   - memoryPos: (B, HW, D) positional encoding
    /// - Returns: (B, N_prompt, D)
    public func callAsFunction(boxes: MLXArray? = nil, memory: MLXArray? = nil, memoryPos: MLXArray? = nil) -> MLXArray {
        let b: Int
        if let mem = memory { b = mem.dim(0) }
        else if let bx = boxes { b = bx.dim(0) }
        else { fatalError("GeometryEncoder requires at least boxes or memory") }
        let d = clsEmbed.weight.dim(1)

        var tokens: [MLXArray] = []
        tokens.append(broadcast(clsEmbed.weight[.newAxis, 0...], to: [b, 1, d]))

        if let bx = boxes, bx.dim(1) > 0 {
            tokens.append(encodeBoxes(bx))
        }

        var src = concatenated(tokens, axis: 1)
        for layer in layers {
            src = layer(src, memory: memory ?? src, memoryPos: memoryPos)
        }
        src = outputLayerNorm(src)
        src = finalProj(src)
        return src
    }
}
