// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "mlx-swift-sam3",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .tvOS(.v17),
        .visionOS(.v1),
    ],
    products: [
        .library(name: "MLXSAM3", targets: ["MLXSAM3"]),
        .executable(name: "ImageInfer", targets: ["ImageInfer"]),
        .executable(name: "Sam31ImageInfer", targets: ["Sam31ImageInfer"]),
        .executable(name: "VideoTrack", targets: ["VideoTrack"]),
        .executable(name: "Sam31VideoTrack", targets: ["Sam31VideoTrack"]),
        .executable(name: "Sam31VideoTrackFull", targets: ["Sam31VideoTrackFull"]),
        .executable(name: "sam3-bench", targets: ["sam3-bench"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.3")),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "MLXSAM3",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXLinalg", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/MLXSAM3"
        ),
        .executableTarget(
            name: "ImageInfer",
            dependencies: [
                "MLXSAM3",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Examples/ImageInfer"
        ),
        .executableTarget(
            name: "Sam31ImageInfer",
            dependencies: [
                "MLXSAM3",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Examples/Sam31ImageInfer"
        ),
        .executableTarget(
            name: "VideoTrack",
            dependencies: [
                "MLXSAM3",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Examples/VideoTrack"
        ),
        .executableTarget(
            name: "Sam31VideoTrack",
            dependencies: [
                "MLXSAM3",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Examples/Sam31VideoTrack"
        ),
        .executableTarget(
            name: "Sam31VideoTrackFull",
            dependencies: [
                "MLXSAM3",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Examples/Sam31VideoTrackFull"
        ),
        .executableTarget(
            name: "sam3-bench",
            dependencies: [
                "MLXSAM3",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Tools/sam3-bench"
        ),
        .testTarget(
            name: "MLXSAM3Tests",
            dependencies: ["MLXSAM3"],
            path: "Tests/MLXSAM3Tests"
        ),
    ]
)
