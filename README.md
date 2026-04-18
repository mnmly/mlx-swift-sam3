xcodebuild -scheme ImageInfer -destination 'platform=macOS' build -derivedDataPath .build/xcode && .build/xcode/Build/Products/Debug/ImageInfer --image /Users/mnmly/Downloads/default.jpeg --prompt "animal" --tokenizer weights/mlx-sam3-new --weights weights/mlx-sam3-new --output    out.png --threshold 0.3


SNAP=~/.cache/huggingface/hub/models--mlx-community--sam3.1-bf16/snapshots/a992e302ea9b0f03f41dfd93414a4fd0e818f65b
 xcodebuild -scheme Sam31ImageInfer -destination 'platform=macOS' build -derivedDataPath .build/xcode && .build/xcode/Build/Products/Debug/Sam31ImageInfer --image /Users/mnmly/Downloads/default.jpeg --prompt "animals" --tokenizer weights/mlx-sam3.1 --weights $SNAP --output out.png --threshold 0.3