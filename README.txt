This folder must contain: face_detection_yunet_2023mar.onnx
(OpenCV's official YuNet face detector model, ~233KB)

build_exe.bat downloads this file automatically before building.
If needed manually, download from:
https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx

If this file is missing, the program still works: it will
automatically fall back to a less accurate face detector (Haar Cascade),
and if that also fails, to border/margin-based auto-crop.
