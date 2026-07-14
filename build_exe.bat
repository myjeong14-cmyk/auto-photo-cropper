@echo off
echo ============================================
echo  IDPhotoEditor - Build Script
echo ============================================

pip install -r requirements.txt

if not exist "models" mkdir "models"

if not exist "models\face_detection_yunet_2023mar.onnx" (
    echo Downloading YuNet face detection model...
    curl -L -o "models\face_detection_yunet_2023mar.onnx" "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx"
)

pyinstaller --noconfirm --onedir --windowed --noupx --name "IDPhotoEditor" --add-data "models;models" main.py

echo.
echo Build finished. Check dist\IDPhotoEditor\IDPhotoEditor.exe
echo Zip the whole dist\IDPhotoEditor folder to distribute it.
pause
