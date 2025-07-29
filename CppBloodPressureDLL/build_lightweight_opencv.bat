@echo off
echo Building BloodPressureDLL with lightweight OpenCV...

:: ビルドディレクトリを作成
if not exist build mkdir build
cd build

:: 軽量版OpenCVを使用してビルド
cmake -G "Visual Studio 17 2022" -A x64 ^
    -DUSE_LIGHTWEIGHT_OPENCV=ON ^
    -DUSE_STATIC_OPENCV=OFF ^
    -DUSE_INDIVIDUAL_OPENCV_LIBS=OFF ^
    ..

if %ERRORLEVEL% neq 0 (
    echo CMake configuration failed
    exit /b 1
)

:: ビルド実行
cmake --build . --config Release

if %ERRORLEVEL% neq 0 (
    echo Build failed
    exit /b 1
)

echo Build completed successfully!
echo Lightweight OpenCV used - opencv480_world.dll not required
echo Estimated size reduction: 60MB -> 13MB

cd .. 
