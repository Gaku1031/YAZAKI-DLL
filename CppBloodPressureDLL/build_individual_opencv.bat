@echo off
echo Building BloodPressureDLL with individual OpenCV libraries...

:: ビルドディレクトリを作成
if not exist build mkdir build
cd build

:: 個別ライブラリを使用してビルド
cmake -G "Visual Studio 17 2022" -A x64 ^
    -DUSE_INDIVIDUAL_OPENCV_LIBS=ON ^
    -DUSE_STATIC_OPENCV=OFF ^
    -DUSE_LIGHTWEIGHT_OPENCV=OFF ^
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
echo Individual OpenCV libraries used - opencv480_world.dll not required
echo Estimated size reduction: 60MB -> 33MB

cd .. 
