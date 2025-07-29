@echo off
echo Building custom lightweight OpenCV for BloodPressureDLL...

:: 必要な環境変数を設定
set OPENCV_VERSION=4.8.0
set BUILD_DIR=opencv_build
set INSTALL_DIR=opencv_install

:: ビルドディレクトリを作成
if not exist %BUILD_DIR% mkdir %BUILD_DIR%
cd %BUILD_DIR%

:: CMake設定（必要最小限のモジュールのみ）
cmake -G "Visual Studio 17 2022" -A x64 ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_INSTALL_PREFIX=../%INSTALL_DIR% ^
    -DBUILD_SHARED_LIBS=ON ^
    -DBUILD_TESTS=OFF ^
    -DBUILD_PERF_TESTS=OFF ^
    -DBUILD_EXAMPLES=OFF ^
    -DBUILD_DOCS=OFF ^
    -DBUILD_opencv_apps=OFF ^
    -DBUILD_opencv_calib3d=OFF ^
    -DBUILD_opencv_features2d=OFF ^
    -DBUILD_opencv_flann=OFF ^
    -DBUILD_opencv_highgui=OFF ^
    -DBUILD_opencv_ml=OFF ^
    -DBUILD_opencv_photo=OFF ^
    -DBUILD_opencv_python=OFF ^
    -DBUILD_opencv_python_bindings_generator=OFF ^
    -DBUILD_opencv_stitching=OFF ^
    -DBUILD_opencv_superres=OFF ^
    -DBUILD_opencv_ts=OFF ^
    -DBUILD_opencv_video=OFF ^
    -DBUILD_opencv_videoio=OFF ^
    -DBUILD_opencv_videostab=OFF ^
    -DBUILD_opencv_contrib=OFF ^
    -DBUILD_opencv_java=OFF ^
    -DBUILD_opencv_js=OFF ^
    -DBUILD_opencv_objc=OFF ^
    -DBUILD_opencv_world=ON ^
    -DWITH_1394=OFF ^
    -DWITH_ADE=OFF ^
    -DWITH_CAROTENE=OFF ^
    -DWITH_CLP=OFF ^
    -DWITH_CUBLAS=OFF ^
    -DWITH_CUDA=OFF ^
    -DWITH_CUDNN=OFF ^
    -DWITH_DIRECTX=OFF ^
    -DWITH_EIGEN=OFF ^
    -DWITH_FFMPEG=OFF ^
    -DWITH_GSTREAMER=OFF ^
    -DWITH_GTK=OFF ^
    -DWITH_HALIDE=OFF ^
    -DWITH_IPP=OFF ^
    -DWITH_IPP_IW=OFF ^
    -DWITH_ITT=OFF ^
    -DWITH_JASPER=OFF ^
    -DWITH_JPEG=ON ^
    -DWITH_LAPACK=OFF ^
    -DWITH_MFX=OFF ^
    -DWITH_MSMF=OFF ^
    -DWITH_OPENCL=OFF ^
    -DWITH_OPENCLAMDBLAS=OFF ^
    -DWITH_OPENCLAMDFFT=OFF ^
    -DWITH_OPENCL_SVM=OFF ^
    -DWITH_OPENEXR=OFF ^
    -DWITH_OPENGL=OFF ^
    -DWITH_OPENJPEG=OFF ^
    -DWITH_OPENMP=OFF ^
    -DWITH_OPENNI=OFF ^
    -DWITH_OPENNI2=OFF ^
    -DWITH_OPENVX=OFF ^
    -DWITH_PNG=ON ^
    -DWITH_PROTOBUF=OFF ^
    -DWITH_PTHREADS_PF=OFF ^
    -DWITH_PVAPI=OFF ^
    -DWITH_QT=OFF ^
    -DWITH_TBB=OFF ^
    -DWITH_TIFF=OFF ^
    -DWITH_VA=OFF ^
    -DWITH_VA_INTEL=OFF ^
    -DWITH_VTK=OFF ^
    -DWITH_WEBP=OFF ^
    -DWITH_WIN32UI=OFF ^
    -DWITH_XIMEA=OFF ^
    -DWITH_XINE=OFF ^
    -DWITH_XSHARP=OFF ^
    -DWITH_ZLIB=ON ^
    -DWITH_OPENCV_DNN=ON ^
    -DWITH_OPENCV_OBJDETECT=ON ^
    -DWITH_OPENCV_IMGPROC=ON ^
    -DWITH_OPENCV_IMGCODECS=ON ^
    -DWITH_OPENCV_CORE=ON ^
    ../opencv

if %ERRORLEVEL% neq 0 (
    echo CMake configuration failed
    exit /b 1
)

:: ビルド実行
cmake --build . --config Release --target install

if %ERRORLEVEL% neq 0 (
    echo Build failed
    exit /b 1
)

echo Custom OpenCV build completed successfully!
echo Install directory: %INSTALL_DIR%
echo Estimated size reduction: 60MB -> 15-20MB

cd .. 
