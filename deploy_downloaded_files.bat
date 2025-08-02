@echo off
echo Deploying downloaded Blood Pressure DLL files...
echo ================================================

REM Create necessary directories
if not exist "CppBloodPressureDLL\models" mkdir "CppBloodPressureDLL\models"
if not exist "CppBloodPressureDLL\include" mkdir "CppBloodPressureDLL\include"

REM Copy main DLL to root directory
if exist "BloodPressureDLL.dll" (
    echo Copying BloodPressureDLL.dll to root...
    copy "BloodPressureDLL.dll" ".\" >nul
    echo ✓ BloodPressureDLL.dll deployed
) else (
    echo ERROR: BloodPressureDLL.dll not found
    exit /b 1
)

REM Copy dependency DLLs to root directory
echo Copying dependency DLLs...
for %%f in (
    abseil_dll.dll
    jpeg62.dll
    libgcc_s_seh-1.dll
    libgfortran-5.dll
    liblapack.dll
    liblzma.dll
    libpng16.dll
    libprotobuf.dll
    libquadmath-0.dll
    libsharpyuv.dll
    libwebp.dll
    libwebpdecoder.dll
    libwebpdemux.dll
    libwebpmux.dll
    libwinpthread-1.dll
    onnxruntime.dll
    openblas.dll
    opencv_core4.dll
    opencv_dnn4.dll
    opencv_imgcodecs4.dll
    opencv_imgproc4.dll
    opencv_objdetect4.dll
    tiff.dll
    zlib1.dll
) do (
    if exist "%%f" (
        copy "%%f" ".\" >nul
        echo ✓ %%f deployed
    ) else (
        echo WARNING: %%f not found
    )
)

REM Copy model files
echo Copying model files...
for %%f in (
    diastolicbloodpressure.onnx
    systolicbloodpressure.onnx
    opencv_face_detector.pbtxt
    opencv_face_detector_uint8.pb
    shape_predictor_68_face_landmarks.dat
) do (
    if exist "models\%%f" (
        copy "models\%%f" "CppBloodPressureDLL\models\" >nul
        echo ✓ %%f deployed to CppBloodPressureDLL\models\
    ) else (
        echo WARNING: models\%%f not found
    )
)

REM Copy header file
if exist "include\BloodPressureDLL.h" (
    copy "include\BloodPressureDLL.h" "CppBloodPressureDLL\include\" >nul
    echo ✓ BloodPressureDLL.h deployed to CppBloodPressureDLL\include\
) else (
    echo WARNING: include\BloodPressureDLL.h not found
)

REM Copy documentation files
if exist "INTEGRATION_GUIDE.md" (
    copy "INTEGRATION_GUIDE.md" ".\" >nul
    echo ✓ INTEGRATION_GUIDE.md deployed
) else (
    echo WARNING: INTEGRATION_GUIDE.md not found
)

if exist "README.md" (
    copy "README.md" ".\" >nul
    echo ✓ README.md deployed
) else (
    echo WARNING: README.md not found
)

echo.
echo Deployment completed!
echo.
echo Files deployed:
echo - Main DLL: BloodPressureDLL.dll
echo - Dependencies: 24 DLL files
echo - Models: 5 model files in CppBloodPressureDLL\models\
echo - Headers: BloodPressureDLL.h in CppBloodPressureDLL\include\
echo - Documentation: INTEGRATION_GUIDE.md, README.md
echo.
echo Ready for testing with GitHub Actions! 
