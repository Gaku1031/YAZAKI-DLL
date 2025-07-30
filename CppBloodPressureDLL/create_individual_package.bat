@echo off
echo Creating package with individual OpenCV libraries...

:: パッケージディレクトリを作成
set PACKAGE_DIR=package_individual
if exist %PACKAGE_DIR% rmdir /s /q %PACKAGE_DIR%
mkdir %PACKAGE_DIR%

:: 個別ライブラリでビルド
call build_individual_opencv.bat

if %ERRORLEVEL% neq 0 (
    echo Build failed, cannot create package
    exit /b 1
)

:: 必要なファイルをコピー
echo Copying files to package...

:: DLLファイル
copy build\Release\BloodPressureDLL.dll %PACKAGE_DIR%\

:: 個別OpenCVライブラリ（opencv480_world.dllの代わり）
copy "%OpenCV_DIR%\bin\opencv_core480.dll" %PACKAGE_DIR%\
copy "%OpenCV_DIR%\bin\opencv_imgproc480.dll" %PACKAGE_DIR%\
copy "%OpenCV_DIR%\bin\opencv_imgcodecs480.dll" %PACKAGE_DIR%\
copy "%OpenCV_DIR%\bin\opencv_objdetect480.dll" %PACKAGE_DIR%\
copy "%OpenCV_DIR%\bin\opencv_dnn480.dll" %PACKAGE_DIR%\

:: その他の依存関係
copy "%OpenCV_DIR%\bin\opencv_videoio480.dll" %PACKAGE_DIR%\
copy "%OpenCV_DIR%\bin\opencv_highgui480.dll" %PACKAGE_DIR%\

:: dlibライブラリ
copy "%dlib_DIR%\bin\dlib19.24.0_release_64bit_msvc1929.dll" %PACKAGE_DIR%\

:: ONNX Runtime
copy "package\onnxruntime.dll" %PACKAGE_DIR%\
copy "package\onnxruntime_providers_shared.dll" %PACKAGE_DIR%\

:: モデルファイル
if not exist %PACKAGE_DIR%\models mkdir %PACKAGE_DIR%\models
copy "models\*.onnx" %PACKAGE_DIR%\models\
copy "models\*.pb" %PACKAGE_DIR%\models\
copy "models\*.pbtxt" %PACKAGE_DIR%\models\

:: dlibモデルファイル
copy "models\shape_predictor_68_face_landmarks.dat" %PACKAGE_DIR%\models\

:: ヘッダーファイル
if not exist %PACKAGE_DIR%\include mkdir %PACKAGE_DIR%\include
copy "include\*.h" %PACKAGE_DIR%\include\

:: ドキュメント
copy "README.md" %PACKAGE_DIR%\
copy "INTEGRATION_GUIDE.md" %PACKAGE_DIR%\

:: パッケージ情報ファイル
echo Individual OpenCV Libraries Package > %PACKAGE_DIR%\PACKAGE_INFO.txt
echo Created: %date% %time% >> %PACKAGE_DIR%\PACKAGE_INFO.txt
echo OpenCV: Individual libraries (not world) >> %PACKAGE_DIR%\PACKAGE_INFO.txt
echo Size reduction: 60MB -> 33MB >> %PACKAGE_DIR%\PACKAGE_INFO.txt
echo Required DLLs: >> %PACKAGE_DIR%\PACKAGE_INFO.txt
echo - opencv_core480.dll >> %PACKAGE_DIR%\PACKAGE_INFO.txt
echo - opencv_imgproc480.dll >> %PACKAGE_DIR%\PACKAGE_INFO.txt
echo - opencv_imgcodecs480.dll >> %PACKAGE_DIR%\PACKAGE_INFO.txt
echo - opencv_objdetect480.dll >> %PACKAGE_DIR%\PACKAGE_INFO.txt
echo - opencv_dnn480.dll >> %PACKAGE_DIR%\PACKAGE_INFO.txt
echo - opencv_videoio480.dll >> %PACKAGE_DIR%\PACKAGE_INFO.txt
echo - opencv_highgui480.dll >> %PACKAGE_DIR%\PACKAGE_INFO.txt

echo Package created successfully in %PACKAGE_DIR%
echo Total size: 
dir %PACKAGE_DIR% /s | find "File(s)" 
