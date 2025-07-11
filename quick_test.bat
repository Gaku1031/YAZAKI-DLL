@echo off
echo === クイックテスト実行 ===
echo.

REM Visual Studio環境の確認
echo Visual Studio環境確認中...
where csc.exe >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Developer Command Prompt環境をセットアップ中...
    if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    ) else (
        echo ✗ Visual Studio 2022 が見つかりません
        echo 手動で Developer Command Prompt を起動してください
        pause
        exit /b 1
    )
)

REM テストディレクトリクリーンアップ
if exist test_environment rmdir /s /q test_environment
mkdir test_environment

REM 必要最小限のファイルコピー
echo 必要ファイルをコピー中...

REM DLL
if exist "build\dist\BloodPressureEstimation.dll" (
    copy "build\dist\BloodPressureEstimation.dll" "test_environment\"
    echo ✓ DLL コピー完了
) else (
    echo ✗ DLL が見つかりません。simple_build.bat を先に実行してください
    pause
    exit /b 1
)

REM Pythonモジュール
if exist "bp_estimation_simple.py" (
    copy "bp_estimation_simple.py" "test_environment\"
    echo ✓ Python モジュール コピー完了
) else (
    echo ✗ Python モジュールが見つかりません
    pause
    exit /b 1
)

REM サンプル動画
if exist "sample-data" (
    xcopy "sample-data" "test_environment\sample-data\" /E /I /Q
    echo ✓ サンプル動画 コピー完了
) else (
    echo ⚠️ サンプル動画がありません
)

REM シンプルテストコード
copy "SimpleBloodPressureTest.cs" "test_environment\"
echo ✓ テストコード コピー完了

REM models ディレクトリ
mkdir "test_environment\models"

REM テストディレクトリに移動
cd test_environment

echo.
echo コピーされたファイル:
dir /b

echo.
echo シンプルテストをコンパイル中...
csc SimpleBloodPressureTest.cs

if %ERRORLEVEL% EQU 0 (
    echo ✓ コンパイル成功
    echo.
    echo テスト実行中...
    echo.
    SimpleBloodPressureTest.exe
) else (
    echo ✗ コンパイル失敗
    echo.
    echo 手動でコンパイルしてください:
    echo cd test_environment
    echo csc SimpleBloodPressureTest.cs
    echo SimpleBloodPressureTest.exe
)

cd ..
echo.
pause