@echo off
echo === 血圧推定DLLテスト環境セットアップ ===
echo.

REM テストディレクトリ作成
echo テストディレクトリ準備中...
if exist test_environment (
    echo 既存のtest_environmentを削除中...
    rmdir /s /q test_environment
)
mkdir test_environment
echo ✓ test_environment作成完了

REM 必要ファイルコピー
echo.
echo 必要ファイルコピー中...

REM DLL確認・コピー
if exist "build\dist\BloodPressureEstimation.dll" (
    copy "build\dist\BloodPressureEstimation.dll" "test_environment\"
    echo ✓ BloodPressureEstimation.dll コピー完了
) else (
    echo ✗ エラー: build\dist\BloodPressureEstimation.dll が見つかりません
    echo まず simple_build.bat を実行してDLLをビルドしてください
    pause
    exit /b 1
)

REM Pythonモジュールコピー
if exist "bp_estimation_simple.py" (
    copy "bp_estimation_simple.py" "test_environment\"
    echo ✓ bp_estimation_simple.py コピー完了
) else (
    echo ✗ エラー: bp_estimation_simple.py が見つかりません
    echo create_cpp_wrapper_dll.py を実行してファイルを生成してください
    pause
    exit /b 1
)

REM C#テストコードコピー
if exist "BloodPressureTest.cs" (
    copy "BloodPressureTest.cs" "test_environment\"
    echo ✓ BloodPressureTest.cs コピー完了
) else (
    echo ✗ エラー: BloodPressureTest.cs が見つかりません
    pause
    exit /b 1
)

REM サンプル動画ディレクトリコピー
if exist "sample-data" (
    xcopy "sample-data" "test_environment\sample-data\" /E /I
    echo ✓ sample-data コピー完了
) else (
    echo ⚠️ 警告: sample-data ディレクトリが見つかりません
    echo サンプル動画を手動で配置してください
)

REM modelsディレクトリ作成（空でも可）
mkdir "test_environment\models" 2>nul
echo ✓ models ディレクトリ準備完了

echo.
echo === セットアップ完了 ===
echo.

REM C#コンパイル
echo C#テストプログラムコンパイル中...
cd test_environment

csc BloodPressureTest.cs
if %ERRORLEVEL% EQU 0 (
    echo ✓ C#コンパイル成功
    echo.
    
    echo === 実行可能ファイル ===
    echo BloodPressureTest.exe が作成されました
    echo.
    
    echo === テスト実行方法 ===
    echo 1. cd test_environment
    echo 2. BloodPressureTest.exe
    echo.
    
    echo === ファイル構成確認 ===
    dir /b
    echo.
    
    echo テスト実行しますか? (Y/N)
    set /p choice="入力: "
    if /i "%choice%"=="Y" (
        echo.
        echo === テスト実行開始 ===
        BloodPressureTest.exe
    ) else (
        echo.
        echo テストは後で手動実行してください
        echo コマンド: cd test_environment && BloodPressureTest.exe
    )
) else (
    echo ✗ C#コンパイル失敗
    echo csc.exe がPATHに含まれていることを確認してください
)

cd ..
echo.
pause