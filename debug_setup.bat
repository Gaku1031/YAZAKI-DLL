@echo off
echo === デバッグ用セットアップスクリプト ===
echo.

REM 現在のディレクトリ確認
echo 現在のディレクトリ: %CD%
echo.

REM C#コンパイラー確認
echo C#コンパイラー確認中...
where csc.exe >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✓ csc.exe が見つかりました
    csc.exe
) else (
    echo ✗ csc.exe が見つかりません
    echo.
    echo Visual Studio Developer Command Prompt を使用してください
    echo または、以下のコマンドを実行してください:
    echo call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    echo.
    pause
    exit /b 1
)

echo.

REM 必要ファイル確認
echo 必要ファイル確認中...

echo DLL確認:
if exist "build\dist\BloodPressureEstimation.dll" (
    echo ✓ build\dist\BloodPressureEstimation.dll
) else (
    echo ✗ build\dist\BloodPressureEstimation.dll が見つかりません
    echo まず simple_build.bat を実行してください
)

echo Pythonモジュール確認:
if exist "bp_estimation_simple.py" (
    echo ✓ bp_estimation_simple.py
) else (
    echo ✗ bp_estimation_simple.py が見つかりません
    echo create_cpp_wrapper_dll.py を実行してください
)

echo C#テストコード確認:
if exist "BloodPressureTest.cs" (
    echo ✓ BloodPressureTest.cs
) else (
    echo ✗ BloodPressureTest.cs が見つかりません
)

echo サンプル動画確認:
if exist "sample-data\100万画素.webm" (
    echo ✓ sample-data\100万画素.webm
) else (
    echo ✗ sample-data\100万画素.webm が見つかりません
)

echo.

REM テストディレクトリ作成
echo テストディレクトリ作成中...
if exist test_environment (
    echo 既存のtest_environmentを削除中...
    rmdir /s /q test_environment
)
mkdir test_environment
echo ✓ test_environment 作成完了

echo.

REM ファイルコピー（詳細ログ付き）
echo ファイルコピー中...

if exist "build\dist\BloodPressureEstimation.dll" (
    copy "build\dist\BloodPressureEstimation.dll" "test_environment\" >nul
    if %ERRORLEVEL% EQU 0 (
        echo ✓ BloodPressureEstimation.dll コピー完了
    ) else (
        echo ✗ BloodPressureEstimation.dll コピー失敗
    )
) else (
    echo ⚠️ BloodPressureEstimation.dll をスキップ
)

if exist "bp_estimation_simple.py" (
    copy "bp_estimation_simple.py" "test_environment\" >nul
    if %ERRORLEVEL% EQU 0 (
        echo ✓ bp_estimation_simple.py コピー完了
    ) else (
        echo ✗ bp_estimation_simple.py コピー失敗
    )
) else (
    echo ⚠️ bp_estimation_simple.py をスキップ
)

if exist "BloodPressureTest.cs" (
    copy "BloodPressureTest.cs" "test_environment\" >nul
    if %ERRORLEVEL% EQU 0 (
        echo ✓ BloodPressureTest.cs コピー完了
    ) else (
        echo ✗ BloodPressureTest.cs コピー失敗
    )
) else (
    echo ⚠️ BloodPressureTest.cs をスキップ
)

if exist "sample-data" (
    xcopy "sample-data" "test_environment\sample-data\" /E /I /Q >nul
    if %ERRORLEVEL% EQU 0 (
        echo ✓ sample-data コピー完了
    ) else (
        echo ✗ sample-data コピー失敗
    )
) else (
    echo ⚠️ sample-data をスキップ
)

mkdir "test_environment\models" 2>nul
echo ✓ models ディレクトリ作成完了

echo.

REM test_environmentに移動してコンパイル
echo test_environmentディレクトリに移動...
cd test_environment

echo.
echo コピーされたファイル確認:
dir /b

echo.

REM C#コンパイル（詳細出力）
echo C#コンパイル実行中...
echo コマンド: csc BloodPressureTest.cs
echo.

csc BloodPressureTest.cs

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✓ C#コンパイル成功
    
    echo.
    echo 生成されたファイル:
    if exist "BloodPressureTest.exe" (
        echo ✓ BloodPressureTest.exe 作成完了
        dir BloodPressureTest.exe
    ) else (
        echo ✗ BloodPressureTest.exe が作成されませんでした
    )
    
    echo.
    echo テスト実行しますか? (Y/N)
    set /p choice="入力: "
    if /i "%choice%"=="Y" (
        echo.
        echo === テスト実行 ===
        BloodPressureTest.exe
    )
    
) else (
    echo.
    echo ✗ C#コンパイル失敗
    echo.
    echo 考えられる原因:
    echo 1. csc.exe がPATHに含まれていない
    echo 2. C#ソースコードにエラーがある
    echo 3. .NET Framework が不足している
    echo.
    echo 解決方法:
    echo 1. Visual Studio Developer Command Prompt を使用
    echo 2. Visual Studio 2022 の インストール確認
)

cd ..
echo.
pause