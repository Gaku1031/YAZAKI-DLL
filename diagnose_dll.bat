@echo off
echo === DLL依存関係診断ツール ===
echo.

REM 現在のディレクトリ確認
echo 現在のディレクトリ: %CD%
echo.

REM DLLファイル確認
echo 1. DLLファイル確認
if exist "BloodPressureEstimation.dll" (
    echo ✓ BloodPressureEstimation.dll が存在します
    
    REM ファイル情報表示
    for %%F in (BloodPressureEstimation.dll) do (
        echo   サイズ: %%~zF bytes
        echo   日時: %%~tF
    )
) else (
    echo ✗ BloodPressureEstimation.dll が見つかりません
    exit /b 1
)

echo.

REM 依存関係確認
echo 2. DLL依存関係確認
echo dumpbin /dependents を実行中...
dumpbin /dependents BloodPressureEstimation.dll 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ⚠️ dumpbin が利用できません
    echo Visual Studio がインストールされていることを確認してください
)

echo.

REM Python環境確認
echo 3. Python環境確認
python --version 2>nul
if %ERRORLEVEL% EQU 0 (
    echo ✓ Python が利用可能です
    
    REM Python DLL確認
    python -c "import sys; print('Python実行可能ファイル:', sys.executable)"
    python -c "import sys; import os; print('Python DLL パス:', os.path.dirname(sys.executable) + '\\python' + str(sys.version_info.major) + str(sys.version_info.minor) + '.dll')"
    
    REM Python DLLの存在確認
    for /f "tokens=*" %%i in ('python -c "import sys; import os; print(os.path.dirname(sys.executable) + '\\python' + str(sys.version_info.major) + str(sys.version_info.minor) + '.dll')"') do set PYTHON_DLL_PATH=%%i
    
    echo.
    echo Python DLL確認: %PYTHON_DLL_PATH%
    if exist "%PYTHON_DLL_PATH%" (
        echo ✓ Python DLL が見つかりました
    ) else (
        echo ✗ Python DLL が見つかりません
        echo これが問題の原因の可能性があります
    )
) else (
    echo ✗ Python が利用できません
)

echo.

REM Visual C++ Redistributable確認
echo 4. Visual C++ Redistributable確認
reg query "HKLM\SOFTWARE\Classes\Installer\Dependencies" /s /f "Microsoft Visual C++" 2>nul | find "2022" >nul
if %ERRORLEVEL% EQU 0 (
    echo ✓ Visual C++ 2022 Redistributable が検出されました
) else (
    echo ⚠️ Visual C++ 2022 Redistributable が見つかりません
    echo https://aka.ms/vs/17/release/vc_redist.x64.exe からダウンロードしてください
)

echo.

REM PATH環境変数確認
echo 5. PATH環境変数確認
echo Python実行可能ファイルのディレクトリがPATHに含まれているか確認中...
python -c "import sys; import os; print(os.path.dirname(sys.executable))" > temp_python_path.txt 2>nul
if %ERRORLEVEL% EQU 0 (
    set /p PYTHON_DIR=<temp_python_path.txt
    del temp_python_path.txt
    
    echo Python ディレクトリ: %PYTHON_DIR%
    echo %PATH% | find /i "%PYTHON_DIR%" >nul
    if %ERRORLEVEL% EQU 0 (
        echo ✓ Python ディレクトリがPATHに含まれています
    ) else (
        echo ⚠️ Python ディレクトリがPATHに含まれていません
    )
) else (
    if exist temp_python_path.txt del temp_python_path.txt
)

echo.

REM 解決方法提案
echo === 解決方法 ===
echo.
echo 問題が見つかった場合の解決手順:
echo.
echo 1. Python DLLが見つからない場合:
echo    - Python DLLを同じディレクトリにコピー
echo    - または、システムPATHにPythonディレクトリを追加
echo.
echo 2. Visual C++ Redistributableが不足している場合:
echo    - Microsoft Visual C++ 2022 Redistributable をインストール
echo    - https://aka.ms/vs/17/release/vc_redist.x64.exe
echo.
echo 3. PATH設定の問題:
echo    - PythonのインストールディレクトリをシステムPATHに追加
echo.

pause