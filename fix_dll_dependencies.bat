@echo off
echo === DLL依存関係修正ツール ===
echo.

REM 現在のディレクトリ確認
echo 現在のディレクトリ: %CD%
echo.

REM Python DLL自動コピー
echo 1. Python DLL自動コピー
echo Python DLLの場所を特定中...

python -c "import sys; import os; print(os.path.dirname(sys.executable) + '\\python' + str(sys.version_info.major) + str(sys.version_info.minor) + '.dll')" > temp_python_dll.txt 2>nul
if %ERRORLEVEL% EQU 0 (
    set /p PYTHON_DLL_PATH=<temp_python_dll.txt
    del temp_python_dll.txt
    
    echo Python DLL パス: %PYTHON_DLL_PATH%
    
    if exist "%PYTHON_DLL_PATH%" (
        echo Python DLLをコピー中...
        copy "%PYTHON_DLL_PATH%" .
        if %ERRORLEVEL% EQU 0 (
            echo ✓ Python DLL コピー完了
        ) else (
            echo ✗ Python DLL コピー失敗
        )
    ) else (
        echo ✗ Python DLL が見つかりません: %PYTHON_DLL_PATH%
    )
) else (
    echo ✗ Python DLL パスの取得に失敗
    if exist temp_python_dll.txt del temp_python_dll.txt
)

echo.

REM 追加で必要なDLLの確認とコピー
echo 2. 追加DLLの確認
echo Python実行ディレクトリから追加DLLをコピー中...

python -c "import sys; import os; print(os.path.dirname(sys.executable))" > temp_python_dir.txt 2>nul
if %ERRORLEVEL% EQU 0 (
    set /p PYTHON_DIR=<temp_python_dir.txt
    del temp_python_dir.txt
    
    echo Python ディレクトリ: %PYTHON_DIR%
    
    REM よく必要になるDLLをコピー
    set DLL_LIST=python311.dll python3.dll vcruntime140.dll vcruntime140_1.dll msvcp140.dll
    
    for %%d in (%DLL_LIST%) do (
        if exist "%PYTHON_DIR%\%%d" (
            if not exist "%%d" (
                copy "%PYTHON_DIR%\%%d" .
                echo ✓ %%d をコピー
            ) else (
                echo - %%d は既に存在
            )
        )
    )
) else (
    if exist temp_python_dir.txt del temp_python_dir.txt
    echo ⚠️ Python ディレクトリの取得に失敗
)

echo.

REM DLLsディレクトリからの追加DLL
echo 3. Python DLLsディレクトリの確認
python -c "import sys; import os; print(os.path.dirname(sys.executable) + '\\DLLs')" > temp_dlls_dir.txt 2>nul
if %ERRORLEVEL% EQU 0 (
    set /p DLLS_DIR=<temp_dlls_dir.txt
    del temp_dlls_dir.txt
    
    if exist "%DLLS_DIR%" (
        echo DLLs ディレクトリ: %DLLS_DIR%
        
        REM よく必要になるPython拡張DLLをコピー
        set EXT_DLL_LIST=_ctypes.pyd _socket.pyd select.pyd unicodedata.pyd
        
        for %%d in (%EXT_DLL_LIST%) do (
            if exist "%DLLS_DIR%\%%d" (
                if not exist "%%d" (
                    copy "%DLLS_DIR%\%%d" .
                    echo ✓ %%d をコピー
                ) else (
                    echo - %%d は既に存在
                )
            )
        )
    ) else (
        echo ⚠️ DLLs ディレクトリが見つかりません
    )
) else (
    if exist temp_dlls_dir.txt del temp_dlls_dir.txt
)

echo.

REM 現在のディレクトリのDLLファイル一覧
echo 4. 現在のディレクトリのDLLファイル
echo コピーされたDLLファイル:
dir *.dll /b 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo DLLファイルが見つかりません
)

echo.
dir *.pyd /b 2>nul
if %ERRORLEVEL% EQU 0 (
    echo Python拡張ファイル:
    dir *.pyd /b
)

echo.

REM テスト実行
echo 5. 修正後テスト
echo BloodPressureTest.exe を実行してテストしますか? (Y/N)
set /p choice="入力: "
if /i "%choice%"=="Y" (
    echo.
    echo === テスト実行 ===
    if exist "BloodPressureTest.exe" (
        BloodPressureTest.exe
    ) else if exist "SimpleBloodPressureTest.exe" (
        SimpleBloodPressureTest.exe
    ) else (
        echo ✗ テスト実行ファイルが見つかりません
        echo 先にC#プログラムをコンパイルしてください
    )
) else (
    echo テストをスキップしました
    echo 手動で BloodPressureTest.exe を実行してください
)

echo.
echo === 修正完了 ===
echo.
echo 依存関係の問題が解決しない場合:
echo 1. Visual C++ 2022 Redistributable をインストール
echo 2. Python を管理者権限で再インストール
echo 3. システム環境変数 PATH にPythonディレクトリを追加
echo.
pause