"""
Windows 64-bit DLLビルドスクリプト
cx_Freezeを使用してPythonコードをWindows 64bit DLLとしてコンパイル
"""

from cx_Freeze import setup, Executable
import sys
import os

# ビルドオプション
build_exe_options = {
    "packages": [
        "cv2", "mediapipe", "numpy", "scipy", "sklearn", "pandas", 
        "joblib", "pywt", "threading", "ctypes", "logging", "collections",
        "datetime", "time", "json", "pathlib", "re"
    ],
    "excludes": [
        "tkinter", "matplotlib", "PyQt5", "PyQt6", "PySide2", "PySide6",
        "test", "unittest", "distutils", "email", "html", "http", "urllib",
        "xml", "xmlrpc", "pydoc", "doctest", "argparse", "calendar",
        "concurrent", "configparser", "contextlib", "copy", "copyreg",
        "difflib", "enum", "functools", "getopt", "glob", "gzip", "hashlib",
        "hmac", "importlib", "inspect", "io", "itertools", "linecache",
        "locale", "operator", "pickle", "pkgutil", "platform", "posixpath",
        "random", "re", "shutil", "signal", "socket", "stat", "string",
        "struct", "subprocess", "sysconfig", "tempfile", "textwrap", "tokenize",
        "traceback", "types", "warnings", "weakref", "zipfile", "zlib"
    ],
    "include_files": [
        ("models/", "models/"),
        ("sample-data/", "sample-data/"),
        ("bp_estimation_dll.py", "bp_estimation_dll.py"),
        ("dll_interface.py", "dll_interface.py"),
        ("BloodPressureEstimation.h", "BloodPressureEstimation.h")
    ],
    "zip_include_packages": "*",
    "zip_exclude_packages": [],
    "silent": True,
    "optimize": 2,
    "include_msvcrt": True
}

# DLL用の実行可能ファイル設定（64bit）
dll_executable = Executable(
    "dll_interface.py",
    base=None,
    target_name="BloodPressureEstimation.dll",
    icon=None  # アイコンがある場合は指定
)

# スタンドアロン実行ファイル（64bit）
exe_executable = Executable(
    "bp_estimation_dll.py",
    base=None,
    target_name="BloodPressureEstimation.exe",
    icon=None  # アイコンがある場合は指定
)

setup(
    name="BloodPressureEstimationDLL",
    version="1.0.0",
    description="血圧推定DLL - rPPGアルゴリズムによる動画からの血圧推定（64-bit Windows対応）",
    author="IKI Japan/Yazaki",
    options={"build_exe": build_exe_options},
    executables=[dll_executable, exe_executable]
)

print("""
=== Windows 64-bit DLL ビルド手順 ===

1. Windows 64-bit環境でPython 3.12をインストール
2. 必要なパッケージをインストール:
   pip install -r requirements.txt
   pip install cx_Freeze

3. DLLビルド実行:
   python build_windows_dll_64bit.py build

4. 生成物:
   - build/exe.win-amd64-3.12/BloodPressureEstimation.dll
   - build/exe.win-amd64-3.12/BloodPressureEstimation.exe
   - 依存ライブラリ（モデルファイル、DLLファイル等）

=== DLL使用方法 ===

C/C++からの呼び出し例:

```c
#include <windows.h>
#include "BloodPressureEstimation.h"

// コールバック関数
void OnBPResult(const char* request_id, int sbp, int dbp, 
                const char* csv_data, const BPErrorInfo* errors) {
    printf("血圧結果: %s - SBP:%d, DBP:%d\\n", request_id, sbp, dbp);
}

int main() {
    // DLL初期化
    if (InitializeDLL("models")) {
        printf("DLL初期化成功\\n");
        
        // 血圧解析実行
        int error_count = StartBloodPressureAnalysis(
            "test_001", 170, 70, BP_SEX_MALE, 
            "video.webm", OnBPResult
        );
        
        if (error_count == 0) {
            printf("血圧解析開始成功\\n");
        }
    }
    
    return 0;
}
```

=== 注意事項 ===

1. 64bit環境でのみ動作
2. Visual Studio 2019/2022 Build Toolsが必要
3. 依存ライブラリは自動的にバンドルされる
4. モデルファイルは別途配布が必要
""") 
