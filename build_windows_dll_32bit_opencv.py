"""
Windows 32-bit DLLビルドスクリプト (OpenCV DNN版)
cx_Freezeを使用してOpenCV DNN版PythonコードをWindows 32bit DLLとしてコンパイル
"""

from cx_Freeze import setup, Executable
import sys
import os

# ビルドオプション (OpenCV DNN版)
build_exe_options = {
    "packages": [
        "cv2", "numpy", "scipy", "sklearn", "pandas", 
        "joblib", "pywt", "threading", "ctypes", "logging", "collections",
        "datetime", "time", "json", "pathlib", "re"
    ],
    "excludes": [
        "mediapipe",  # MediaPipeを除外
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
        ("opencv_face_detector_uint8.pb", "opencv_face_detector_uint8.pb"),
        ("opencv_face_detector.pbtxt", "opencv_face_detector.pbtxt"),
        ("sample-data/", "sample-data/"),
        ("bp_estimation_dll.py", "bp_estimation_dll.py"),
        ("dll_interface.py", "dll_interface.py")
    ],
    "zip_include_packages": "*",
    "zip_exclude_packages": [],
    "silent": True,
    "optimize": 2,
    "include_msvcrt": True,
    "build_exe": "build/exe.win32-3.12"  # 32bit指定
}

# DLL用の実行可能ファイル設定（32bit）
dll_executable = Executable(
    "dll_interface.py",
    base=None,
    target_name="BloodPressureEstimation32.dll",
    icon=None
)

# スタンドアロン実行ファイル（32bit）
exe_executable = Executable(
    "bp_estimation_dll.py",
    base=None,
    target_name="BloodPressureEstimation32.exe",
    icon=None
)

setup(
    name="BloodPressureEstimationDLL32",
    version="1.0.0",
    description="血圧推定DLL - OpenCV DNN版による32bit Windows対応",
    author="IKI Japan/Yazaki",
    options={"build_exe": build_exe_options},
    executables=[dll_executable, exe_executable]
)

print("""
=== Windows 32-bit DLL ビルド手順 (OpenCV DNN版) ===

1. Windows 32-bit環境でPython 3.12 (32bit版) をインストール
2. 必要なパッケージをインストール:
   pip install opencv-python numpy scipy scikit-learn pandas joblib pywavelets
   pip install cx_Freeze

3. OpenCV DNN モデルファイルの準備:
   - opencv_face_detector_uint8.pb
   - opencv_face_detector.pbtxt

4. DLLビルド実行:
   python build_windows_dll_32bit_opencv.py build

5. 生成物:
   - build/exe.win32-3.12/BloodPressureEstimation32.dll
   - build/exe.win32-3.12/BloodPressureEstimation32.exe
   - 依存ライブラリ（OpenCV DNN、機械学習モデル等）

=== 配布ファイル構成 ===

BloodPressureEstimation32_DLL/
├── BloodPressureEstimation32.dll        # メイン32bit DLL
├── BloodPressureEstimation.h            # C/C++ヘッダーファイル
├── opencv_face_detector_uint8.pb        # OpenCV DNN顔検出モデル
├── opencv_face_detector.pbtxt           # OpenCV DNN設定ファイル
├── models/                              # 機械学習モデル
│   ├── model_sbp.pkl
│   └── model_dbp.pkl
├── python312.dll                       # Python ランタイム
├── opencv_world4xx.dll                 # OpenCV DLL
└── 他の依存DLL

=== Visual Studio での使用方法 ===

1. プロジェクト設定:
   - プラットフォーム: Win32 (x86)
   - 文字セット: Unicode

2. インクルードパス追加:
   - BloodPressureEstimation.h のパス

3. ライブラリパス追加:
   - BloodPressureEstimation32.dll のパス

4. 実行時DLL配置:
   - BloodPressureEstimation32.dll
   - python312.dll
   - opencv_world4xx.dll
   - models/ フォルダ
   - OpenCV DNN モデルファイル

=== C++サンプルコード ===

```cpp
#include <windows.h>
#include <iostream>
#include "BloodPressureEstimation.h"

// コールバック関数
void OnBPResult(const char* request_id, int sbp, int dbp, 
                const char* csv_data, const BPErrorInfo* errors) {
    if (errors == nullptr) {
        std::cout << "血圧結果: " << request_id 
                  << " - SBP:" << sbp << ", DBP:" << dbp << std::endl;
    } else {
        std::cout << "エラー: " << errors->message << std::endl;
    }
}

int main() {
    // DLL初期化
    if (InitializeDLL("models")) {
        std::cout << "DLL初期化成功 (OpenCV DNN版)" << std::endl;
        
        // 血圧解析実行
        const char* error_code = StartBloodPressureAnalysis(
            "20250707083524932_9000000001_0000012345",
            170, 70, BP_SEX_MALE,
            "C:\\\\Videos\\\\test.webm",
            OnBPResult
        );
        
        if (error_code == nullptr) {
            std::cout << "血圧解析開始成功" << std::endl;
            
            // 処理完了まで待機
            while (true) {
                const char* status = GetBloodPressureStatus(
                    "20250707083524932_9000000001_0000012345"
                );
                if (strcmp(status, BP_STATUS_NONE) == 0) {
                    break;
                }
                Sleep(1000);
            }
        } else {
            std::cout << "エラー: " << error_code << std::endl;
        }
    } else {
        std::cout << "DLL初期化失敗" << std::endl;
    }
    
    return 0;
}
```

=== 注意事項 (OpenCV DNN版) ===

1. 32bit環境専用 (64bit環境では動作しない)
2. MediaPipeではなくOpenCV DNNを使用
3. 顔検出精度は若干劣るが、32bit対応が可能
4. OpenCV DNN モデルファイルの配布が必要
5. Visual C++ 2019/2022 Redistributable (x86) が必要

""")