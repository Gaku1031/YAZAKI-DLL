"""
Windows 32-bit DLLビルドスクリプト
cx_Freezeを使用してPythonコードをWindows DLLとしてコンパイル
"""

from cx_Freeze import setup, Executable
import sys
import os

# ビルドオプション
build_exe_options = {
    "packages": [
        "cv2", "mediapipe", "numpy", "scipy", "sklearn", "pandas", 
        "joblib", "pywt", "threading", "ctypes", "logging", "collections",
        "datetime", "time"
    ],
    "excludes": [
        "tkinter", "matplotlib", "PyQt5", "PyQt6", "PySide2", "PySide6",
        "test", "unittest", "distutils", "email", "html", "http", "urllib",
        "xml", "xmlrpc"
    ],
    "include_files": [
        ("models/", "models/"),
        ("sample-data/", "sample-data/"),
        ("bp_estimation_dll.py", "bp_estimation_dll.py"),
        ("dll_interface.py", "dll_interface.py")
    ],
    "zip_include_packages": "*",
    "zip_exclude_packages": [],
    "silent": True,
    "optimize": 2
}

# DLL用の実行可能ファイル設定
dll_executable = Executable(
    "dll_interface.py",
    base=None,
    target_name="BloodPressureEstimation.dll"
)

# スタンドアロン実行ファイル
exe_executable = Executable(
    "bp_estimation_dll.py",
    base=None,
    target_name="BloodPressureEstimation.exe"
)

setup(
    name="BloodPressureEstimationDLL",
    version="1.0.0",
    description="血圧推定DLL - rPPGアルゴリズムによる動画からの血圧推定（32-bit Windows対応）",
    author="IKI Japan/Yazaki",
    options={"build_exe": build_exe_options},
    executables=[dll_executable, exe_executable]
)

print("""
=== Windows 32-bit DLL ビルド手順 ===

1. Windows 32-bit環境でPython 3.12をインストール
2. 必要なパッケージをインストール:
   pip install -r requirements.txt
   pip install cx_Freeze

3. DLLビルド実行:
   python build_windows_dll.py build

4. 生成物:
   - build/exe.win32-3.12/BloodPressureEstimation.dll
   - build/exe.win32-3.12/BloodPressureEstimation.exe
   - 依存ライブラリ（モデルファイル、DLLファイル等）

=== DLL使用方法 ===

C/C++からの呼び出し例:

```c
#include <windows.h>

// DLL関数の型定義
typedef BOOL (*InitializeDLLFunc)(const char* model_dir);
typedef int (*StartBPAnalysisFunc)(const char* request_id, int height, int weight, 
                                   int sex, const char* movie_path, void* callback);
typedef BOOL (*CancelBPProcessingFunc)(const char* request_id);
typedef const char* (*GetBPStatusFunc)(const char* request_id);
typedef const char* (*GetDLLVersionFunc)();

// コールバック関数の型定義
typedef void (*BPCallbackFunc)(const char* request_id, int sbp, int dbp, 
                               const char* csv_data, void* errors);

int main() {
    // DLLロード
    HINSTANCE hDLL = LoadLibrary(L"BloodPressureEstimation.dll");
    if (!hDLL) return -1;
    
    // 関数取得
    InitializeDLLFunc InitializeDLL = (InitializeDLLFunc)GetProcAddress(hDLL, "InitializeDLL");
    StartBPAnalysisFunc StartBPAnalysis = (StartBPAnalysisFunc)GetProcAddress(hDLL, "StartBloodPressureAnalysis");
    // ... 他の関数も同様に取得
    
    // DLL初期化
    if (InitializeDLL("models")) {
        printf("DLL初期化成功\\n");
        
        // 血圧解析実行
        StartBPAnalysis("test_001", 170, 70, 1, "video.webm", callback_function);
    }
    
    FreeLibrary(hDLL);
    return 0;
}
```

=== DLL関数仕様 ===

1. InitializeDLL(model_dir)
   - モデルディレクトリを指定してDLLを初期化
   - 戻り値: 成功=TRUE, 失敗=FALSE

2. StartBloodPressureAnalysis(request_id, height, weight, sex, movie_path, callback)
   - 血圧解析を非同期で開始
   - 戻り値: エラー数（0=成功）

3. CancelBloodPressureProcessing(request_id)
   - 指定したリクエストIDの処理を中断
   - 戻り値: 成功=TRUE, 失敗=FALSE

4. GetBloodPressureStatus(request_id)
   - 処理状況を取得
   - 戻り値: "none" | "processing"

5. GetDLLVersion()
   - DLLバージョンを取得
   - 戻り値: バージョン文字列

=== コールバック仕様 ===
void callback(const char* request_id, int sbp, int dbp, const char* csv_data, ErrorInfo* errors)
- request_id: リクエストID
- sbp: 収縮期血圧 (mmHg)
- dbp: 拡張期血圧 (mmHg)
- csv_data: PPGローデータ (CSV形式)
- errors: エラー情報配列 (NULL=エラーなし)
""")