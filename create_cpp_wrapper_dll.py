"""
C/C++ラッパーDLL作成スクリプト
C#から呼び出し可能なDLLをC++で作成し、内部でPythonエンジンを呼び出す
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def create_cpp_wrapper():
    """C++ラッパーDLL作成"""
    print("=== C++ラッパーDLL作成 ===")
    
    # C++ヘッダーファイル
    header_code = '''#pragma once

#ifdef BLOODPRESSURE_EXPORTS
#define BLOODPRESSURE_API __declspec(dllexport)
#else
#define BLOODPRESSURE_API __declspec(dllimport)
#endif

extern "C" {
    // コールバック関数型定義
    typedef void(*AnalysisCallback)(const char* requestId, int maxBloodPressure, 
                                   int minBloodPressure, const char* measureRowData, 
                                   const char* errors);

    // エクスポート関数
    BLOODPRESSURE_API bool InitializeDLL(const char* modelDir);
    BLOODPRESSURE_API const char* StartBloodPressureAnalysisRequest(
        const char* requestId, int height, int weight, int sex, 
        const char* moviePath, AnalysisCallback callback);
    BLOODPRESSURE_API const char* GetProcessingStatus(const char* requestId);
    BLOODPRESSURE_API bool CancelBloodPressureAnalysis(const char* requestId);
    BLOODPRESSURE_API const char* GetVersionInfo();
}
'''

    with open("BloodPressureEstimation.h", "w", encoding="utf-8") as f:
        f.write(header_code)
    
    # C++実装ファイル
    cpp_code = '''#include "BloodPressureEstimation.h"
#include <Python.h>
#include <string>
#include <map>
#include <memory>
#include <iostream>

// グローバル変数
static bool g_initialized = false;
static PyObject* g_bp_module = nullptr;
static PyObject* g_estimator = nullptr;
static std::map<std::string, std::string> g_status_cache;

// 文字列バッファ（戻り値用）
static std::string g_last_error;
static std::string g_last_status;
static std::string g_version_info;

// Pythonエンジン初期化
bool InitializePython() {
    if (Py_IsInitialized()) {
        return true;
    }

    try {
        // Pythonインタープリター初期化
        Py_Initialize();
        if (!Py_IsInitialized()) {
            g_last_error = "Python initialization failed";
            return false;
        }

        // sys.pathにカレントディレクトリ追加
        PyRun_SimpleString("import sys");
        PyRun_SimpleString("sys.path.append('.')");
        
        // 血圧推定モジュールインポート
        g_bp_module = PyImport_ImportModule("bp_estimation_simple");
        if (!g_bp_module) {
            PyErr_Print();
            g_last_error = "Failed to import bp_estimation_simple module";
            return false;
        }

        // エスティメーターインスタンス取得
        PyObject* estimator_class = PyObject_GetAttrString(g_bp_module, "BPEstimator");
        if (!estimator_class) {
            g_last_error = "Failed to get BPEstimator class";
            return false;
        }

        g_estimator = PyObject_CallObject(estimator_class, nullptr);
        Py_DECREF(estimator_class);
        
        if (!g_estimator) {
            g_last_error = "Failed to create BPEstimator instance";
            return false;
        }

        std::cout << "Python engine initialized successfully" << std::endl;
        return true;
    }
    catch (...) {
        g_last_error = "Exception during Python initialization";
        return false;
    }
}

// Python関数呼び出しヘルパー
PyObject* CallPythonMethod(const char* method_name, PyObject* args = nullptr) {
    if (!g_estimator) {
        return nullptr;
    }

    PyObject* method = PyObject_GetAttrString(g_estimator, method_name);
    if (!method) {
        return nullptr;
    }

    PyObject* result = PyObject_CallObject(method, args);
    Py_DECREF(method);
    
    return result;
}

// エクスポート関数実装
extern "C" {

BLOODPRESSURE_API bool InitializeDLL(const char* modelDir) {
    try {
        if (g_initialized) {
            return true;
        }

        // Pythonエンジン初期化
        if (!InitializePython()) {
            return false;
        }

        // Pythonエスティメーター初期化
        PyObject* args = PyTuple_New(1);
        PyTuple_SetItem(args, 0, PyUnicode_FromString(modelDir ? modelDir : "models"));
        
        PyObject* result = CallPythonMethod("initialize", args);
        Py_DECREF(args);
        
        if (result && PyBool_Check(result)) {
            g_initialized = (result == Py_True);
            Py_DECREF(result);
        } else {
            g_initialized = false;
        }

        if (g_initialized) {
            std::cout << "DLL initialized successfully" << std::endl;
        }

        return g_initialized;
    }
    catch (...) {
        g_last_error = "Exception in InitializeDLL";
        return false;
    }
}

BLOODPRESSURE_API const char* StartBloodPressureAnalysisRequest(
    const char* requestId, int height, int weight, int sex, 
    const char* moviePath, AnalysisCallback callback) {
    
    try {
        if (!g_initialized) {
            g_last_error = "1001"; // DLL_NOT_INITIALIZED
            return g_last_error.c_str();
        }

        // Python関数呼び出し
        PyObject* args = PyTuple_New(5);
        PyTuple_SetItem(args, 0, PyUnicode_FromString(requestId));
        PyTuple_SetItem(args, 1, PyLong_FromLong(height));
        PyTuple_SetItem(args, 2, PyLong_FromLong(weight));
        PyTuple_SetItem(args, 3, PyLong_FromLong(sex));
        PyTuple_SetItem(args, 4, PyUnicode_FromString(moviePath));

        PyObject* result = CallPythonMethod("start_analysis", args);
        Py_DECREF(args);

        if (result) {
            if (result == Py_None) {
                g_last_error = ""; // 成功
            } else if (PyUnicode_Check(result)) {
                const char* error_str = PyUnicode_AsUTF8(result);
                g_last_error = error_str ? error_str : "Unknown error";
            }
            Py_DECREF(result);
        } else {
            g_last_error = "1006"; // INTERNAL_PROCESSING_ERROR
        }

        return g_last_error.empty() ? nullptr : g_last_error.c_str();
    }
    catch (...) {
        g_last_error = "1006";
        return g_last_error.c_str();
    }
}

BLOODPRESSURE_API const char* GetProcessingStatus(const char* requestId) {
    try {
        if (!g_initialized) {
            g_last_status = "none";
            return g_last_status.c_str();
        }

        PyObject* args = PyTuple_New(1);
        PyTuple_SetItem(args, 0, PyUnicode_FromString(requestId));

        PyObject* result = CallPythonMethod("get_status", args);
        Py_DECREF(args);

        if (result && PyUnicode_Check(result)) {
            const char* status_str = PyUnicode_AsUTF8(result);
            g_last_status = status_str ? status_str : "none";
            Py_DECREF(result);
        } else {
            g_last_status = "none";
        }

        return g_last_status.c_str();
    }
    catch (...) {
        g_last_status = "none";
        return g_last_status.c_str();
    }
}

BLOODPRESSURE_API bool CancelBloodPressureAnalysis(const char* requestId) {
    try {
        if (!g_initialized) {
            return false;
        }

        PyObject* args = PyTuple_New(1);
        PyTuple_SetItem(args, 0, PyUnicode_FromString(requestId));

        PyObject* result = CallPythonMethod("cancel_analysis", args);
        Py_DECREF(args);

        bool cancelled = false;
        if (result && PyBool_Check(result)) {
            cancelled = (result == Py_True);
            Py_DECREF(result);
        }

        return cancelled;
    }
    catch (...) {
        return false;
    }
}

BLOODPRESSURE_API const char* GetVersionInfo() {
    try {
        if (g_version_info.empty()) {
            if (g_initialized && g_estimator) {
                PyObject* result = CallPythonMethod("get_version");
                if (result && PyUnicode_Check(result)) {
                    const char* version_str = PyUnicode_AsUTF8(result);
                    g_version_info = version_str ? version_str : "v1.0.0-cpp-wrapper";
                    Py_DECREF(result);
                }
            }
            
            if (g_version_info.empty()) {
                g_version_info = "v1.0.0-cpp-wrapper";
            }
        }

        return g_version_info.c_str();
    }
    catch (...) {
        g_version_info = "v1.0.0-cpp-wrapper";
        return g_version_info.c_str();
    }
}

} // extern "C"

// DLLエントリポイント
BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) {
    switch (ul_reason_for_call) {
    case DLL_PROCESS_ATTACH:
        std::cout << "BloodPressure DLL attached" << std::endl;
        break;
    case DLL_PROCESS_DETACH:
        if (g_estimator) {
            Py_DECREF(g_estimator);
            g_estimator = nullptr;
        }
        if (g_bp_module) {
            Py_DECREF(g_bp_module);
            g_bp_module = nullptr;
        }
        if (Py_IsInitialized()) {
            Py_Finalize();
        }
        std::cout << "BloodPressure DLL detached" << std::endl;
        break;
    }
    return TRUE;
}
'''

    with open("BloodPressureEstimation.cpp", "w", encoding="utf-8") as f:
        f.write(cpp_code)
    
    print("✓ C++ヘッダー・実装ファイル作成完了")

def create_simple_python_module():
    """シンプルなPythonモジュール作成"""
    print("\\n=== シンプルPythonモジュール作成 ===")
    
    python_code = '''"""
血圧推定用シンプルPythonモジュール
C++ラッパーから呼び出される
"""

import os
import threading
import time
from typing import Dict, Optional

class BPEstimator:
    """血圧推定クラス"""
    
    def __init__(self):
        self.is_initialized = False
        self.processing_requests: Dict[str, str] = {}
        self.version = "1.0.0-cpp-wrapper"
        self.lock = threading.Lock()
        
    def initialize(self, model_dir: str = "models") -> bool:
        """初期化"""
        try:
            print(f"Initializing BP estimator with model_dir: {model_dir}")
            # 実際の初期化処理はここに実装
            self.is_initialized = True
            return True
        except Exception as e:
            print(f"Initialization error: {e}")
            return False
    
    def start_analysis(self, request_id: str, height: int, weight: int, 
                      sex: int, movie_path: str) -> Optional[str]:
        """血圧解析開始"""
        try:
            if not self.is_initialized:
                return "1001"  # DLL_NOT_INITIALIZED
            
            # パラメータ検証
            if not request_id or len(request_id) < 10:
                return "1004"  # INVALID_INPUT_PARAMETERS
                
            if not movie_path or not os.path.exists(movie_path):
                return "1004"  # INVALID_INPUT_PARAMETERS
                
            if not (1 <= sex <= 2):
                return "1004"  # INVALID_INPUT_PARAMETERS
                
            if not (100 <= height <= 250):
                return "1004"  # INVALID_INPUT_PARAMETERS
                
            if not (30 <= weight <= 200):
                return "1004"  # INVALID_INPUT_PARAMETERS
            
            # 処理中チェック
            with self.lock:
                if request_id in self.processing_requests:
                    return "1005"  # REQUEST_DURING_PROCESSING
                
                # 非同期処理開始
                self.processing_requests[request_id] = "processing"
                thread = threading.Thread(
                    target=self._process_analysis,
                    args=(request_id, height, weight, sex, movie_path)
                )
                thread.start()
            
            return None  # 成功
            
        except Exception as e:
            print(f"Analysis start error: {e}")
            return "1006"  # INTERNAL_PROCESSING_ERROR
    
    def _process_analysis(self, request_id: str, height: int, weight: int,
                         sex: int, movie_path: str):
        """血圧解析処理"""
        try:
            print(f"Processing analysis for request: {request_id}")
            
            # 簡易血圧計算
            bmi = weight / ((height / 100) ** 2)
            
            # BMIベース推定
            sbp = max(90, min(180, 120 + int((bmi - 22) * 2)))
            dbp = max(60, min(110, 80 + int((bmi - 22) * 1)))
            
            # 処理時間シミュレート
            time.sleep(2)
            
            print(f"Analysis complete: {request_id}, SBP={sbp}, DBP={dbp}")
            
        except Exception as e:
            print(f"Analysis processing error: {e}")
        finally:
            with self.lock:
                self.processing_requests[request_id] = "none"
    
    def get_status(self, request_id: str) -> str:
        """処理状況取得"""
        with self.lock:
            return self.processing_requests.get(request_id, "none")
    
    def cancel_analysis(self, request_id: str) -> bool:
        """解析中断"""
        with self.lock:
            if request_id in self.processing_requests:
                self.processing_requests[request_id] = "none"
                return True
            return False
    
    def get_version(self) -> str:
        """バージョン情報取得"""
        return f"v{self.version}"

# テスト用
if __name__ == "__main__":
    estimator = BPEstimator()
    
    if estimator.initialize():
        print("✓ 初期化成功")
        print(f"バージョン: {estimator.get_version()}")
        
        # テスト解析
        result = estimator.start_analysis("test_123", 170, 70, 1, "test.webm")
        if result:
            print(f"エラーコード: {result}")
        else:
            print("解析開始成功")
    else:
        print("✗ 初期化失敗")
'''

    with open("bp_estimation_simple.py", "w", encoding="utf-8") as f:
        f.write(python_code)
    
    print("✓ bp_estimation_simple.py 作成完了")

def create_def_file():
    """DEFファイル作成"""
    print("\\n=== DEFファイル作成 ===")
    
    def_code = '''EXPORTS
InitializeDLL
StartBloodPressureAnalysisRequest
GetProcessingStatus
CancelBloodPressureAnalysis
GetVersionInfo'''

    with open("BloodPressureEstimation.def", "w", encoding="utf-8") as f:
        f.write(def_code)
    
    print("✓ BloodPressureEstimation.def 作成完了")

def create_build_script():
    """ビルドスクリプト作成"""
    print("\\n=== ビルドスクリプト作成 ===")
    
    # CMakeファイル
    cmake_code = '''cmake_minimum_required(VERSION 3.16)
project(BloodPressureEstimation)

set(CMAKE_CXX_STANDARD 17)

# Python検索
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# インクルードディレクトリ
include_directories(${Python3_INCLUDE_DIRS})

# ソースファイル
set(SOURCES
    BloodPressureEstimation.cpp
)

# DLL作成
add_library(BloodPressureEstimation SHARED ${SOURCES})

# Python DLLリンク
target_link_libraries(BloodPressureEstimation ${Python3_LIBRARIES})

# DEFファイル使用
set_target_properties(BloodPressureEstimation PROPERTIES
    LINK_FLAGS "/DEF:${CMAKE_CURRENT_SOURCE_DIR}/BloodPressureEstimation.def"
)

# プリプロセッサ定義
target_compile_definitions(BloodPressureEstimation PRIVATE BLOODPRESSURE_EXPORTS)

# 出力ディレクトリ
set_target_properties(BloodPressureEstimation PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/dist
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/dist
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/dist
)
'''

    with open("CMakeLists.txt", "w", encoding="utf-8") as f:
        f.write(cmake_code)
    
    # バッチファイル
    batch_code = '''@echo off
echo === C++ Wrapper DLL Build ===

REM Visual Studio環境設定
call "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat"

REM ビルドディレクトリ作成
if exist build rmdir /s /q build
mkdir build
cd build

REM CMake設定
cmake .. -G "Visual Studio 17 2022" -A x64

REM ビルド実行
cmake --build . --config Release

REM 結果確認
if exist dist\\BloodPressureEstimation.dll (
    echo ✓ DLL作成成功
    echo エクスポート関数確認:
    dumpbin /exports dist\\BloodPressureEstimation.dll
) else (
    echo ✗ DLL作成失敗
)

cd ..
pause
'''

    with open("build_cpp_dll.bat", "w", encoding="utf-8") as f:
        f.write(batch_code)
    
    print("✓ CMakeLists.txt と build_cpp_dll.bat 作成完了")

def create_csharp_test():
    """C#テストコード作成"""
    print("\\n=== C#テストコード作成 ===")
    
    csharp_code = '''using System;
using System.Runtime.InteropServices;

namespace BloodPressureDllTest
{
    public class BloodPressureDll
    {
        private const string DllPath = "BloodPressureEstimation.dll";

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void AnalysisCallback(
            [MarshalAs(UnmanagedType.LPStr)] string requestId,
            int maxBloodPressure,
            int minBloodPressure,
            [MarshalAs(UnmanagedType.LPStr)] string measureRowData,
            [MarshalAs(UnmanagedType.LPStr)] string errors
        );

        [DllImport(DllPath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern bool InitializeDLL([MarshalAs(UnmanagedType.LPStr)] string modelDir);

        [DllImport(DllPath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        [return: MarshalAs(UnmanagedType.LPStr)]
        public static extern string StartBloodPressureAnalysisRequest(
            [MarshalAs(UnmanagedType.LPStr)] string requestId,
            int height, int weight, int sex,
            [MarshalAs(UnmanagedType.LPStr)] string moviePath,
            AnalysisCallback callback);

        [DllImport(DllPath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        [return: MarshalAs(UnmanagedType.LPStr)]
        public static extern string GetProcessingStatus([MarshalAs(UnmanagedType.LPStr)] string requestId);

        [DllImport(DllPath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern bool CancelBloodPressureAnalysis([MarshalAs(UnmanagedType.LPStr)] string requestId);

        [DllImport(DllPath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        [return: MarshalAs(UnmanagedType.LPStr)]
        public static extern string GetVersionInfo();

        public static void TestCppWrapperDLL()
        {
            Console.WriteLine("=== C++ Wrapper DLL テスト ===");

            try
            {
                // 1. DLL初期化
                Console.WriteLine("1. DLL初期化");
                bool initResult = InitializeDLL("models");
                Console.WriteLine($"   結果: {initResult}");

                if (!initResult)
                {
                    Console.WriteLine("DLL初期化に失敗しました");
                    return;
                }

                // 2. バージョン取得
                Console.WriteLine("2. バージョン取得");
                string version = GetVersionInfo();
                Console.WriteLine($"   バージョン: {version}");

                // 3. 処理状況取得
                Console.WriteLine("3. 処理状況取得");
                string status = GetProcessingStatus("test_request");
                Console.WriteLine($"   状況: {status}");

                // 4. 解析リクエスト（無効パラメータ）
                Console.WriteLine("4. 解析リクエスト（無効パラメータ）");
                AnalysisCallback callback = (reqId, sbp, dbp, csvData, errors) =>
                {
                    Console.WriteLine($"   コールバック: {reqId}, SBP={sbp}, DBP={dbp}");
                };

                string errorCode = StartBloodPressureAnalysisRequest(
                    "invalid_id", 170, 70, 1, "test.webm", callback);
                Console.WriteLine($"   エラーコード: {errorCode}");

                // 5. 有効なリクエスト
                Console.WriteLine("5. 有効な解析リクエスト");
                string requestId = $"{DateTime.Now:yyyyMMddHHmmssfff}_1234567890_0987654321";
                
                // ダミーファイル作成
                System.IO.File.WriteAllText("test_video.webm", "dummy");
                
                errorCode = StartBloodPressureAnalysisRequest(
                    requestId, 170, 70, 1, "test_video.webm", callback);
                
                if (string.IsNullOrEmpty(errorCode))
                {
                    Console.WriteLine("   解析開始成功");
                    
                    // 状況確認
                    System.Threading.Thread.Sleep(1000);
                    status = GetProcessingStatus(requestId);
                    Console.WriteLine($"   処理状況: {status}");
                    
                    // 完了待ち
                    System.Threading.Thread.Sleep(3000);
                    status = GetProcessingStatus(requestId);
                    Console.WriteLine($"   最終状況: {status}");
                }
                else
                {
                    Console.WriteLine($"   エラーコード: {errorCode}");
                }

                Console.WriteLine("=== テスト完了 ===");
            }
            catch (DllNotFoundException ex)
            {
                Console.WriteLine($"DLLが見つかりません: {ex.Message}");
                Console.WriteLine("build\\dist\\BloodPressureEstimation.dll が存在することを確認してください");
            }
            catch (EntryPointNotFoundException ex)
            {
                Console.WriteLine($"エントリポイントが見つかりません: {ex.Message}");
                Console.WriteLine("DLLのエクスポート関数を確認してください");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"エラー: {ex.Message}");
            }
        }

        public static void Main(string[] args)
        {
            TestCppWrapperDLL();
            Console.WriteLine("\\nEnterキーで終了...");
            Console.ReadLine();
        }
    }
}'''

    with open("CSharpCppWrapperTest.cs", "w", encoding="utf-8") as f:
        f.write(csharp_code)
    
    print("✓ CSharpCppWrapperTest.cs 作成完了")

def main():
    """メイン処理"""
    print("=== C++ラッパーDLL作成スクリプト ===")
    
    try:
        # 1. C++ラッパー作成
        create_cpp_wrapper()
        
        # 2. シンプルPythonモジュール作成
        create_simple_python_module()
        
        # 3. DEFファイル作成
        create_def_file()
        
        # 4. ビルドスクリプト作成
        create_build_script()
        
        # 5. C#テストコード作成
        create_csharp_test()
        
        print("\\n🎉 C++ラッパーDLL作成完了！")
        print("\\n次の手順:")
        print("1. build_cpp_dll.bat を実行してDLLビルド")
        print("2. csc CSharpCppWrapperTest.cs でテストコンパイル")
        print("3. build\\dist\\BloodPressureEstimation.dll をテスト実行ディレクトリにコピー")
        print("4. bp_estimation_simple.py も同じディレクトリにコピー")
        print("5. CSharpCppWrapperTest.exe 実行")
        
        print("\\n特徴:")
        print("✓ 確実なDLLエクスポート")
        print("✓ C#から直接呼び出し可能")
        print("✓ 内部でPythonエンジン実行")
        print("✓ エラーハンドリング完備")
        
        return True
        
    except Exception as e:
        print(f"\\n❌ エラー: {e}")
        return False

if __name__ == "__main__":
    main()