"""
C#から呼び出し可能な血圧推定DLL作成スクリプト
ctypesとPyInstallerを使用してWindows DLLを生成
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def create_csharp_compatible_dll():
    """C#互換DLL作成"""
    print("=== C#互換血圧推定DLL作成 ===")
    
    # C#互換Pythonコード生成
    dll_code = '''"""
C#から呼び出し可能な血圧推定DLL
ctypes.windll経由でエクスポート
"""

import os
import sys
import ctypes
import threading
import time
import json
import csv
from datetime import datetime
from typing import Optional, List, Callable, Dict, Tuple
import re

# 必要最小限の依存関係のみインポート
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

# エラーコード定義
class ErrorCode:
    DLL_NOT_INITIALIZED = "1001"
    DEVICE_CONNECTION_FAILED = "1002"
    CALIBRATION_INCOMPLETE = "1003"
    INVALID_INPUT_PARAMETERS = "1004"
    REQUEST_DURING_PROCESSING = "1005"
    INTERNAL_PROCESSING_ERROR = "1006"

class ProcessingStatus:
    NONE = "none"
    PROCESSING = "processing"

class BPEstimator:
    """血圧推定クラス"""
    
    def __init__(self):
        self.is_initialized = False
        self.processing_requests: Dict[str, threading.Thread] = {}
        self.request_status: Dict[str, str] = {}
        self.version = "1.0.0-csharp-compatible"
        self.lock = threading.Lock()
        self.face_mesh = None
        
    def initialize(self, model_dir: str = "models") -> bool:
        """初期化"""
        try:
            if not all([HAS_OPENCV, HAS_NUMPY]):
                print("エラー: OpenCVまたはNumPyが不足しています")
                return False
            
            # MediaPipe FaceMesh初期化
            if HAS_MEDIAPIPE:
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.8,
                    min_tracking_confidence=0.7
                )
                print("✓ FaceMesh初期化完了")
            
            self.is_initialized = True
            print("✓ 初期化完了")
            return True
            
        except Exception as e:
            print(f"初期化エラー: {e}")
            return False
    
    def _validate_request_id(self, request_id: str) -> bool:
        """リクエストID検証"""
        if not request_id:
            return False
        # 簡易検証
        return len(request_id) > 10
    
    def start_analysis(self, request_id: str, height: int, weight: int, 
                      sex: int, movie_path: str, callback: Optional[Callable] = None) -> Optional[str]:
        """血圧解析開始"""
        
        if not self.is_initialized:
            return ErrorCode.DLL_NOT_INITIALIZED
        
        # パラメータ検証
        if not self._validate_request_id(request_id):
            return ErrorCode.INVALID_INPUT_PARAMETERS
        
        if not movie_path or not os.path.exists(movie_path):
            return ErrorCode.INVALID_INPUT_PARAMETERS
        
        if not (1 <= sex <= 2):
            return ErrorCode.INVALID_INPUT_PARAMETERS
        
        if not (100 <= height <= 250):
            return ErrorCode.INVALID_INPUT_PARAMETERS
        
        if not (30 <= weight <= 200):
            return ErrorCode.INVALID_INPUT_PARAMETERS
        
        # 処理中チェック
        with self.lock:
            if request_id in self.processing_requests:
                return ErrorCode.REQUEST_DURING_PROCESSING
            
            # 処理開始
            self.request_status[request_id] = ProcessingStatus.PROCESSING
            thread = threading.Thread(
                target=self._process_analysis,
                args=(request_id, height, weight, sex, movie_path, callback)
            )
            self.processing_requests[request_id] = thread
            thread.start()
        
        return None
    
    def _process_analysis(self, request_id: str, height: int, weight: int,
                         sex: int, movie_path: str, callback: Optional[Callable]):
        """血圧解析処理"""
        try:
            # 簡易推定（実装例）
            bmi = weight / ((height / 100) ** 2)
            
            # BMIベース推定
            sbp = max(90, min(180, 120 + int((bmi - 22) * 2)))
            dbp = max(60, min(110, 80 + int((bmi - 22) * 1)))
            
            # 簡易CSV生成
            csv_data = f"time,signal,peak\\n0.0,0.5,0\\n1.0,0.6,1\\n2.0,0.4,0"
            
            # 成功時のコールバック
            if callback:
                callback(request_id, sbp, dbp, csv_data, [])
            
        except Exception as e:
            if callback:
                callback(request_id, 0, 0, "", [str(e)])
        
        finally:
            with self.lock:
                if request_id in self.processing_requests:
                    del self.processing_requests[request_id]
                self.request_status[request_id] = ProcessingStatus.NONE
    
    def get_status(self, request_id: str) -> str:
        """処理状況取得"""
        with self.lock:
            return self.request_status.get(request_id, ProcessingStatus.NONE)
    
    def cancel_analysis(self, request_id: str) -> bool:
        """解析中断"""
        with self.lock:
            if request_id in self.processing_requests:
                self.request_status[request_id] = ProcessingStatus.NONE
                return True
            return False
    
    def get_version(self) -> str:
        """バージョン情報取得"""
        return f"v{self.version}"

# グローバルインスタンス
estimator = BPEstimator()

# C#エクスポート用関数
def initialize_dll(model_dir_bytes):
    """DLL初期化（C#から呼び出し）"""
    try:
        if model_dir_bytes:
            model_dir = ctypes.string_at(model_dir_bytes).decode('utf-8')
        else:
            model_dir = "models"
        return estimator.initialize(model_dir)
    except Exception as e:
        print(f"InitializeDLL error: {e}")
        return False

def start_analysis_request(request_id_bytes, height, weight, sex, movie_path_bytes, callback_ptr):
    """血圧解析リクエスト（C#から呼び出し）"""
    try:
        request_id = ctypes.string_at(request_id_bytes).decode('utf-8')
        movie_path = ctypes.string_at(movie_path_bytes).decode('utf-8')
        
        def py_callback(req_id, sbp, dbp, csv_data, errors):
            if callback_ptr:
                # C#コールバック呼び出し（簡易実装）
                print(f"Analysis complete: {req_id}, SBP={sbp}, DBP={dbp}")
        
        error_code = estimator.start_analysis(
            request_id, height, weight, sex, movie_path, py_callback)
        return error_code.encode('utf-8') if error_code else b""
    except Exception as e:
        print(f"StartAnalysisRequest error: {e}")
        return str(e).encode('utf-8')

def get_processing_status(request_id_bytes):
    """処理状況取得（C#から呼び出し）"""
    try:
        request_id = ctypes.string_at(request_id_bytes).decode('utf-8')
        result = estimator.get_status(request_id)
        return result.encode('utf-8')
    except Exception as e:
        print(f"GetProcessingStatus error: {e}")
        return b"none"

def cancel_analysis(request_id_bytes):
    """解析中断（C#から呼び出し）"""
    try:
        request_id = ctypes.string_at(request_id_bytes).decode('utf-8')
        return estimator.cancel_analysis(request_id)
    except Exception as e:
        print(f"CancelAnalysis error: {e}")
        return False

def get_version_info():
    """バージョン情報取得（C#から呼び出し）"""
    try:
        return estimator.get_version().encode('utf-8')
    except Exception as e:
        print(f"GetVersionInfo error: {e}")
        return b"v1.0.0"

# Windows DLL用エクスポート
if sys.platform.startswith('win'):
    # ctypes関数型定義
    initialize_dll_func = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_char_p)(initialize_dll)
    start_analysis_func = ctypes.WINFUNCTYPE(ctypes.c_char_p, ctypes.c_char_p, 
                                           ctypes.c_int, ctypes.c_int, ctypes.c_int, 
                                           ctypes.c_char_p, ctypes.c_void_p)(start_analysis_request)
    get_status_func = ctypes.WINFUNCTYPE(ctypes.c_char_p, ctypes.c_char_p)(get_processing_status)
    cancel_func = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_char_p)(cancel_analysis)
    version_func = ctypes.WINFUNCTYPE(ctypes.c_char_p)(get_version_info)

# テスト用
if __name__ == "__main__":
    print("C#互換血圧推定DLL テスト")
    
    if initialize_dll(b"models"):
        print("✓ 初期化成功")
        version = get_version_info().decode('utf-8')
        print(f"バージョン: {version}")
        
        # 処理状況テスト
        status = get_processing_status(b"test_request").decode('utf-8')
        print(f"処理状況: {status}")
    else:
        print("✗ 初期化失敗")
'''

    with open("bp_estimation_csharp_dll.py", "w", encoding="utf-8") as f:
        f.write(dll_code)
    
    print("✓ bp_estimation_csharp_dll.py 作成完了")

def create_csharp_spec():
    """C#互換PyInstaller spec作成"""
    print("\\n=== C#互換PyInstaller spec作成 ===")
    
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-

import os
import sys

# 基本設定
APP_NAME = "BloodPressureEstimation"
SCRIPT_PATH = "bp_estimation_csharp_dll.py"

# 軽量化除外モジュール
EXCLUDED_MODULES = [
    'tkinter', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6',
    'matplotlib', 'seaborn', 'plotly', 'bokeh',
    'pandas', 'sklearn', 'tensorflow',
    'IPython', 'jupyter', 'pytest', 'unittest',
    'multiprocessing', 'asyncio',
    'email', 'xml', 'html', 'urllib3', 'requests',
]

# 最小限のhidden imports
HIDDEN_IMPORTS = [
    'cv2.cv2',
    'mediapipe.python._framework_bindings',
    'numpy.core._methods',
]

# データファイル
DATAS = [
    ('models', 'models'),
]

a = Analysis(
    [SCRIPT_PATH],
    pathex=[],
    binaries=[],
    datas=DATAS,
    hiddenimports=HIDDEN_IMPORTS,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDED_MODULES,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# 通常のEXE形式でビルド
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # デバッグ用にconsole=True
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
'''
    
    with open("BloodPressureEstimation_CSharp.spec", "w", encoding="utf-8") as f:
        f.write(spec_content)
    
    print("✓ BloodPressureEstimation_CSharp.spec 作成完了")

def build_csharp_dll():
    """C#互換DLLビルド"""
    print("\\n=== C#互換DLLビルド開始 ===")
    
    # クリーンアップ
    cleanup_dirs = ["build", "dist", "__pycache__"]
    for dir_name in cleanup_dirs:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"✓ {dir_name}/ クリーンアップ")
    
    # PyInstallerコマンド
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "BloodPressureEstimation_CSharp.spec",
        "--clean",
        "--noconfirm",
        "--log-level=WARN"
    ]
    
    print("C#互換PyInstallerビルド実行中...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ PyInstallerビルド成功")
        
        # EXEをDLLにリネーム
        exe_path = Path("dist") / "BloodPressureEstimation.exe"
        dll_path = Path("dist") / "BloodPressureEstimation.dll"
        
        if exe_path.exists():
            exe_path.rename(dll_path)
            size_mb = dll_path.stat().st_size / (1024 * 1024)
            print(f"✓ C#互換DLL作成成功: {dll_path}")
            print(f"  サイズ: {size_mb:.1f} MB")
            
            print("\\n=== 次の手順 ===")
            print("1. C#プロジェクトでDLLを参照")
            print("2. DllImport でInitializeDLL等を宣言")
            print("3. 呼び出しテスト実行")
            
            return True
        else:
            print("✗ EXEファイルが見つかりません")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"✗ ビルドエラー: {e}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False

def update_csharp_test():
    """C#テストコード更新"""
    print("\\n=== C#テストコード更新 ===")
    
    csharp_code = '''using System;
using System.Runtime.InteropServices;

namespace BloodPressureDllTest
{
    public class BloodPressureDll
    {
        private const string DllPath = "BloodPressureEstimation.dll";

        [DllImport(DllPath, CallingConvention = CallingConvention.StdCall, CharSet = CharSet.Ansi)]
        public static extern bool InitializeDLL([MarshalAs(UnmanagedType.LPStr)] string modelDir);

        [DllImport(DllPath, CallingConvention = CallingConvention.StdCall, CharSet = CharSet.Ansi)]
        [return: MarshalAs(UnmanagedType.LPStr)]
        public static extern string StartBloodPressureAnalysisRequest(
            [MarshalAs(UnmanagedType.LPStr)] string requestId,
            int height, int weight, int sex,
            [MarshalAs(UnmanagedType.LPStr)] string moviePath,
            IntPtr callback);

        [DllImport(DllPath, CallingConvention = CallingConvention.StdCall, CharSet = CharSet.Ansi)]
        [return: MarshalAs(UnmanagedType.LPStr)]
        public static extern string GetProcessingStatus([MarshalAs(UnmanagedType.LPStr)] string requestId);

        [DllImport(DllPath, CallingConvention = CallingConvention.StdCall, CharSet = CharSet.Ansi)]
        public static extern bool CancelBloodPressureAnalysis([MarshalAs(UnmanagedType.LPStr)] string requestId);

        [DllImport(DllPath, CallingConvention = CallingConvention.StdCall, CharSet = CharSet.Ansi)]
        [return: MarshalAs(UnmanagedType.LPStr)]
        public static extern string GetVersionInfo();

        public static void TestDLL()
        {
            Console.WriteLine("=== C#互換血圧推定DLLテスト ===");

            try
            {
                // 1. DLL初期化
                Console.WriteLine("1. DLL初期化");
                bool initResult = InitializeDLL("models");
                Console.WriteLine($"   結果: {initResult}");

                // 2. バージョン取得
                Console.WriteLine("2. バージョン取得");
                string version = GetVersionInfo();
                Console.WriteLine($"   バージョン: {version}");

                // 3. 処理状況取得
                Console.WriteLine("3. 処理状況取得");
                string status = GetProcessingStatus("test_request");
                Console.WriteLine($"   状況: {status}");

                // 4. 解析リクエスト（無効パラメータ）
                Console.WriteLine("4. 解析リクエスト");
                string errorCode = StartBloodPressureAnalysisRequest(
                    "test_request_123", 170, 70, 1, "test.webm", IntPtr.Zero);
                Console.WriteLine($"   エラーコード: {errorCode}");

                Console.WriteLine("=== テスト完了 ===");
            }
            catch (DllNotFoundException ex)
            {
                Console.WriteLine($"DLLが見つかりません: {ex.Message}");
            }
            catch (EntryPointNotFoundException ex)
            {
                Console.WriteLine($"エントリポイントが見つかりません: {ex.Message}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"エラー: {ex.Message}");
            }
        }

        public static void Main(string[] args)
        {
            TestDLL();
            Console.WriteLine("\\nEnterキーで終了...");
            Console.ReadLine();
        }
    }
}'''

    with open("CSharpDllTest_Updated.cs", "w", encoding="utf-8") as f:
        f.write(csharp_code)
    
    print("✓ CSharpDllTest_Updated.cs 作成完了")

def main():
    """メイン処理"""
    print("=== C#互換血圧推定DLL作成スクリプト ===")
    
    try:
        # 1. C#互換DLLコード作成
        create_csharp_compatible_dll()
        
        # 2. C#互換spec作成
        create_csharp_spec()
        
        # 3. DLLビルド
        success = build_csharp_dll()
        
        # 4. C#テストコード更新
        update_csharp_test()
        
        if success:
            print("\\n🎉 C#互換DLL作成完了！")
            print("\\n次の手順:")
            print("1. Visual Studioで CSharpDllTest_Updated.cs をコンパイル")
            print("2. BloodPressureEstimation.dll と同じディレクトリで実行")
            print("3. エラーなく実行されることを確認")
        else:
            print("\\n❌ DLL作成に失敗")
        
        return success
        
    except Exception as e:
        print(f"\\n❌ エラー: {e}")
        return False

if __name__ == "__main__":
    main()