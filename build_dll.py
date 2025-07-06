"""
血圧推定DLLビルダー
ctypesを使用してDLLインターフェースを提供
"""

import ctypes
from ctypes import *
import os
import sys

# パスを追加してモジュールをインポート
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from bp_estimation_dll import (
    initialize_dll, start_bp_analysis, cancel_bp_processing,
    get_bp_status, get_dll_version, test_callback
)

# DLL関数の型定義
class ErrorInfo(Structure):
    _fields_ = [
        ("code", c_char_p),
        ("message", c_char_p),
        ("is_retriable", c_bool)
    ]

# コールバック関数の型定義
CallbackFunc = CFUNCTYPE(None, c_char_p, c_int, c_int, c_char_p, POINTER(ErrorInfo))

# DLL関数の定義
class BloodPressureDLL:
    def __init__(self):
        self.dll_initialized = False
    
    # DLL初期化
    @staticmethod
    def InitializeDLL(model_dir_bytes):
        """
        DLL初期化
        Args:
            model_dir_bytes: モデルディレクトリパス（bytes）
        Returns:
            bool: 初期化成功フラグ
        """
        try:
            model_dir = model_dir_bytes.decode('utf-8') if model_dir_bytes else "models"
            return initialize_dll(model_dir)
        except Exception as e:
            print(f"InitializeDLL Error: {e}")
            return False
    
    # 血圧解析開始
    @staticmethod
    def StartBloodPressureAnalysis(request_id_bytes, height, weight, sex, 
                                 movie_path_bytes, callback_func):
        """
        血圧解析開始
        Args:
            request_id_bytes: リクエストID（bytes）
            height: 身長（int）
            weight: 体重（int）
            sex: 性別（int）
            movie_path_bytes: 動画パス（bytes）
            callback_func: コールバック関数
        Returns:
            int: エラー数
        """
        try:
            request_id = request_id_bytes.decode('utf-8')
            movie_path = movie_path_bytes.decode('utf-8')
            
            # Pythonコールバック関数をCタイプから変換
            def python_callback(req_id, sbp, dbp, csv_data, errors):
                if callback_func:
                    # エラー配列の変換は省略（基本実装）
                    callback_func(req_id.encode(), sbp, dbp, csv_data.encode(), None)
            
            return start_bp_analysis(request_id, height, weight, sex, movie_path, python_callback)
        except Exception as e:
            print(f"StartBloodPressureAnalysis Error: {e}")
            return 1
    
    # 血圧解析中断
    @staticmethod
    def CancelBloodPressureProcessing(request_id_bytes):
        """
        血圧解析中断
        Args:
            request_id_bytes: リクエストID（bytes）
        Returns:
            bool: 中断成功フラグ
        """
        try:
            request_id = request_id_bytes.decode('utf-8')
            return cancel_bp_processing(request_id)
        except Exception as e:
            print(f"CancelBloodPressureProcessing Error: {e}")
            return False
    
    # 血圧解析状況取得
    @staticmethod
    def GetBloodPressureStatus(request_id_bytes):
        """
        血圧解析状況取得
        Args:
            request_id_bytes: リクエストID（bytes）
        Returns:
            bytes: ステータス文字列
        """
        try:
            request_id = request_id_bytes.decode('utf-8')
            status = get_bp_status(request_id)
            return status.encode('utf-8')
        except Exception as e:
            print(f"GetBloodPressureStatus Error: {e}")
            return b"error"
    
    # DLLバージョン取得
    @staticmethod
    def GetDLLVersion():
        """
        DLLバージョン取得
        Returns:
            bytes: バージョン文字列
        """
        try:
            version = get_dll_version()
            return version.encode('utf-8')
        except Exception as e:
            print(f"GetDLLVersion Error: {e}")
            return b"error"

# DLL関数のエクスポート用
dll_instance = BloodPressureDLL()

# C言語インターフェース関数
def c_initialize_dll(model_dir_ptr):
    model_dir_bytes = ctypes.string_at(model_dir_ptr) if model_dir_ptr else b"models"
    return dll_instance.InitializeDLL(model_dir_bytes)

def c_start_bp_analysis(request_id_ptr, height, weight, sex, movie_path_ptr, callback_ptr):
    request_id_bytes = ctypes.string_at(request_id_ptr)
    movie_path_bytes = ctypes.string_at(movie_path_ptr)
    return dll_instance.StartBloodPressureAnalysis(
        request_id_bytes, height, weight, sex, movie_path_bytes, callback_ptr)

def c_cancel_bp_processing(request_id_ptr):
    request_id_bytes = ctypes.string_at(request_id_ptr)
    return dll_instance.CancelBloodPressureProcessing(request_id_bytes)

def c_get_bp_status(request_id_ptr):
    request_id_bytes = ctypes.string_at(request_id_ptr)
    status_bytes = dll_instance.GetBloodPressureStatus(request_id_bytes)
    # 静的バッファを使用（実装上の簡略化）
    global status_buffer
    status_buffer = status_bytes
    return ctypes.c_char_p(status_buffer)

def c_get_dll_version():
    version_bytes = dll_instance.GetDLLVersion()
    global version_buffer
    version_buffer = version_bytes
    return ctypes.c_char_p(version_buffer)

# グローバルバッファ
status_buffer = b""
version_buffer = b""

# DLLエクスポート用の関数プロトタイプ定義
exported_functions = {
    'InitializeDLL': (c_initialize_dll, [ctypes.c_char_p], ctypes.c_bool),
    'StartBloodPressureAnalysis': (c_start_bp_analysis, 
                                 [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, 
                                  ctypes.c_int, ctypes.c_char_p, CallbackFunc], ctypes.c_int),
    'CancelBloodPressureProcessing': (c_cancel_bp_processing, [ctypes.c_char_p], ctypes.c_bool),
    'GetBloodPressureStatus': (c_get_bp_status, [ctypes.c_char_p], ctypes.c_char_p),
    'GetDLLVersion': (c_get_dll_version, [], ctypes.c_char_p)
}

if __name__ == "__main__":
    print("血圧推定DLLビルダー")
    print("エクスポート関数:")
    for func_name in exported_functions.keys():
        print(f"  - {func_name}")
    
    # 基本テスト
    print("\n基本テスト実行:")
    
    # 初期化テスト
    if c_initialize_dll(b"models"):
        print("✓ DLL初期化成功")
        
        # バージョン取得テスト
        version = c_get_dll_version()
        print(f"✓ DLLバージョン: {version.decode()}")
        
        # ステータス取得テスト
        test_id = b"test_request_001"
        status = c_get_bp_status(test_id)
        print(f"✓ 初期ステータス: {status.decode()}")
        
    else:
        print("✗ DLL初期化失敗")