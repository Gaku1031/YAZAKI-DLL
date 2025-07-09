"""
血圧推定DLL - Windows 64-bit DLL インターフェース
ctypesを使用してWindows DLLとして公開
"""

import ctypes
from ctypes import *
import os
import sys
import threading

# パスを追加してモジュールをインポート
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from bp_estimation_dll import (
    initialize_dll, start_bp_analysis, cancel_bp_processing,
    get_bp_status, get_dll_version, test_callback, ErrorInfo
)

# Windows DLL用のグローバル変数
_dll_initialized = False
_string_buffers = {}  # 文字列バッファを保持

# エラー情報構造体
class DLLErrorInfo(Structure):
    _fields_ = [
        ("code", c_char_p),
        ("message", c_char_p),
        ("is_retriable", c_bool)
    ]

# コールバック関数の型定義
CallbackFuncType = CFUNCTYPE(None, c_char_p, c_int, c_int, c_char_p, POINTER(DLLErrorInfo))

def get_string_buffer(key, value):
    """文字列バッファを管理"""
    _string_buffers[key] = value.encode('utf-8') if isinstance(value, str) else value
    return _string_buffers[key]

# =============================================================================
# DLL公開関数
# =============================================================================

def InitializeDLL(model_dir_ptr=None):
    """
    DLL初期化
    Args:
        model_dir_ptr: モデルディレクトリパス（c_char_p）
    Returns:
        bool: 初期化成功フラグ
    """
    global _dll_initialized
    try:
        if model_dir_ptr:
            model_dir = ctypes.string_at(model_dir_ptr).decode('utf-8')
        else:
            model_dir = "models"
        
        result = initialize_dll(model_dir)
        _dll_initialized = result
        return result
    except Exception as e:
        print(f"InitializeDLL Error: {e}")
        return False

def StartBloodPressureAnalysis(request_id_ptr, height, weight, sex, movie_path_ptr, callback_ptr):
    """
    血圧解析開始
    Args:
        request_id_ptr: リクエストID（c_char_p）
        height: 身長（c_int）
        weight: 体重（c_int）
        sex: 性別（c_int）
        movie_path_ptr: 動画パス（c_char_p）
        callback_ptr: コールバック関数（CallbackFuncType）
    Returns:
        int: エラー数
    """
    try:
        if not _dll_initialized:
            return 1
        
        request_id = ctypes.string_at(request_id_ptr).decode('utf-8')
        movie_path = ctypes.string_at(movie_path_ptr).decode('utf-8')
        
        # Pythonコールバック関数を作成
        def python_callback(req_id, sbp, dbp, csv_data, errors):
            if callback_ptr:
                try:
                    # エラー配列の変換
                    if errors:
                        error_array = (DLLErrorInfo * len(errors))()
                        for i, err in enumerate(errors):
                            error_array[i].code = get_string_buffer(f"err_code_{i}", err.code)
                            error_array[i].message = get_string_buffer(f"err_msg_{i}", err.message)
                            error_array[i].is_retriable = err.is_retriable
                        error_ptr = ctypes.cast(error_array, POINTER(DLLErrorInfo))
                    else:
                        error_ptr = None
                    
                    # コールバック実行
                    callback_ptr(
                        get_string_buffer("req_id", req_id),
                        sbp,
                        dbp,
                        get_string_buffer("csv_data", csv_data),
                        error_ptr
                    )
                except Exception as e:
                    print(f"Callback Error: {e}")
        
        return start_bp_analysis(request_id, height, weight, sex, movie_path, python_callback)
    except Exception as e:
        print(f"StartBloodPressureAnalysis Error: {e}")
        return 1

def CancelBloodPressureProcessing(request_id_ptr):
    """
    血圧解析中断
    Args:
        request_id_ptr: リクエストID（c_char_p）
    Returns:
        bool: 中断成功フラグ
    """
    try:
        if not _dll_initialized:
            return False
        
        request_id = ctypes.string_at(request_id_ptr).decode('utf-8')
        return cancel_bp_processing(request_id)
    except Exception as e:
        print(f"CancelBloodPressureProcessing Error: {e}")
        return False

def GetBloodPressureStatus(request_id_ptr):
    """
    血圧解析状況取得
    Args:
        request_id_ptr: リクエストID（c_char_p）
    Returns:
        c_char_p: ステータス文字列
    """
    try:
        if not _dll_initialized:
            return get_string_buffer("status_error", "error")
        
        request_id = ctypes.string_at(request_id_ptr).decode('utf-8')
        status = get_bp_status(request_id)
        return get_string_buffer("status", status)
    except Exception as e:
        print(f"GetBloodPressureStatus Error: {e}")
        return get_string_buffer("status_error", "error")

def GetDLLVersion():
    """
    DLLバージョン取得
    Returns:
        c_char_p: バージョン文字列
    """
    try:
        version = get_dll_version()
        return get_string_buffer("version", version)
    except Exception as e:
        print(f"GetDLLVersion Error: {e}")
        return get_string_buffer("version_error", "error")

# =============================================================================
# ctypes関数ラッパー（Windows DLL用）
# =============================================================================

# 関数プロトタイプの設定
InitializeDLL.argtypes = [c_char_p]
InitializeDLL.restype = c_bool

StartBloodPressureAnalysis.argtypes = [c_char_p, c_int, c_int, c_int, c_char_p, CallbackFuncType]
StartBloodPressureAnalysis.restype = c_int

CancelBloodPressureProcessing.argtypes = [c_char_p]
CancelBloodPressureProcessing.restype = c_bool

GetBloodPressureStatus.argtypes = [c_char_p]
GetBloodPressureStatus.restype = c_char_p

GetDLLVersion.argtypes = []
GetDLLVersion.restype = c_char_p

# =============================================================================
# DLLエントリーポイント（Windows用）
# =============================================================================

def DllMain(hModule, reason, reserved):
    """DLLメイン関数（Windows DLL用）"""
    if reason == 1:  # DLL_PROCESS_ATTACH
        print("血圧推定DLL - ロード完了")
    elif reason == 0:  # DLL_PROCESS_DETACH
        print("血圧推定DLL - アンロード")
    return True

# =============================================================================
# テスト関数
# =============================================================================

def test_dll_interface():
    """DLLインターフェースのテスト"""
    print("=== DLLインターフェーステスト ===")
    
    # 初期化テスト
    model_dir = b"models"
    if InitializeDLL(c_char_p(model_dir)):
        print("✓ DLL初期化成功")
        
        # バージョン取得テスト
        version_ptr = GetDLLVersion()
        version = ctypes.string_at(version_ptr).decode('utf-8')
        print(f"✓ DLLバージョン: {version}")
        
        # ステータス取得テスト
        test_id = b"test_request_001"
        status_ptr = GetBloodPressureStatus(c_char_p(test_id))
        status = ctypes.string_at(status_ptr).decode('utf-8')
        print(f"✓ 初期ステータス: {status}")
        
        # テストコールバック
        def test_dll_callback(req_id_ptr, sbp, dbp, csv_ptr, errors_ptr):
            req_id = ctypes.string_at(req_id_ptr).decode('utf-8')
            csv_data = ctypes.string_at(csv_ptr).decode('utf-8')
            print(f"コールバック受信: {req_id} - SBP:{sbp}, DBP:{dbp}, CSV:{len(csv_data)}文字")
        
        callback_func = CallbackFuncType(test_dll_callback)
        
        # 血圧解析テスト（サンプルデータがある場合）
        sample_video_path = "sample-data/100万画素.webm"
        sample_video = sample_video_path.encode('utf-8')
        if os.path.exists(sample_video_path):
            print("✓ サンプル動画でのテスト開始...")
            error_count = StartBloodPressureAnalysis(
                c_char_p(test_id), 170, 70, 1, c_char_p(sample_video), callback_func
            )
            print(f"✓ 血圧解析開始: エラー数={error_count}")
            
            # 処理完了まで待機
            import time
            while True:
                status_ptr = GetBloodPressureStatus(c_char_p(test_id))
                status = ctypes.string_at(status_ptr).decode('utf-8')
                print(f"処理状況: {status}")
                if status == "none":
                    break
                time.sleep(2)
        else:
            print("⚠ サンプル動画が見つからないため、血圧解析テストをスキップ")
        
    else:
        print("✗ DLL初期化失敗")

if __name__ == "__main__":
    test_dll_interface()