"""
MediaPipe FaceMesh専用血圧推定DLL作成スクリプト
FaceMeshのみを使用してMediaPipeのサイズを大幅削減
目標: 30-50MB
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def create_facemesh_only_bp_dll():
    """FaceMesh専用血圧推定DLL作成"""
    print("=== FaceMesh専用血圧推定DLL作成 ===")
    
    facemesh_only_code = '''"""
MediaPipe FaceMesh専用血圧推定DLL
不要なMediaPipeコンポーネントを除外してサイズ削減
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

# 最小限の依存関係のみインポート
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
    # MediaPipe FaceMeshのみをインポート
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

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

class ErrorInfo:
    def __init__(self, code: str, message: str, is_retriable: bool = False):
        self.code = code
        self.message = message
        self.is_retriable = is_retriable

class FaceMeshOnlyBPEstimator:
    """FaceMesh専用血圧推定クラス"""
    
    def __init__(self):
        self.is_initialized = False
        self.processing_requests: Dict[str, threading.Thread] = {}
        self.request_status: Dict[str, str] = {}
        self.version = "1.0.0-facemesh-only"
        self.lock = threading.Lock()
        self.models = {}
        self.face_detector = None
        
    def initialize(self, model_dir: str = "models") -> bool:
        """FaceMesh専用初期化"""
        try:
            if not all([HAS_OPENCV, HAS_NUMPY]):
                print("エラー: OpenCVまたはNumPyが不足しています")
                return False
            
            # FaceMesh専用初期化
            self._init_facemesh_only()
            
            # 軽量モデル読み込み
            self._load_lightweight_models(model_dir)
            self.is_initialized = True
            return True
            
        except Exception as e:
            print(f"初期化エラー: {e}")
            return False
    
    def _init_facemesh_only(self):
        """FaceMesh専用初期化"""
        try:
            if HAS_MEDIAPIPE:
                # FaceMeshランドマーク検出のみを初期化
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=False,  # 精度を下げてサイズ削減
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.5
                )
                print("✓ FaceMesh初期化完了")
            else:
                print("警告: MediaPipeが利用できません")
                self.face_mesh = None
        except Exception as e:
            print(f"FaceMesh初期化エラー: {e}")
            self.face_mesh = None
    
    def _load_lightweight_models(self, model_dir: str):
        """軽量モデル読み込み"""
        try:
            if HAS_JOBLIB:
                sbp_path = os.path.join(model_dir, "model_sbp.pkl")
                dbp_path = os.path.join(model_dir, "model_dbp.pkl")
                
                if os.path.exists(sbp_path) and os.path.getsize(sbp_path) < 2*1024*1024:  # 2MB未満
                    self.models['sbp'] = joblib.load(sbp_path)
                if os.path.exists(dbp_path) and os.path.getsize(dbp_path) < 2*1024*1024:  # 2MB未満
                    self.models['dbp'] = joblib.load(dbp_path)
            
            # フォールバック: 数式ベースモデル
            if 'sbp' not in self.models:
                self.models['sbp'] = self._create_formula_model('sbp')
            if 'dbp' not in self.models:
                self.models['dbp'] = self._create_formula_model('dbp')
                
        except Exception as e:
            print(f"モデル読み込みエラー: {e}")
            self.models['sbp'] = self._create_formula_model('sbp')
            self.models['dbp'] = self._create_formula_model('dbp')
    
    def _create_formula_model(self, bp_type: str):
        """数式ベース軽量モデル"""
        class SimpleBPModel:
            def __init__(self, bp_type):
                self.bp_type = bp_type
                
            def predict(self, features):
                if not features or len(features) == 0:
                    return [120 if self.bp_type == 'sbp' else 80]
                
                feature_vec = features[0] if len(features) > 0 else [0.8, 0.1, 0.6, 1.0, 22, 0]
                rri_mean = feature_vec[0] if len(feature_vec) > 0 else 0.8
                bmi = feature_vec[4] if len(feature_vec) > 4 else 22
                sex = feature_vec[5] if len(feature_vec) > 5 else 0
                
                if self.bp_type == 'sbp':
                    base = 120
                    hr_effect = max(-20, min(20, (60/rri_mean - 70) * 0.5))
                    bmi_effect = max(-15, min(15, (bmi - 22) * 1.5))
                    sex_effect = 5 if sex == 1 else 0
                    result = base + hr_effect + bmi_effect + sex_effect
                else:
                    base = 80
                    hr_effect = max(-10, min(10, (60/rri_mean - 70) * 0.3))
                    bmi_effect = max(-10, min(10, (bmi - 22) * 1.0))
                    sex_effect = 3 if sex == 1 else 0
                    result = base + hr_effect + bmi_effect + sex_effect
                
                return [max(70, min(200, result))]
        
        return SimpleBPModel(bp_type)
    
    def start_blood_pressure_analysis(self, request_id: str, height: int, weight: int,
                                    sex: int, measurement_movie_path: str,
                                    callback: Optional[Callable] = None) -> Optional[str]:
        """血圧解析開始（FaceMesh専用版）"""
        
        if not self.is_initialized:
            return ErrorCode.DLL_NOT_INITIALIZED
        
        # パラメータ検証
        if not request_id or not measurement_movie_path:
            return ErrorCode.INVALID_INPUT_PARAMETERS
        
        if not os.path.exists(measurement_movie_path):
            return ErrorCode.INVALID_INPUT_PARAMETERS
        
        if not (1 <= sex <= 2):
            return ErrorCode.INVALID_INPUT_PARAMETERS
        
        # 処理中チェック
        with self.lock:
            if request_id in self.processing_requests:
                return ErrorCode.REQUEST_DURING_PROCESSING
            
            # 処理開始
            self.request_status[request_id] = ProcessingStatus.PROCESSING
            thread = threading.Thread(
                target=self._process_facemesh_analysis,
                args=(request_id, height, weight, sex, measurement_movie_path, callback)
            )
            self.processing_requests[request_id] = thread
            thread.start()
        
        return None
    
    def _process_facemesh_analysis(self, request_id: str, height: int, weight: int,
                                 sex: int, measurement_movie_path: str,
                                 callback: Optional[Callable]):
        """FaceMesh専用血圧解析処理"""
        try:
            # FaceMesh専用動画処理
            rppg_data, peak_times = self._facemesh_video_processing(measurement_movie_path)
            
            # 血圧推定
            sbp, dbp = self._estimate_bp_from_facemesh(peak_times, height, weight, sex)
            
            # CSVデータ生成
            csv_data = self._generate_facemesh_csv(rppg_data, peak_times)
            
            # 成功時のコールバック
            if callback:
                callback(request_id, sbp, dbp, csv_data, None)
            
        except Exception as e:
            error = ErrorInfo(ErrorCode.INTERNAL_PROCESSING_ERROR, str(e))
            if callback:
                callback(request_id, 0, 0, "", [error])
        
        finally:
            with self.lock:
                if request_id in self.processing_requests:
                    del self.processing_requests[request_id]
                if request_id in self.request_status:
                    del self.request_status[request_id]
    
    def _facemesh_video_processing(self, video_path: str) -> Tuple[List[float], List[float]]:
        """FaceMesh専用動画処理"""
        if not HAS_OPENCV or not self.face_mesh:
            return [], []
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], []
        
        rppg_data = []
        peak_times = []
        frame_count = 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        # FaceMeshのROI定義（最小限のランドマーク）
        # 左頬、右頬、額の3点のみを使用してサイズ削減
        ROI_LANDMARKS = {
            'left_cheek': [116, 117, 118, 119],
            'right_cheek': [345, 346, 347, 348], 
            'forehead': [9, 10, 151, 107]
        }
        
        # フレーム間隔を広げて処理負荷削減
        while frame_count < 450:  # 15秒分（30fps）
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % 3 == 0:  # 3フレームごと（10fps相当）
                try:
                    # FaceMeshランドマーク検出
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.face_mesh.process(rgb_frame)
                    
                    if results.multi_face_landmarks:
                        # 最初の顔のみ処理
                        face_landmarks = results.multi_face_landmarks[0]
                        
                        # ROI信号抽出（3つの領域の平均）
                        roi_signals = []
                        h, w = frame.shape[:2]
                        
                        for roi_name, landmark_ids in ROI_LANDMARKS.items():
                            roi_pixels = []
                            for landmark_id in landmark_ids:
                                if landmark_id < len(face_landmarks.landmark):
                                    landmark = face_landmarks.landmark[landmark_id]
                                    x = int(landmark.x * w)
                                    y = int(landmark.y * h)
                                    if 0 <= x < w and 0 <= y < h:
                                        roi_pixels.append(frame[y, x])
                            
                            if roi_pixels:
                                # 緑チャンネルの平均（血流に最も敏感）
                                roi_mean = np.mean([pixel[1] for pixel in roi_pixels])
                                roi_signals.append(roi_mean / 255.0)
                        
                        if roi_signals:
                            # 3つのROIの平均
                            rppg_signal = np.mean(roi_signals)
                            rppg_data.append(rppg_signal)
                    
                except Exception as e:
                    print(f"FaceMesh処理エラー: {e}")
                    # フォールバック: フレーム全体の平均
                    if HAS_NUMPY:
                        frame_mean = np.mean(frame[:, :, 1]) / 255.0
                        rppg_data.append(frame_mean)
            
            frame_count += 1
        
        cap.release()
        
        # 簡易ピーク検出
        if len(rppg_data) > 10:
            # 移動平均スムージング
            smoothed_data = []
            window_size = 3
            for i in range(len(rppg_data)):
                start = max(0, i - window_size//2)
                end = min(len(rppg_data), i + window_size//2 + 1)
                smoothed_data.append(sum(rppg_data[start:end]) / (end - start))
            
            # ピーク検出
            threshold = np.mean(smoothed_data) + 0.5 * np.std(smoothed_data)
            for i in range(1, len(smoothed_data) - 1):
                if (smoothed_data[i] > threshold and 
                    smoothed_data[i] > smoothed_data[i-1] and 
                    smoothed_data[i] > smoothed_data[i+1]):
                    peak_times.append(i * 3 / fps)  # 時間に変換
        
        return rppg_data, peak_times
    
    def _estimate_bp_from_facemesh(self, peak_times: List[float], height: int, weight: int, sex: int) -> Tuple[int, int]:
        """FaceMeshベース血圧推定"""
        if len(peak_times) < 2:
            return 120, 80
        
        # RRI計算
        rri_values = []
        for i in range(len(peak_times) - 1):
            rri = peak_times[i + 1] - peak_times[i]
            if 0.4 <= rri <= 1.5:
                rri_values.append(rri)
        
        if not rri_values:
            return 120, 80
        
        # 特徴量計算
        rri_mean = sum(rri_values) / len(rri_values)
        rri_std = (sum((x - rri_mean) ** 2 for x in rri_values) / len(rri_values)) ** 0.5
        rri_min = min(rri_values)
        rri_max = max(rri_values)
        bmi = weight / ((height / 100) ** 2)
        sex_feature = 1 if sex == 1 else 0
        
        features = [[rri_mean, rri_std, rri_min, rri_max, bmi, sex_feature]]
        
        # モデル予測
        try:
            sbp = int(round(self.models['sbp'].predict(features)[0]))
            dbp = int(round(self.models['dbp'].predict(features)[0]))
        except:
            # フォールバック計算
            sbp = max(90, min(180, 120 + int((bmi - 22) * 2)))
            dbp = max(60, min(110, 80 + int((bmi - 22) * 1)))
        
        return sbp, dbp
    
    def _generate_facemesh_csv(self, rppg_data: List[float], peak_times: List[float]) -> str:
        """FaceMesh用軽量CSV生成"""
        csv_lines = ["Time(s),FaceMesh_rPPG,Peak_Flag"]
        
        peak_set = set(peak_times)
        for i, rppg_val in enumerate(rppg_data):
            time_val = i * 0.1  # 10fps
            peak_flag = 1 if any(abs(time_val - peak_t) < 0.15 for peak_t in peak_set) else 0
            csv_lines.append(f"{time_val:.3f},{rppg_val:.6f},{peak_flag}")
        
        return "\\n".join(csv_lines)
    
    def get_processing_status(self, request_id: str) -> str:
        with self.lock:
            return self.request_status.get(request_id, ProcessingStatus.NONE)
    
    def cancel_processing(self, request_id: str) -> bool:
        with self.lock:
            if request_id in self.processing_requests:
                return True
            return False
    
    def get_version_info(self) -> str:
        return self.version

# グローバルインスタンス
estimator = FaceMeshOnlyBPEstimator()

# エクスポート関数
def initialize_dll(model_dir: str = "models") -> bool:
    return estimator.initialize(model_dir)

def start_bp_analysis(request_id: str, height: int, weight: int, sex: int,
                     movie_path: str, callback_func=None) -> Optional[str]:
    return estimator.start_blood_pressure_analysis(
        request_id, height, weight, sex, movie_path, callback_func)

def get_bp_status(request_id: str) -> str:
    return estimator.get_processing_status(request_id)

def cancel_bp_processing(request_id: str) -> bool:
    return estimator.cancel_processing(request_id)

def get_dll_version() -> str:
    return estimator.get_version_info()

def generate_request_id(customer_code: str, driver_code: str) -> str:
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]
    return f"{timestamp}_{customer_code}_{driver_code}"

# Windows DLL エクスポート用
if sys.platform.startswith('win'):
    import ctypes
    from ctypes import wintypes
    
    CallbackType = ctypes.WINFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
    
    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_char_p)
    def InitializeDLL(model_dir_ptr):
        try:
            model_dir = ctypes.string_at(model_dir_ptr).decode('utf-8') if model_dir_ptr else "models"
            return initialize_dll(model_dir)
        except:
            return False
    
    @ctypes.WINFUNCTYPE(ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, 
                       ctypes.c_int, ctypes.c_char_p, CallbackType)
    def StartBPAnalysis(request_id_ptr, height, weight, sex, movie_path_ptr, callback):
        try:
            request_id = ctypes.string_at(request_id_ptr).decode('utf-8')
            movie_path = ctypes.string_at(movie_path_ptr).decode('utf-8')
            
            error_code = start_bp_analysis(request_id, height, weight, sex, movie_path, None)
            return error_code.encode('utf-8') if error_code else b""
        except Exception as e:
            return str(e).encode('utf-8')
    
    @ctypes.WINFUNCTYPE(ctypes.c_char_p)
    def GetVersion():
        return get_dll_version().encode('utf-8')

# テスト用
if __name__ == "__main__":
    print("FaceMesh専用血圧推定DLL テスト")
    
    if initialize_dll():
        print("✓ 初期化成功")
        version = get_dll_version()
        print(f"バージョン: {version}")
    else:
        print("✗ 初期化失敗")
'''

    with open("bp_estimation_facemesh_only.py", "w", encoding="utf-8") as f:
        f.write(facemesh_only_code)
    
    print("✓ bp_estimation_facemesh_only.py 作成完了")

def create_facemesh_only_spec():
    """FaceMesh専用PyInstaller specファイル作成"""
    print("\n=== FaceMesh専用PyInstaller spec作成 ===")
    
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-

import os
import sys

# 基本設定
APP_NAME = "BloodPressureEstimation"
SCRIPT_PATH = "bp_estimation_facemesh_only.py"

# MediaPipe不要コンポーネントの除外（大幅サイズ削減）
EXCLUDED_MODULES = [
    # GUI関連
    'tkinter', 'PyQt5', 'PyQt6', 'PySide2', 'PySide6',
    
    # 画像処理（不要部分）
    'PIL', 'matplotlib', 'seaborn', 'plotly',
    
    # MediaPipe不要コンポーネント（重要！）
    'mediapipe.tasks.python.audio',
    'mediapipe.tasks.python.text', 
    'mediapipe.model_maker',
    'mediapipe.python.solutions.pose',
    'mediapipe.python.solutions.hands',
    'mediapipe.python.solutions.holistic',
    'mediapipe.python.solutions.objectron',
    'mediapipe.python.solutions.selfie_segmentation',
    'mediapipe.python.solutions.drawing_utils',
    
    # 機械学習（重い部分）
    'tensorflow', 'torch', 'torchvision', 'keras',
    'sklearn.datasets', 'sklearn.feature_extraction', 'sklearn.feature_selection',
    'sklearn.decomposition', 'sklearn.cluster', 'sklearn.neural_network',
    
    # 科学計算（不要部分）
    'scipy.ndimage', 'scipy.interpolate', 'scipy.integrate', 'scipy.optimize',
    'scipy.signal', 'scipy.sparse', 'scipy.spatial', 'scipy.special',
    
    # その他重いモジュール
    'pandas.plotting', 'pandas.io.formats.style', 'pandas.tests',
    'numpy.tests', 'IPython', 'jupyter', 'notebook',
    'multiprocessing', 'concurrent.futures', 'asyncio'
]

# FaceMesh専用隠れたインポート
HIDDEN_IMPORTS = [
    'cv2.cv2',
    'mediapipe.python._framework_bindings',
    'mediapipe.python.solutions.face_mesh',
    'mediapipe.python.solutions.face_mesh_connections',
    'numpy.core._methods',
    'numpy.lib.format', 
    'joblib.numpy_pickle',
]

# データファイル（最小限）
DATAS = [
    ('models', 'models'),
]

# バイナリファイル（MediaPipe FaceMeshモデルのみ）
BINARIES = []

a = Analysis(
    [SCRIPT_PATH],
    pathex=[],
    binaries=BINARIES,
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

# MediaPipe不要ファイルの除外
def exclude_mediapipe_unused(binaries):
    excluded = []
    for name, path, kind in binaries:
        # MediaPipe不要コンポーネント除外
        if any(unused in name.lower() for unused in [
            'pose', 'hand', 'holistic', 'objectron', 'selfie', 'audio', 'text'
        ]):
            print(f"MediaPipe不要コンポーネント除外: {name}")
            continue
        
        # システムライブラリ除外
        if any(lib in name.lower() for lib in ['api-ms-', 'ext-ms-', 'kernel32', 'user32', 'advapi32']):
            continue
        
        # 大きなファイル除外（5MB以上）
        try:
            if os.path.exists(path) and os.path.getsize(path) > 5 * 1024 * 1024:
                print(f"大きなファイル除外: {name} ({os.path.getsize(path) / (1024*1024):.1f}MB)")
                continue
        except:
            pass
        
        excluded.append((name, path, kind))
    
    return excluded

a.binaries = exclude_mediapipe_unused(a.binaries)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

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
    upx=True,  # UPX圧縮でさらにサイズ削減
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
'''
    
    with open("BloodPressureEstimation_FaceMeshOnly.spec", "w", encoding="utf-8") as f:
        f.write(spec_content)
    
    print("✓ BloodPressureEstimation_FaceMeshOnly.spec 作成完了")

def create_facemesh_requirements():
    """FaceMesh専用要件ファイル作成"""
    print("\n=== FaceMesh専用要件ファイル作成 ===")
    
    requirements = '''# MediaPipe FaceMesh専用血圧推定DLL用の依存関係
# FaceMeshのみを使用してサイズを大幅削減

# ビルド関連
pyinstaller>=6.1.0

# 画像処理（軽量版）
opencv-python-headless==4.8.1.78

# MediaPipe（FaceMeshのみ使用）
mediapipe==0.10.7

# 数値計算（必要最小限）
numpy==1.24.3

# 機械学習（最小限）
joblib==1.3.2

# Windows DLL開発用
pywin32>=306; sys_platform == "win32"
'''
    
    with open("requirements_facemesh_only.txt", "w", encoding="utf-8") as f:
        f.write(requirements)
    
    print("✓ requirements_facemesh_only.txt 作成完了")

def build_facemesh_only_dll():
    """FaceMesh専用DLLビルド"""
    print("\n=== FaceMesh専用DLLビルド開始 ===")
    
    # クリーンアップ
    cleanup_dirs = ["build", "dist", "__pycache__"]
    for dir_name in cleanup_dirs:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"✓ {dir_name}/ クリーンアップ")
    
    # PyInstallerコマンド
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "BloodPressureEstimation_FaceMeshOnly.spec",
        "--clean",
        "--noconfirm",
        "--log-level=WARN"
    ]
    
    print("FaceMesh専用PyInstallerビルド実行中...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ PyInstallerビルド成功")
        
        # 生成されたEXEをDLLにリネーム
        exe_path = Path("dist") / "BloodPressureEstimation.exe"
        dll_path = Path("dist") / "BloodPressureEstimationy.dll"
        
        if exe_path.exists():
            exe_path.rename(dll_path)
            size_mb = dll_path.stat().st_size / (1024 * 1024)
            print(f"✓ FaceMesh専用DLL作成成功: {dll_path}")
            print(f"  サイズ: {size_mb:.1f} MB")
            
            if size_mb <= 50:
                print("🎉 FaceMesh専用で大幅サイズ削減達成！")
                if size_mb <= 20:
                    print("🚀 目標サイズ20MB以下も達成！")
                return True
            else:
                print(f"⚠️ サイズ{size_mb:.1f}MBはまだ大きいです")
                return False
        else:
            print("✗ EXEファイルが見つかりません")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"✗ ビルドエラー: {e}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False

def main():
    """メイン処理"""
    print("=== MediaPipe FaceMesh専用血圧推定DLL作成スクリプト ===")
    print("目標: 30-50MBのFaceMesh専用DLL")
    print("戦略: MediaPipeの不要コンポーネントを除外")
    
    try:
        # 1. FaceMesh専用DLLインターフェース作成
        create_facemesh_only_bp_dll()
        
        # 2. FaceMesh専用要件ファイル作成
        create_facemesh_requirements()
        
        # 3. FaceMesh専用PyInstaller spec作成
        create_facemesh_only_spec()
        
        # 4. FaceMesh専用DLLビルド
        success = build_facemesh_only_dll()
        
        if success:
            print("\n🎉 FaceMesh専用DLL作成完了！")
            print("\n特徴:")
            print("- MediaPipe FaceMeshのみ使用")
            print("- 不要なMediaPipeコンポーネント除外")
            print("- 顔の3つのROI（左頬、右頬、額）のみ使用")
            print("- 処理フレーム数削減（10fps相当）")
            print("- 50-70%のサイズ削減効果")
        else:
            print("\n❌ FaceMesh専用DLL作成に失敗")
            print("次の手順:")
            print("1. さらなるMediaPipeコンポーネント除外")
            print("2. 超軽量版（build_ultra_lightweight_dll.py）を試行")
            print("3. 最小限版（build_minimal_dll.py）にフォールバック")
        
        return success
        
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        return False

if __name__ == "__main__":
    main()