"""
血圧推定DLL
30秒WebM動画からrPPGアルゴリズムを使用してRRIを取得し、機械学習モデルで血圧を推定する
"""

import os
import sys
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import joblib
import threading
import time
import csv
from datetime import datetime
from collections import deque
from ctypes import *
from typing import Dict, List, Tuple, Optional, Callable
from scipy import signal
from scipy.signal import butter, find_peaks, lfilter
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
import pywt
from scipy.stats import zscore
import logging
import re

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MediaPipeの初期化
mp_face_mesh = mp.solutions.face_mesh

# 顔のROIのランドマーク番号
FACE_ROI = [118, 119, 120, 47, 217, 174, 196, 197, 419, 399, 437, 277,
            349, 348, 347, 280, 425, 426, 391, 393, 164, 167, 165, 206, 205, 50, 118]

# エラーコード定義
class ErrorCode:
    DLL_NOT_INITIALIZED = "1001"
    DEVICE_CONNECTION_FAILED = "1002"
    CALIBRATION_INCOMPLETE = "1003"
    INVALID_INPUT_PARAMETERS = "1004"
    REQUEST_DURING_PROCESSING = "1005"
    INTERNAL_PROCESSING_ERROR = "1006"

# エラー情報構造体
class ErrorInfo:
    def __init__(self, code: str, message: str, is_retriable: bool = False):
        self.code = code
        self.message = message
        self.is_retriable = is_retriable

# 処理状態
class ProcessingStatus:
    NONE = "none"
    PROCESSING = "processing"

# バンドパスフィルター
class BandpassFilter:
    def __init__(self, fs: float, low_freq: float, high_freq: float, order: int = 5):
        self.fs = fs
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.order = order
        self.high_b, self.high_a = signal.butter(
            self.order, self.low_freq / (self.fs / 2), btype='high')
        self.low_b, self.low_a = signal.butter(
            self.order, self.high_freq / (self.fs / 2), btype='low')

    def apply(self, signal_in: np.ndarray) -> np.ndarray:
        filtered_signal = signal.filtfilt(self.high_b, self.high_a, signal_in)
        filtered_signal = signal.filtfilt(self.low_b, self.low_a, filtered_signal)
        return filtered_signal

# rPPG信号処理クラス
class RPPGProcessor:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.1, min_tracking_confidence=0.1)
        self.window_size = 30
        self.skin_means = deque(maxlen=300)
        self.timestamps = deque(maxlen=300)
        self.roi_pixels_list = deque(maxlen=300)

    def process_video(self, video_path: str) -> Tuple[List[float], List[float], List[float]]:
        """
        動画を処理してrPPG信号とピーク時間を取得
        Returns: (rppg_data, time_data, peak_times)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"動画を開けませんでした: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30

        rppg_data = []
        time_data = []
        frame_count = 0

        # データ初期化
        self.skin_means.clear()
        self.timestamps.clear()
        self.roi_pixels_list.clear()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = frame_count / fps
            frame_count += 1

            # フレーム処理
            if len(frame.shape) == 3:
                self._process_color_frame(frame, current_time)
            else:
                self._process_grayscale_frame(frame, current_time)

            # rPPG信号計算
            if len(self.skin_means) > 30:
                rppg_signal = self._calculate_rppg_signal()
                if len(rppg_signal) > 0:
                    rppg_data.append(rppg_signal[-1])
                    time_data.append(current_time)

        cap.release()

        # ピーク検出
        if len(rppg_data) > 0:
            rppg_array = np.array(rppg_data)
            peaks, _ = find_peaks(rppg_array, distance=10)
            peak_times = [time_data[i] for i in peaks if i < len(time_data)]
        else:
            peak_times = []

        return rppg_data, time_data, peak_times

    def _process_color_frame(self, frame: np.ndarray, current_time: float):
        """カラーフレームの処理"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape
                points = []
                for idx in FACE_ROI:
                    x = int(face_landmarks.landmark[idx].x * w)
                    y = int(face_landmarks.landmark[idx].y * h)
                    points.append([x, y])

                points = np.array(points, dtype=np.int32)
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [points], 255)

                roi = cv2.bitwise_and(frame, frame, mask=mask)
                roi_ycbcr = cv2.cvtColor(roi, cv2.COLOR_RGB2YCrCb)
                skin_mask = cv2.inRange(roi_ycbcr, (0, 100, 130), (255, 140, 175))
                skin_pixels = cv2.bitwise_and(roi, roi, mask=skin_mask)
                skin_mean = cv2.mean(skin_pixels, skin_mask)[:3]

                self.skin_means.append(skin_mean)
                self.timestamps.append(current_time)

    def _process_grayscale_frame(self, frame: np.ndarray, current_time: float):
        """グレースケールフレームの処理"""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w = gray.shape[:2]
                points = []
                for idx in FACE_ROI:
                    x = int(face_landmarks.landmark[idx].x * w)
                    y = int(face_landmarks.landmark[idx].y * h)
                    points.append([x, y])

                points = np.array(points, dtype=np.int32)
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [points], 255)

                roi_pixels = gray[mask == 255]
                self.roi_pixels_list.append(roi_pixels)
                self.timestamps.append(current_time)

    def _calculate_rppg_signal(self) -> np.ndarray:
        """POSベースのrPPG信号抽出"""
        fs = 30
        lowcut = 0.7
        highcut = 3.0
        order = 7
        
        try:
            if len(self.skin_means) <= 10:
                return np.array([])

            skin_means = np.array(list(self.skin_means)[10:])
            if not np.all(np.isfinite(skin_means)):
                return np.array([])

            C = skin_means.T
            frames = C.shape[1]
            L = self.window_size
            if frames < L:
                return np.array([])

            H = np.zeros(frames)
            for f in range(frames - L + 1):
                block = C[:, f:f+L].T
                mu_C = np.mean(block, axis=0)
                mu_C[mu_C == 0] = 1e-8
                C_normed = block / mu_C
                M = np.array([[0, 1, -1], [-2, 1, 1]])
                S = np.dot(M, C_normed.T)
                alpha = np.std(S[0, :]) / (np.std(S[1, :]) + 1e-8)
                P = S[0, :] + alpha * S[1, :]
                P_std = np.std(P) if np.std(P) != 0 else 1e-8
                P_normalized = (P - np.mean(P)) / P_std
                H[f:f + L] += P_normalized

            pulse = -1 * H
            pulse_z = zscore(pulse)
            pulse_baseline = pulse_z - signal.convolve(pulse_z, np.ones(6)/6, mode='same')

            filtered = self._bandpass_filter(pulse_baseline, lowcut, highcut, fs, order)
            norm_sig = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-8)
            return norm_sig
        except Exception as e:
            logger.error(f"rPPG信号計算エラー: {e}")
            return np.array([])

    def _bandpass_filter(self, data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 3) -> np.ndarray:
        """バンドパスフィルター"""
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return lfilter(b, a, data)

# 血圧推定クラス
class BloodPressureEstimator:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.model_sbp = None
        self.model_dbp = None
        self._load_models()

    def _load_models(self):
        """学習済みモデルの読み込み"""
        try:
            sbp_path = os.path.join(self.model_dir, "model_sbp.pkl")
            dbp_path = os.path.join(self.model_dir, "model_dbp.pkl")
            
            if not os.path.exists(sbp_path) or not os.path.exists(dbp_path):
                raise FileNotFoundError(f"モデルファイルが見つかりません: {sbp_path}, {dbp_path}")
            
            self.model_sbp = joblib.load(sbp_path)
            self.model_dbp = joblib.load(dbp_path)
            logger.info("モデルの読み込みが完了しました")
        except Exception as e:
            logger.error(f"モデル読み込みエラー: {e}")
            raise

    def estimate_bp(self, peak_times: List[float], height: int, weight: int, sex: int) -> Tuple[int, int]:
        """
        血圧推定
        Args:
            peak_times: ピーク時間のリスト
            height: 身長(cm)
            weight: 体重(kg)
            sex: 性別(1=男性, 2=女性)
        Returns:
            (収縮期血圧, 拡張期血圧)
        """
        if len(peak_times) < 2:
            raise ValueError("ピークが検出されませんでした")

        # RRIの計算
        rri_values = []
        for i in range(len(peak_times) - 1):
            rri = peak_times[i + 1] - peak_times[i]
            if 0.4 <= rri <= 1.2:  # 異常値除去
                rri_values.append(rri)

        if len(rri_values) == 0:
            raise ValueError("有効なRRIが検出されませんでした")

        # 特徴量の計算
        rri_array = np.array(rri_values)
        bmi = weight / ((height / 100) ** 2)
        sex_feature = 1 if sex == 1 else 0  # 1=男性, 2=女性 → 1=男性, 0=女性

        feature_vector = np.array([
            rri_array.mean(),
            rri_array.std(),
            rri_array.min(),
            rri_array.max(),
            bmi,
            sex_feature
        ]).reshape(1, -1)

        # 血圧推定
        pred_sbp = self.model_sbp.predict(feature_vector)[0]
        pred_dbp = self.model_dbp.predict(feature_vector)[0]

        # 整数に変換（仕様に合わせて）
        return int(round(pred_sbp)), int(round(pred_dbp))

    def generate_ppg_csv(self, rppg_data: List[float], time_data: List[float], 
                        peak_times: List[float]) -> str:
        """PPGローデータのCSV生成"""
        csv_data = []
        
        # ヘッダー
        csv_data.append("Time(s),rPPG_Signal,Peak_Flag")
        
        # データ行
        peak_set = set(peak_times)
        for i, (time_val, rppg_val) in enumerate(zip(time_data, rppg_data)):
            peak_flag = 1 if any(abs(time_val - peak_t) < 0.1 for peak_t in peak_set) else 0
            csv_data.append(f"{time_val:.3f},{rppg_val:.6f},{peak_flag}")
        
        return "\n".join(csv_data)

# DLLメインクラス
class BloodPressureDLL:
    def __init__(self):
        self.is_initialized = False
        self.processing_requests: Dict[str, threading.Thread] = {}
        self.request_status: Dict[str, str] = {}
        self.rppg_processor = RPPGProcessor()
        self.bp_estimator = None
        self.version = "1.0.0"
        self.lock = threading.Lock()

    def initialize(self, model_dir: str = "models") -> bool:
        """DLL初期化"""
        try:
            self.bp_estimator = BloodPressureEstimator(model_dir)
            self.is_initialized = True
            logger.info("DLL初期化が完了しました")
            return True
        except Exception as e:
            logger.error(f"DLL初期化エラー: {e}")
            return False

    def start_blood_pressure_analysis(self, request_id: str, height: int, weight: int, 
                                    sex: int, measurement_movie_path: str, 
                                    callback: Optional[Callable] = None) -> Optional[str]:
        """血圧解析開始"""
        
        # 初期化チェック
        if not self.is_initialized:
            return ErrorCode.DLL_NOT_INITIALIZED
        
        # パラメータ検証
        if not request_id or not measurement_movie_path:
            return ErrorCode.INVALID_INPUT_PARAMETERS
        
        # リクエストID形式の検証 (${yyyyMMddHHmmssfff}_${顧客コード}_${乗務員コード})
        if not self._validate_request_id(request_id):
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
                target=self._process_blood_pressure_analysis,
                args=(request_id, height, weight, sex, measurement_movie_path, callback)
            )
            self.processing_requests[request_id] = thread
            thread.start()
        
        return None  # 成功時はNone

    def _process_blood_pressure_analysis(self, request_id: str, height: int, weight: int,
                                       sex: int, measurement_movie_path: str,
                                       callback: Optional[Callable]):
        """血圧解析処理（非同期）"""
        try:
            # rPPG処理
            rppg_data, time_data, peak_times = self.rppg_processor.process_video(measurement_movie_path)
            
            # 血圧推定
            sbp, dbp = self.bp_estimator.estimate_bp(peak_times, height, weight, sex)
            
            # CSVデータ生成
            csv_data = self.bp_estimator.generate_ppg_csv(rppg_data, time_data, peak_times)
            
            # 成功時のコールバック
            if callback:
                callback(request_id, sbp, dbp, csv_data, None)
            
            logger.info(f"血圧解析完了: {request_id} - SBP: {sbp}, DBP: {dbp}")
            
        except Exception as e:
            logger.error(f"血圧解析エラー: {e}")
            error = ErrorInfo(ErrorCode.INTERNAL_PROCESSING_ERROR, str(e))
            if callback:
                callback(request_id, 0, 0, "", [error])
        
        finally:
            # 処理完了処理
            with self.lock:
                if request_id in self.processing_requests:
                    del self.processing_requests[request_id]
                if request_id in self.request_status:
                    del self.request_status[request_id]

    def cancel_processing(self, request_id: str) -> bool:
        """処理中断"""
        with self.lock:
            if request_id in self.processing_requests:
                # 実際の中断処理（スレッドの停止は困難なため、フラグによる制御推奨）
                logger.info(f"処理中断要求: {request_id}")
                return True
            return False

    def get_processing_status(self, request_id: str) -> str:
        """処理状況取得"""
        with self.lock:
            return self.request_status.get(request_id, ProcessingStatus.NONE)

    def get_version_info(self) -> str:
        """バージョン情報取得"""
        return self.version
    
    def _validate_request_id(self, request_id: str) -> bool:
        """リクエストID形式の検証"""
        # 形式: ${yyyyMMddHHmmssfff}_${顧客コード}_${乗務員コード}
        # 例: 20250707083524932_9000000001_0000012345
        pattern = r'^\d{17}_\w+_\w+$'
        return re.match(pattern, request_id) is not None

# グローバルDLLインスタンス
dll_instance = BloodPressureDLL()

# C言語スタイルのエクスポート関数
def initialize_dll(model_dir: str = "models") -> bool:
    """DLL初期化"""
    return dll_instance.initialize(model_dir)

def start_bp_analysis(request_id: str, height: int, weight: int, sex: int,
                     movie_path: str, callback_func=None) -> Optional[str]:
    """血圧解析開始"""
    error_code = dll_instance.start_blood_pressure_analysis(
        request_id, height, weight, sex, movie_path, callback_func)
    return error_code

def cancel_bp_processing(request_id: str) -> bool:
    """血圧解析中断"""
    return dll_instance.cancel_processing(request_id)

def get_bp_status(request_id: str) -> str:
    """血圧解析状況取得"""
    return dll_instance.get_processing_status(request_id)

def get_dll_version() -> str:
    """DLLバージョン取得"""
    return dll_instance.get_version_info()

def generate_request_id(customer_code: str, driver_code: str) -> str:
    """リクエストID生成"""
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]  # yyyyMMddHHmmssfff
    return f"{timestamp}_{customer_code}_{driver_code}"

# テスト用のコールバック関数
def test_callback(request_id: str, max_bp: int, min_bp: int, csv_data: str, errors: List[ErrorInfo]):
    """テスト用コールバック"""
    print(f"=== 血圧解析結果 ===")
    print(f"Request ID: {request_id}")
    print(f"最高血圧: {max_bp} mmHg")
    print(f"最低血圧: {min_bp} mmHg")
    print(f"CSVデータサイズ: {len(csv_data)} 文字")
    
    if errors:
        print("エラー:")
        for error in errors:
            print(f"  - {error.code}: {error.message}")
    
    # CSVファイルに保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"bp_result_{request_id}_{timestamp}.csv"
    with open(csv_filename, 'w', encoding='utf-8') as f:
        f.write(csv_data)
    print(f"CSVファイル保存: {csv_filename}")

if __name__ == "__main__":
    # テスト実行
    print("血圧推定DLL テスト開始")
    
    # 初期化
    if initialize_dll():
        print("DLL初期化成功")
        
        # テスト用パラメータ
        test_request_id = generate_request_id("9000000001", "0000012345")
        test_height = 170
        test_weight = 70
        test_sex = 1  # 男性
        test_movie_path = "sample-data/100万画素.webm"
        
        # 血圧解析実行
        error_code = start_bp_analysis(
            test_request_id, test_height, test_weight, test_sex,
            test_movie_path, test_callback
        )
        
        if error_code is None:
            print("血圧解析開始成功")
            
            # 処理状況監視
            while True:
                status = get_bp_status(test_request_id)
                print(f"処理状況: {status}")
                if status == ProcessingStatus.NONE:
                    break
                time.sleep(1)
                
        else:
            print(f"血圧解析開始失敗: {error_code}")
    else:
        print("DLL初期化失敗")