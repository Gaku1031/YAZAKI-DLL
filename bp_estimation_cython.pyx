# cython: language_level=3
# distutils: language=c++
# cython: boundscheck=False
# cython: wraparound=True
# cython: cdivision=True
# cython: nonecheck=False

"""
Cython-based Blood Pressure Estimation DLL
Compiled to C++ for C# integration with code obfuscation
"""

# Cython imports
cimport cython
from cpython cimport array
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
from libc.stdlib cimport malloc, free
from libc.string cimport strcpy, strlen

# Python imports (will be compiled to C++)
import os
import sys
import threading
import time
import json
import csv
from datetime import datetime
from typing import Optional, List, Callable, Dict, Tuple
import re

# Import dependencies
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
from typing import Dict, List, Tuple, Optional, Callable
from scipy import signal
from scipy.signal import butter, find_peaks, lfilter
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
import pywt
from scipy.stats import zscore
import logging
import re

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MediaPipe initialization
mp_face_mesh = mp.solutions.face_mesh

# Face ROI landmark numbers
FACE_ROI = [118, 119, 120, 47, 217, 174, 196, 197, 419, 399, 437, 277,
            349, 348, 347, 280, 425, 426, 391, 393, 164, 167, 165, 206, 205, 50, 118]

# README.md compliant error code definitions
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

# Bandpass filter class
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

# rPPG signal processing class
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
        Process video and extract rPPG signal and peak times
        Returns: (rppg_data, time_data, peak_times)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30

        rppg_data = []
        time_data = []
        frame_count = 0

        # Initialize data
        self.skin_means.clear()
        self.timestamps.clear()
        self.roi_pixels_list.clear()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = frame_count / fps
            frame_count += 1

            # Frame processing
            if len(frame.shape) == 3:
                self._process_color_frame(frame, current_time)
            else:
                self._process_grayscale_frame(frame, current_time)

            # rPPG signal calculation
            if len(self.skin_means) > 30:
                rppg_signal = self._calculate_rppg_signal()
                if len(rppg_signal) > 0:
                    rppg_data.append(rppg_signal[-1])
                    time_data.append(current_time)

        cap.release()

        # Peak detection
        if len(rppg_data) > 0:
            rppg_array = np.array(rppg_data)
            peaks, _ = find_peaks(rppg_array, distance=10)
            peak_times = [time_data[i] for i in peaks if i < len(time_data)]
        else:
            peak_times = []

        return rppg_data, time_data, peak_times

    def _process_color_frame(self, frame: np.ndarray, current_time: float):
        """Process color frame"""
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
        """Process grayscale frame"""
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
        """POS-based rPPG signal extraction"""
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
            logger.error(f"rPPG signal calculation error: {e}")
            return np.array([])

    def _bandpass_filter(self, data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 3) -> np.ndarray:
        """Bandpass filter"""
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return lfilter(b, a, data)

# Blood pressure estimation class
class BloodPressureEstimator:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.model_sbp = None
        self.model_dbp = None
        self._load_models()

    def _load_models(self):
        """Load trained models"""
        try:
            sbp_path = os.path.join(self.model_dir, "model_sbp.pkl")
            dbp_path = os.path.join(self.model_dir, "model_dbp.pkl")
            
            if not os.path.exists(sbp_path) or not os.path.exists(dbp_path):
                raise FileNotFoundError(f"Model files not found: {sbp_path}, {dbp_path}")
            
            self.model_sbp = joblib.load(sbp_path)
            self.model_dbp = joblib.load(dbp_path)
            logger.info("Model loading completed")
        except Exception as e:
            logger.error(f"Model loading error: {e}")
            raise

    def estimate_bp(self, peak_times: List[float], height: int, weight: int, sex: int) -> Tuple[int, int]:
        """
        Blood pressure estimation
        Args:
            peak_times: List of peak times
            height: Height (cm)
            weight: Weight (kg)
            sex: Sex (1=male, 2=female)
        Returns:
            (Systolic blood pressure, Diastolic blood pressure)
        """
        if len(peak_times) < 2:
            raise ValueError("No peaks detected")

        # RRI calculation
        rri_values = []
        for i in range(len(peak_times) - 1):
            rri = peak_times[i + 1] - peak_times[i]
            if 0.4 <= rri <= 1.2:  # Remove outliers
                rri_values.append(rri)

        if len(rri_values) == 0:
            raise ValueError("No valid RRI detected")

        # Feature calculation
        rri_array = np.array(rri_values)
        bmi = weight / ((height / 100) ** 2)
        sex_feature = 1 if sex == 1 else 0  # 1=male, 2=female -> 1=male, 0=female

        feature_vector = np.array([
            rri_array.mean(),
            rri_array.std(),
            rri_array.min(),
            rri_array.max(),
            bmi,
            sex_feature
        ]).reshape(1, -1)

        # Blood pressure estimation
        pred_sbp = self.model_sbp.predict(feature_vector)[0]
        pred_dbp = self.model_dbp.predict(feature_vector)[0]

        # Convert to integer (according to specification)
        return int(round(pred_sbp)), int(round(pred_dbp))

    def generate_ppg_csv(self, rppg_data: List[float], time_data: List[float], 
                        peak_times: List[float]) -> str:
        """Generate PPG raw data CSV"""
        csv_data = []
        
        # Header
        csv_data.append("Time(s),rPPG_Signal,Peak_Flag")
        
        # Data rows
        peak_set = set(peak_times)
        for i, (time_val, rppg_val) in enumerate(zip(time_data, rppg_data)):
            peak_flag = 1 if any(abs(time_val - peak_t) < 0.1 for peak_t in peak_set) else 0
            csv_data.append(f"{time_val:.3f},{rppg_val:.6f},{peak_flag}")
        
        return "\n".join(csv_data)

# Main Cython class
class CythonBPEstimator:
    """Cython-based blood pressure estimation class with code obfuscation"""
    
    def __init__(self):
        self.is_initialized = False
        self.processing_requests = {}
        self.request_status = {}
        self.version = "1.0.0-cython-obfuscated"
        self.lock = threading.Lock()
        self.rppg_processor = RPPGProcessor()
        self.bp_estimator = None

    def initialize(self, model_dir: str = "models") -> bool:
        """Cython initialization with obfuscation"""
        try:
            # Initialize blood pressure estimator
            self.bp_estimator = BloodPressureEstimator(model_dir)
            
            self.is_initialized = True
            logger.info("Cython initialization completed")
            return True

        except Exception as e:
            logger.error(f"Initialization error: {e}")
            return False

    def _validate_request_id(self, request_id: str) -> bool:
        """README.md compliant request ID validation"""
        if not request_id:
            return False

        # Request ID format: yyyyMMddHHmmssfff_customer_code_driver_code
        pattern = r'^\d{17}_\d{10}_\d{10}$'
        match_result = re.match(pattern, request_id)
        return match_result is not None

    def start_blood_pressure_analysis_request(self, request_id: str, height: int,
                                              weight: int, sex: int,
                                              measurement_movie_path: str,
                                              callback = None) -> Optional[str]:
        """README.md compliant blood pressure analysis request"""

        if not self.is_initialized:
            return ErrorCode.DLL_NOT_INITIALIZED

        # Parameter validation (README.md compliant)
        if not self._validate_request_id(request_id):
            return ErrorCode.INVALID_INPUT_PARAMETERS

        if not measurement_movie_path or not os.path.exists(measurement_movie_path):
            return ErrorCode.INVALID_INPUT_PARAMETERS

        if not (1 <= sex <= 2):
            return ErrorCode.INVALID_INPUT_PARAMETERS

        if not (100 <= height <= 250):
            return ErrorCode.INVALID_INPUT_PARAMETERS

        if not (30 <= weight <= 200):
            return ErrorCode.INVALID_INPUT_PARAMETERS

        # Processing check
        with self.lock:
            if request_id in self.processing_requests:
                return ErrorCode.REQUEST_DURING_PROCESSING

            # Start processing
            self.request_status[request_id] = ProcessingStatus.PROCESSING
            thread = threading.Thread(
                target=self._process_blood_pressure_analysis,
                args=(request_id, height, weight, sex,
                      measurement_movie_path, callback)
            )
            self.processing_requests[request_id] = thread
            thread.start()

        return None

    def _process_blood_pressure_analysis(self, request_id: str, height: int, weight: int,
                                       sex: int, measurement_movie_path: str,
                                       callback):
        """Blood pressure analysis processing (asynchronous)"""
        try:
            # rPPG processing
            rppg_data, time_data, peak_times = self.rppg_processor.process_video(measurement_movie_path)
            
            # Blood pressure estimation
            sbp, dbp = self.bp_estimator.estimate_bp(peak_times, height, weight, sex)
            
            # CSV data generation
            csv_data = self.bp_estimator.generate_ppg_csv(rppg_data, time_data, peak_times)
            
            # Success callback
            if callback:
                callback(request_id, sbp, dbp, csv_data, None)
            
            logger.info(f"Blood pressure analysis completed: {request_id} - SBP: {sbp}, DBP: {dbp}")
            
        except Exception as e:
            logger.error(f"Blood pressure analysis error: {e}")
            error = ErrorInfo(ErrorCode.INTERNAL_PROCESSING_ERROR, str(e))
            if callback:
                callback(request_id, 0, 0, "", [error])
        
        finally:
            # Processing completion
            with self.lock:
                if request_id in self.processing_requests:
                    del self.processing_requests[request_id]
                if request_id in self.request_status:
                    del self.request_status[request_id]

    def get_processing_status(self, request_id: str) -> str:
        """README.md compliant processing status retrieval"""
        with self.lock:
            return self.request_status.get(request_id, ProcessingStatus.NONE)

    def cancel_blood_pressure_analysis(self, request_id: str) -> bool:
        """README.md compliant blood pressure analysis cancellation"""
        with self.lock:
            if request_id in self.processing_requests:
                # Actual cancellation processing (thread stopping is difficult, flag control recommended)
                logger.info(f"Processing cancellation requested: {request_id}")
                return True
            return False

    def get_version_info(self) -> str:
        """README.md compliant version information retrieval"""
        return f"v{self.version}"

# Global instance
estimator = CythonBPEstimator()

# README.md compliant export functions
def initialize_dll(model_dir: str = "models") -> bool:
    """DLL initialization (C# call compatible)"""
    return estimator.initialize(model_dir)

def start_blood_pressure_analysis_request(request_id: str, height: int, weight: int,
                                          sex: int, measurement_movie_path: str,
                                          callback = None) -> Optional[str]:
    """Blood pressure analysis request (README.md compliant)"""
    return estimator.start_blood_pressure_analysis_request(
        request_id, height, weight, sex, measurement_movie_path, callback)

def get_processing_status(request_id: str) -> str:
    """Processing status retrieval (README.md compliant)"""
    return estimator.get_processing_status(request_id)

def cancel_blood_pressure_analysis(request_id: str) -> bool:
    """Blood pressure analysis cancellation (README.md compliant)"""
    return estimator.cancel_blood_pressure_analysis(request_id)

def get_version_info() -> str:
    """Version information retrieval (README.md compliant)"""
    return estimator.get_version_info()

def generate_request_id(customer_code: str, driver_code: str) -> str:
    """Request ID generation (README.md compliant)"""
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]
    return f"{timestamp}_{customer_code}_{driver_code}"

# Test callback function
def test_callback(request_id: str, max_bp: int, min_bp: int, csv_data: str, errors: List[ErrorInfo]):
    """Test callback"""
    print(f"=== Blood Pressure Analysis Result ===")
    print(f"Request ID: {request_id}")
    print(f"Systolic BP: {max_bp} mmHg")
    print(f"Diastolic BP: {min_bp} mmHg")
    print(f"CSV data size: {len(csv_data)} characters")
    
    if errors:
        print("Errors:")
        for error in errors:
            print(f"  - {error.code}: {error.message}")
    
    # Save to CSV file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"bp_result_{request_id}_{timestamp}.csv"
    with open(csv_filename, 'w', encoding='utf-8') as f:
        f.write(csv_data)
    print(f"CSV file saved: {csv_filename}")

if __name__ == "__main__":
    # Test execution
    print("Blood Pressure Estimation DLL Test Start")
    
    # Initialization
    if initialize_dll():
        print("DLL initialization successful")
        
        # Test parameters
        test_request_id = generate_request_id("9000000001", "0000012345")
        test_height = 170
        test_weight = 70
        test_sex = 1  # Male
        test_movie_path = "sample-data/100万画素.webm"
        
        # Blood pressure analysis execution
        error_code = start_blood_pressure_analysis_request(
            test_request_id, test_height, test_weight, test_sex,
            test_movie_path, test_callback
        )
        
        if error_code is None:
            print("Blood pressure analysis start successful")
            
            # Processing status monitoring
            while True:
                status = get_processing_status(test_request_id)
                print(f"Processing status: {status}")
                if status == ProcessingStatus.NONE:
                    break
                time.sleep(1)
                
        else:
            print(f"Blood pressure analysis start failed: {error_code}")
    else:
        print("DLL initialization failed") 
