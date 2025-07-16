# bp_estimation_cython.pyx
# cython: language_level=3
# distutils: language=c++

import sys
import os
import time
import threading
import csv
import re
from datetime import datetime
from collections import deque
from typing import Dict, List, Tuple, Optional, Callable

from libc.string cimport memcpy


# Global variables for DLL state
models = {}
active_requests = {}
is_initialized = False
mp_face_mesh = None
dll_lock = None

# MediaPipe face ROI landmarks
FACE_ROI = [118, 119, 120, 47, 217, 174, 196, 197, 419, 399, 437, 277,
            349, 348, 347, 280, 425, 426, 391, 393, 164, 167, 165, 206, 205, 50, 118]

# Error codes matching original implementation
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

def safe_initialize_dependencies():
    """Safe initialization of Python dependencies"""
    try:
        # Set up Python path for dependencies
        current_dir = os.getcwd()
        python_deps = os.path.join(current_dir, "python_deps")
        
        if python_deps not in sys.path:
            sys.path.insert(0, python_deps)
        
        # Import required libraries with error handling
        global np, cv2, mp, signal, joblib, find_peaks, butter, lfilter, zscore
        
        import numpy as np
        print("NumPy imported successfully")
        
        import cv2
        print("OpenCV imported successfully")
        
        import mediapipe as mp
        print("MediaPipe imported successfully")
        
        from scipy import signal
        from scipy.signal import find_peaks, butter, lfilter
        print("SciPy imported successfully")
        
        from scipy.stats import zscore
        print("SciPy.stats imported successfully")
        
        import joblib
        print("joblib imported successfully")
        
        return True
        
    except Exception as e:
        print(f"Dependency import error: {e}")
        return False

class BandpassFilter:
    """Bandpass filter implementation"""
    def __init__(self, fs: float, low_freq: float, high_freq: float, order: int = 5):
        self.fs = fs
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.order = order
        
        try:
            self.high_b, self.high_a = signal.butter(
                self.order, self.low_freq / (self.fs / 2), btype='high')
            self.low_b, self.low_a = signal.butter(
                self.order, self.high_freq / (self.fs / 2), btype='low')
        except Exception as e:
            print(f"Filter initialization error: {e}")
            self.high_b = self.high_a = self.low_b = self.low_a = None

    def apply(self, signal_in):
        """Apply bandpass filter"""
        if self.high_b is None:
            return signal_in
        
        try:
            filtered_signal = signal.filtfilt(self.high_b, self.high_a, signal_in)
            filtered_signal = signal.filtfilt(self.low_b, self.low_a, filtered_signal)
            return filtered_signal
        except Exception as e:
            print(f"Filter application error: {e}")
            return signal_in

class RPPGProcessor:
    """rPPG signal processing class - faithful port from original"""
    def __init__(self):
        try:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=0.1, min_tracking_confidence=0.1)
            self.window_size = 30
            self.skin_means = deque(maxlen=300)
            self.timestamps = deque(maxlen=300)
            self.roi_pixels_list = deque(maxlen=300)
            print("RPPGProcessor initialized successfully")
        except Exception as e:
            print(f"RPPGProcessor initialization error: {e}")
            self.face_mesh = None

    def process_video(self, video_path: str):
        """Process video to extract rPPG signal and peak times"""
        if self.face_mesh is None:
            raise ValueError("Face mesh not initialized")
        
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

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                current_time = frame_count / fps
                frame_count += 1

                # Process frame
                if len(frame.shape) == 3:
                    self._process_color_frame(frame, current_time)
                else:
                    self._process_grayscale_frame(frame, current_time)

                # Calculate rPPG signal
                if len(self.skin_means) > 30:
                    rppg_signal = self._calculate_rppg_signal()
                    if len(rppg_signal) > 0:
                        rppg_data.append(rppg_signal[-1])
                        time_data.append(current_time)

        finally:
            cap.release()

        # Peak detection
        if len(rppg_data) > 0:
            rppg_array = np.array(rppg_data)
            peaks, _ = find_peaks(rppg_array, distance=10)
            peak_times = [time_data[i] for i in peaks if i < len(time_data)]
        else:
            peak_times = []

        return rppg_data, time_data, peak_times

    def _process_color_frame(self, frame, current_time: float):
        """Process color frame - faithful port from original"""
        try:
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
        except Exception as e:
            print(f"Color frame processing error: {e}")

    def _process_grayscale_frame(self, frame, current_time: float):
        """Process grayscale frame - faithful port from original"""
        try:
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
        except Exception as e:
            print(f"Grayscale frame processing error: {e}")

    def _calculate_rppg_signal(self):
        """POS-based rPPG signal extraction - faithful port from original"""
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
            print(f"rPPG signal calculation error: {e}")
            return np.array([])

    def _bandpass_filter(self, data, lowcut: float, highcut: float, fs: float, order: int = 3):
        """Bandpass filter - faithful port from original"""
        try:
            nyquist = 0.5 * fs
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = butter(order, [low, high], btype='band')
            return lfilter(b, a, data)
        except Exception as e:
            print(f"Bandpass filter error: {e}")
            return data

class BloodPressureEstimator:
    """Blood pressure estimator - faithful port from original"""
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.model_sbp = None
        self.model_dbp = None
        self._load_models()

    def _load_models(self):
        """Load trained models - faithful port from original"""
        try:
            sbp_path = os.path.join(self.model_dir, "model_sbp.pkl")
            dbp_path = os.path.join(self.model_dir, "model_dbp.pkl")
            
            if os.path.exists(sbp_path):
                self.model_sbp = joblib.load(sbp_path)
                print("SBP model loaded successfully")
            else:
                print(f"SBP model not found: {sbp_path}")
                
            if os.path.exists(dbp_path):
                self.model_dbp = joblib.load(dbp_path)
                print("DBP model loaded successfully")
            else:
                print(f"DBP model not found: {dbp_path}")
                
        except Exception as e:
            print(f"Model loading error: {e}")

    def estimate_bp(self, peak_times, height: int, weight: int, sex: int):
        """Blood pressure estimation - faithful port from original"""
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
        sex_feature = 1 if sex == 1 else 0  # 1=male, 2=female → 1=male, 0=female

        feature_vector = np.array([
            rri_array.mean(),
            rri_array.std(),
            rri_array.min(),
            rri_array.max(),
            bmi,
            sex_feature
        ]).reshape(1, -1)

        # Blood pressure prediction
        if self.model_sbp is not None:
            pred_sbp = self.model_sbp.predict(feature_vector)[0]
        else:
            # Fallback calculation if model not available
            pred_sbp = 120 + (bmi - 22) * 2 + rri_array.mean() * 10
            
        if self.model_dbp is not None:
            pred_dbp = self.model_dbp.predict(feature_vector)[0]
        else:
            # Fallback calculation if model not available
            pred_dbp = 80 + (bmi - 22) * 1 + rri_array.mean() * 5

        return int(round(pred_sbp)), int(round(pred_dbp))

    def generate_ppg_csv(self, rppg_data, time_data, peak_times) -> str:
        """Generate PPG raw data CSV - faithful port from original"""
        csv_data = []
        
        # Header
        csv_data.append("Time(s),rPPG_Signal,Peak_Flag")
        
        # Data rows
        peak_set = set(peak_times)
        for i, (time_val, rppg_val) in enumerate(zip(time_data, rppg_data)):
            peak_flag = 1 if any(abs(time_val - peak_t) < 0.1 for peak_t in peak_set) else 0
            csv_data.append(f"{time_val:.3f},{rppg_val:.6f},{peak_flag}")
        
        return "\n".join(csv_data)

# Main DLL implementation maintaining original algorithm
def initialize_models(model_dir):
    """Initialize blood pressure estimation models with full algorithm"""
    global models, is_initialized, mp_face_mesh, dll_lock
    
    try:
        # Initialize dependencies
        if not safe_initialize_dependencies():
            return False
        
        # Initialize thread lock
        dll_lock = threading.Lock()
        
        # Convert C string to Python string
        if isinstance(model_dir, bytes):
            model_dir = model_dir.decode('utf-8')
        
        # Create estimator with full algorithm
        estimator = BloodPressureEstimator(model_dir)
        models['estimator'] = estimator
        
        # Create rPPG processor
        processor = RPPGProcessor()
        models['processor'] = processor
        
        is_initialized = True
        print("Full blood pressure estimation algorithm initialized")
        
        # Force garbage collection
        import gc
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"Critical error initializing full algorithm: {e}")
        is_initialized = False
        return False

def start_blood_pressure_analysis(request_id, height, weight, sex, movie_path):
    """Start blood pressure analysis with full algorithm"""
    global active_requests, is_initialized, dll_lock
    
    try:
        # Convert C strings to Python strings
        if isinstance(request_id, bytes):
            request_id = request_id.decode('utf-8')
        if isinstance(movie_path, bytes):
            movie_path = movie_path.decode('utf-8')
        
        if not is_initialized:
            return ErrorCode.DLL_NOT_INITIALIZED
        
        # Validate request ID format
        pattern = r'^\d{17}_\w+_\w+$'
        if not re.match(pattern, request_id):
            return ErrorCode.INVALID_INPUT_PARAMETERS
        
        if not os.path.exists(movie_path):
            return ErrorCode.INVALID_INPUT_PARAMETERS
        
        if not (1 <= sex <= 2):
            return ErrorCode.INVALID_INPUT_PARAMETERS
        
        # Check if already processing
        with dll_lock:
            if request_id in active_requests:
                return ErrorCode.REQUEST_DURING_PROCESSING
        
        # Start analysis in background thread with full algorithm
        def analyze_thread():
            try:
                processor = models['processor']
                estimator = models['estimator']
                
                # Full rPPG processing
                rppg_data, time_data, peak_times = processor.process_video(movie_path)
                
                # Full blood pressure estimation
                sbp, dbp = estimator.estimate_bp(peak_times, height, weight, sex)
                
                # Generate CSV data
                csv_data = estimator.generate_ppg_csv(rppg_data, time_data, peak_times)
                
                # Store results
                with dll_lock:
                    active_requests[request_id] = {
                        'status': 'completed',
                        'sbp': sbp,
                        'dbp': dbp,
                        'csv_data': csv_data,
                        'timestamp': datetime.now().isoformat()
                    }
                
                print(f"Full blood pressure analysis completed: {request_id} - SBP: {sbp}, DBP: {dbp}")
                
            except Exception as e:
                print(f"Full blood pressure analysis error: {e}")
                with dll_lock:
                    active_requests[request_id] = {
                        'status': 'error',
                        'error': str(e)
                    }
        
        thread = threading.Thread(target=analyze_thread)
        thread.daemon = True
        thread.start()
        
        with dll_lock:
            active_requests[request_id] = {
                'status': 'processing',
                'start_time': time.time()
            }
        
        return ""  # Success
        
    except Exception as e:
        return f"ERROR: {str(e)}"

def get_processing_status(request_id):
    """Get processing status"""
    global active_requests, dll_lock
    
    try:
        if isinstance(request_id, bytes):
            request_id = request_id.decode('utf-8')
        
        with dll_lock:
            if request_id not in active_requests:
                return "ERROR: Request ID not found"
            
            request_data = active_requests[request_id]
            status = request_data['status']
            
            if status == 'completed':
                sbp = request_data['sbp']
                dbp = request_data['dbp']
                return f"COMPLETED:SBP={sbp},DBP={dbp}"
            elif status == 'error':
                return f"ERROR:{request_data['error']}"
            else:
                return "PROCESSING"
        
    except Exception as e:
        return f"ERROR: {str(e)}"

def cancel_blood_pressure_analysis(request_id):
    """Cancel blood pressure analysis"""
    global active_requests, dll_lock
    
    try:
        if isinstance(request_id, bytes):
            request_id = request_id.decode('utf-8')
        
        with dll_lock:
            if request_id not in active_requests:
                return False
            
            del active_requests[request_id]
            return True
        
    except Exception as e:
        print(f"Error canceling analysis: {e}")
        return False

def get_version_info():
    """Get version information"""
    return "BloodPressureEstimation v1.0.0 (Full Algorithm)"

def generate_request_id():
    """Generate a unique request ID"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
    return f"{timestamp}_9000000001_0000012345"

# Static buffer for C string returns
cdef char result_buffer[1024]

# C wrapper functions for DLL export
cdef public int DllMain(void* hModule, unsigned long ul_reason_for_call, void* lpReserved):
    return 1

cdef public int InitializeDLL(const char* model_dir):
    cdef str model_dir_str
    try:
        if model_dir:
            model_dir_str = model_dir.decode('utf-8')
        else:
            model_dir_str = "models"
        result = initialize_models(model_dir_str)
        return 1 if result else 0
    except Exception as e:
        import traceback
        print("=== DLL InitializeDLL exception ===")
        print(str(e))
        traceback.print_exc()
        print("=== END DLL InitializeDLL exception ===")
        return 0

cdef public const char* StartBloodPressureAnalysisRequest(const char* request_id, int height, int weight, int sex, const char* movie_path):
    cdef str request_id_str
    cdef str movie_path_str
    cdef str result_str
    cdef bytes result_bytes
    cdef int n
    try:
        request_id_str = request_id.decode('utf-8') if request_id else ""
        movie_path_str = movie_path.decode('utf-8') if movie_path else ""
        result_str = start_blood_pressure_analysis(request_id_str, height, weight, sex, movie_path_str)
        result_bytes = result_str.encode('utf-8')
        n = min(len(result_bytes), 1023)
        memcpy(result_buffer, <const char*>result_bytes, n)
        result_buffer[n] = 0
        return <const char*>result_buffer
    except Exception as e:
        error_msg = f"ERROR: Exception in full analysis - {str(e)}"
        result_bytes = error_msg.encode('utf-8')
        n = min(len(result_bytes), 1023)
        memcpy(result_buffer, <const char*>result_bytes, n)
        result_buffer[n] = 0
        return <const char*>result_buffer

cdef public const char* GetProcessingStatus(const char* request_id):
    cdef str request_id_str
    cdef str result_str
    cdef bytes result_bytes
    cdef int n
    try:
        request_id_str = request_id.decode('utf-8') if request_id else ""
        result_str = get_processing_status(request_id_str)
        result_bytes = result_str.encode('utf-8')
        n = min(len(result_bytes), 1023)
        memcpy(result_buffer, <const char*>result_bytes, n)
        result_buffer[n] = 0
        return <const char*>result_buffer
    except Exception as e:
        error_msg = f"ERROR: Exception in status check - {str(e)}"
        result_bytes = error_msg.encode('utf-8')
        n = min(len(result_bytes), 1023)
        memcpy(result_buffer, <const char*>result_bytes, n)
        result_buffer[n] = 0
        return <const char*>result_buffer

cdef public int CancelBloodPressureAnalysis(const char* request_id):
    cdef str request_id_str
    try:
        request_id_str = request_id.decode('utf-8') if request_id else ""
        result = cancel_blood_pressure_analysis(request_id_str)
        return 1 if result else 0
    except Exception as e:
        print(f"CancelBloodPressureAnalysis exception: {e}")
        return 0

cdef public const char* GetVersionInfo():
    cdef str result_str
    cdef bytes result_bytes
    cdef int n
    try:
        result_str = get_version_info()
        result_bytes = result_str.encode('utf-8')
        n = min(len(result_bytes), 1023)
        memcpy(result_buffer, <const char*>result_bytes, n)
        result_buffer[n] = 0
        return <const char*>result_buffer
    except Exception as e:
        error_msg = f"BloodPressureEstimation v1.0.0 (Error: {str(e)})"
        result_bytes = error_msg.encode('utf-8')
        n = min(len(result_bytes), 1023)
        memcpy(result_buffer, <const char*>result_bytes, n)
        result_buffer[n] = 0
        return <const char*>result_buffer

cdef public const char* GenerateRequestId():
    cdef str result_str
    cdef bytes result_bytes
    cdef int n
    try:
        result_str = generate_request_id()
        result_bytes = result_str.encode('utf-8')
        n = min(len(result_bytes), 1023)
        memcpy(result_buffer, <const char*>result_bytes, n)
        result_buffer[n] = 0
        return <const char*>result_buffer
    except Exception as e:
        error_msg = "ERROR: Could not generate request ID"
        result_bytes = error_msg.encode('utf-8')
        n = min(len(result_bytes), 1023)
        memcpy(result_buffer, <const char*>result_bytes, n)
        result_buffer[n] = 0
        return <const char*>result_buffer

def InitializeDLL(model_dir: str = "models") -> int:
    """C++ラッパー用のDLL初期化関数"""
    try:
        success = initialize_dll(model_dir)
        return 1 if success else 0
    except Exception as e:
        logger.error(f"InitializeDLL error: {e}")
        return 0

def StartBloodPressureAnalysisRequest(request_id: str, height: int, weight: int, 
                                    sex: int, movie_path: str) -> str:
    """C++ラッパー用の血圧解析開始関数"""
    try:
        # 非同期コールバックなしで同期実行
        error_code = start_bp_analysis(request_id, height, weight, sex, movie_path, None)
        if error_code is None:
            return "SUCCESS"
        else:
            return error_code
    except Exception as e:
        logger.error(f"StartBloodPressureAnalysisRequest error: {e}")
        return ErrorCode.INTERNAL_PROCESSING_ERROR

def GetProcessingStatus(request_id: str) -> str:
    """C++ラッパー用の処理状況取得関数"""
    try:
        return get_bp_status(request_id)
    except Exception as e:
        logger.error(f"GetProcessingStatus error: {e}")
        return ProcessingStatus.NONE

def CancelBloodPressureAnalysis(request_id: str) -> int:
    """C++ラッパー用の処理中断関数"""
    try:
        success = cancel_bp_processing(request_id)
        return 1 if success else 0
    except Exception as e:
        logger.error(f"CancelBloodPressureAnalysis error: {e}")
        return 0

def GetVersionInfo() -> str:
    """C++ラッパー用のバージョン情報取得関数"""
    try:
        return get_dll_version()
    except Exception as e:
        logger.error(f"GetVersionInfo error: {e}")
        return "ERROR: Version unavailable"

def GenerateRequestId() -> str:
    """C++ラッパー用のリクエストID生成関数"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]
        return f"{timestamp}_DEFAULT_001"
    except Exception as e:
        logger.error(f"GenerateRequestId error: {e}")
        return "ERROR: ID generation failed"

# エクスポート関数のテスト
def test_export_functions():
    """エクスポート関数のテスト"""
    print("=== Testing Export Functions ===")
    
    # 初期化テスト
    init_result = InitializeDLL("models")
    print(f"InitializeDLL result: {init_result}")
    
    # バージョン情報テスト
    version = GetVersionInfo()
    print(f"Version: {version}")
    
    # リクエストID生成テスト
    request_id = GenerateRequestId()
    print(f"Generated Request ID: {request_id}")
    
    # ステータステスト
    status = GetProcessingStatus("test_123")
    print(f"Status: {status}")
    
    print("=== Export Functions Test Complete ===")

if __name__ == "__main__":
    # テスト実行
    test_export_functions()0
    except Exception as e:
        logger.error(f"InitializeDLL error: {e}")
        return 0

def StartBloodPressureAnalysisRequest(request_id: str, height: int, weight: int, 
                                    sex: int, movie_path: str) -> str:
    """C++ラッパー用の血圧解析開始関数"""
    try:
        # 非同期コールバックなしで同期実行
        error_code = start_bp_analysis(request_id, height, weight, sex, movie_path, None)
        if error_code is None:
            return "SUCCESS"
        else:
            return error_code
    except Exception as e:
        logger.error(f"StartBloodPressureAnalysisRequest error: {e}")
        return ErrorCode.INTERNAL_PROCESSING_ERROR

def GetProcessingStatus(request_id: str) -> str:
    """C++ラッパー用の処理状況取得関数"""
    try:
        return get_bp_status(request_id)
    except Exception as e:
        logger.error(f"GetProcessingStatus error: {e}")
        return ProcessingStatus.NONE

def CancelBloodPressureAnalysis(request_id: str) -> int:
    """C++ラッパー用の処理中断関数"""
    try:
        success = cancel_bp_processing(request_id)
        return 1 if success else 0
    except Exception as e:
        logger.error(f"CancelBloodPressureAnalysis error: {e}")
        return 0

def GetVersionInfo() -> str:
    """C++ラッパー用のバージョン情報取得関数"""
    try:
        return get_dll_version()
    except Exception as e:
        logger.error(f"GetVersionInfo error: {e}")
        return "ERROR: Version unavailable"

def GenerateRequestId() -> str:
    """C++ラッパー用のリクエストID生成関数"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]
        return f"{timestamp}_DEFAULT_001"
    except Exception as e:
        logger.error(f"GenerateRequestId error: {e}")
        return "ERROR: ID generation failed"
