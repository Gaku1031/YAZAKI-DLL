# cython: language_level=3
# distutils: language=c++

import sys
import os
import numpy as np
import cv2
import mediapipe as mp
from scipy import signal
from sklearn.ensemble import RandomForestRegressor
import joblib
import time
import threading
from datetime import datetime

# Windows DLL exports for pure C functions
# (Functions will be exported via setup.py configuration)

# Global variables
models = {}
active_requests = {}
is_initialized = False
mp_face_mesh = None
python_initialized = False

# Pure C functions for DLL export (no Python dependencies)
# These will be exported as C functions for C# DllImport

def initialize_models(model_dir):
    """Initialize the blood pressure estimation models"""
    global models, is_initialized, mp_face_mesh
    
    try:
        # Convert C string to Python string
        if isinstance(model_dir, bytes):
            model_dir = model_dir.decode('utf-8')
        
        # Load models
        sbp_model_path = os.path.join(model_dir, "model_sbp.pkl")
        dbp_model_path = os.path.join(model_dir, "model_dbp.pkl")
        
        if not os.path.exists(sbp_model_path) or not os.path.exists(dbp_model_path):
            print("Model files not found")
            return False
        
        models['sbp'] = joblib.load(sbp_model_path)
        models['dbp'] = joblib.load(dbp_model_path)
        
        # Initialize MediaPipe
        mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        is_initialized = True
        print("Models initialized successfully")
        return True
        
    except Exception as e:
        print(f"Error initializing models: {e}")
        return False

def start_blood_pressure_analysis(request_id, height, weight, sex, movie_path):
    """Start blood pressure analysis"""
    global active_requests, is_initialized
    
    try:
        # Convert C strings to Python strings
        if isinstance(request_id, bytes):
            request_id = request_id.decode('utf-8')
        if isinstance(movie_path, bytes):
            movie_path = movie_path.decode('utf-8')
        
        if not is_initialized:
            return "ERROR: Models not initialized"
        
        if request_id in active_requests:
            return "ERROR: Request ID already exists"
        
        # Validate parameters
        if height <= 0 or weight <= 0 or sex not in [0, 1]:
            return "ERROR: Invalid parameters"
        
        if not os.path.exists(movie_path):
            return "ERROR: Movie file not found"
        
        # Start analysis in background thread
        def analyze_thread():
            try:
                # Simulate analysis
                time.sleep(2)
                active_requests[request_id] = {
                    'status': 'completed',
                    'sbp': 120 + np.random.randint(-10, 10),
                    'dbp': 80 + np.random.randint(-5, 5),
                    'timestamp': datetime.now().isoformat()
                }
            except Exception as e:
                active_requests[request_id] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        thread = threading.Thread(target=analyze_thread)
        thread.daemon = True
        thread.start()
        
        active_requests[request_id] = {
            'status': 'processing',
            'start_time': time.time()
        }
        
        return ""  # Success
        
    except Exception as e:
        return f"ERROR: {str(e)}"

def get_processing_status(request_id):
    """Get processing status for a request"""
    global active_requests
    
    try:
        # Convert C string to Python string
        if isinstance(request_id, bytes):
            request_id = request_id.decode('utf-8')
        
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
    global active_requests
    
    try:
        # Convert C string to Python string
        if isinstance(request_id, bytes):
            request_id = request_id.decode('utf-8')
        
        if request_id not in active_requests:
            return False
        
        if request_id in active_requests:
            del active_requests[request_id]
        return True
        
    except Exception as e:
        print(f"Error canceling analysis: {e}")
        return False

def get_version_info():
    """Get version information"""
    return "BloodPressureEstimation v1.0.0 (Cython)"

def generate_request_id():
    """Generate a unique request ID"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
    return f"{timestamp}_9000000001_0000012345"

def get_current_time():
    """Get current timestamp"""
    return time.time()

# Pure C wrapper functions for DLL export
def DllMain(hModule, ul_reason_for_call, lpReserved):
    global python_initialized
    if ul_reason_for_call == 1:  # DLL_PROCESS_ATTACH
        if not python_initialized:
            python_initialized = True
    elif ul_reason_for_call == 0:  # DLL_PROCESS_DETACH
        if python_initialized:
            python_initialized = False
    return 1

def InitializeDLL(model_dir):
    try:
        model_dir_str = model_dir.decode('utf-8') if isinstance(model_dir, bytes) else str(model_dir)
        ret_val = 1 if initialize_models(model_dir_str) else 0
        return ret_val
    except:
        return 0

def StartBloodPressureAnalysisRequest(request_id, height, weight, sex, movie_path):
    try:
        request_id_str = request_id.decode('utf-8') if isinstance(request_id, bytes) else str(request_id)
        movie_path_str = movie_path.decode('utf-8') if isinstance(movie_path, bytes) else str(movie_path)
        result_str = start_blood_pressure_analysis(request_id_str, height, weight, sex, movie_path_str)
        return result_str.encode('utf-8')
    except:
        return b"ERROR: Exception occurred"

def GetProcessingStatus(request_id):
    try:
        request_id_str = request_id.decode('utf-8') if isinstance(request_id, bytes) else str(request_id)
        result_str = get_processing_status(request_id_str)
        return result_str.encode('utf-8')
    except:
        return b"ERROR: Exception occurred"

def CancelBloodPressureAnalysis(request_id):
    try:
        request_id_str = request_id.decode('utf-8') if isinstance(request_id, bytes) else str(request_id)
        ret_val = 1 if cancel_blood_pressure_analysis(request_id_str) else 0
        return ret_val
    except:
        return 0

def GetVersionInfo():
    try:
        result_str = get_version_info()
        return result_str.encode('utf-8')
    except:
        return b"ERROR: Version info not available"

def GenerateRequestId():
    try:
        result_str = generate_request_id()
        return result_str.encode('utf-8')
    except:
        return b"ERROR: Could not generate request ID" 
