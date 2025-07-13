# cython: language_level=3
# distutils: language=c++
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

"""
Cython DLL Wrapper for C# Integration
Exports Windows DLL functions for blood pressure estimation
"""

# Cython imports
cimport cython
from cpython cimport array
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
from libc.stdlib cimport malloc, free
from libc.string cimport strcpy, strlen

# Windows-specific imports
IF UNAME_SYSNAME == "Windows":
    from libcpp cimport bool
    from cpython.bytes cimport PyBytes_AsString, PyBytes_FromString
    from cpython.unicode cimport PyUnicode_AsUTF8String, PyUnicode_FromString

# Import the main Cython module
from bp_estimation_cython cimport *

# Windows DLL export functions
IF UNAME_SYSNAME == "Windows":
    import ctypes
    from ctypes import wintypes
    
    # Callback type definition for C#
    CallbackType = ctypes.WINFUNCTYPE(
        None,                    # No return value
        ctypes.c_char_p,        # requestId
        ctypes.c_int,           # maxBloodPressure
        ctypes.c_int,           # minBloodPressure
        ctypes.c_char_p,        # measureRowData
        ctypes.c_void_p         # errors
    )
    
    # DLL Export Functions
    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_char_p)
    def InitializeDLL(model_dir_ptr):
        """DLL initialization (Windows DLL export)"""
        try:
            if model_dir_ptr:
                model_dir = ctypes.string_at(model_dir_ptr).decode('utf-8')
            else:
                model_dir = "models"
            return initialize_dll(model_dir)
        except Exception as e:
            print(f"InitializeDLL error: {e}")
            return False
    
    @ctypes.WINFUNCTYPE(ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
                        ctypes.c_int, ctypes.c_char_p, CallbackType)
    def StartBloodPressureAnalysisRequest(request_id_ptr, height, weight, sex,
                                          movie_path_ptr, callback):
        """Blood pressure analysis request (Windows DLL export)"""
        try:
            request_id = ctypes.string_at(request_id_ptr).decode('utf-8')
            movie_path = ctypes.string_at(movie_path_ptr).decode('utf-8')
            
            def py_callback(req_id, sbp, dbp, csv_data, errors):
                if callback:
                    callback(req_id.encode('utf-8'), sbp, dbp,
                             csv_data.encode('utf-8'), None)
            
            error_code = start_blood_pressure_analysis_request(
                request_id, height, weight, sex, movie_path, py_callback)
            return error_code.encode('utf-8') if error_code else b""
        except Exception as e:
            return str(e).encode('utf-8')
    
    @ctypes.WINFUNCTYPE(ctypes.c_char_p, ctypes.c_char_p)
    def GetProcessingStatus(request_id_ptr):
        """Processing status retrieval (Windows DLL export)"""
        try:
            request_id = ctypes.string_at(request_id_ptr).decode('utf-8')
            return get_processing_status(request_id).encode('utf-8')
        except Exception as e:
            return b"none"
    
    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_char_p)
    def CancelBloodPressureAnalysis(request_id_ptr):
        """Blood pressure analysis cancellation (Windows DLL export)"""
        try:
            request_id = ctypes.string_at(request_id_ptr).decode('utf-8')
            return cancel_blood_pressure_analysis(request_id)
        except Exception as e:
            return False
    
    @ctypes.WINFUNCTYPE(ctypes.c_char_p)
    def GetVersionInfo():
        """Version information retrieval (Windows DLL export)"""
        return get_version_info().encode('utf-8')
    
    @ctypes.WINFUNCTYPE(ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p)
    def GenerateRequestId(customer_code_ptr, driver_code_ptr):
        """Request ID generation (Windows DLL export)"""
        try:
            customer_code = ctypes.string_at(customer_code_ptr).decode('utf-8')
            driver_code = ctypes.string_at(driver_code_ptr).decode('utf-8')
            return generate_request_id(customer_code, driver_code).encode('utf-8')
        except Exception as e:
            return str(e).encode('utf-8')

# Non-Windows fallback functions
ELSE:
    def InitializeDLL(model_dir_ptr=None):
        """DLL initialization (non-Windows)"""
        try:
            model_dir = model_dir_ptr.decode('utf-8') if model_dir_ptr else "models"
            return initialize_dll(model_dir)
        except Exception as e:
            print(f"InitializeDLL error: {e}")
            return False
    
    def StartBloodPressureAnalysisRequest(request_id_ptr, height, weight, sex,
                                          movie_path_ptr, callback=None):
        """Blood pressure analysis request (non-Windows)"""
        try:
            request_id = request_id_ptr.decode('utf-8')
            movie_path = movie_path_ptr.decode('utf-8')
            
            error_code = start_blood_pressure_analysis_request(
                request_id, height, weight, sex, movie_path, callback)
            return error_code.encode('utf-8') if error_code else b""
        except Exception as e:
            return str(e).encode('utf-8')
    
    def GetProcessingStatus(request_id_ptr):
        """Processing status retrieval (non-Windows)"""
        try:
            request_id = request_id_ptr.decode('utf-8')
            return get_processing_status(request_id).encode('utf-8')
        except Exception as e:
            return b"none"
    
    def CancelBloodPressureAnalysis(request_id_ptr):
        """Blood pressure analysis cancellation (non-Windows)"""
        try:
            request_id = request_id_ptr.decode('utf-8')
            return cancel_blood_pressure_analysis(request_id)
        except Exception as e:
            return False
    
    def GetVersionInfo():
        """Version information retrieval (non-Windows)"""
        return get_version_info().encode('utf-8')
    
    def GenerateRequestId(customer_code_ptr, driver_code_ptr):
        """Request ID generation (non-Windows)"""
        try:
            customer_code = customer_code_ptr.decode('utf-8')
            driver_code = driver_code_ptr.decode('utf-8')
            return generate_request_id(customer_code, driver_code).encode('utf-8')
        except Exception as e:
            return str(e).encode('utf-8')

# Test function for verification
def test_cython_dll():
    """Test function to verify Cython DLL functionality"""
    print("Testing Cython DLL functionality...")
    
    # Test initialization
    if InitializeDLL():
        print("DLL initialization successful")
    else:
        print("DLL initialization failed")
        return False
    
    # Test version info
    version = GetVersionInfo().decode('utf-8')
    print(f"Version: {version}")
    
    # Test request ID generation
    customer_code = "9000000001"
    driver_code = "0000012345"
    request_id = GenerateRequestId(customer_code.encode('utf-8'), 
                                  driver_code.encode('utf-8')).decode('utf-8')
    print(f"Generated request ID: {request_id}")
    
    # Test processing status
    status = GetProcessingStatus(request_id.encode('utf-8')).decode('utf-8')
    print(f"Processing status: {status}")
    
    print("All Cython DLL tests passed")
    return True

if __name__ == "__main__":
    test_cython_dll() 
