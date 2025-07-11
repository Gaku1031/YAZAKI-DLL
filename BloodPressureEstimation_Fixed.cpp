#include "BloodPressureEstimation_Fixed.h"
#include <Python.h>
#include <string>
#include <map>
#include <memory>
#include <iostream>

// Global variables
static bool g_initialized = false;
static PyObject* g_bp_module = nullptr;
static PyObject* g_estimator = nullptr;
static std::map<std::string, std::string> g_status_cache;

// String buffers for return values
static std::string g_last_error;
static std::string g_last_status;
static std::string g_version_info;

// Python engine initialization
bool InitializePython() {
    if (Py_IsInitialized()) {
        return true;
    }

    try {
        // Initialize Python interpreter
        Py_Initialize();
        if (!Py_IsInitialized()) {
            g_last_error = "Python initialization failed";
            return false;
        }

        // Add current directory to sys.path
        PyRun_SimpleString("import sys");
        PyRun_SimpleString("sys.path.append('.')");
        
        // Import blood pressure estimation module
        g_bp_module = PyImport_ImportModule("bp_estimation_simple");
        if (!g_bp_module) {
            PyErr_Print();
            g_last_error = "Failed to import bp_estimation_simple module";
            return false;
        }

        // Get estimator instance
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

// Python method call helper
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

// Export function implementations
extern "C" {

BLOODPRESSURE_API bool InitializeDLL(const char* modelDir) {
    try {
        if (g_initialized) {
            return true;
        }

        // Initialize Python engine
        if (!InitializePython()) {
            return false;
        }

        // Initialize Python estimator
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

        // Call Python function
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
                g_last_error = ""; // Success
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

// DLL entry point
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