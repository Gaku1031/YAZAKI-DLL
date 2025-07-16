#include "BloodPressureWrapper.h"
#include <Python.h>
#include <windows.h>
#include <string>
#include <mutex>
#include <iostream>

static std::mutex pyMutex;
static bool pythonInitialized = false;
static PyObject* pModule = nullptr;
static std::string lastError;

static void ClearPythonError() {
    if (PyErr_Occurred()) {
        PyObject *ptype, *pvalue, *ptraceback;
        PyErr_Fetch(&ptype, &pvalue, &ptraceback);
        
        if (pvalue) {
            PyObject* str = PyObject_Str(pvalue);
            if (str) {
                const char* errorStr = PyUnicode_AsUTF8(str);
                if (errorStr) {
                    lastError = std::string("Python Error: ") + errorStr;
                }
                Py_DECREF(str);
            }
        }
        
        Py_XDECREF(ptype);
        Py_XDECREF(pvalue);
        Py_XDECREF(ptraceback);
    }
}

static bool SafeInitializePython() {
    try {
        if (pythonInitialized) {
            return true;
        }

        // Python環境の設定
        const char* pythonHome = ".";  // 現在のディレクトリを設定
        const wchar_t* pythonHomeW = L".";
        
        // PythonHomeを設定（重要）
        Py_SetPythonHome(const_cast<wchar_t*>(pythonHomeW));
        
        // Python初期化
        if (!Py_IsInitialized()) {
            Py_Initialize();
        }
        
        if (!Py_IsInitialized()) {
            lastError = "Failed to initialize Python interpreter";
            return false;
        }

        // GIL状態の確認と設定
        if (!PyGILState_Check()) {
            PyEval_InitThreads();
        }

        pythonInitialized = true;
        return true;
        
    } catch (...) {
        lastError = "Exception during Python initialization";
        return false;
    }
}

extern "C" __declspec(dllexport)
int InitializeBP(const char* model_dir) {
    std::lock_guard<std::mutex> lock(pyMutex);
    
    try {
        // 安全なPython初期化
        if (!SafeInitializePython()) {
            return 0;
        }
        
        // GIL確保
        PyGILState_STATE gstate = PyGILState_Ensure();
        
        try {
            // sys.pathの設定（詳細）
            PyRun_SimpleString(
                "import sys\n"
                "import os\n"
                "current_dir = os.path.dirname(os.path.abspath('.'))\n"
                "python_deps = os.path.join(current_dir, 'python_deps')\n"
                "if current_dir not in sys.path:\n"
                "    sys.path.insert(0, current_dir)\n"
                "if python_deps not in sys.path:\n"
                "    sys.path.insert(0, python_deps)\n"
                "if os.path.join(python_deps, 'numpy') not in sys.path:\n"
                "    sys.path.insert(0, os.path.join(python_deps, 'numpy'))\n"
                "if os.path.join(python_deps, 'cv2') not in sys.path:\n"
                "    sys.path.insert(0, os.path.join(python_deps, 'cv2'))\n"
                "print('Python paths configured')"
            );
            
            if (PyErr_Occurred()) {
                ClearPythonError();
                PyGILState_Release(gstate);
                return 0;
            }

            // 依存関係のプリロード
            PyRun_SimpleString(
                "try:\n"
                "    import numpy as np\n"
                "    print(f'NumPy {np.__version__} loaded successfully')\n"
                "except Exception as e:\n"
                "    print(f'NumPy load failed: {e}')\n"
                "try:\n"
                "    import cv2\n"
                "    print(f'OpenCV {cv2.__version__} loaded successfully')\n"
                "except Exception as e:\n"
                "    print(f'OpenCV load failed: {e}')"
            );

            // Cython拡張モジュール（pyd）をimport
            if (pModule) {
                Py_DECREF(pModule);
                pModule = nullptr;
            }
            
            pModule = PyImport_ImportModule("BloodPressureEstimation");
            if (!pModule) {
                ClearPythonError();
                PyGILState_Release(gstate);
                return 0;
            }

            int result = 0;
            PyObject* pFunc = PyObject_GetAttrString(pModule, "InitializeDLL");
            if (pFunc && PyCallable_Check(pFunc)) {
                PyObject* pArgs = Py_BuildValue("(s)", model_dir ? model_dir : "models");
                if (pArgs) {
                    PyObject* pResult = PyObject_CallObject(pFunc, pArgs);
                    if (pResult) {
                        if (PyLong_Check(pResult)) {
                            result = (int)PyLong_AsLong(pResult);
                        } else if (PyBool_Check(pResult)) {
                            result = PyObject_IsTrue(pResult);
                        }
                        Py_DECREF(pResult);
                    } else {
                        ClearPythonError();
                    }
                    Py_DECREF(pArgs);
                }
                Py_DECREF(pFunc);
            } else {
                ClearPythonError();
                Py_XDECREF(pFunc);
            }
            
            PyGILState_Release(gstate);
            return result;
            
        } catch (...) {
            PyGILState_Release(gstate);
            lastError = "Exception in Python execution";
            return 0;
        }
        
    } catch (...) {
        lastError = "Exception in InitializeBP";
        return 0;
    }
}

static std::string CallStringFunc(const char* funcName, const char* fmt = nullptr, ...) {
    std::lock_guard<std::mutex> lock(pyMutex);
    
    if (!pythonInitialized || !pModule) {
        return "ERROR: Not initialized";
    }
    
    try {
        PyGILState_STATE gstate = PyGILState_Ensure();
        
        try {
            std::string result = "";
            PyObject* pFunc = PyObject_GetAttrString(pModule, funcName);
            
            if (pFunc && PyCallable_Check(pFunc)) {
                PyObject* pArgs = nullptr;
                
                if (fmt) {
                    va_list args;
                    va_start(args, fmt);
                    pArgs = Py_VaBuildValue(fmt, args);
                    va_end(args);
                }
                
                PyObject* pResult = PyObject_CallObject(pFunc, pArgs);
                if (pResult) {
                    if (PyUnicode_Check(pResult)) {
                        const char* resultStr = PyUnicode_AsUTF8(pResult);
                        if (resultStr) {
                            result = resultStr;
                        }
                    }
                    Py_DECREF(pResult);
                } else {
                    ClearPythonError();
                    result = "ERROR: Exception in " + std::string(funcName);
                }
                
                if (pArgs) Py_DECREF(pArgs);
                Py_DECREF(pFunc);
            } else {
                ClearPythonError();
                result = "ERROR: Function not found: " + std::string(funcName);
                Py_XDECREF(pFunc);
            }
            
            PyGILState_Release(gstate);
            return result;
            
        } catch (...) {
            PyGILState_Release(gstate);
            return "ERROR: Exception in function call";
        }
        
    } catch (...) {
        return "ERROR: Exception in CallStringFunc";
    }
}

extern "C" __declspec(dllexport)
const char* StartBloodPressureAnalysisRequest(const char* request_id, int height, int weight, int sex, const char* movie_path) {
    static thread_local std::string ret;
    ret = CallStringFunc("StartBloodPressureAnalysisRequest", "siiis", request_id, height, weight, sex, movie_path);
    return ret.c_str();
}

extern "C" __declspec(dllexport)
const char* GetProcessingStatus(const char* request_id) {
    static thread_local std::string ret;
    ret = CallStringFunc("GetProcessingStatus", "s", request_id);
    return ret.c_str();
}

extern "C" __declspec(dllexport)
int CancelBloodPressureAnalysis(const char* request_id) {
    std::lock_guard<std::mutex> lock(pyMutex);
    
    if (!pythonInitialized || !pModule) return 0;
    
    try {
        PyGILState_STATE gstate = PyGILState_Ensure();
        
        try {
            int result = 0;
            PyObject* pFunc = PyObject_GetAttrString(pModule, "CancelBloodPressureAnalysis");
            
            if (pFunc && PyCallable_Check(pFunc)) {
                PyObject* pArgs = Py_BuildValue("(s)", request_id);
                if (pArgs) {
                    PyObject* pResult = PyObject_CallObject(pFunc, pArgs);
                    if (pResult) {
                        result = PyObject_IsTrue(pResult);
                        Py_DECREF(pResult);
                    } else {
                        ClearPythonError();
                    }
                    Py_DECREF(pArgs);
                }
                Py_DECREF(pFunc);
            } else {
                ClearPythonError();
                Py_XDECREF(pFunc);
            }
            
            PyGILState_Release(gstate);
            return result;
            
        } catch (...) {
            PyGILState_Release(gstate);
            return 0;
        }
        
    } catch (...) {
        return 0;
    }
}

extern "C" __declspec(dllexport)
const char* GetVersionInfo() {
    static thread_local std::string ret;
    ret = CallStringFunc("GetVersionInfo");
    return ret.c_str();
}

extern "C" __declspec(dllexport)
const char* GenerateRequestId() {
    static thread_local std::string ret;
    ret = CallStringFunc("GenerateRequestId");
    return ret.c_str();
}

// エラー情報取得関数（デバッグ用）
extern "C" __declspec(dllexport)
const char* GetLastError() {
    static std::string error_copy = lastError;
    return error_copy.c_str();
}

// DLL終了処理
extern "C" __declspec(dllexport)
void CleanupBP() {
    std::lock_guard<std::mutex> lock(pyMutex);
    
    if (pModule) {
        Py_DECREF(pModule);
        pModule = nullptr;
    }
    
    if (pythonInitialized) {
        if (Py_IsInitialized()) {
            Py_Finalize();
        }
        pythonInitialized = false;
    }
}

// DLLエントリポイント
BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) {
    switch (ul_reason_for_call) {
    case DLL_PROCESS_ATTACH:
        // DLL読み込み時の初期化
        break;
    case DLL_PROCESS_DETACH:
        // DLL解放時のクリーンアップ
        CleanupBP();
        break;
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
        break;
    }
    return TRUE;
}
