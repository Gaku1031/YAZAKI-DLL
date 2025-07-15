// CppWrapperDLL.cpp
#include "BloodPressureWrapper.h"
#include <Python.h>
#include <windows.h>
#include <string>
#include <mutex>

static std::mutex pyMutex;
static bool pythonInitialized = false;
static PyObject* pModule = nullptr;

// PythonランタイムとCython DLLの初期化
extern "C" __declspec(dllexport)
int InitializeBP(const char* model_dir) {
    std::lock_guard<std::mutex> lock(pyMutex);
    if (!pythonInitialized) {
        Py_Initialize();
        pythonInitialized = true;
    }
    // GIL確保
    PyGILState_STATE gstate = PyGILState_Ensure();
    // sys.pathにカレントディレクトリとpython_depsを追加
    PyRun_SimpleString("import sys; sys.path.insert(0, './python_deps'); sys.path.insert(0, './')");
    // Cython拡張モジュール（pyd）をimport
    if (pModule) {
        Py_DECREF(pModule);
        pModule = nullptr;
    }
    pModule = PyImport_ImportModule("BloodPressureEstimation");
    int result = 0;
    if (pModule) {
        PyObject* pFunc = PyObject_GetAttrString(pModule, "InitializeDLL");
        if (pFunc && PyCallable_Check(pFunc)) {
            PyObject* pArgs = Py_BuildValue("(s)", model_dir);
            PyObject* pResult = PyObject_CallObject(pFunc, pArgs);
            if (pResult) {
                result = PyLong_AsLong(pResult);
                Py_DECREF(pResult);
            } else {
                PyErr_Print();
            }
            Py_DECREF(pArgs);
            Py_DECREF(pFunc);
        } else {
            PyErr_Print();
            Py_XDECREF(pFunc);
        }
    } else {
        PyErr_Print();
    }
    PyGILState_Release(gstate);
    return result;
}

static std::string CallStringFunc(const char* funcName, const char* fmt = nullptr, ...) {
    std::lock_guard<std::mutex> lock(pyMutex);
    if (!pythonInitialized || !pModule) return "ERROR: Not initialized";
    PyGILState_STATE gstate = PyGILState_Ensure();
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
            result = PyUnicode_AsUTF8(pResult);
            Py_DECREF(pResult);
        } else {
            PyErr_Print();
            result = "ERROR: Exception in ";
            result += funcName;
        }
        if (pArgs) Py_DECREF(pArgs);
        Py_DECREF(pFunc);
    } else {
        PyErr_Print();
        result = "ERROR: Not initialized";
        Py_XDECREF(pFunc);
    }
    PyGILState_Release(gstate);
    return result;
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
    PyGILState_STATE gstate = PyGILState_Ensure();
    int result = 0;
    PyObject* pFunc = PyObject_GetAttrString(pModule, "CancelBloodPressureAnalysis");
    if (pFunc && PyCallable_Check(pFunc)) {
        PyObject* pArgs = Py_BuildValue("(s)", request_id);
        PyObject* pResult = PyObject_CallObject(pFunc, pArgs);
        if (pResult) {
            result = PyObject_IsTrue(pResult);
            Py_DECREF(pResult);
        } else {
            PyErr_Print();
        }
        Py_DECREF(pArgs);
        Py_DECREF(pFunc);
    } else {
        PyErr_Print();
        Py_XDECREF(pFunc);
    }
    PyGILState_Release(gstate);
    return result;
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
