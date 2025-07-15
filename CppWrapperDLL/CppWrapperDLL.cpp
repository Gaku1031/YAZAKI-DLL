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
int InitializeBP(const char* model_dir) {
    std::lock_guard<std::mutex> lock(pyMutex);
    if (!pythonInitialized) {
        if (!Py_IsInitialized()) {
            Py_Initialize();
        }
        pythonInitialized = true;
    }
    // sys.pathにカレントディレクトリとpython_depsを追加
    PyRun_SimpleString("import sys; sys.path.insert(0, './python_deps'); sys.path.insert(0, './')");
    // Cython拡張モジュール（pyd）をimport
    if (pModule) {
        Py_DECREF(pModule);
        pModule = nullptr;
    }
    pModule = PyImport_ImportModule("BloodPressureEstimation");
    if (!pModule) {
        PyErr_Print();
        return 0;
    }
    // Python関数呼び出し例（InitializeDLL）
    PyObject* pFunc = PyObject_GetAttrString(pModule, "InitializeDLL");
    if (pFunc && PyCallable_Check(pFunc)) {
        PyObject* pArgs = Py_BuildValue("(s)", model_dir);
        PyObject* pResult = PyObject_CallObject(pFunc, pArgs);
        int result = 0;
        if (pResult) {
            result = PyLong_AsLong(pResult);
            Py_DECREF(pResult);
        } else {
            PyErr_Print();
        }
        Py_DECREF(pArgs);
        Py_DECREF(pFunc);
        return result;
    }
    PyErr_Print();
    Py_XDECREF(pFunc);
    return 0;
}

const char* StartBloodPressureAnalysisRequest(const char* request_id, int height, int weight, int sex, const char* movie_path) {
    std::lock_guard<std::mutex> lock(pyMutex);
    if (!pythonInitialized || !pModule) return "ERROR: Not initialized";
    PyObject* pFunc = PyObject_GetAttrString(pModule, "StartBloodPressureAnalysisRequest");
    if (pFunc && PyCallable_Check(pFunc)) {
        PyObject* pArgs = Py_BuildValue("(siiis)", request_id, height, weight, sex, movie_path);
        PyObject* pResult = PyObject_CallObject(pFunc, pArgs);
        const char* result = "";
        if (pResult) {
            result = _strdup(PyUnicode_AsUTF8(pResult));
            Py_DECREF(pResult);
        } else {
            PyErr_Print();
            result = "ERROR: Exception in StartBloodPressureAnalysisRequest";
        }
        Py_DECREF(pArgs);
        Py_DECREF(pFunc);
        return result;
    }
    PyErr_Print();
    Py_XDECREF(pFunc);
    return "ERROR: Not initialized";
}

const char* GetProcessingStatus(const char* request_id) {
    std::lock_guard<std::mutex> lock(pyMutex);
    if (!pythonInitialized || !pModule) return "ERROR: Not initialized";
    PyObject* pFunc = PyObject_GetAttrString(pModule, "GetProcessingStatus");
    if (pFunc && PyCallable_Check(pFunc)) {
        PyObject* pArgs = Py_BuildValue("(s)", request_id);
        PyObject* pResult = PyObject_CallObject(pFunc, pArgs);
        const char* result = "";
        if (pResult) {
            result = _strdup(PyUnicode_AsUTF8(pResult));
            Py_DECREF(pResult);
        } else {
            PyErr_Print();
            result = "ERROR: Exception in GetProcessingStatus";
        }
        Py_DECREF(pArgs);
        Py_DECREF(pFunc);
        return result;
    }
    PyErr_Print();
    Py_XDECREF(pFunc);
    return "ERROR: Not initialized";
}

int CancelBloodPressureAnalysis(const char* request_id) {
    std::lock_guard<std::mutex> lock(pyMutex);
    if (!pythonInitialized || !pModule) return 0;
    PyObject* pFunc = PyObject_GetAttrString(pModule, "CancelBloodPressureAnalysis");
    if (pFunc && PyCallable_Check(pFunc)) {
        PyObject* pArgs = Py_BuildValue("(s)", request_id);
        PyObject* pResult = PyObject_CallObject(pFunc, pArgs);
        int result = 0;
        if (pResult) {
            result = PyObject_IsTrue(pResult);
            Py_DECREF(pResult);
        } else {
            PyErr_Print();
        }
        Py_DECREF(pArgs);
        Py_DECREF(pFunc);
        return result;
    }
    PyErr_Print();
    Py_XDECREF(pFunc);
    return 0;
}

const char* GetVersionInfo() {
    std::lock_guard<std::mutex> lock(pyMutex);
    if (!pythonInitialized || !pModule) return "ERROR: Not initialized";
    PyObject* pFunc = PyObject_GetAttrString(pModule, "GetVersionInfo");
    if (pFunc && PyCallable_Check(pFunc)) {
        PyObject* pResult = PyObject_CallObject(pFunc, nullptr);
        const char* result = "";
        if (pResult) {
            result = _strdup(PyUnicode_AsUTF8(pResult));
            Py_DECREF(pResult);
        } else {
            PyErr_Print();
            result = "ERROR: Exception in GetVersionInfo";
        }
        Py_DECREF(pFunc);
        return result;
    }
    PyErr_Print();
    Py_XDECREF(pFunc);
    return "ERROR: Not initialized";
}

const char* GenerateRequestId() {
    std::lock_guard<std::mutex> lock(pyMutex);
    if (!pythonInitialized || !pModule) return "ERROR: Not initialized";
    PyObject* pFunc = PyObject_GetAttrString(pModule, "GenerateRequestId");
    if (pFunc && PyCallable_Check(pFunc)) {
        PyObject* pResult = PyObject_CallObject(pFunc, nullptr);
        const char* result = "";
        if (pResult) {
            result = _strdup(PyUnicode_AsUTF8(pResult));
            Py_DECREF(pResult);
        } else {
            PyErr_Print();
            result = "ERROR: Exception in GenerateRequestId";
        }
        Py_DECREF(pFunc);
        return result;
    }
    PyErr_Print();
    Py_XDECREF(pFunc);
    return "ERROR: Not initialized";
} 
