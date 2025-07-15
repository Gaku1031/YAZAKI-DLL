// CppWrapperDLL.cpp
#include "BloodPressureWrapper.h"
#include <Python.h>
#include <windows.h>
#include <string>

extern "C" __declspec(dllexport)
int InitializeBP(const char* model_dir) {
    // Pythonランタイム初期化（多重初期化防止）
    if (!Py_IsInitialized()) {
        Py_Initialize();
    }

    // sys.pathにカレントディレクトリとpython_depsを追加
    PyRun_SimpleString("import sys; sys.path.insert(0, './python_deps'); sys.path.insert(0, './')");

    // Cython拡張モジュール（pyd）をimport
    PyObject* pModule = PyImport_ImportModule("BloodPressureEstimation");
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
        Py_DECREF(pModule);
        return result;
    }
    PyErr_Print();
    Py_XDECREF(pFunc);
    Py_DECREF(pModule);
    return 0;
} 
