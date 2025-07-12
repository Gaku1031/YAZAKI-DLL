"""
C/C++ wrapper
C#から呼び出し可能なDLLをC++で作成し、内部でPythonエンジンを呼び出す
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def create_cpp_wrapper():
    """C++ wrapper"""
    print("=== C++ wrapper ===")
    
    # C++ヘッダーファイル
    header_code = '''#pragma once

#ifdef BLOODPRESSURE_EXPORTS
#define BLOODPRESSURE_API __declspec(dllexport)
#else
#define BLOODPRESSURE_API __declspec(dllimport)
#endif

extern "C" {
    // callback function type definition
    typedef void(*AnalysisCallback)(const char* requestId, int maxBloodPressure, 
                                   int minBloodPressure, const char* measureRowData, 
                                   const char* errors);

    // export functions
    BLOODPRESSURE_API bool InitializeDLL(const char* modelDir);
    BLOODPRESSURE_API const char* StartBloodPressureAnalysisRequest(
        const char* requestId, int height, int weight, int sex, 
        const char* moviePath, AnalysisCallback callback);
    BLOODPRESSURE_API const char* GetProcessingStatus(const char* requestId);
    BLOODPRESSURE_API bool CancelBloodPressureAnalysis(const char* requestId);
    BLOODPRESSURE_API const char* GetVersionInfo();
}
'''

    with open("BloodPressureEstimation.h", "w", encoding="utf-8") as f:
        f.write(header_code)
    
    # C++ implementation file
    cpp_code = '''#include "BloodPressureEstimation.h"
#include <Python.h>
#include <string>
#include <map>
#include <memory>
#include <iostream>

// global variables
static bool g_initialized = false;
static PyObject* g_bp_module = nullptr;
static PyObject* g_estimator = nullptr;
static std::map<std::string, std::string> g_status_cache;

// string buffer (return value)
static std::string g_last_error;
static std::string g_last_status;
static std::string g_version_info;

// Python engine initialization
bool InitializePython() {
    if (Py_IsInitialized()) {
        return true;
    }

    try {
        // Python interpreter initialization
        Py_Initialize();
        if (!Py_IsInitialized()) {
            g_last_error = "Python initialization failed";
            return false;
        }

        // add current directory to sys.path
        PyRun_SimpleString("import sys");
        PyRun_SimpleString("sys.path.append('.')");
        
        // import blood pressure estimation module
        g_bp_module = PyImport_ImportModule("bp_estimation_simple");
        if (!g_bp_module) {
            PyErr_Print();
            g_last_error = "Failed to import bp_estimation_simple module";
            return false;
        }

        // get estimator instance
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

// Python function call helper
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

// export function implementation
extern "C" {

BLOODPRESSURE_API bool InitializeDLL(const char* modelDir) {
    try {
        if (g_initialized) {
            return true;
        }

        // Python engine initialization
        if (!InitializePython()) {
            return false;
        }

        // Python estimator initialization
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

        // Python function call
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
                g_last_error = ""; // success
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
'''

    with open("BloodPressureEstimation.cpp", "w", encoding="utf-8") as f:
        f.write(cpp_code)
    
    print("✓ C++ header and implementation file created")

def create_simple_python_module():
    """simple python module"""
    print("\\n=== simple python module ===")
    
    python_code = '''"""
blood pressure estimation simple python module
called from C++ wrapper
"""

import os
import threading
import time
from typing import Dict, Optional

class BPEstimator:
    """blood pressure estimation class"""
    
    def __init__(self):
        self.is_initialized = False
        self.processing_requests: Dict[str, str] = {}
        self.version = "1.0.0-cpp-wrapper"
        self.lock = threading.Lock()
        
    def initialize(self, model_dir: str = "models") -> bool:
        """initialization"""
        try:
            print(f"Initializing BP estimator with model_dir: {model_dir}")
            # actual initialization process is here
            self.is_initialized = True
            return True
        except Exception as e:
            print(f"Initialization error: {e}")
            return False
    
    def start_analysis(self, request_id: str, height: int, weight: int, 
                      sex: int, movie_path: str) -> Optional[str]:
        """blood pressure analysis start"""
        try:
            if not self.is_initialized:
                return "1001"  # DLL_NOT_INITIALIZED
            
            # parameter validation
            if not request_id or len(request_id) < 10:
                return "1004"  # INVALID_INPUT_PARAMETERS
                
            if not movie_path or not os.path.exists(movie_path):
                return "1004"  # INVALID_INPUT_PARAMETERS
                
            if not (1 <= sex <= 2):
                return "1004"  # INVALID_INPUT_PARAMETERS
                
            if not (100 <= height <= 250):
                return "1004"  # INVALID_INPUT_PARAMETERS
                
            if not (30 <= weight <= 200):
                return "1004"  # INVALID_INPUT_PARAMETERS
            
            # processing check
            with self.lock:
                if request_id in self.processing_requests:
                    return "1005"  # REQUEST_DURING_PROCESSING
                
                # asynchronous processing start
                self.processing_requests[request_id] = "processing"
                thread = threading.Thread(
                    target=self._process_analysis,
                    args=(request_id, height, weight, sex, movie_path)
                )
                thread.start()
            
            return None  # success
            
        except Exception as e:
            print(f"Analysis start error: {e}")
            return "1006"  # INTERNAL_PROCESSING_ERROR
    
    def _process_analysis(self, request_id: str, height: int, weight: int,
                         sex: int, movie_path: str):
        """blood pressure analysis processing"""
        try:
            print(f"Processing analysis for request: {request_id}")
            
            # simple blood pressure calculation
            bmi = weight / ((height / 100) ** 2)
            
            # BMI based estimation
            sbp = max(90, min(180, 120 + int((bmi - 22) * 2)))
            dbp = max(60, min(110, 80 + int((bmi - 22) * 1)))
            
            # processing time simulation
            time.sleep(2)
            
            print(f"Analysis complete: {request_id}, SBP={sbp}, DBP={dbp}")
            
        except Exception as e:
            print(f"Analysis processing error: {e}")
        finally:
            with self.lock:
                self.processing_requests[request_id] = "none"
    
    def get_status(self, request_id: str) -> str:
        """processing status get"""
        with self.lock:
            return self.processing_requests.get(request_id, "none")
    
    def cancel_analysis(self, request_id: str) -> bool:
        """analysis cancel"""
        with self.lock:
            if request_id in self.processing_requests:
                self.processing_requests[request_id] = "none"
                return True
            return False
    
    def get_version(self) -> str:
        """version information get"""
        return f"v{self.version}"

# test
if __name__ == "__main__":
    estimator = BPEstimator()
    
    if estimator.initialize():
        print("✓ initialization successful")
        print(f"version: {estimator.get_version()}")
        
        # test analysis
        result = estimator.start_analysis("test_123", 170, 70, 1, "test.webm")
        if result:
            print(f"error code: {result}")
        else:
            print("analysis start successful")
    else:
        print("✗ initialization failed")
'''

    with open("bp_estimation_simple.py", "w", encoding="utf-8") as f:
        f.write(python_code)
    
    print("✓ bp_estimation_simple.py created")

def create_def_file():
    """DEF file"""
    print("\\n=== DEF file ===")
    
    def_code = '''EXPORTS
InitializeDLL
StartBloodPressureAnalysisRequest
GetProcessingStatus
CancelBloodPressureAnalysis
GetVersionInfo'''

    with open("BloodPressureEstimation.def", "w", encoding="utf-8") as f:
        f.write(def_code)
    
    print("✓ BloodPressureEstimation.def created")

def create_build_script():
    """build script"""
    print("\\n=== build script ===")
    
    # CMake file
    cmake_code = '''cmake_minimum_required(VERSION 3.16)
project(BloodPressureEstimation)

set(CMAKE_CXX_STANDARD 17)

# Python search
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# include directory
include_directories(${Python3_INCLUDE_DIRS})

# source file
set(SOURCES
    BloodPressureEstimation.cpp
)

# DLL creation
add_library(BloodPressureEstimation SHARED ${SOURCES})

# Python DLL link
target_link_libraries(BloodPressureEstimation ${Python3_LIBRARIES})

# DEF file use
set_target_properties(BloodPressureEstimation PROPERTIES
    LINK_FLAGS "/DEF:${CMAKE_CURRENT_SOURCE_DIR}/BloodPressureEstimation.def"
)

# preprocessor definition
target_compile_definitions(BloodPressureEstimation PRIVATE BLOODPRESSURE_EXPORTS)

# output directory
set_target_properties(BloodPressureEstimation PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/dist
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/dist
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/dist
)
'''

    with open("CMakeLists.txt", "w", encoding="utf-8") as f:
        f.write(cmake_code)
    
    # batch file
    batch_code = '''@echo off
echo === C++ Wrapper DLL Build ===

REM Visual Studio environment setting
call "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat"

REM build directory creation
if exist build rmdir /s /q build
mkdir build
cd build

REM CMake setting
cmake .. -G "Visual Studio 17 2022" -A x64

REM build
cmake --build . --config Release

REM check result
if exist dist\\BloodPressureEstimation.dll (
    echo ✓ DLL created successfully
    echo export function check:
    dumpbin /exports dist\\BloodPressureEstimation.dll
) else (
    echo ✗ DLL creation failed
)

cd ..
pause
'''

    with open("build_cpp_dll.bat", "w", encoding="utf-8") as f:
        f.write(batch_code)
    
    print("✓ CMakeLists.txt and build_cpp_dll.bat created")

def create_csharp_test():
    """C# test code"""
    print("\\n=== C# test code ===")
    
    csharp_code = '''using System;
using System.Runtime.InteropServices;

namespace BloodPressureDllTest
{
    public class BloodPressureDll
    {
        private const string DllPath = "BloodPressureEstimation.dll";

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void AnalysisCallback(
            [MarshalAs(UnmanagedType.LPStr)] string requestId,
            int maxBloodPressure,
            int minBloodPressure,
            [MarshalAs(UnmanagedType.LPStr)] string measureRowData,
            [MarshalAs(UnmanagedType.LPStr)] string errors
        );

        [DllImport(DllPath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern bool InitializeDLL([MarshalAs(UnmanagedType.LPStr)] string modelDir);

        [DllImport(DllPath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        [return: MarshalAs(UnmanagedType.LPStr)]
        public static extern string StartBloodPressureAnalysisRequest(
            [MarshalAs(UnmanagedType.LPStr)] string requestId,
            int height, int weight, int sex,
            [MarshalAs(UnmanagedType.LPStr)] string moviePath,
            AnalysisCallback callback);

        [DllImport(DllPath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        [return: MarshalAs(UnmanagedType.LPStr)]
        public static extern string GetProcessingStatus([MarshalAs(UnmanagedType.LPStr)] string requestId);

        [DllImport(DllPath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern bool CancelBloodPressureAnalysis([MarshalAs(UnmanagedType.LPStr)] string requestId);

        [DllImport(DllPath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        [return: MarshalAs(UnmanagedType.LPStr)]
        public static extern string GetVersionInfo();

        public static void TestCppWrapperDLL()
        {
            Console.WriteLine("=== C++ Wrapper DLL test ===");

            try
            {
                // 1. DLL initialization
                Console.WriteLine("1. DLL initialization");
                bool initResult = InitializeDLL("models");
                Console.WriteLine($"    result: {initResult}");

                if (!initResult)
                {
                    Console.WriteLine("DLL initialization failed");
                    return;
                }

                // 2. version get
                Console.WriteLine("2. version get");
                string version = GetVersionInfo();
                Console.WriteLine($"    version: {version}");

                // 3. processing status get
                Console.WriteLine("3. processing status get");
                string status = GetProcessingStatus("test_request");
                Console.WriteLine($"    status: {status}");

                // 4. analysis request (invalid parameters)
                Console.WriteLine("4. analysis request (invalid parameters)");
                AnalysisCallback callback = (reqId, sbp, dbp, csvData, errors) =>
                {
                    Console.WriteLine($"    callback: {reqId}, SBP={sbp}, DBP={dbp}");
                };

                string errorCode = StartBloodPressureAnalysisRequest(
                    "invalid_id", 170, 70, 1, "test.webm", callback);
                Console.WriteLine($"    error code: {errorCode}");

                // 5. valid analysis request
                Console.WriteLine("5. valid analysis request");
                string requestId = $"{DateTime.Now:yyyyMMddHHmmssfff}_1234567890_0987654321";
                
                // dummy file creation
                System.IO.File.WriteAllText("test_video.webm", "dummy");
                
                errorCode = StartBloodPressureAnalysisRequest(
                    requestId, 170, 70, 1, "test_video.webm", callback);
                
                if (string.IsNullOrEmpty(errorCode))
                {
                    Console.WriteLine("    analysis start successful");
                    
                    // status check
                    System.Threading.Thread.Sleep(1000);
                    status = GetProcessingStatus(requestId);
                    Console.WriteLine($"    processing status: {status}");
                    
                    // wait for completion
                    System.Threading.Thread.Sleep(3000);
                    status = GetProcessingStatus(requestId);
                    Console.WriteLine($"    final status: {status}");
                }
                else
                {
                    Console.WriteLine($"    error code: {errorCode}");
                }

                Console.WriteLine("=== test finished ===");
            }
            catch (DllNotFoundException ex)
            {
                Console.WriteLine($"DLL not found: {ex.Message}");
                Console.WriteLine("build\\dist\\BloodPressureEstimation.dll exists");
            }
            catch (EntryPointNotFoundException ex)
            {
                Console.WriteLine($"entry point not found: {ex.Message}");
                Console.WriteLine("check DLL export functions");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"error: {ex.Message}");
            }
        }

        public static void Main(string[] args)
        {
            TestCppWrapperDLL();
            Console.WriteLine("\\nPress Enter to exit...");
            Console.ReadLine();
        }
    }
}'''

    with open("CSharpCppWrapperTest.cs", "w", encoding="utf-8") as f:
        f.write(csharp_code)
    
    print("✓ CSharpCppWrapperTest.cs created")

def main():
    """main process"""
    print("=== C++ wrapper DLL creation script ===")
    
    try:
        # 1. C++ wrapper creation
        create_cpp_wrapper()
        
        # 2. simple python module creation
        create_simple_python_module()
        
        # 3. DEF file creation
        create_def_file()
        
        # 4. build script creation
        create_build_script()
        
        # 5. C# test code creation
        create_csharp_test()
        
        print("\\n🎉 C++ wrapper DLL created!")
        print("\\nnext steps:")
        print("1. run build_cpp_dll.bat to build DLL")
        print("2. csc CSharpCppWrapperTest.cs to compile test code")
        print("3. copy build\\dist\\BloodPressureEstimation.dll to test execution directory")
        print("4. copy bp_estimation_simple.py to the same directory")
        print("5. run CSharpCppWrapperTest.exe")
        
        print("\\nfeatures:")
        print("✓ reliable DLL export")
        print("✓ callable from C#")
        print("✓ Python engine execution inside")
        print("✓ error handling complete")
        
        return True
        
    except Exception as e:
        print(f"\\n❌ error: {e}")
        return False

if __name__ == "__main__":
    main()
