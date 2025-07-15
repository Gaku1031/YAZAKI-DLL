// CppWrapperDLL.cpp
#include "BloodPressureWrapper.h"
#include <windows.h>
#include <string>
#include <mutex>

static std::mutex dllMutex;
static HMODULE hCythonDll = nullptr;

// Cython DLL関数ポインタ型
using fnInitializeDLL = int(*)(const char*);
using fnStartBloodPressureAnalysisRequest = const char*(*)(const char*, int, int, int, const char*);
using fnGetProcessingStatus = const char*(*)(const char*);
using fnCancelBloodPressureAnalysis = int(*)(const char*);
using fnGetVersionInfo = const char* (*)();
using fnGenerateRequestId = const char* (*)();

static fnInitializeDLL pInitializeDLL = nullptr;
static fnStartBloodPressureAnalysisRequest pStartRequest = nullptr;
static fnGetProcessingStatus pGetStatus = nullptr;
static fnCancelBloodPressureAnalysis pCancel = nullptr;
static fnGetVersionInfo pGetVersion = nullptr;
static fnGenerateRequestId pGenReqId = nullptr;

static bool LoadCythonDll() {
    if (hCythonDll) return true;
    hCythonDll = LoadLibraryA("BloodPressureEstimation.dll");
    if (!hCythonDll) return false;
    pInitializeDLL = (fnInitializeDLL)GetProcAddress(hCythonDll, "InitializeDLL");
    pStartRequest = (fnStartBloodPressureAnalysisRequest)GetProcAddress(hCythonDll, "StartBloodPressureAnalysisRequest");
    pGetStatus = (fnGetProcessingStatus)GetProcAddress(hCythonDll, "GetProcessingStatus");
    pCancel = (fnCancelBloodPressureAnalysis)GetProcAddress(hCythonDll, "CancelBloodPressureAnalysis");
    pGetVersion = (fnGetVersionInfo)GetProcAddress(hCythonDll, "GetVersionInfo");
    pGenReqId = (fnGenerateRequestId)GetProcAddress(hCythonDll, "GenerateRequestId");
    return pInitializeDLL && pStartRequest && pGetStatus && pCancel && pGetVersion && pGenReqId;
}

extern "C" __declspec(dllexport)
int InitializeBP(const char* model_dir) {
    std::lock_guard<std::mutex> lock(dllMutex);
    if (!LoadCythonDll()) return 0;
    return pInitializeDLL ? pInitializeDLL(model_dir) : 0;
}

extern "C" __declspec(dllexport)
const char* StartBloodPressureAnalysisRequest(const char* request_id, int height, int weight, int sex, const char* movie_path) {
    std::lock_guard<std::mutex> lock(dllMutex);
    if (!LoadCythonDll() || !pStartRequest) return "ERROR: DLL not loaded";
    return pStartRequest(request_id, height, weight, sex, movie_path);
}

extern "C" __declspec(dllexport)
const char* GetProcessingStatus(const char* request_id) {
    std::lock_guard<std::mutex> lock(dllMutex);
    if (!LoadCythonDll() || !pGetStatus) return "ERROR: DLL not loaded";
    return pGetStatus(request_id);
}

extern "C" __declspec(dllexport)
int CancelBloodPressureAnalysis(const char* request_id) {
    std::lock_guard<std::mutex> lock(dllMutex);
    if (!LoadCythonDll() || !pCancel) return 0;
    return pCancel(request_id);
}

extern "C" __declspec(dllexport)
const char* GetVersionInfo() {
    std::lock_guard<std::mutex> lock(dllMutex);
    if (!LoadCythonDll() || !pGetVersion) return "ERROR: DLL not loaded";
    return pGetVersion();
}

extern "C" __declspec(dllexport)
const char* GenerateRequestId() {
    std::lock_guard<std::mutex> lock(dllMutex);
    if (!LoadCythonDll() || !pGenReqId) return "ERROR: DLL not loaded";
    return pGenReqId();
} 
