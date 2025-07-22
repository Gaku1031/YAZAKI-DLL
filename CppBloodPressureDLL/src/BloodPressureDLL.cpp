#include "../include/BloodPressureDLL.h"
#include "rppg.h"
#include "bp_estimator.h"
#include <string>
#include <map>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <sstream>
#include <set>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <cstdio>
#include <vector>
#include <onnxruntime_cxx_api.h>

#ifdef _WIN32
#include <windows.h>
#endif

namespace {
    // Safe initialization helper to avoid static initialization order fiasco
    template<typename T>
    T& getSafeStatic() {
        static T instance;
        return instance;
    }
    
    // std::string version = "1.0.0";
    
    // Lazy initialization for complex objects
    std::mutex& getSafeMutex() { return getSafeStatic<std::mutex>(); }
    std::map<std::string, std::thread>& getSafeThreads() { return getSafeStatic<std::map<std::string, std::thread>>(); }
    std::map<std::string, std::string>& getSafeStatus() { return getSafeStatic<std::map<std::string, std::string>>(); }
    std::mutex& getSafeCallbackMutex() { return getSafeStatic<std::mutex>(); }
    // std::string& getSafeCSVStr() { return getSafeStatic<std::string>(); }
    // std::string& getSafeErrorsStr() { return getSafeStatic<std::string>(); }
    
    std::atomic<bool> initialized{false};
    BloodPressureEstimator* g_estimator = nullptr;
    
    // thread_local std::string tl_return_str;
    // thread_local std::string tl_error_str;
    // thread_local std::string tl_status_str;
    // thread_local std::string tl_id_str;
    // thread_local std::string tl_version_str;
    
    // Constants for common responses
    const char* const STATUS_NONE = "none";
    const char* const STATUS_PROCESSING = "processing";
    const char* const EMPTY_STRING = "";
}

// CSV生成用のヘルパー関数
std::string generateCSV(const std::vector<double>& rppg_signal, 
                       const std::vector<double>& time_data,
                       const std::vector<double>& peak_times) {
    std::stringstream csv;
    csv << "Time(s),rPPG_Signal,Peak_Flag\n";
    
    if (rppg_signal.size() != time_data.size()) {
        return "Error: Signal and time data size mismatch";
    }
    
    // ピーク時間をセットに変換（高速検索用）
    std::set<double> peak_set(peak_times.begin(), peak_times.end());
    
    for (size_t i = 0; i < rppg_signal.size(); ++i) {
        double time_val = time_data[i];
        double rppg_val = rppg_signal[i];
        
        // ピークフラグの判定（0.1秒以内の近似値）
        int peak_flag = 0;
        for (double peak_time : peak_set) {
            if (std::abs(time_val - peak_time) < 0.1) {
                peak_flag = 1;
                break;
            }
        }
        
        csv << std::fixed << std::setprecision(3) << time_val << ","
            << std::fixed << std::setprecision(6) << rppg_val << ","
            << peak_flag << "\n";
    }
    
    return csv.str();
}

bool file_exists(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return f.good();
}

std::string get_dll_architecture(const std::string& dll_path) {
    std::ifstream file(dll_path, std::ios::binary);
    if (!file) return "not found";
    file.seekg(0, std::ios::end);
    size_t filesize = file.tellg();
    if (filesize < 0x40) return "file too small";
    file.seekg(0x3C);
    int pe_offset = 0;
    file.read(reinterpret_cast<char*>(&pe_offset), 4);
    if (pe_offset <= 0 || pe_offset + 6 > filesize) return "invalid PE offset";
    file.seekg(pe_offset);
    char sig[4] = {};
    file.read(sig, 4);
    if (sig[0] != 'P' || sig[1] != 'E' || sig[2] != 0 || sig[3] != 0) return "no PE signature";
    short machine = 0;
    file.read(reinterpret_cast<char*>(&machine), 2);
    unsigned char* p = reinterpret_cast<unsigned char*>(&machine);
    unsigned short machine_le = p[0] | (p[1] << 8);
    if (machine_le == 0x8664) return "x64";
    if (machine_le == 0x14c) return "x86";
    char buf[32];
    snprintf(buf, sizeof(buf), "unknown(0x%04x)", machine_le);
    return buf;
}

extern "C" {

int InitializeBP(char* outBuf, int bufSize, const char* modelDir) {
    if (!outBuf || bufSize <= 0) return -1;
#ifdef _WIN32
    HMODULE h = LoadLibraryA("onnxruntime.dll");
    if (!h) {
        DWORD err = GetLastError();
        snprintf(outBuf, bufSize, "[ERROR] LoadLibraryA(onnxruntime.dll) failed. GetLastError=%lu", (unsigned long)err);
        return -1;
    }
#endif
    printf("[DEBUG] Enter InitializeBP\n"); fflush(stdout);
    try {
        std::ostringstream oss;
        oss << "[DEBUG] try block entered\n";
        // 1. onnxruntime.dllの存在チェック
        std::string dll_path = "onnxruntime.dll";
        oss << "[CHECK] onnxruntime.dll: ";
        if (!file_exists(dll_path)) {
            oss << "NOT FOUND\n";
            snprintf(outBuf, bufSize, "%s", oss.str().c_str());
            return -1;
        }
        oss << "FOUND\n";
        // 2. bit数チェック
        std::string arch = get_dll_architecture(dll_path);
        oss << "[CHECK] onnxruntime.dll architecture: " << arch << "\n";
        if (arch != "x64") {
            oss << "ERROR: onnxruntime.dll is not x64\n";
            snprintf(outBuf, bufSize, "%s", oss.str().c_str());
            return -1;
        }
        // 3. 依存DLLチェック（zlib.dll, opencv_world480.dllなど）
        std::vector<std::string> deps = {"zlib.dll", "opencv_world480.dll"};
        for (const auto& dep : deps) {
            oss << "[CHECK] " << dep << ": ";
            if (!file_exists(dep)) {
                oss << "NOT FOUND\n";
                snprintf(outBuf, bufSize, "%s", oss.str().c_str());
                return -1;
            }
            oss << "FOUND\n";
        }
        // 4. モデルファイルチェック（既存のまま）
        std::string modelPath = modelDir ? modelDir : "models";
        oss << "[CHECK] modelPath: " << modelPath << "\n";
        oss << "[STEP] Ort::Env OK\n";
        Ort::SessionOptions session_options;
        printf("[DEBUG] After SessionOptions\n"); fflush(stdout);
        session_options.SetIntraOpNumThreads(1);
        oss << "[STEP] SessionOptions OK\n";
        snprintf(outBuf, bufSize, "%s", oss.str().c_str());
        return 0;
    } catch (const std::exception& e) {
        printf("[DEBUG] Caught std::exception: %s\n", e.what()); fflush(stdout);
        snprintf(outBuf, bufSize, "InitializeBP exception: %s", e.what());
        return -1;
    } catch (...) {
        printf("[DEBUG] Caught unknown exception\n"); fflush(stdout);
        snprintf(outBuf, bufSize, "InitializeBP unknown exception");
        return -1;
    }
}

int StartBloodPressureAnalysisRequest(char* outBuf, int bufSize,
    const char* requestId, int height, int weight, int sex,
    const char* moviePath, BPCallback callback)
{
    try {
        if (!initialized) {
            snprintf(outBuf, bufSize, "1001: DLL_NOT_INITIALIZED");
            return -1;
        }
        std::string reqId(requestId ? requestId : "");
        {
            std::lock_guard<std::mutex> lock(getSafeMutex());
            if (getSafeThreads().count(reqId)) {
                snprintf(outBuf, bufSize, "1005: REQUEST_DURING_PROCESSING");
                return -1;
            }
            getSafeStatus()[reqId] = STATUS_PROCESSING;
        }
        std::string thread_request_id = requestId ? std::string(requestId) : "";
        RPPGProcessor rppg;
        try {
            RPPGResult r = rppg.processVideo(moviePath);
            auto bp = g_estimator->estimate_bp(r.peak_times, height, weight, sex);
            {
                std::lock_guard<std::mutex> lock(getSafeCallbackMutex());
                // getSafeCSVStr() = generateCSV(r.rppg_signal, r.time_data, r.peak_times);
                // getSafeErrorsStr() = EMPTY_JSON;
                if (callback) {
                    callback(thread_request_id.c_str(), bp.first, bp.second, EMPTY_STRING, EMPTY_STRING);
                }
            }
            snprintf(outBuf, bufSize, "OK");
            return 0;
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(getSafeCallbackMutex());
            // getSafeErrorsStr() = std::string("[{\"code\":\"1006\",\"message\":\"") + e.what() + "\",\"isRetriable\":false}]";
            if (callback) {
                callback(thread_request_id.c_str(), 0, 0, EMPTY_STRING, EMPTY_STRING);
            }
            snprintf(outBuf, bufSize, "StartBloodPressureAnalysisRequest inner exception: %s", e.what());
            return -1;
        }
        {
            std::lock_guard<std::mutex> lock(getSafeMutex());
            getSafeStatus()[reqId] = STATUS_NONE;
            getSafeThreads().erase(reqId);
        }
    } catch (const std::exception& e) {
        snprintf(outBuf, bufSize, "StartBloodPressureAnalysisRequest exception: %s", e.what());
        return -1;
    } catch (...) {
        snprintf(outBuf, bufSize, "StartBloodPressureAnalysisRequest unknown exception");
        return -1;
    }
}

int GetProcessingStatus(char* outBuf, int bufSize, const char* requestId) {
    try {
        if (!requestId) {
            snprintf(outBuf, bufSize, "%s", STATUS_NONE);
            return 0;
        }
        std::string reqId(requestId);
        {
            std::lock_guard<std::mutex> lock(getSafeMutex());
            auto it = getSafeStatus().find(reqId);
            if (it != getSafeStatus().end()) {
                snprintf(outBuf, bufSize, "%s", it->second.c_str());
                return 0;
            }
        }
        snprintf(outBuf, bufSize, "%s", STATUS_NONE);
        return 0;
    } catch (const std::exception& e) {
        snprintf(outBuf, bufSize, "GetProcessingStatus exception: %s", e.what());
        return -1;
    } catch (...) {
        snprintf(outBuf, bufSize, "GetProcessingStatus unknown exception");
        return -1;
    }
}

int CancelBloodPressureAnalysis(char* outBuf, int bufSize, const char* requestId) {
    snprintf(outBuf, bufSize, "OK");
    return 0;
}

int GetVersionInfo(char* outBuf, int bufSize) {
    if (!outBuf || bufSize <= 0) return -1;
    outBuf[0] = 'V';
    if (bufSize > 1) outBuf[1] = '\0';
    return 0;
}

int GenerateRequestId(char* outBuf, int bufSize) {
    try {
        auto now = std::chrono::system_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
        std::ostringstream oss;
        oss << ms << "_CUSTOMER_DRIVER";
        snprintf(outBuf, bufSize, "%s", oss.str().c_str());
        return 0;
    } catch (const std::exception& e) {
        snprintf(outBuf, bufSize, "GenerateRequestId exception: %s", e.what());
        return -1;
    } catch (...) {
        snprintf(outBuf, bufSize, "GenerateRequestId unknown exception");
        return -1;
    }
}

int AnalyzeBloodPressureFromImages(char* outBuf, int bufSize,
    const char** imagePaths, int numImages, int height, int weight, int sex, BPCallback callback) {
    try {
        if (!initialized) {
            snprintf(outBuf, bufSize, "AnalyzeBloodPressureFromImages not initialized");
            return -1;
        }
        if (!g_estimator) {
            snprintf(outBuf, bufSize, "AnalyzeBloodPressureFromImages estimator is null");
            return -1;
        }
        std::vector<std::string> paths;
        for (int i = 0; i < numImages; ++i) {
            if (imagePaths[i]) {
                paths.emplace_back(imagePaths[i]);
            }
        }
        RPPGProcessor rppg;
        RPPGResult r = rppg.processImagesFromPaths(paths);
        auto bp = g_estimator->estimate_bp(r.peak_times, height, weight, sex);
        {
            std::lock_guard<std::mutex> lock(getSafeCallbackMutex());
            // getSafeCSVStr() = generateCSV(r.rppg_signal, r.time_data, r.peak_times);
            // getSafeErrorsStr() = EMPTY_JSON;
            if (callback) {
                callback(EMPTY_STRING, bp.first, bp.second, EMPTY_STRING, EMPTY_STRING);
            }
        }
        snprintf(outBuf, bufSize, "OK");
        return 0;
    } catch (const std::exception& e) {
        snprintf(outBuf, bufSize, "AnalyzeBloodPressureFromImages exception: %s", e.what());
        return -1;
    } catch (...) {
        snprintf(outBuf, bufSize, "AnalyzeBloodPressureFromImages unknown exception");
        return -1;
    }
}

#ifdef _WIN32
// DLL Entry Point - Critical for detecting early loading issues
BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) {
    return TRUE;
}
#endif

} 
