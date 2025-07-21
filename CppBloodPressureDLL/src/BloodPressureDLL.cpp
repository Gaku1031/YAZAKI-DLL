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
    
    std::string version = "1.0.0";
    
    // Lazy initialization for complex objects
    std::mutex& getSafeMutex() { return getSafeStatic<std::mutex>(); }
    std::map<std::string, std::thread>& getSafeThreads() { return getSafeStatic<std::map<std::string, std::thread>>(); }
    std::map<std::string, std::string>& getSafeStatus() { return getSafeStatic<std::map<std::string, std::string>>(); }
    std::mutex& getSafeCallbackMutex() { return getSafeStatic<std::mutex>(); }
    std::string& getSafeCSVStr() { return getSafeStatic<std::string>(); }
    std::string& getSafeErrorsStr() { return getSafeStatic<std::string>(); }
    
    std::atomic<bool> initialized{false};
    BloodPressureEstimator* g_estimator = nullptr;
    
    // Thread-safe string buffer management with longer lifetime
    thread_local std::string tl_return_str;
    thread_local std::string tl_error_str;
    thread_local std::string tl_status_str;
    thread_local std::string tl_id_str;
    thread_local std::string tl_version_str;
    
    // Constants for common responses
    const char* const STATUS_NONE = "none";
    const char* const STATUS_PROCESSING = "processing";
    const char* const VERSION_INFO = "BloodPressureDLL v1.0.0";
    const char* const EMPTY_JSON = "[]";
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

extern "C" {

int InitializeBP(const char* modelDir) {
    printf("[DLL] InitializeBP called\n"); fflush(stdout);
    try {
        std::lock_guard<std::mutex> lock(getSafeMutex());
        if (g_estimator) {
            delete g_estimator;
            g_estimator = nullptr;
        }
        std::string modelPath = modelDir ? modelDir : "models";
        
        // Add detailed logging for debugging
        printf("[DLL] InitializeBP: Creating estimator with model path: %s\n", modelPath.c_str()); fflush(stdout);
        
        g_estimator = new BloodPressureEstimator(modelPath);
        initialized = true;
        printf("[DLL] InitializeBP success\n"); fflush(stdout);
        return 1;
    } catch (const std::exception& e) {
        printf("[DLL] InitializeBP std::exception: %s\n", e.what()); fflush(stdout);
        std::ofstream log("dll_error.log", std::ios::app);
        log << "InitializeBP exception: " << e.what() << std::endl;
        log.close();
        initialized = false;
        return 0;
    } catch (...) {
        printf("[DLL] InitializeBP unknown exception\n"); fflush(stdout);
        std::ofstream log("dll_error.log", std::ios::app);
        log << "InitializeBP unknown exception" << std::endl;
        log.close();
        initialized = false;
        return 0;
    }
}

const char* StartBloodPressureAnalysisRequest(
    const char* requestId, int height, int weight, int sex,
    const char* moviePath, BPCallback callback)
{
    printf("[DLL] StartBloodPressureAnalysisRequest called\n"); fflush(stdout);
    try {
        if (!initialized) {
            tl_error_str = "1001: DLL_NOT_INITIALIZED";
            printf("[DLL] StartBloodPressureAnalysisRequest not initialized\n"); fflush(stdout);
            return tl_error_str.c_str();
        }
        std::string reqId(requestId ? requestId : "");
        {
            std::lock_guard<std::mutex> lock(getSafeMutex());
            if (getSafeThreads().count(reqId)) {
                tl_error_str = "1005: REQUEST_DURING_PROCESSING";
                printf("[DLL] StartBloodPressureAnalysisRequest already processing\n"); fflush(stdout);
                return tl_error_str.c_str();
            }
            getSafeStatus()[reqId] = STATUS_PROCESSING;
        }
        std::string thread_request_id = requestId ? std::string(requestId) : "";
        RPPGProcessor rppg;
        try {
            printf("[DLL] StartBloodPressureAnalysisRequest: processVideo start\n"); fflush(stdout);
            RPPGResult r = rppg.processVideo(moviePath);
            printf("[DLL] StartBloodPressureAnalysisRequest: processVideo end\n"); fflush(stdout);
            auto bp = g_estimator->estimate_bp(r.peak_times, height, weight, sex);
            printf("[DLL] StartBloodPressureAnalysisRequest: estimate_bp end\n"); fflush(stdout);
            
            // Thread-safe callback data preparation
            {
                std::lock_guard<std::mutex> lock(getSafeCallbackMutex());
                getSafeCSVStr() = generateCSV(r.rppg_signal, r.time_data, r.peak_times);
                getSafeErrorsStr() = EMPTY_JSON;
                if (callback) {
                    callback(thread_request_id.c_str(), bp.first, bp.second, getSafeCSVStr().c_str(), getSafeErrorsStr().c_str());
                }
            }
            printf("[DLL] StartBloodPressureAnalysisRequest: callback end\n"); fflush(stdout);
        } catch (const std::exception& e) {
            printf("[DLL] StartBloodPressureAnalysisRequest inner std::exception: %s\n", e.what()); fflush(stdout);
            std::ofstream log("dll_error.log", std::ios::app);
            log << "StartBloodPressureAnalysisRequest inner exception: " << e.what() << std::endl;
            log.close();
            
            std::lock_guard<std::mutex> lock(getSafeCallbackMutex());
            getSafeErrorsStr() = std::string("[{\"code\":\"1006\",\"message\":\"") + e.what() + "\",\"isRetriable\":false}]";
            if (callback) {
                callback(thread_request_id.c_str(), 0, 0, EMPTY_STRING, getSafeErrorsStr().c_str());
            }
        }
        {
            std::lock_guard<std::mutex> lock(getSafeMutex());
            getSafeStatus()[reqId] = STATUS_NONE;
            getSafeThreads().erase(reqId);
        }
        printf("[DLL] StartBloodPressureAnalysisRequest end\n"); fflush(stdout);
        return EMPTY_STRING;
    } catch (const std::exception& e) {
        printf("[DLL] StartBloodPressureAnalysisRequest std::exception: %s\n", e.what()); fflush(stdout);
        std::ofstream log("dll_error.log", std::ios::app);
        log << "StartBloodPressureAnalysisRequest exception: " << e.what() << std::endl;
        log.close();
        tl_error_str = std::string("StartBloodPressureAnalysisRequest failed: ") + e.what();
        return tl_error_str.c_str();
    } catch (...) {
        printf("[DLL] StartBloodPressureAnalysisRequest unknown exception\n"); fflush(stdout);
        std::ofstream log("dll_error.log", std::ios::app);
        log << "StartBloodPressureAnalysisRequest unknown exception" << std::endl;
        log.close();
        tl_error_str = "StartBloodPressureAnalysisRequest failed: unknown error";
        return tl_error_str.c_str();
    }
}

const char* GetProcessingStatus(const char* requestId) {
    printf("[DLL] GetProcessingStatus called\n"); fflush(stdout);
    try {
        if (!requestId) {
            printf("[DLL] GetProcessingStatus: requestId is null\n"); fflush(stdout);
            return STATUS_NONE;
        }
        
        std::string reqId(requestId);
        printf("[DLL] GetProcessingStatus: requestId=%s\n", reqId.c_str()); fflush(stdout);
        
        {
            std::lock_guard<std::mutex> lock(getSafeMutex());
            auto it = getSafeStatus().find(reqId);
            if (it != getSafeStatus().end()) {
                tl_status_str = it->second;
                printf("[DLL] GetProcessingStatus: returning status=%s\n", tl_status_str.c_str()); fflush(stdout);
                return tl_status_str.c_str();
            }
        }
        
        printf("[DLL] GetProcessingStatus: returning default status\n"); fflush(stdout);
        return STATUS_NONE;
    } catch (const std::exception& e) {
        printf("[DLL] GetProcessingStatus std::exception: %s\n", e.what()); fflush(stdout);
        std::ofstream log("dll_error.log", std::ios::app);
        log << "GetProcessingStatus exception: " << e.what() << std::endl;
        log.close();
        tl_status_str = std::string("GetProcessingStatus failed: ") + e.what();
        return tl_status_str.c_str();
    } catch (...) {
        printf("[DLL] GetProcessingStatus unknown exception\n"); fflush(stdout);
        std::ofstream log("dll_error.log", std::ios::app);
        log << "GetProcessingStatus unknown exception" << std::endl;
        log.close();
        tl_status_str = "GetProcessingStatus failed: unknown error";
        return tl_status_str.c_str();
    }
}

int CancelBloodPressureAnalysis(const char* requestId) {
    // スレッドの強制停止は非推奨。フラグ管理で対応推奨
    return 0;
}

int GetVersionInfo(char* outBuf, int bufSize) {
    if (outBuf && bufSize > 0) {
        outBuf[0] = 'A';
        if (bufSize > 1) outBuf[1] = '\0';
    }
    return 0;
}

const char* GenerateRequestId() {
    printf("[DLL] GenerateRequestId called\n"); fflush(stdout);
    try {
        auto now = std::chrono::system_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
        std::ostringstream oss;
        oss << ms << "_CUSTOMER_DRIVER";
        tl_id_str = oss.str();
        printf("[DLL] GenerateRequestId returning id: %s\n", tl_id_str.c_str()); fflush(stdout);
        return tl_id_str.c_str();
    } catch (const std::exception& e) {
        printf("[DLL] GenerateRequestId std::exception: %s\n", e.what()); fflush(stdout);
        std::ofstream log("dll_error.log", std::ios::app);
        log << "GenerateRequestId exception: " << e.what() << std::endl;
        log.close();
        tl_error_str = std::string("GenerateRequestId failed: ") + e.what();
        return tl_error_str.c_str();
    } catch (...) {
        printf("[DLL] GenerateRequestId unknown exception\n"); fflush(stdout);
        std::ofstream log("dll_error.log", std::ios::app);
        log << "GenerateRequestId unknown exception" << std::endl;
        log.close();
        tl_error_str = "GenerateRequestId failed: unknown error";
        return tl_error_str.c_str();
    }
}

int AnalyzeBloodPressureFromImages(const char** imagePaths, int numImages, int height, int weight, int sex, BPCallback callback) {
    printf("[DLL] AnalyzeBloodPressureFromImages called with %d images\n", numImages); fflush(stdout);
    try {
        if (!initialized) {
            printf("[DLL] AnalyzeBloodPressureFromImages not initialized\n"); fflush(stdout);
            return 1001;
        }
        if (!g_estimator) {
            printf("[DLL] AnalyzeBloodPressureFromImages estimator is null\n"); fflush(stdout);
            return 1001;
        }
        
        std::vector<std::string> paths;
        for (int i = 0; i < numImages; ++i) {
            if (imagePaths[i]) {
                paths.emplace_back(imagePaths[i]);
                if (i < 5 || i == numImages - 1) {
                    printf("[DLL] Image[%d]: %s\n", i, imagePaths[i]); fflush(stdout);
                }
            }
        }
        printf("[DLL] Total valid images: %zu\n", paths.size()); fflush(stdout);
        
        RPPGProcessor rppg;
        printf("[DLL] AnalyzeBloodPressureFromImages: processImagesFromPaths start\n"); fflush(stdout);
        RPPGResult r = rppg.processImagesFromPaths(paths);
        printf("[DLL] AnalyzeBloodPressureFromImages: processImagesFromPaths end\n"); fflush(stdout);
        auto bp = g_estimator->estimate_bp(r.peak_times, height, weight, sex);
        printf("[DLL] AnalyzeBloodPressureFromImages: estimate_bp end (SBP=%d, DBP=%d)\n", bp.first, bp.second); fflush(stdout);
        
        // Thread-safe callback data preparation
        {
            std::lock_guard<std::mutex> lock(getSafeCallbackMutex());
            getSafeCSVStr() = generateCSV(r.rppg_signal, r.time_data, r.peak_times);
            getSafeErrorsStr() = EMPTY_JSON;
            if (callback) {
                callback(EMPTY_STRING, bp.first, bp.second, getSafeCSVStr().c_str(), getSafeErrorsStr().c_str());
            }
        }
        printf("[DLL] AnalyzeBloodPressureFromImages: callback end\n"); fflush(stdout);
        return 0;
    } catch (const std::exception& e) {
        printf("[DLL] AnalyzeBloodPressureFromImages std::exception: %s\n", e.what()); fflush(stdout);
        std::ofstream log("dll_error.log", std::ios::app);
        log << "AnalyzeBloodPressureFromImages exception: " << e.what() << std::endl;
        log.close();
        
        std::lock_guard<std::mutex> lock(getSafeCallbackMutex());
        getSafeErrorsStr() = std::string("[{\"code\":\"1006\",\"message\":\"") + e.what() + "\",\"isRetriable\":false}]";
        if (callback) {
            callback(EMPTY_STRING, 0, 0, EMPTY_STRING, getSafeErrorsStr().c_str());
        }
        return 1006;
    } catch (...) {
        printf("[DLL] AnalyzeBloodPressureFromImages unknown exception\n"); fflush(stdout);
        std::ofstream log("dll_error.log", std::ios::app);
        log << "AnalyzeBloodPressureFromImages unknown exception" << std::endl;
        log.close();
        
        std::lock_guard<std::mutex> lock(getSafeCallbackMutex());
        getSafeErrorsStr() = "[{\"code\":\"1006\",\"message\":\"unknown error\",\"isRetriable\":false}]";
        if (callback) {
            callback(EMPTY_STRING, 0, 0, EMPTY_STRING, getSafeErrorsStr().c_str());
        }
        return 1006;
    }
}

#ifdef _WIN32
// DLL Entry Point - Critical for detecting early loading issues
BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) {
    switch (ul_reason_for_call) {
    case DLL_PROCESS_ATTACH:
        {
            // Create a log file immediately upon DLL loading
            FILE* log = fopen("dll_load.log", "w");
            if (log) {
                fprintf(log, "[DLL_MAIN] DLL_PROCESS_ATTACH - DLL loading started\n");
                fprintf(log, "[DLL_MAIN] Module handle: %p\n", hModule);
                fclose(log);
            }
            
            // Disable thread library calls to prevent threading issues during startup
            DisableThreadLibraryCalls(hModule);
            
            // Basic environment check
            try {
                FILE* log2 = fopen("dll_load.log", "a");
                if (log2) {
                    fprintf(log2, "[DLL_MAIN] Environment check passed\n");
                    fclose(log2);
                }
            } catch (...) {
                // Even basic operations are failing
                FILE* log2 = fopen("dll_load.log", "a");
                if (log2) {
                    fprintf(log2, "[DLL_MAIN] ERROR: Exception during basic environment check\n");
                    fclose(log2);
                }
                return FALSE;
            }
        }
        break;
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
        break;
    case DLL_PROCESS_DETACH:
        {
            FILE* log = fopen("dll_load.log", "a");
            if (log) {
                fprintf(log, "[DLL_MAIN] DLL_PROCESS_DETACH - DLL unloading\n");
                fclose(log);
            }
        }
        break;
    }
    return TRUE;
}
#endif

} 
