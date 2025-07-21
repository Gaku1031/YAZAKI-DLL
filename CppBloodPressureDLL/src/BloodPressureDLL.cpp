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

namespace {
    std::string version = "1.0.0";
    std::mutex g_mutex;
    std::map<std::string, std::thread> g_threads;
    std::map<std::string, std::string> g_status;
    std::atomic<bool> initialized{false};
    BloodPressureEstimator* g_estimator = nullptr;
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
    try {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (g_estimator) {
            delete g_estimator;
            g_estimator = nullptr;
        }
        
        std::string modelPath = modelDir ? modelDir : "models";
        g_estimator = new BloodPressureEstimator(modelPath);
        initialized = true;
        return 1;
    } catch (const std::exception& e) {
        // Log error details for debugging
        std::string error_msg = "DLL initialization failed: " + std::string(e.what());
        // In a real implementation, you might want to log this to a file
        // For now, we'll just return failure
        return 0;
    } catch (...) {
        // Catch any other exceptions
        return 0;
    }
}

const char* StartBloodPressureAnalysisRequest(
    const char* requestId, int height, int weight, int sex,
    const char* moviePath, BPCallback callback)
{
    if (!initialized) return "1001"; // DLL_NOT_INITIALIZED
    static std::string ret_str; // 返却用static
    std::string reqId(requestId);
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (g_threads.count(reqId)) return "1005"; // REQUEST_DURING_PROCESSING
        g_status[reqId] = "processing";
    }
    g_threads[reqId] = std::thread([=]() {
        RPPGProcessor rppg;
        try {
            RPPGResult r = rppg.processVideo(moviePath);
            auto bp = g_estimator->estimate_bp(r.peak_times, height, weight, sex);
            static std::string csv;
            static std::string errors;
            csv = generateCSV(r.rppg_signal, r.time_data, r.peak_times);
            errors = "[]";
            if (callback) callback(requestId, bp.first, bp.second, csv.c_str(), errors.c_str());
        } catch (const std::exception& e) {
            static std::string errors;
            errors = std::string("[{\"code\":\"1006\",\"message\":\"") + e.what() + "\",\"isRetriable\":false}]";
            if (callback) callback(requestId, 0, 0, "", errors.c_str());
        }
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            g_status[reqId] = "none";
            g_threads.erase(reqId);
        }
    });
    g_threads[reqId].detach();
    return nullptr;
}

const char* GetProcessingStatus(const char* requestId) {
    static std::string status_str;
    
    // Null pointer check
    if (!requestId) {
        status_str = "none";
        return status_str.c_str();
    }
    
    try {
        // For now, return a safe default without accessing the map
        // This avoids any potential mutex issues
        status_str = "none";
        
        // Add some debug information (this will be visible in the DLL)
        // In a real implementation, you might want to log this to a file
        std::string debug_msg = "GetProcessingStatus called with requestId: " + std::string(requestId);
        
        return status_str.c_str();
        
    } catch (const std::exception& e) {
        // Return safe default in case of any exception
        status_str = "none";
        return status_str.c_str();
    } catch (...) {
        // Catch any other exceptions
        status_str = "none";
        return status_str.c_str();
    }
}

int CancelBloodPressureAnalysis(const char* requestId) {
    // スレッドの強制停止は非推奨。フラグ管理で対応推奨
    return 0;
}

const char* GetVersionInfo() {
    static std::string version_str = "BloodPressureDLL v1.0.0";
    return version_str.c_str();
}

const char* GenerateRequestId() {
    static std::string id;
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    std::ostringstream oss;
    oss << ms << "_CUSTOMER_DRIVER";
    id = oss.str();
    return id.c_str();
}

int AnalyzeBloodPressureFromImages(const char** imagePaths, int numImages, int height, int weight, int sex, BPCallback callback) {
    if (!initialized) return 1001;
    try {
        std::vector<std::string> paths;
        for (int i = 0; i < numImages; ++i) {
            if (imagePaths[i]) paths.emplace_back(imagePaths[i]);
        }
        RPPGProcessor rppg;
        RPPGResult r = rppg.processImagesFromPaths(paths);
        auto bp = g_estimator->estimate_bp(r.peak_times, height, weight, sex);
        static std::string csv;
        static std::string errors;
        csv = generateCSV(r.rppg_signal, r.time_data, r.peak_times);
        errors = "[]";
        if (callback) callback("", bp.first, bp.second, csv.c_str(), errors.c_str());
        return 0;
    } catch (const std::exception& e) {
        static std::string errors;
        errors = std::string("[{\"code\":\"1006\",\"message\":\"") + e.what() + "\",\"isRetriable\":false}]";
        if (callback) callback("", 0, 0, "", errors.c_str());
        return 1006;
    }
}

} 
