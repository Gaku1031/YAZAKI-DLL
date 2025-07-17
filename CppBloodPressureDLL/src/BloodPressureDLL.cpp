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
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_estimator) delete g_estimator;
    g_estimator = new BloodPressureEstimator(modelDir ? modelDir : "models");
    initialized = true;
    return 1;
}

const char* StartBloodPressureAnalysisRequest(
    const char* requestId, int height, int weight, int sex,
    const char* moviePath, BPCallback callback)
{
    if (!initialized) return "1001"; // DLL_NOT_INITIALIZED
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
            
            // CSV生成
            std::string csv = generateCSV(r.rppg_signal, r.time_data, r.peak_times);
            
            std::string errors = "[]";
            if (callback) callback(requestId, bp.first, bp.second, csv.c_str(), errors.c_str());
        } catch (const std::exception& e) {
            std::string errors = std::string("[{\"code\":\"1006\",\"message\":\"") + e.what() + "\",\"isRetriable\":false}]";
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
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_status.find(requestId);
    if (it != g_status.end()) return it->second.c_str();
    return "none";
}

int CancelBloodPressureAnalysis(const char* requestId) {
    // スレッドの強制停止は非推奨。フラグ管理で対応推奨
    return 0;
}

const char* GetVersionInfo() {
    return version.c_str();
}

const char* GenerateRequestId() {
    static thread_local std::string id;
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    std::ostringstream oss;
    oss << ms << "_CUSTOMER_DRIVER";
    id = oss.str();
    return id.c_str();
}

} 
