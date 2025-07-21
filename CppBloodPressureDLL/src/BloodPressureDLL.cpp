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

static std::string empty_str = "";
static std::string safe_request_id;

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
        std::ofstream log("dll_error.log", std::ios::app);
        log << "InitializeBP exception: " << e.what() << std::endl;
        return 0;
    } catch (...) {
        std::ofstream log("dll_error.log", std::ios::app);
        log << "InitializeBP unknown exception" << std::endl;
        return 0;
    }
}

const char* StartBloodPressureAnalysisRequest(
    const char* requestId, int height, int weight, int sex,
    const char* moviePath, BPCallback callback)
{
    try {
        if (!initialized) {
            static thread_local std::string err = "1001: DLL_NOT_INITIALIZED";
            return err.c_str();
        }
        static std::string ret_str;
        std::string reqId(requestId);
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            if (g_threads.count(reqId)) {
                static thread_local std::string err = "1005: REQUEST_DURING_PROCESSING";
                return err.c_str();
            }
            g_status[reqId] = "processing";
        }
        static thread_local std::string thread_request_id = requestId ? std::string(requestId) : "";
        RPPGProcessor rppg;
        try {
            RPPGResult r = rppg.processVideo(moviePath);
            auto bp = g_estimator->estimate_bp(r.peak_times, height, weight, sex);
            static thread_local std::string csv;
            static thread_local std::string errors;
            csv = generateCSV(r.rppg_signal, r.time_data, r.peak_times);
            errors = "[]";
            if (callback) callback(thread_request_id.c_str(), bp.first, bp.second, csv.c_str(), errors.c_str());
        } catch (const std::exception& e) {
            std::ofstream log("dll_error.log", std::ios::app);
            log << "StartBloodPressureAnalysisRequest inner exception: " << e.what() << std::endl;
            static thread_local std::string errors;
            errors = std::string("[{\"code\":\"1006\",\"message\":\"") + e.what() + "\",\"isRetriable\":false}]";
            if (callback) callback(thread_request_id.c_str(), 0, 0, empty_str.c_str(), errors.c_str());
        }
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            g_status[reqId] = "none";
            g_threads.erase(reqId);
        }
        static thread_local std::string ok = "";
        return ok.c_str();
    } catch (const std::exception& e) {
        std::ofstream log("dll_error.log", std::ios::app);
        log << "StartBloodPressureAnalysisRequest exception: " << e.what() << std::endl;
        static thread_local std::string err;
        err = std::string("StartBloodPressureAnalysisRequest failed: ") + e.what();
        return err.c_str();
    } catch (...) {
        std::ofstream log("dll_error.log", std::ios::app);
        log << "StartBloodPressureAnalysisRequest unknown exception" << std::endl;
        static thread_local std::string err = "StartBloodPressureAnalysisRequest failed: unknown error";
        return err.c_str();
    }
}

const char* GetProcessingStatus(const char* requestId) {
    static thread_local std::string status_str;
    try {
        if (!requestId) {
            status_str = "none";
            return status_str.c_str();
        }
        status_str = "none";
        std::string debug_msg = "GetProcessingStatus called with requestId: " + std::string(requestId);
        return status_str.c_str();
    } catch (const std::exception& e) {
        std::ofstream log("dll_error.log", std::ios::app);
        log << "GetProcessingStatus exception: " << e.what() << std::endl;
        status_str = std::string("GetProcessingStatus failed: ") + e.what();
        return status_str.c_str();
    } catch (...) {
        std::ofstream log("dll_error.log", std::ios::app);
        log << "GetProcessingStatus unknown exception" << std::endl;
        status_str = "GetProcessingStatus failed: unknown error";
        return status_str.c_str();
    }
}

int CancelBloodPressureAnalysis(const char* requestId) {
    // スレッドの強制停止は非推奨。フラグ管理で対応推奨
    return 0;
}

const char* GetVersionInfo() {
    try {
        static thread_local std::string version_str = "BloodPressureDLL v1.0.0";
        return version_str.c_str();
    } catch (const std::exception& e) {
        std::ofstream log("dll_error.log", std::ios::app);
        log << "GetVersionInfo exception: " << e.what() << std::endl;
        static thread_local std::string err;
        err = std::string("GetVersionInfo failed: ") + e.what();
        return err.c_str();
    } catch (...) {
        std::ofstream log("dll_error.log", std::ios::app);
        log << "GetVersionInfo unknown exception" << std::endl;
        static thread_local std::string err = "GetVersionInfo failed: unknown error";
        return err.c_str();
    }
}

const char* GenerateRequestId() {
    try {
        static thread_local std::string id;
        auto now = std::chrono::system_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
        std::ostringstream oss;
        oss << ms << "_CUSTOMER_DRIVER";
        id = oss.str();
        return id.c_str();
    } catch (const std::exception& e) {
        std::ofstream log("dll_error.log", std::ios::app);
        log << "GenerateRequestId exception: " << e.what() << std::endl;
        static thread_local std::string err;
        err = std::string("GenerateRequestId failed: ") + e.what();
        return err.c_str();
    } catch (...) {
        std::ofstream log("dll_error.log", std::ios::app);
        log << "GenerateRequestId unknown exception" << std::endl;
        static thread_local std::string err = "GenerateRequestId failed: unknown error";
        return err.c_str();
    }
}

int AnalyzeBloodPressureFromImages(const char** imagePaths, int numImages, int height, int weight, int sex, BPCallback callback) {
    try {
        if (!initialized) return 1001;
        static thread_local std::string thread_request_id = "";
        std::vector<std::string> paths;
        for (int i = 0; i < numImages; ++i) {
            if (imagePaths[i]) paths.emplace_back(imagePaths[i]);
        }
        RPPGProcessor rppg;
        RPPGResult r = rppg.processImagesFromPaths(paths);
        auto bp = g_estimator->estimate_bp(r.peak_times, height, weight, sex);
        static thread_local std::string csv;
        static thread_local std::string errors;
        csv = generateCSV(r.rppg_signal, r.time_data, r.peak_times);
        errors = "[]";
        if (callback) callback(thread_request_id.c_str(), bp.first, bp.second, csv.c_str(), errors.c_str());
        return 0;
    } catch (const std::exception& e) {
        std::ofstream log("dll_error.log", std::ios::app);
        log << "AnalyzeBloodPressureFromImages exception: " << e.what() << std::endl;
        static thread_local std::string thread_request_id = "";
        static thread_local std::string errors;
        errors = std::string("[{\"code\":\"1006\",\"message\":\"") + e.what() + "\",\"isRetriable\":false}]";
        if (callback) callback(thread_request_id.c_str(), 0, 0, empty_str.c_str(), errors.c_str());
        return 1006;
    } catch (...) {
        std::ofstream log("dll_error.log", std::ios::app);
        log << "AnalyzeBloodPressureFromImages unknown exception" << std::endl;
        static thread_local std::string thread_request_id = "";
        static thread_local std::string errors = "[{\"code\":\"1006\",\"message\":\"unknown error\",\"isRetriable\":false}]";
        if (callback) callback(thread_request_id.c_str(), 0, 0, empty_str.c_str(), errors.c_str());
        return 1006;
    }
}

} 
