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
#include <numeric> // For std::accumulate
#include <algorithm> // For std::min_element, std::max_element

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
    // std::mutex& getSafeMutex() { return getSafeStatic<std::mutex>(); }
    // std::map<std::string, std::thread>& getSafeThreads() { return getSafeStatic<std::map<std::string, std::thread>>(); }
    // std::map<std::string, std::string>& getSafeStatus() { return getSafeStatic<std::map<std::string, std::string>>(); }
    // std::mutex& getSafeCallbackMutex() { return getSafeStatic<std::mutex>(); }
    // std::string& getSafeCSVStr() { return getSafeStatic<std::string>(); }
    // std::string& getSafeErrorsStr() { return getSafeStatic<std::string>(); }
    
    std::atomic<bool> initialized{false};
    static std::unique_ptr<BloodPressureEstimator> g_estimator;
    static std::string g_model_dir;
    
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

typedef void(*BPCallback)(const char* requestId, int sbp, int dbp, const char* csv, const char* errorsJson);

__declspec(dllexport)
int InitializeBP(char* outBuf, int bufSize, const char* modelDir) {
    if (!outBuf || bufSize <= 0) return -1;
    try {
        // 既存のインスタンスを解放
        g_estimator.reset();
        // モデルファイル存在チェック
        std::string sbp_path = std::string(modelDir) + "/systolicbloodpressure.onnx";
        std::string dbp_path = std::string(modelDir) + "/diastolicbloodpressure.onnx";
        if (!file_exists(sbp_path) || !file_exists(dbp_path)) {
            std::string msg = "Model file not found: " + sbp_path + " or " + dbp_path;
            snprintf(outBuf, bufSize, "%s", msg.c_str());
            FILE* f = fopen("dll_error.log", "a");
            if (f) { fprintf(f, "%s\n", msg.c_str()); fclose(f); }
            return 1;
        }
        // 推論器初期化
        g_estimator = std::make_unique<BloodPressureEstimator>(modelDir);
        g_model_dir = modelDir;
        snprintf(outBuf, bufSize, "ORT_ENV_OK");
        return 0;
    } catch (const std::exception& e) {
        snprintf(outBuf, bufSize, "InitializeBP exception: %s", e.what());
        FILE* f = fopen("dll_error.log", "a");
        if (f) { fprintf(f, "InitializeBP exception: %s\n", e.what()); fclose(f); }
        g_estimator.reset();
        return 1;
    }
}
__declspec(dllexport)
int GetVersionInfo(char* outBuf, int bufSize) {
    if (!outBuf || bufSize <= 0) return -1;
    const char* msg = "VERSION";
    int n = 0;
    while (msg[n] && n < bufSize - 1) {
        outBuf[n] = msg[n];
        ++n;
    }
    outBuf[n] = '\0';
    return 0;
}
__declspec(dllexport)
int EstimateBloodPressure(
    double* peak_times, int peak_count,
    int height, int weight, int sex,
    int* sbp, int* dbp)
{
    try {
        if (!g_estimator) return -1;
        // ピーク時刻配列: std::vector<double> peak_times
        std::vector<double> peaks(peak_times, peak_times + peak_count);
        auto result = g_estimator->estimate_bp(peaks, height, weight, sex);
        if (sbp) *sbp = result.first;
        if (dbp) *dbp = result.second;
        return 0;
    } catch (const std::exception& e) {
        FILE* f = fopen("dll_error.log", "a");
        if (f) {
            fprintf(f, "EstimateBloodPressure exception: %s\n", e.what());
            fclose(f);
        }
        return -1;
    } catch (...) {
        FILE* f = fopen("dll_error.log", "a");
        if (f) {
            fprintf(f, "EstimateBloodPressure unknown exception\n");
            fclose(f);
        }
        return -1;
    }
}
__declspec(dllexport)
int EstimateBloodPressureFromVideo(
    const char* videoPath,
    int height, int weight, int sex,
    int* sbp, int* dbp)
{
    try {
        if (!g_estimator) return -1;
        // Get model directory from stored global variable
        std::string model_dir = g_model_dir.empty() ? "models" : g_model_dir;
        RPPGProcessor rppg(model_dir);
        RPPGResult rppg_result = rppg.processVideo(videoPath);
        if (rppg_result.peak_times.empty()) {
            FILE* f = fopen("dll_error.log", "a");
            if (f) {
                fprintf(f, "EstimateBloodPressureFromVideo: No peaks detected or video read error: %s\n", videoPath);
                fclose(f);
            }
            return -1;
        }
        // ピーク時刻配列: std::vector<double> peak_times
        std::vector<double> peaks(rppg_result.peak_times.begin(), rppg_result.peak_times.end());
        auto result = g_estimator->estimate_bp(peaks, height, weight, sex);
        if (sbp) *sbp = result.first;
        if (dbp) *dbp = result.second;
        return 0;
    } catch (const std::exception& e) {
        FILE* f = fopen("dll_error.log", "a");
        if (f) {
            fprintf(f, "EstimateBloodPressureFromVideo exception: %s\n", e.what());
            fclose(f);
        }
        return -1;
    } catch (...) {
        FILE* f = fopen("dll_error.log", "a");
        if (f) {
            fprintf(f, "EstimateBloodPressureFromVideo unknown exception\n");
            fclose(f);
        }
        return -1;
    }
}
extern "C" {
__declspec(dllexport)
int StartBloodPressureAnalysisRequest(
    char* outBuf, int bufSize,
    const char* requestId,
    int height, int weight, int sex,
    const char* moviePath,
    BPCallback callback)
{
    try {
        if (!g_estimator) {
            if (callback) callback(requestId, 0, 0, "", "[{\"code\":1001,\"message\":\"DLL not initialized\",\"isRetriable\":false}]");
            snprintf(outBuf, bufSize, "DLL not initialized");
            return 1001;
        }
        std::string model_dir = g_model_dir.empty() ? "models" : g_model_dir;
        RPPGProcessor rppg(model_dir);
        RPPGResult rppg_result = rppg.processVideo(moviePath);
        if (rppg_result.peak_times.empty()) {
            if (callback) callback(requestId, 0, 0, "", "[{\"code\":1006,\"message\":\"No peaks detected or video read error\",\"isRetriable\":false}]");
            snprintf(outBuf, bufSize, "No peaks detected or video read error");
            return 1006;
        }
        // ピーク時刻配列: std::vector<double> peak_times
        std::vector<double> peaks(rppg_result.peak_times.begin(), rppg_result.peak_times.end());
        auto result = g_estimator->estimate_bp(peaks, height, weight, sex);
        std::string csv = generateCSV(rppg_result.rppg_signal, rppg_result.time_data, rppg_result.peak_times);
        if (callback) callback(requestId, result.first, result.second, csv.c_str(), "[]");
        snprintf(outBuf, bufSize, "OK");
        return 0;
    } catch (const std::exception& e) {
        if (callback) callback(requestId, 0, 0, "", (std::string("[{\"code\":1006,\"message\":\"") + e.what() + "\",\"isRetriable\":false}]").c_str());
        snprintf(outBuf, bufSize, "%s", e.what());
        return 1006;
    } catch (...) {
        if (callback) callback(requestId, 0, 0, "", "[{\"code\":1006,\"message\":\"Unknown exception\",\"isRetriable\":false}]");
        snprintf(outBuf, bufSize, "Unknown exception");
        return 1006;
    }
}
} // extern "C"
