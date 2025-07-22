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
    // std::mutex& getSafeMutex() { return getSafeStatic<std::mutex>(); }
    // std::map<std::string, std::thread>& getSafeThreads() { return getSafeStatic<std::map<std::string, std::thread>>(); }
    // std::map<std::string, std::string>& getSafeStatus() { return getSafeStatic<std::map<std::string, std::string>>(); }
    // std::mutex& getSafeCallbackMutex() { return getSafeStatic<std::mutex>(); }
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
__declspec(dllexport)
int InitializeBP(char* outBuf, int bufSize, const char* modelDir) {
    if (!outBuf || bufSize <= 0) return -1;
    const char* msg = "OK";
    int n = 0;
    while (msg[n] && n < bufSize - 1) {
        outBuf[n] = msg[n];
        ++n;
    }
    outBuf[n] = '\0';
    return 0;
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
} 
