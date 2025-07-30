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
#include <cstring> // For strncpy

#ifdef _WIN32
#include <windows.h>
#endif

// 詳細タイミング計測用の構造体
struct DetailedTiming {
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    std::string stage_name;
    
    void start(const std::string& name) {
        stage_name = name;
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    void end() {
        end_time = std::chrono::high_resolution_clock::now();
    }
    
    double get_duration_ms() const {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000.0;
    }
};

// グローバルタイミング記録
static std::vector<DetailedTiming> g_timing_log;

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
    
    // タイミング計測ヘルパー関数
    void start_timing(const std::string& stage_name) {
        DetailedTiming timing;
        timing.start(stage_name);
        g_timing_log.push_back(timing);
    }
    
    void end_timing() {
        if (!g_timing_log.empty()) {
            g_timing_log.back().end();
        }
    }
    
    std::string get_timing_summary() {
        std::stringstream ss;
        ss << "\n=== DETAILED TIMING ANALYSIS ===\n";
        
        double total_time = 0.0;
        for (const auto& timing : g_timing_log) {
            double duration = timing.get_duration_ms();
            total_time += duration;
            ss << std::fixed << std::setprecision(2) 
               << timing.stage_name << ": " << duration << " ms\n";
        }
        
        ss << "Total time: " << total_time << " ms\n";
        ss << "=== TIMING BREAKDOWN ===\n";
        
        // 各段階の割合を計算
        for (const auto& timing : g_timing_log) {
            double duration = timing.get_duration_ms();
            double percentage = (total_time > 0) ? (duration / total_time) * 100.0 : 0.0;
            ss << std::fixed << std::setprecision(1) 
               << timing.stage_name << ": " << percentage << "%\n";
        }
        
        return ss.str();
    }
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
    const char* msg = "v1.0.0";
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
        // Process video using original method (temporarily disable direct processing)
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
        
        // タイミング情報をファイルに出力
        printf("[DEBUG] About to get timing summaries...\n");
        std::string rppg_timing = rppg.get_timing_summary();
        printf("[DEBUG] RPPG timing length: %zu\n", rppg_timing.length());
        
        std::string bp_timing = g_estimator->get_timing_summary();
        printf("[DEBUG] BP timing length: %zu\n", bp_timing.length());
        
        std::string timing_info = rppg_timing + bp_timing;
        printf("[DEBUG] Combined timing length: %zu\n", timing_info.length());
        
        if (timing_info.empty()) {
            printf("[DEBUG] WARNING: Timing info is empty!\n");
            timing_info = "=== NO TIMING DATA AVAILABLE ===\n";
        }
        
        printf("%s", timing_info.c_str());
        fflush(stdout);
        
        printf("[DEBUG] Timing info preview: %.200s...\n", timing_info.c_str());
        
        // タイミング情報をファイルに保存
        std::ofstream timing_file("detailed_timing.log");
        if (timing_file.is_open()) {
            timing_file << "=== DETAILED TIMING ANALYSIS ===" << std::endl;
            timing_file << timing_info << std::endl;
            timing_file.close();
            printf("[DEBUG] Timing file saved successfully\n");
        } else {
            printf("[DEBUG] Failed to open timing file for writing\n");
        }
        
        // タイミング情報をJSON形式でエラーフィールドに含める
        std::string escaped_timing = timing_info;
        // 改行とタブをエスケープ
        size_t pos = 0;
        while ((pos = escaped_timing.find('\n', pos)) != std::string::npos) {
            escaped_timing.replace(pos, 1, "\\n");
            pos += 2;
        }
        pos = 0;
        while ((pos = escaped_timing.find('\t', pos)) != std::string::npos) {
            escaped_timing.replace(pos, 1, "\\t");
            pos += 2;
        }
        pos = 0;
        while ((pos = escaped_timing.find('"', pos)) != std::string::npos) {
            escaped_timing.replace(pos, 1, "\\\"");
            pos += 2;
        }
        std::string timing_json = "{\"timing_info\":\"" + escaped_timing + "\"}";
        printf("[DEBUG] JSON length: %zu\n", timing_json.length());
        printf("[DEBUG] JSON preview: %.100s...\n", timing_json.c_str());
        printf("[DEBUG] About to call callback function...\n");
        printf("[DEBUG] Callback function pointer: %p\n", (void*)callback);
        if (callback) {
            printf("[DEBUG] Calling callback function...\n");
            callback(requestId, result.first, result.second, csv.c_str(), timing_json.c_str());
            printf("[DEBUG] Callback function called successfully\n");
        } else {
            printf("[DEBUG] Callback function is null!\n");
        }
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
__declspec(dllexport)
int AnalyzeBloodPressureFromImages(
    char* outBuf, int bufSize,
    const char** imagePaths, int numImages, int height, int weight, int sex, BPCallback callback)
{
    try {
        if (!g_estimator) {
            if (callback) callback("", 0, 0, "", "[{\"code\":1001,\"message\":\"DLL not initialized\",\"isRetriable\":false}]");
            snprintf(outBuf, bufSize, "DLL not initialized");
            return 1001;
        }
        
        if (!imagePaths || numImages <= 0) {
            if (callback) callback("", 0, 0, "", "[{\"code\":1002,\"message\":\"Invalid image paths or count\",\"isRetriable\":false}]");
            snprintf(outBuf, bufSize, "Invalid image paths or count");
            return 1002;
        }
        
        std::string model_dir = g_model_dir.empty() ? "models" : g_model_dir;
        RPPGProcessor rppg(model_dir);
        
        // 画像パスをstd::vectorに変換
        std::vector<std::string> image_paths;
        for (int i = 0; i < numImages; ++i) {
            if (imagePaths[i]) {
                image_paths.push_back(imagePaths[i]);
            }
        }
        
        if (image_paths.empty()) {
            if (callback) callback("", 0, 0, "", "[{\"code\":1003,\"message\":\"No valid image paths provided\",\"isRetriable\":false}]");
            snprintf(outBuf, bufSize, "No valid image paths provided");
            return 1003;
        }
        
        RPPGResult rppg_result = rppg.processImagesFromPaths(image_paths, 30.0);
        if (rppg_result.peak_times.empty()) {
            if (callback) callback("", 0, 0, "", "[{\"code\":1006,\"message\":\"No peaks detected from images\",\"isRetriable\":false}]");
            snprintf(outBuf, bufSize, "No peaks detected from images");
            return 1006;
        }
        
        // ピーク時刻配列: std::vector<double> peak_times
        std::vector<double> peaks(rppg_result.peak_times.begin(), rppg_result.peak_times.end());
        auto result = g_estimator->estimate_bp(peaks, height, weight, sex);
        std::string csv = generateCSV(rppg_result.rppg_signal, rppg_result.time_data, rppg_result.peak_times);
        
        // タイミング情報を取得
        std::string rppg_timing = rppg.get_timing_summary();
        std::string bp_timing = g_estimator->get_timing_summary();
        std::string timing_info = rppg_timing + bp_timing;
        
        // Debug prints for timing info
        printf("[DEBUG] About to get timing summaries...\n");
        printf("[DEBUG] RPPG timing length: %zu\n", rppg_timing.length());
        printf("[DEBUG] BP timing length: %zu\n", bp_timing.length());
        printf("[DEBUG] Combined timing length: %zu\n", timing_info.length());
        if (timing_info.empty()) {
            printf("[DEBUG] WARNING: Timing info is empty!\n");
            timing_info = "=== NO TIMING DATA AVAILABLE ===\n";
        }
        printf("%s", timing_info.c_str()); // Print to stdout
        fflush(stdout);
        printf("[DEBUG] Timing info preview: %.200s...\n", timing_info.c_str());
        
        // Save to file
        std::ofstream timing_file("detailed_timing.log");
        if (timing_file.is_open()) {
            timing_file << "=== DETAILED TIMING ANALYSIS ===" << std::endl;
            timing_file << timing_info << std::endl;
            timing_file.close();
            printf("[DEBUG] Timing file saved successfully\n");
        } else {
            printf("[DEBUG] Failed to open timing file for writing\n");
        }
        
        // JSON escape and pass to callback
        std::string escaped_timing = timing_info;
        size_t pos = 0;
        while ((pos = escaped_timing.find('\n', pos)) != std::string::npos) {
            escaped_timing.replace(pos, 1, "\\n");
            pos += 2;
        }
        pos = 0;
        while ((pos = escaped_timing.find('\t', pos)) != std::string::npos) {
            escaped_timing.replace(pos, 1, "\\t");
            pos += 2;
        }
        pos = 0;
        while ((pos = escaped_timing.find('"', pos)) != std::string::npos) {
            escaped_timing.replace(pos, 1, "\\\"");
            pos += 2;
        }
        std::string timing_json = "{\"timing_info\":\"" + escaped_timing + "\"}";
        printf("[DEBUG] JSON length: %zu\n", timing_json.length());
        printf("[DEBUG] JSON preview: %.100s...\n", timing_json.c_str());
        
        // Call callback
        printf("[DEBUG] About to call callback function...\n");
        printf("[DEBUG] Callback function pointer: %p\n", (void*)callback);
        if (callback) {
            printf("[DEBUG] Calling callback function...\n");
            callback("", result.first, result.second, csv.c_str(), timing_json.c_str());
            printf("[DEBUG] Callback function called successfully\n");
        } else {
            printf("[DEBUG] Callback function is null!\n");
        }
        
        snprintf(outBuf, bufSize, "OK");
        return 0;
        
    } catch (const std::exception& e) {
        if (callback) callback("", 0, 0, "", (std::string("[{\"code\":1006,\"message\":\"") + e.what() + "\",\"isRetriable\":false}]").c_str());
        snprintf(outBuf, bufSize, "%s", e.what());
        return 1006;
    } catch (...) {
        if (callback) callback("", 0, 0, "", "[{\"code\":1006,\"message\":\"Unknown exception\",\"isRetriable\":false}]");
        snprintf(outBuf, bufSize, "Unknown exception");
        return 1006;
    }
}

// GenerateRequestId関数の実装
extern "C" __declspec(dllexport)
int GenerateRequestId(char* outBuf, int bufSize) {
    try {
        // 現在のタイムスタンプを取得
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
        
        // タイムスタンプを文字列に変換
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
        ss << "_" << std::setfill('0') << std::setw(3) << ms.count();
        
        std::string requestId = "REQ_" + ss.str();
        
        // バッファサイズをチェック
        if (static_cast<int>(requestId.length()) >= bufSize) {
            return -1; // バッファが小さすぎる
        }
        
        // 結果をコピー
        strncpy(outBuf, requestId.c_str(), bufSize - 1);
        outBuf[bufSize - 1] = '\0'; // 確実にnull終端
        return 0; // 成功
        
    } catch (const std::exception& e) {
        if (bufSize > 0) {
            strncpy(outBuf, "ERROR", bufSize - 1);
            outBuf[bufSize - 1] = '\0';
        }
        return -1;
    } catch (...) {
        if (bufSize > 0) {
            strncpy(outBuf, "ERROR", bufSize - 1);
            outBuf[bufSize - 1] = '\0';
        }
        return -1;
    }
}

} // extern "C"
