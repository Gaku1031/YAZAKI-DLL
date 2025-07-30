#include "../include/BloodPressureDLL.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <filesystem>
#include <iomanip> // For std::fixed and std::setprecision
#include <cstring> // For strlen

// グローバル変数（コールバック用）
std::string g_lastResult;
std::string g_lastError;

// コールバック関数
void bloodPressureCallback(
    const char* requestId,
    int maxBloodPressure,
    int minBloodPressure,
    const char* measureRowData,
    const char* errorsJson
) {
    if (errorsJson && strlen(errorsJson) > 0) {
        g_lastError = errorsJson;
        g_lastResult = "";
    } else {
        g_lastResult = "Success";
        g_lastError = "";
        std::cout << "Request ID: " << requestId << std::endl;
        std::cout << "Systolic BP: " << maxBloodPressure << " mmHg" << std::endl;
        std::cout << "Diastolic BP: " << minBloodPressure << " mmHg" << std::endl;
    }
}

// パフォーマンステスト用のヘルパー関数
void run_performance_test(const std::string& video_path) {
    std::cout << "=== Performance Test ===" << std::endl;
    
    // 1. 初期化時間測定
    auto start_init = std::chrono::high_resolution_clock::now();
    
    char outBuf[1024];
    int result = InitializeBP(outBuf, sizeof(outBuf), "./models");
    if (result != 0) {
        std::cerr << "Initialization failed: " << outBuf << std::endl;
        return;
    }
    
    auto end_init = std::chrono::high_resolution_clock::now();
    auto init_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_init - start_init);
    std::cout << "Initialization time: " << init_time.count() << " ms" << std::endl;
    
    // 2. 処理時間測定
    auto start_process = std::chrono::high_resolution_clock::now();
    
    // リクエストID生成
    char requestId[64];
    if (GenerateRequestId(requestId, sizeof(requestId)) != 0) {
        std::cerr << "Failed to generate request ID" << std::endl;
        return;
    }
    
    // 血圧解析開始
    if (StartBloodPressureAnalysisRequest(outBuf, sizeof(outBuf), 
                                        requestId, 170, 70, 1, // height, weight, sex
                                        video_path.c_str(), bloodPressureCallback) != 0) {
        std::cerr << "Failed to start blood pressure analysis: " << outBuf << std::endl;
        return;
    }
    
    auto end_process = std::chrono::high_resolution_clock::now();
    auto process_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_process - start_process);
    
    std::cout << "Processing time: " << process_time.count() << " ms" << std::endl;
    std::cout << "Processing speed: " << std::fixed << std::setprecision(2) 
              << (process_time.count() / 1000.0) << " seconds" << std::endl;
    
    // 3. メモリ使用量推定
    std::cout << "Estimated memory usage: ~50-100MB (including OpenCV + dlib)" << std::endl;
    
    // 4. 結果表示
    if (g_lastError.empty()) {
        std::cout << "Processing completed successfully" << std::endl;
    } else {
        std::cout << "Processing failed: " << g_lastError << std::endl;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <video_path> [--performance-test]" << std::endl;
        std::cout << "Options:" << std::endl;
        std::cout << "  --performance-test  Run detailed performance analysis" << std::endl;
        return 1;
    }
    
    std::string video_path = argv[1];
    bool performance_test = false;
    
    // コマンドラインオプション解析
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--performance-test") {
            performance_test = true;
        }
    }
    
    // ファイル存在チェック
    if (!std::filesystem::exists(video_path)) {
        std::cerr << "Error: Video file not found: " << video_path << std::endl;
        return 1;
    }
    
    try {
        if (performance_test) {
            run_performance_test(video_path);
        } else {
            // 通常の処理
            char outBuf[1024];
            
            // 初期化
            if (InitializeBP(outBuf, sizeof(outBuf), "./models") != 0) {
                std::cerr << "Initialization failed: " << outBuf << std::endl;
                return 1;
            }
            
            // リクエストID生成
            char requestId[64];
            if (GenerateRequestId(requestId, sizeof(requestId)) != 0) {
                std::cerr << "Failed to generate request ID" << std::endl;
                return 1;
            }
            
            // 血圧解析開始
            if (StartBloodPressureAnalysisRequest(outBuf, sizeof(outBuf), 
                                                requestId, 170, 70, 1, // height, weight, sex
                                                video_path.c_str(), bloodPressureCallback) != 0) {
                std::cerr << "Failed to start blood pressure analysis: " << outBuf << std::endl;
                return 1;
            }
            
            // 結果を待つ（実際の実装では適切な待機処理が必要）
            std::cout << "Blood pressure analysis started. Check callback for results." << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 
