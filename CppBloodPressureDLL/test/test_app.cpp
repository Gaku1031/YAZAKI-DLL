#include "../include/BloodPressureDLL.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <filesystem>
#include <iomanip> // For std::fixed and std::setprecision

// パフォーマンステスト用のヘルパー関数
void run_performance_test(const std::string& video_path) {
    std::cout << "=== Performance Test ===" << std::endl;
    
    // 1. 初期化時間測定
    auto start_init = std::chrono::high_resolution_clock::now();
    BloodPressureDLL bp_dll;
    auto end_init = std::chrono::high_resolution_clock::now();
    auto init_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_init - start_init);
    std::cout << "Initialization time: " << init_time.count() << " ms" << std::endl;
    
    // 2. 処理時間測定
    auto start_process = std::chrono::high_resolution_clock::now();
    BloodPressureResult result = bp_dll.estimateBloodPressure(video_path);
    auto end_process = std::chrono::high_resolution_clock::now();
    auto process_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_process - start_process);
    
    std::cout << "Processing time: " << process_time.count() << " ms" << std::endl;
    std::cout << "Processing speed: " << std::fixed << std::setprecision(2) 
              << (process_time.count() / 1000.0) << " seconds" << std::endl;
    
    // 3. メモリ使用量推定
    std::cout << "Estimated memory usage: ~50-100MB (including OpenCV + dlib)" << std::endl;
    
    // 4. 結果表示
    if (result.success) {
        std::cout << "Systolic BP: " << result.systolic << " mmHg" << std::endl;
        std::cout << "Diastolic BP: " << result.diastolic << " mmHg" << std::endl;
        std::cout << "Heart Rate: " << result.heart_rate << " bpm" << std::endl;
    } else {
        std::cout << "Processing failed: " << result.error_message << std::endl;
    }
    
    // 5. タイミング詳細（利用可能な場合）
    std::string timing_summary = bp_dll.getTimingSummary();
    if (!timing_summary.empty()) {
        std::cout << "\n=== Detailed Timing ===" << std::endl;
        std::cout << timing_summary << std::endl;
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
            BloodPressureDLL bp_dll;
            BloodPressureResult result = bp_dll.estimateBloodPressure(video_path);
            
            if (result.success) {
                std::cout << "Blood Pressure Estimation Results:" << std::endl;
                std::cout << "Systolic: " << result.systolic << " mmHg" << std::endl;
                std::cout << "Diastolic: " << result.diastolic << " mmHg" << std::endl;
                std::cout << "Heart Rate: " << result.heart_rate << " bpm" << std::endl;
            } else {
                std::cerr << "Error: " << result.error_message << std::endl;
                return 1;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 
