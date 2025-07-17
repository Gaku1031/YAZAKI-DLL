#include "../include/BloodPressureDLL.h"
#include <iostream>
#include <thread>

void result_callback(const char* requestId, int maxBP, int minBP, const char* csv, const char* errorsJson) {
    std::cout << "[CALLBACK] RequestID: " << requestId << std::endl;
    std::cout << "  MaxBP: " << maxBP << ", MinBP: " << minBP << std::endl;
    std::cout << "  CSV: " << csv << std::endl;
    std::cout << "  Errors: " << errorsJson << std::endl;
}

int main() {
    std::cout << "DLL Version: " << GetVersionInfo() << std::endl;
    if (!InitializeBP("../models")) {
        std::cerr << "DLL初期化失敗" << std::endl;
        return 1;
    }
    std::string reqId = GenerateRequestId();
    std::cout << "RequestID: " << reqId << std::endl;
    const char* err = StartBloodPressureAnalysisRequest(
        reqId.c_str(), 170, 70, 1, "sample.webm", result_callback);
    if (err) {
        std::cerr << "解析開始失敗: " << err << std::endl;
        return 1;
    }
    // ステータス監視
    while (true) {
        std::string status = GetProcessingStatus(reqId.c_str());
        std::cout << "Status: " << status << std::endl;
        if (status == "none") break;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    std::cout << "テスト完了" << std::endl;
    return 0;
} 
