#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// 初期化・終了処理
__declspec(dllexport) int InitializeBP(const char* model_dir);
__declspec(dllexport) void CleanupBP();

// 血圧解析機能
__declspec(dllexport) const char* StartBloodPressureAnalysisRequest(
    const char* request_id, 
    int height, 
    int weight, 
    int sex, 
    const char* movie_path
);

__declspec(dllexport) const char* GetProcessingStatus(const char* request_id);
__declspec(dllexport) int CancelBloodPressureAnalysis(const char* request_id);

// ユーティリティ関数
__declspec(dllexport) const char* GetVersionInfo();
__declspec(dllexport) const char* GenerateRequestId();

// デバッグ・エラー処理
__declspec(dllexport) const char* GetLastError();

#ifdef __cplusplus
}
#endif
