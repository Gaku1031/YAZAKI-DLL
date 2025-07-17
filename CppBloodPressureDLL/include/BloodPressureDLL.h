#pragma once
#ifdef _WIN32
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif

extern "C" {
// コールバック型
typedef void(*BPCallback)(
    const char* requestId,
    int maxBloodPressure,
    int minBloodPressure,
    const char* measureRowData,
    const char* errorsJson // エラーはJSON文字列で渡す
);
// DLL初期化
DLL_EXPORT int InitializeBP(const char* modelDir);
// 血圧解析リクエスト
DLL_EXPORT const char* StartBloodPressureAnalysisRequest(
    const char* requestId, int height, int weight, int sex,
    const char* moviePath, BPCallback callback);
// 処理状況取得
DLL_EXPORT const char* GetProcessingStatus(const char* requestId);
// 血圧解析中断
DLL_EXPORT int CancelBloodPressureAnalysis(const char* requestId);
// バージョン情報取得
DLL_EXPORT const char* GetVersionInfo();
// リクエストID生成
DLL_EXPORT const char* GenerateRequestId();
} 
