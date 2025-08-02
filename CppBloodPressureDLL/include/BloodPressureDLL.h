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
DLL_EXPORT int InitializeBP(char* outBuf, int bufSize, const char* modelDir);
// 血圧解析リクエスト
DLL_EXPORT int StartBloodPressureAnalysisRequest(char* outBuf, int bufSize,
    const char* requestId, int height, int weight, int sex,
    const char* moviePath, BPCallback callback);
// 処理状況取得
DLL_EXPORT int GetProcessingStatus(char* outBuf, int bufSize, const char* requestId);
// 血圧解析中断
DLL_EXPORT int CancelBloodPressureAnalysis(char* outBuf, int bufSize, const char* requestId);
// バージョン情報取得
DLL_EXPORT int GetVersionInfo(char* outBuf, int bufSize);
// リクエストID生成
DLL_EXPORT int GenerateRequestId(char* outBuf, int bufSize);
// 画像配列から血圧推定（C#から呼び出し用）
DLL_EXPORT int AnalyzeBloodPressureFromImages(char* outBuf, int bufSize,
    const char** imagePaths, int numImages, int height, int weight, int sex, BPCallback callback);
} 
