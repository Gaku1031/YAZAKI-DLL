#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#ifdef BLOODPRESSURE_EXPORTS
#define BLOODPRESSURE_API __declspec(dllexport)
#else
#define BLOODPRESSURE_API __declspec(dllimport)
#endif

extern "C" {
    // Callback function type definition
    typedef void(*AnalysisCallback)(const char* requestId, int maxBloodPressure, 
                                   int minBloodPressure, const char* measureRowData, 
                                   const char* errors);

    // Export functions
    BLOODPRESSURE_API bool InitializeDLL(const char* modelDir);
    BLOODPRESSURE_API const char* StartBloodPressureAnalysisRequest(
        const char* requestId, int height, int weight, int sex, 
        const char* moviePath, AnalysisCallback callback);
    BLOODPRESSURE_API const char* GetProcessingStatus(const char* requestId);
    BLOODPRESSURE_API bool CancelBloodPressureAnalysis(const char* requestId);
    BLOODPRESSURE_API const char* GetVersionInfo();
}