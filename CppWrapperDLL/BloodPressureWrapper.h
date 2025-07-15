#pragma once

#ifdef __cplusplus
extern "C" {
#endif

__declspec(dllexport) int InitializeBP(const char* model_dir);
__declspec(dllexport) const char* StartBloodPressureAnalysisRequest(const char* request_id, int height, int weight, int sex, const char* movie_path);
__declspec(dllexport) const char* GetProcessingStatus(const char* request_id);
__declspec(dllexport) int CancelBloodPressureAnalysis(const char* request_id);
__declspec(dllexport) const char* GetVersionInfo();
__declspec(dllexport) const char* GenerateRequestId();

#ifdef __cplusplus
}
#endif 
