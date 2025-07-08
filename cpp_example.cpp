/**
 * @file cpp_example.cpp
 * @brief 血圧推定DLL C++使用例
 * @version 1.0.0
 */

#include <windows.h>
#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include "BloodPressureEstimation.h"

// グローバル変数
bool g_processing_completed = false;
std::string g_result_message;

/**
 * @brief 血圧解析結果コールバック関数
 */
void OnBloodPressureResult(const char* requestId, int maxBloodPressure, int minBloodPressure, 
                          const char* measureRowData, const BPErrorInfo* errors) {
    std::cout << "=== 血圧解析結果 ===" << std::endl;
    std::cout << "Request ID: " << requestId << std::endl;
    std::cout << "最高血圧: " << maxBloodPressure << " mmHg" << std::endl;
    std::cout << "最低血圧: " << minBloodPressure << " mmHg" << std::endl;
    std::cout << "CSVデータサイズ: " << strlen(measureRowData) << " 文字" << std::endl;
    
    if (errors != nullptr) {
        std::cout << "エラー発生:" << std::endl;
        std::cout << "  コード: " << errors->code << std::endl;
        std::cout << "  メッセージ: " << errors->message << std::endl;
        std::cout << "  再試行可能: " << (errors->is_retriable ? "はい" : "いいえ") << std::endl;
    } else {
        std::cout << "解析成功（エラーなし）" << std::endl;
    }
    
    g_processing_completed = true;
    g_result_message = "解析完了";
}

/**
 * @brief DLL関数ポインタ型定義
 */
typedef BOOL (*InitializeDLLFunc)(const char*);
typedef const char* (*StartBPAnalysisFunc)(const char*, int, int, int, const char*, BPAnalysisCallback);
typedef BOOL (*CancelBPProcessingFunc)(const char*);
typedef const char* (*GetBPStatusFunc)(const char*);
typedef const char* (*GetDLLVersionFunc)(void);
typedef const char* (*GenerateRequestIDFunc)(const char*, const char*, char*);
typedef BOOL (*ValidateMovieFileFunc)(const char*);

int main() {
    std::cout << "血圧推定DLL C++サンプル" << std::endl;
    std::cout << "========================" << std::endl;
    
    // DLLロード
    HINSTANCE hDLL = LoadLibrary(L"BloodPressureEstimation.dll");
    if (!hDLL) {
        std::cerr << "エラー: DLLをロードできませんでした" << std::endl;
        return -1;
    }
    
    // 関数ポインタ取得
    InitializeDLLFunc InitializeDLL = (InitializeDLLFunc)GetProcAddress(hDLL, "InitializeDLL");
    StartBPAnalysisFunc StartBPAnalysis = (StartBPAnalysisFunc)GetProcAddress(hDLL, "StartBloodPressureAnalysis");
    CancelBPProcessingFunc CancelBPProcessing = (CancelBPProcessingFunc)GetProcAddress(hDLL, "CancelBloodPressureProcessing");
    GetBPStatusFunc GetBPStatus = (GetBPStatusFunc)GetProcAddress(hDLL, "GetBloodPressureStatus");
    GetDLLVersionFunc GetDLLVersion = (GetDLLVersionFunc)GetProcAddress(hDLL, "GetDLLVersion");
    GenerateRequestIDFunc GenerateRequestID = (GenerateRequestIDFunc)GetProcAddress(hDLL, "GenerateRequestID");
    ValidateMovieFileFunc ValidateMovieFile = (ValidateMovieFileFunc)GetProcAddress(hDLL, "ValidateMovieFile");
    
    if (!InitializeDLL || !StartBPAnalysis || !GetBPStatus || !GetDLLVersion) {
        std::cerr << "エラー: 必要な関数を取得できませんでした" << std::endl;
        FreeLibrary(hDLL);
        return -1;
    }
    
    // DLLバージョン表示
    std::cout << "DLLバージョン: " << GetDLLVersion() << std::endl;
    
    // DLL初期化
    std::cout << "\nDLL初期化中..." << std::endl;
    if (!InitializeDLL("models")) {
        std::cerr << "エラー: DLL初期化に失敗しました" << std::endl;
        FreeLibrary(hDLL);
        return -1;
    }
    std::cout << "DLL初期化成功" << std::endl;
    
    // リクエストID生成
    char request_id_buffer[64];
    const char* request_id = GenerateRequestID("9000000001", "0000012345", request_id_buffer);
    std::cout << "生成されたリクエストID: " << request_id << std::endl;
    
    // 動画ファイルパス設定
    const char* movie_path = "sample-data\\100万画素.webm";
    
    // 動画ファイル検証
    if (ValidateMovieFile && ValidateMovieFile(movie_path)) {
        std::cout << "動画ファイル検証成功: " << movie_path << std::endl;
    } else {
        std::cout << "警告: 動画ファイル検証に失敗しました（処理は続行されます）" << std::endl;
    }
    
    // 生体パラメータ設定
    int height = 170;  // cm
    int weight = 70;   // kg
    int sex = BP_SEX_MALE;  // 男性
    
    std::cout << "\n血圧解析開始..." << std::endl;
    std::cout << "  身長: " << height << " cm" << std::endl;
    std::cout << "  体重: " << weight << " kg" << std::endl;
    std::cout << "  性別: " << (sex == BP_SEX_MALE ? "男性" : "女性") << std::endl;
    std::cout << "  動画: " << movie_path << std::endl;
    
    // 血圧解析開始
    const char* error_code = StartBPAnalysis(request_id, height, weight, sex, movie_path, OnBloodPressureResult);
    
    if (error_code != nullptr) {
        std::cerr << "エラー: 血圧解析開始に失敗しました（エラーコード: " << error_code << "）" << std::endl;
        FreeLibrary(hDLL);
        return -1;
    }
    
    std::cout << "血圧解析開始成功（非同期処理）" << std::endl;
    
    // 処理状況監視
    std::cout << "\n処理状況監視..." << std::endl;
    while (!g_processing_completed) {
        const char* status = GetBPStatus(request_id);
        std::cout << "処理状況: " << status << std::endl;
        
        if (strcmp(status, BP_STATUS_NONE) == 0 && g_processing_completed) {
            break;
        }
        
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
    
    std::cout << "\n" << g_result_message << std::endl;
    
    // クリーンアップ
    FreeLibrary(hDLL);
    
    std::cout << "\nプログラム終了。Enterキーを押してください..." << std::endl;
    std::cin.get();
    
    return 0;
}

/**
 * @brief コンパイル方法
 * 
 * Visual Studio Developer Command Prompt で以下を実行:
 * 
 * cl /EHsc cpp_example.cpp /link BloodPressureEstimation.lib
 * 
 * または、Visual Studio プロジェクトで:
 * 1. 新しいC++コンソールプロジェクト作成
 * 2. このファイルを追加
 * 3. BloodPressureEstimation.h をインクルードディレクトリに追加
 * 4. BloodPressureEstimation.lib をライブラリディレクトリに追加
 * 5. プロジェクトプロパティで「追加の依存ファイル」に BloodPressureEstimation.lib を追加
 * 6. プラットフォームを「Win32 (x86)」に設定
 * 7. ビルド実行
 */