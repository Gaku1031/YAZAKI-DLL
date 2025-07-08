/**
 * @file BloodPressureEstimation.h
 * @brief 血圧推定DLL - C/C++インターフェース定義
 * @version 1.0.0
 * @date 2025-01-06
 * @author IKI Japan/Yazaki
 */

#ifndef BLOOD_PRESSURE_ESTIMATION_H
#define BLOOD_PRESSURE_ESTIMATION_H

#ifdef __cplusplus
extern "C" {
#endif

#include <windows.h>

// =============================================================================
// 型定義
// =============================================================================

/**
 * @brief エラー情報構造体
 */
typedef struct {
    const char* code;        ///< エラーコード
    const char* message;     ///< エラーメッセージ
    BOOL is_retriable;       ///< 再試行可能フラグ
} BPErrorInfo;

/**
 * @brief 血圧解析結果コールバック関数型
 * @param requestId リクエストID
 * @param maxBloodPressure 最高血圧 (mmHg)
 * @param minBloodPressure 最低血圧 (mmHg)
 * @param measureRowData PPGローデータ (CSV形式)
 * @param errors エラー情報配列 (NULL=エラーなし)
 */
typedef void (*BPAnalysisCallback)(
    const char* requestId,
    int maxBloodPressure,
    int minBloodPressure,
    const char* measureRowData,
    const BPErrorInfo* errors
);

// =============================================================================
// エラーコード定数
// =============================================================================

#define BP_ERROR_DLL_NOT_INITIALIZED     "1001"  ///< DLL未初期化
#define BP_ERROR_DEVICE_CONNECTION_FAILED "1002"  ///< デバイス接続失敗
#define BP_ERROR_CALIBRATION_INCOMPLETE   "1003"  ///< キャリブレーション未完了
#define BP_ERROR_INVALID_PARAMETERS       "1004"  ///< 入力パラメータ不正
#define BP_ERROR_REQUEST_DURING_PROCESSING "1005" ///< 測定中リクエスト不可
#define BP_ERROR_INTERNAL_PROCESSING      "1006"  ///< DLL内部処理エラー

// =============================================================================
// 処理状態定数
// =============================================================================

#define BP_STATUS_NONE         "none"       ///< 未受付
#define BP_STATUS_PROCESSING   "processing" ///< 処理中

// =============================================================================
// 性別定数
// =============================================================================

#define BP_SEX_MALE    1  ///< 男性
#define BP_SEX_FEMALE  2  ///< 女性

// =============================================================================
// DLL関数宣言
// =============================================================================

/**
 * @brief DLL初期化
 * @param model_dir モデルディレクトリパス (NULL=デフォルト:"models")
 * @return 成功=TRUE, 失敗=FALSE
 * @note DLL使用前に必ず呼び出すこと
 * 
 * @example
 * @code
 * if (InitializeDLL("models")) {
 *     printf("DLL初期化成功\n");
 * } else {
 *     printf("DLL初期化失敗\n");
 * }
 * @endcode
 */
__declspec(dllexport) BOOL __stdcall InitializeDLL(const char* model_dir);

/**
 * @brief 血圧解析開始
 * @param requestId リクエストID (形式: "${yyyyMMddHHmmssfff}_${顧客コード}_${乗務員コード}")
 * @param height 身長 (cm, 整数)
 * @param weight 体重 (kg, 整数)
 * @param sex 性別 (BP_SEX_MALE=1, BP_SEX_FEMALE=2)
 * @param measurementMoviePath WebM動画ファイルの絶対パス
 * @param callback 結果通知用コールバック関数 (NULL可)
 * @return エラーコード文字列 (NULL=成功, エラー発生時はエラーコード)
 * @note 非同期処理。結果はコールバックで通知される
 * 
 * @example
 * @code
 * void OnBPResult(const char* req_id, int sbp, int dbp, const char* csv, const BPErrorInfo* errors) {
 *     printf("血圧結果: %s - SBP:%d, DBP:%d\n", req_id, sbp, dbp);
 * }
 * 
 * const char* error_code = StartBloodPressureAnalysis(
 *     "20250707083524932_9000000001_0000012345",
 *     170, 70, BP_SEX_MALE,
 *     "C:\\Videos\\measurement.webm",
 *     OnBPResult
 * );
 * @endcode
 */
__declspec(dllexport) const char* __stdcall StartBloodPressureAnalysis(
    const char* requestId,
    int height,
    int weight,
    int sex,
    const char* measurementMoviePath,
    BPAnalysisCallback callback
);

/**
 * @brief 血圧解析処理中断
 * @param requestId 中断対象のリクエストID
 * @return 成功=TRUE, 失敗=FALSE
 * @note 処理中の解析を強制中断する
 * 
 * @example
 * @code
 * if (CancelBloodPressureProcessing("20250707083524932_9000000001_0000012345")) {
 *     printf("処理中断成功\n");
 * }
 * @endcode
 */
__declspec(dllexport) BOOL __stdcall CancelBloodPressureProcessing(const char* requestId);

/**
 * @brief 血圧解析処理状況取得
 * @param requestId 状況確認対象のリクエストID
 * @return 処理状況文字列 (BP_STATUS_NONE | BP_STATUS_PROCESSING)
 * @note 処理完了はコールバックで通知されるため、状況には含まれない
 * 
 * @example
 * @code
 * const char* status = GetBloodPressureStatus("20250707083524932_9000000001_0000012345");
 * if (strcmp(status, BP_STATUS_PROCESSING) == 0) {
 *     printf("処理中...\n");
 * }
 * @endcode
 */
__declspec(dllexport) const char* __stdcall GetBloodPressureStatus(const char* requestId);

/**
 * @brief DLLバージョン情報取得
 * @return バージョン文字列
 * 
 * @example
 * @code
 * printf("DLLバージョン: %s\n", GetDLLVersion());
 * @endcode
 */
__declspec(dllexport) const char* __stdcall GetDLLVersion(void);

// =============================================================================
// ユーティリティ関数
// =============================================================================

/**
 * @brief リクエストID生成ヘルパー
 * @param customerCode 顧客コード
 * @param driverCode 乗務員コード
 * @param buffer 生成されたリクエストIDを格納するバッファ (最低48文字)
 * @return 生成されたリクエストID文字列
 * @note バッファサイズは48文字以上確保すること
 * 
 * @example
 * @code
 * char request_id[64];
 * GenerateRequestID("9000000001", "0000012345", request_id);
 * printf("生成されたID: %s\n", request_id);
 * @endcode
 */
__declspec(dllexport) const char* __stdcall GenerateRequestID(const char* customerCode, const char* driverCode, char* buffer);

/**
 * @brief 動画ファイル検証
 * @param moviePath 動画ファイルパス
 * @return 有効=TRUE, 無効=FALSE
 * @note WebM形式、30秒、30fps、1280x720の条件をチェック
 * 
 * @example
 * @code
 * if (ValidateMovieFile("C:\\Videos\\test.webm")) {
 *     printf("動画ファイル有効\n");
 * }
 * @endcode
 */
__declspec(dllexport) BOOL __stdcall ValidateMovieFile(const char* moviePath);

#ifdef __cplusplus
}
#endif

#endif // BLOOD_PRESSURE_ESTIMATION_H

/**
 * @mainpage 血圧推定DLL API リファレンス
 * 
 * @section intro_sec 概要
 * 
 * 血圧推定DLLは、30秒のWebM動画ファイルからrPPG（remote PhotoPlethysmoGraphy）アルゴリズムを
 * 使用して心拍変動（RRI）を計測し、機械学習モデルにより血圧を推定するライブラリです。
 * 
 * @section features_sec 主な機能
 * 
 * - MediaPipeによる高精度顔検出
 * - POS（Plane Orthogonal to Skin）アルゴリズムによるrPPG信号抽出
 * - Random Forestモデルによる血圧推定
 * - 非同期処理とコールバック通知
 * - 32bit Windows DLL形式
 * 
 * @section usage_sec 使用方法
 * 
 * 1. InitializeDLL() でDLLを初期化
 * 2. StartBloodPressureAnalysis() で解析開始
 * 3. コールバック関数で結果を受信
 * 4. 必要に応じてGetBloodPressureStatus()で状況確認
 * 
 * @section requirements_sec 動作要件
 * 
 * - Windows 10/11 (32bit)
 * - モデルファイル: model_sbp.pkl, model_dbp.pkl
 * - 入力動画: WebM形式、30秒、30fps、1280x720
 * 
 * @section example_sec サンプルコード
 * 
 * 詳細なサンプルコードについては、examples/ ディレクトリを参照してください。
 */