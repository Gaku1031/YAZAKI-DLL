using System;
using System.Runtime.InteropServices;
using System.Threading;

namespace BloodPressureDllTest
{
    public class BloodPressureDll
    {
        private const string DllPath = "BloodPressureEstimation.dll";

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void AnalysisCallback(
            [MarshalAs(UnmanagedType.LPStr)] string requestId,
            int maxBloodPressure,
            int minBloodPressure,
            [MarshalAs(UnmanagedType.LPStr)] string measureRowData,
            [MarshalAs(UnmanagedType.LPStr)] string errors
        );

        [DllImport(DllPath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern bool InitializeDLL([MarshalAs(UnmanagedType.LPStr)] string modelDir);

        [DllImport(DllPath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        [return: MarshalAs(UnmanagedType.LPStr)]
        public static extern string StartBloodPressureAnalysisRequest(
            [MarshalAs(UnmanagedType.LPStr)] string requestId,
            int height, int weight, int sex,
            [MarshalAs(UnmanagedType.LPStr)] string moviePath,
            AnalysisCallback callback);

        [DllImport(DllPath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        [return: MarshalAs(UnmanagedType.LPStr)]
        public static extern string GetProcessingStatus([MarshalAs(UnmanagedType.LPStr)] string requestId);

        [DllImport(DllPath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern bool CancelBloodPressureAnalysis([MarshalAs(UnmanagedType.LPStr)] string requestId);

        [DllImport(DllPath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        [return: MarshalAs(UnmanagedType.LPStr)]
        public static extern string GetVersionInfo();

        public static void TestBalancedDLL()
        {
            Console.WriteLine("=== Blood Pressure DLL Integration Test ===");

            try
            {
                // 1. DLL初期化
                Console.WriteLine("1. DLL initialization");
                bool initResult = InitializeDLL("models");
                Console.WriteLine($"    Result: {initResult}");

                if (!initResult)
                {
                    Console.WriteLine("DLL initialization failed");
                    return;
                }

                // 2. バージョン情報取得
                Console.WriteLine("2. Version information");
                string version = GetVersionInfo();
                Console.WriteLine($"    Version: {version}");

                // 3. 処理状況取得
                Console.WriteLine("3. Processing status check");
                string status = GetProcessingStatus("test_request");
                Console.WriteLine($"    Status: {status}");

                // 4. 無効パラメータでのテスト
                Console.WriteLine("4. Invalid parameters test");
                AnalysisCallback callback = (reqId, sbp, dbp, csvData, errors) =>
                {
                    Console.WriteLine($"    Callback: {reqId}, SBP={sbp}, DBP={dbp}");
                };

                string errorCode = StartBloodPressureAnalysisRequest(
                    "invalid_id", 170, 70, 1, "test.webm", callback);
                Console.WriteLine($"    Error code: {errorCode}");

                // 5. 有効な解析リクエスト
                Console.WriteLine("5. Valid analysis request");
                string requestId = $"{DateTime.Now:yyyyMMddHHmmssfff}_9000000001_0000012345";
                
                // ダミーファイル作成
                System.IO.File.WriteAllText("test_video.webm", "dummy video data");
                
                errorCode = StartBloodPressureAnalysisRequest(
                    requestId, 170, 70, 1, "test_video.webm", callback);
                
                if (string.IsNullOrEmpty(errorCode))
                {
                    Console.WriteLine("    Analysis start successful");
                    
                    // 状況監視
                    Thread.Sleep(1000);
                    status = GetProcessingStatus(requestId);
                    Console.WriteLine($"    Processing status: {status}");
                    
                    // 完了待機
                    Thread.Sleep(3000);
                    status = GetProcessingStatus(requestId);
                    Console.WriteLine($"    Final status: {status}");
                }
                else
                {
                    Console.WriteLine($"    Error code: {errorCode}");
                }

                Console.WriteLine("=== Test completed successfully ===");
            }
            catch (DllNotFoundException ex)
            {
                Console.WriteLine($"DLL not found: {ex.Message}");
                throw;
            }
            catch (EntryPointNotFoundException ex)
            {
                Console.WriteLine($"Entry point not found: {ex.Message}");
                throw;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
                throw;
            }
        }

        public static void TestProductionLikeAnalysis()
        {
            Console.WriteLine("=== 本番同等・血圧推定自動テスト ===");
            string sampleMovie = "100万画素.webm";
            string requestId = $"{DateTime.Now:yyyyMMddHHmmssfff}_9000000001_0000012345";
            int height = 170;
            int weight = 70;
            int sex = 1; // 男性
            bool callbackCalled = false;
            int resultSBP = 0, resultDBP = 0;
            string resultCSV = null;
            string resultError = null;

            AnalysisCallback callback = (reqId, sbp, dbp, csvData, errors) =>
            {
                callbackCalled = true;
                resultSBP = sbp;
                resultDBP = dbp;
                resultCSV = csvData;
                resultError = errors;
                Console.WriteLine($"[Callback] RequestID: {reqId}, SBP: {sbp}, DBP: {dbp}");
                if (!string.IsNullOrEmpty(errors))
                {
                    Console.WriteLine($"[Callback] Error: {errors}");
                }
                if (!string.IsNullOrEmpty(csvData))
                {
                    Console.WriteLine($"[Callback] CSVデータ長: {csvData.Length}");
                }
            };

            // DLL初期化
            Console.WriteLine("1. DLL初期化");
            bool initResult = InitializeDLL("models");
            Console.WriteLine($"    Result: {initResult}");
            if (!initResult)
            {
                Console.WriteLine("DLL初期化失敗");
                return;
            }

            // サンプル動画存在確認
            if (!System.IO.File.Exists(sampleMovie))
            {
                Console.WriteLine($"サンプル動画が見つかりません: {sampleMovie}");
                return;
            }

            // 血圧推定リクエスト
            Console.WriteLine("2. 血圧推定リクエスト送信");
            string errorCode = StartBloodPressureAnalysisRequest(
                requestId, height, weight, sex, sampleMovie, callback);
            if (!string.IsNullOrEmpty(errorCode))
            {
                Console.WriteLine($"    エラーコード: {errorCode}");
                return;
            }
            Console.WriteLine("    血圧推定開始 成功");

            // ステータス監視・完了待ち
            Console.WriteLine("3. ステータス監視");
            int maxWaitSec = 120; // 最大2分待つ
            int waited = 0;
            while (waited < maxWaitSec)
            {
                string status = GetProcessingStatus(requestId);
                Console.WriteLine($"    Status: {status}");
                if (status == "none") break;
                Thread.Sleep(2000);
                waited += 2;
            }
            if (!callbackCalled)
            {
                Console.WriteLine("    [ERROR] コールバックが呼ばれませんでした");
            }
            else
            {
                Console.WriteLine($"4. 血圧推定結果: SBP={resultSBP}, DBP={resultDBP}");
                if (!string.IsNullOrEmpty(resultCSV))
                {
                    Console.WriteLine($"   CSVデータ長: {resultCSV.Length}");
                }
                if (!string.IsNullOrEmpty(resultError))
                {
                    Console.WriteLine($"   エラー: {resultError}");
                }
            }
            Console.WriteLine("=== 本番同等テスト終了 ===");
        }

        public static void Main(string[] args)
        {
            TestBalancedDLL();
            Console.WriteLine("\nPress Enter to run 本番同等テスト...");
            Console.ReadLine();
            TestProductionLikeAnalysis();
            Console.WriteLine("\nPress Enter to exit...");
            Console.ReadLine();
        }
    }
}
