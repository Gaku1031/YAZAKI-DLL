using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;

namespace BloodPressureTest
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
    }

    public class TestResults
    {
        public string RequestId { get; set; }
        public int Height { get; set; }
        public int Weight { get; set; }
        public int Sex { get; set; }
        public string VideoFile { get; set; }
        public int SystolicBP { get; set; }
        public int DiastolicBP { get; set; }
        public string Status { get; set; }
        public string ErrorCode { get; set; }
        public string CsvData { get; set; }
        public DateTime StartTime { get; set; }
        public DateTime EndTime { get; set; }
        public bool Success { get; set; }
    }

    class Program
    {
        private static TestResults currentTest = null;
        private static bool analysisCompleted = false;

        static void Main(string[] args)
        {
            Console.WriteLine("=== 血圧推定DLL実動テスト ===");
            Console.WriteLine();

            try
            {
                // 1. 環境確認
                Console.WriteLine("1. 環境確認");
                if (!CheckEnvironment())
                {
                    return;
                }

                // 2. DLL初期化
                Console.WriteLine("\\n2. DLL初期化");
                if (!InitializeDLL())
                {
                    return;
                }

                // 3. サンプル動画でのテスト
                Console.WriteLine("\\n3. サンプル動画による血圧推定テスト");
                RunSampleVideoTest();

                Console.WriteLine("\\n=== テスト完了 ===");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"エラー: {ex.Message}");
                Console.WriteLine($"詳細: {ex}");
            }

            Console.WriteLine("\\nEnterキーで終了...");
            Console.ReadLine();
        }

        static bool CheckEnvironment()
        {
            Console.WriteLine("  環境ファイル確認中...");

            // DLL確認
            if (!File.Exists("BloodPressureEstimation.dll"))
            {
                Console.WriteLine("  ✗ BloodPressureEstimation.dll が見つかりません");
                Console.WriteLine("    build\\dist\\ からDLLをコピーしてください");
                return false;
            }
            Console.WriteLine("  ✓ BloodPressureEstimation.dll 確認");

            // Pythonモジュール確認
            if (!File.Exists("bp_estimation_simple.py"))
            {
                Console.WriteLine("  ✗ bp_estimation_simple.py が見つかりません");
                Console.WriteLine("    Pythonモジュールをコピーしてください");
                return false;
            }
            Console.WriteLine("  ✓ bp_estimation_simple.py 確認");

            // サンプル動画確認
            string sampleVideo = @"sample-data\\100万画素.webm";
            if (!File.Exists(sampleVideo))
            {
                Console.WriteLine($"  ✗ サンプル動画が見つかりません: {sampleVideo}");
                return false;
            }
            Console.WriteLine($"  ✓ サンプル動画確認: {sampleVideo}");

            return true;
        }

        static bool InitializeDLL()
        {
            try
            {
                Console.WriteLine("  DLL初期化中...");
                bool result = BloodPressureDll.InitializeDLL("models");
                
                if (result)
                {
                    Console.WriteLine("  ✓ DLL初期化成功");
                    
                    string version = BloodPressureDll.GetVersionInfo();
                    Console.WriteLine($"  バージョン: {version}");
                    return true;
                }
                else
                {
                    Console.WriteLine("  ✗ DLL初期化失敗");
                    return false;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  ✗ DLL初期化エラー: {ex.Message}");
                return false;
            }
        }

        static void RunSampleVideoTest()
        {
            string sampleVideo = @"sample-data\\100万画素.webm";
            
            // テストケース定義
            var testCases = new[]
            {
                new { Height = 170, Weight = 70, Sex = 1, Name = "男性 170cm 70kg" },
                new { Height = 160, Weight = 55, Sex = 2, Name = "女性 160cm 55kg" },
                new { Height = 175, Weight = 80, Sex = 1, Name = "男性 175cm 80kg" }
            };

            Console.WriteLine($"  使用動画: {sampleVideo}");
            Console.WriteLine($"  動画サイズ: {new FileInfo(sampleVideo).Length / 1024} KB");
            Console.WriteLine();

            foreach (var testCase in testCases)
            {
                Console.WriteLine($"--- {testCase.Name} ---");
                RunSingleTest(sampleVideo, testCase.Height, testCase.Weight, testCase.Sex);
                Console.WriteLine();
            }
        }

        static void RunSingleTest(string videoPath, int height, int weight, int sex)
        {
            try
            {
                // リクエストID生成
                string requestId = GenerateRequestId("9000000001", "0000012345");
                Console.WriteLine($"  リクエストID: {requestId}");

                // テスト結果初期化
                currentTest = new TestResults
                {
                    RequestId = requestId,
                    Height = height,
                    Weight = weight,
                    Sex = sex,
                    VideoFile = videoPath,
                    StartTime = DateTime.Now,
                    Success = false
                };

                analysisCompleted = false;

                // コールバック関数定義
                BloodPressureDll.AnalysisCallback callback = (reqId, sbp, dbp, csvData, errors) =>
                {
                    Console.WriteLine($"  📊 解析完了コールバック:");
                    Console.WriteLine($"    リクエストID: {reqId}");
                    Console.WriteLine($"    最高血圧: {sbp} mmHg");
                    Console.WriteLine($"    最低血圧: {dbp} mmHg");
                    Console.WriteLine($"    CSVデータサイズ: {csvData?.Length ?? 0} 文字");
                    
                    if (!string.IsNullOrEmpty(errors))
                    {
                        Console.WriteLine($"    エラー: {errors}");
                    }

                    // 結果保存
                    if (currentTest != null)
                    {
                        currentTest.SystolicBP = sbp;
                        currentTest.DiastolicBP = dbp;
                        currentTest.CsvData = csvData;
                        currentTest.ErrorCode = errors;
                        currentTest.EndTime = DateTime.Now;
                        currentTest.Success = sbp > 0 && dbp > 0;
                    }

                    analysisCompleted = true;
                };

                // 血圧解析開始
                Console.WriteLine("  🚀 血圧解析開始...");
                string errorCode = BloodPressureDll.StartBloodPressureAnalysisRequest(
                    requestId, height, weight, sex, videoPath, callback);

                if (!string.IsNullOrEmpty(errorCode))
                {
                    Console.WriteLine($"  ✗ 解析開始エラー: {errorCode}");
                    return;
                }

                Console.WriteLine("  ✓ 解析リクエスト送信成功");

                // 処理状況監視
                Console.WriteLine("  ⏳ 解析処理中...");
                MonitorProgress(requestId);

                // 結果表示
                DisplayTestResult();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  ✗ テストエラー: {ex.Message}");
            }
        }

        static void MonitorProgress(string requestId)
        {
            int timeout = 60; // 60秒タイムアウト
            int elapsed = 0;

            while (!analysisCompleted && elapsed < timeout)
            {
                Thread.Sleep(1000);
                elapsed++;

                string status = BloodPressureDll.GetProcessingStatus(requestId);
                Console.Write($"\\r  状況: {status} ({elapsed}s)");

                if (status == "none")
                {
                    break;
                }
            }

            Console.WriteLine();

            if (elapsed >= timeout)
            {
                Console.WriteLine("  ⚠️ タイムアウト - 解析を中断します");
                BloodPressureDll.CancelBloodPressureAnalysis(requestId);
            }
        }

        static void DisplayTestResult()
        {
            if (currentTest == null)
            {
                Console.WriteLine("  ✗ テスト結果なし");
                return;
            }

            Console.WriteLine($"  📋 テスト結果:");
            Console.WriteLine($"    処理時間: {(currentTest.EndTime - currentTest.StartTime).TotalSeconds:F1}秒");
            
            if (currentTest.Success)
            {
                Console.WriteLine($"    ✓ 成功");
                Console.WriteLine($"    最高血圧: {currentTest.SystolicBP} mmHg");
                Console.WriteLine($"    最低血圧: {currentTest.DiastolicBP} mmHg");
                
                // BMI計算
                double bmi = currentTest.Weight / Math.Pow(currentTest.Height / 100.0, 2);
                Console.WriteLine($"    BMI: {bmi:F1}");
                
                // 血圧分類
                string bpCategory = ClassifyBloodPressure(currentTest.SystolicBP, currentTest.DiastolicBP);
                Console.WriteLine($"    血圧分類: {bpCategory}");

                // CSVデータ保存
                if (!string.IsNullOrEmpty(currentTest.CsvData))
                {
                    string csvFileName = $"result_{currentTest.RequestId}.csv";
                    File.WriteAllText(csvFileName, currentTest.CsvData);
                    Console.WriteLine($"    CSVファイル: {csvFileName}");
                }
            }
            else
            {
                Console.WriteLine($"    ✗ 失敗");
                if (!string.IsNullOrEmpty(currentTest.ErrorCode))
                {
                    Console.WriteLine($"    エラーコード: {currentTest.ErrorCode}");
                }
            }
        }

        static string ClassifyBloodPressure(int systolic, int diastolic)
        {
            if (systolic < 120 && diastolic < 80)
                return "正常";
            else if (systolic < 130 && diastolic < 80)
                return "正常高値";
            else if (systolic < 140 || diastolic < 90)
                return "高血圧前症";
            else if (systolic < 160 || diastolic < 100)
                return "高血圧 I度";
            else if (systolic < 180 || diastolic < 110)
                return "高血圧 II度";
            else
                return "高血圧 III度";
        }

        static string GenerateRequestId(string customerCode, string driverCode)
        {
            string timestamp = DateTime.Now.ToString("yyyyMMddHHmmssfff");
            return $"{timestamp}_{customerCode}_{driverCode}";
        }
    }
}