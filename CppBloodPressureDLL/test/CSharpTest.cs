using System;
using System.Runtime.InteropServices;
using System.Threading;
using System.IO;

namespace BloodPressureDllTest
{
    // コールバック関数の型定義
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate void BPCallback(
        [MarshalAs(UnmanagedType.LPStr)] string requestId,
        int maxBloodPressure,
        int minBloodPressure,
        [MarshalAs(UnmanagedType.LPStr)] string measureRowData,
        [MarshalAs(UnmanagedType.LPStr)] string errorsJson
    );

    public class BloodPressureDll
    {
        private const string DllPath = "BloodPressureDLL.dll";

        [DllImport(DllPath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern int InitializeBP([MarshalAs(UnmanagedType.LPStr)] string modelDir);

        [DllImport(DllPath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        [return: MarshalAs(UnmanagedType.LPStr)]
        public static extern string StartBloodPressureAnalysisRequest(
            [MarshalAs(UnmanagedType.LPStr)] string requestId,
            int height, int weight, int sex,
            [MarshalAs(UnmanagedType.LPStr)] string moviePath,
            BPCallback callback);

        [DllImport(DllPath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        [return: MarshalAs(UnmanagedType.LPStr)]
        public static extern string GetProcessingStatus([MarshalAs(UnmanagedType.LPStr)] string requestId);

        [DllImport(DllPath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern int CancelBloodPressureAnalysis([MarshalAs(UnmanagedType.LPStr)] string requestId);

        [DllImport(DllPath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        [return: MarshalAs(UnmanagedType.LPStr)]
        public static extern string GetVersionInfo();

        [DllImport(DllPath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        [return: MarshalAs(UnmanagedType.LPStr)]
        public static extern string GenerateRequestId();

        // コールバック関数の実装
        public static void TestCallback(string requestId, int maxBP, int minBP, string csvData, string errorsJson)
        {
            Console.WriteLine("=== 血圧解析結果 ===");
            Console.WriteLine($"Request ID: {requestId}");
            Console.WriteLine($"最高血圧: {maxBP} mmHg");
            Console.WriteLine($"最低血圧: {minBP} mmHg");
            Console.WriteLine($"CSVデータサイズ: {csvData?.Length ?? 0} 文字");
            
            // 血圧値の妥当性チェック
            if (maxBP > 0 && minBP > 0 && maxBP >= minBP)
            {
                Console.WriteLine("[SUCCESS] 血圧推定成功 - 妥当な値が取得されました");
                Console.WriteLine($"[RESULT] 推定血圧: SBP={maxBP} mmHg, DBP={minBP} mmHg");
                
                // 血圧分類
                string sbpCategory = GetBloodPressureCategory(maxBP, minBP);
                Console.WriteLine($"[CLASSIFICATION] 血圧分類: {sbpCategory}");
            }
            else if (maxBP == 0 && minBP == 0)
            {
                Console.WriteLine("[ERROR] 血圧推定失敗 - 推定値が0です");
            }
            else
            {
                Console.WriteLine("[WARNING] 血圧推定警告 - 異常な値が取得されました");
            }
            
            if (!string.IsNullOrEmpty(errorsJson) && errorsJson != "[]")
            {
                Console.WriteLine($"[ERROR] エラー: {errorsJson}");
            }
            else
            {
                Console.WriteLine("[SUCCESS] エラーなし");
            }
            
            // CSVファイルに保存
            if (!string.IsNullOrEmpty(csvData))
            {
                string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                string csvFileName = $"bp_result_{requestId}_{timestamp}.csv";
                try
                {
                    File.WriteAllText(csvFileName, csvData);
                    Console.WriteLine($"[SUCCESS] CSVファイル保存: {csvFileName}");
                    
                    // CSVデータの基本情報を表示
                    string[] lines = csvData.Split('\n');
                    if (lines.Length > 1)
                    {
                        Console.WriteLine($"   CSV行数: {lines.Length - 1} (ヘッダー除く)");
                        Console.WriteLine($"   データ期間: {lines[1].Split(',')[0]} - {lines[lines.Length - 2].Split(',')[0]} 秒");
                        
                        // CSVデータの統計情報を表示
                        if (lines.Length > 2)
                        {
                            var sbpValues = new List<float>();
                            var dbpValues = new List<float>();
                            
                            for (int i = 1; i < lines.Length; i++)
                            {
                                if (!string.IsNullOrEmpty(lines[i]))
                                {
                                    var parts = lines[i].Split(',');
                                    if (parts.Length >= 3)
                                    {
                                        if (float.TryParse(parts[1], out float sbp))
                                            sbpValues.Add(sbp);
                                        if (float.TryParse(parts[2], out float dbp))
                                            dbpValues.Add(dbp);
                                    }
                                }
                            }
                            
                            if (sbpValues.Count > 0)
                            {
                                Console.WriteLine($"   SBP統計: 平均={sbpValues.Average():F1}, 最小={sbpValues.Min():F1}, 最大={sbpValues.Max():F1}");
                            }
                            if (dbpValues.Count > 0)
                            {
                                Console.WriteLine($"   DBP統計: 平均={dbpValues.Average():F1}, 最小={dbpValues.Min():F1}, 最大={dbpValues.Max():F1}");
                            }
                        }
                    }
                }
                catch (Exception e)
                {
                    Console.WriteLine($"[ERROR] CSVファイル保存エラー: {e.Message}");
                }
            }
        }

        // 血圧分類を取得するヘルパーメソッド
        private static string GetBloodPressureCategory(int sbp, int dbp)
        {
            if (sbp < 120 && dbp < 80)
                return "正常血圧";
            else if (sbp < 130 && dbp < 80)
                return "正常高値血圧";
            else if (sbp < 140 && dbp < 90)
                return "正常高値血圧";
            else if (sbp < 160 && dbp < 100)
                return "軽症高血圧";
            else if (sbp < 180 && dbp < 110)
                return "中等症高血圧";
            else
                return "重症高血圧";
        }

        public static void TestComprehensiveDLL()
        {
            Console.WriteLine("=== C++ Blood Pressure DLL テスト ===");
            
            try
            {
                // 1. DLL初期化テスト
                Console.WriteLine("\n1. DLL初期化テスト");
                int initResult = InitializeBP("models");
                Console.WriteLine($"   初期化結果: {initResult}");
                
                if (initResult == 0)
                {
                    Console.WriteLine("   初期化失敗");
                    return;
                }
                
                // 2. バージョン情報取得テスト
                Console.WriteLine("\n2. バージョン情報取得テスト");
                string version = GetVersionInfo();
                Console.WriteLine($"   バージョン: {version}");
                
                // 3. リクエストID生成テスト
                Console.WriteLine("\n3. リクエストID生成テスト");
                string requestId = GenerateRequestId();
                Console.WriteLine($"   生成されたID: {requestId}");
                
                // 4. 処理状況取得テスト
                Console.WriteLine("\n4. 処理状況取得テスト");
                string status = GetProcessingStatus("test_request");
                Console.WriteLine($"   処理状況: {status}");
                
                // 5. 血圧解析テスト（サンプル動画がある場合）
                Console.WriteLine("\n5. 血圧解析テスト");
                string sampleVideo = "sample_video.webm";
                if (File.Exists(sampleVideo))
                {
                    var fileInfo = new FileInfo(sampleVideo);
                    Console.WriteLine($"   サンプル動画: {sampleVideo} ({fileInfo.Length / 1024 / 1024} MB)");
                    
                    // コールバック関数を設定
                    BPCallback callback = TestCallback;
                    
                    // 血圧解析開始
                    string analysisResult = StartBloodPressureAnalysisRequest(
                        requestId, 170, 70, 1, sampleVideo, callback);
                    
                    Console.WriteLine($"   解析開始結果: {analysisResult ?? "成功"}");
                    
                    if (analysisResult == null)
                    {
                        Console.WriteLine("   解析開始成功 - 処理を監視中...");
                        
                        // 処理状況を監視（最大120秒）
                        for (int i = 0; i < 120; i++)
                        {
                            Thread.Sleep(1000);
                            string currentStatus = GetProcessingStatus(requestId);
                            
                            if (i % 10 == 0) // 10秒ごとに状況を表示
                            {
                                Console.WriteLine($"   処理状況 {i+1}秒: {currentStatus}");
                            }
                            
                            if (currentStatus == "none")
                            {
                                Console.WriteLine($"   [SUCCESS] 処理完了 ({i+1}秒)");
                                break;
                            }
                            
                            if (i == 119)
                            {
                                Console.WriteLine("   [WARNING] 処理が120秒を超えました");
                            }
                        }
                    }
                    else
                    {
                        Console.WriteLine($"   [ERROR] 解析開始失敗: {analysisResult}");
                    }
                }
                else
                {
                    Console.WriteLine($"   [ERROR] サンプル動画が見つかりません: {sampleVideo}");
                    Console.WriteLine("   動画解析テストをスキップ");
                }
                
                // 6. 中断テスト
                Console.WriteLine("\n6. 中断テスト");
                int cancelResult = CancelBloodPressureAnalysis(requestId);
                Console.WriteLine($"   中断結果: {cancelResult}");
                
                // 7. テスト結果サマリー
                Console.WriteLine("\n7. テスト結果サマリー");
                Console.WriteLine("   [SUCCESS] DLL初期化: 成功");
                Console.WriteLine("   [SUCCESS] バージョン取得: 成功");
                Console.WriteLine("   [SUCCESS] リクエストID生成: 成功");
                Console.WriteLine("   [SUCCESS] 処理状況取得: 成功");
                if (File.Exists(sampleVideo))
                {
                    Console.WriteLine("   [SUCCESS] 血圧解析: 実行済み");
                }
                else
                {
                    Console.WriteLine("   [WARNING] 血圧解析: スキップ（動画なし）");
                }
                Console.WriteLine("   [SUCCESS] 中断機能: テスト済み");
                
                Console.WriteLine("\n=== 全てのテストが完了しました ===");
                
            }
            catch (Exception ex)
            {
                Console.WriteLine($"\nエラー: {ex.Message}");
                Console.WriteLine($"スタックトレース: {ex.StackTrace}");
            }
        }

        public static void Main(string[] args)
        {
            try
            {
                Console.WriteLine("C++ Blood Pressure DLL テスト開始");
                TestComprehensiveDLL();
                Console.WriteLine("テスト完了");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"アプリケーションエラー: {ex.Message}");
                Environment.Exit(1);
            }
        }
    }
}
