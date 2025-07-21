using System;
using System.Runtime.InteropServices;
using System.Threading;
using System.IO;
using System.Collections.Generic;
using System.Linq;

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
                
                // 事前チェック
                Console.WriteLine("   事前チェック:");
                Console.WriteLine($"   - カレントディレクトリ: {Environment.CurrentDirectory}");
                Console.WriteLine($"   - DLLファイル存在: {File.Exists("BloodPressureDLL.dll")}");
                Console.WriteLine($"   - Modelsディレクトリ存在: {Directory.Exists("models")}");
                
                if (Directory.Exists("models"))
                {
                    Console.WriteLine("   - Modelsディレクトリ内容:");
                    try
                    {
                        foreach (var file in Directory.GetFiles("models", "*", SearchOption.AllDirectories))
                        {
                            try
                            {
                                var fileInfo = new FileInfo(file);
                                var sizeKB = fileInfo.Length / 1024.0;
                                Console.WriteLine($"     {file.Replace(Environment.CurrentDirectory + "\\", "")} ({sizeKB:F2} KB)");
                                
                                // Check if file is actually readable
                                if (fileInfo.Length == 0)
                                {
                                    Console.WriteLine($"       WARNING: File appears to be empty (0 bytes)");
                                    // Try to read a small portion to verify
                                    try
                                    {
                                        using (var stream = File.OpenRead(file))
                                        {
                                            var buffer = new byte[10];
                                            var bytesRead = stream.Read(buffer, 0, buffer.Length);
                                            Console.WriteLine($"       File read test: {bytesRead} bytes read successfully");
                                        }
                                    }
                                    catch (Exception readEx)
                                    {
                                        Console.WriteLine($"       File read test failed: {readEx.Message}");
                                    }
                                }
                            }
                            catch (Exception fileEx)
                            {
                                Console.WriteLine($"     {file.Replace(Environment.CurrentDirectory + "\\", "")} - ERROR: {fileEx.Message}");
                            }
                        }
                    }
                    catch (Exception dirEx)
                    {
                        Console.WriteLine($"   ERROR accessing models directory: {dirEx.Message}");
                    }
                }
                
                // 依存DLLの存在チェック
                var requiredDlls = new[] { "opencv_world480.dll", "onnxruntime.dll", "zlib.dll" };
                Console.WriteLine("   - 依存DLLチェック:");
                foreach (var dll in requiredDlls)
                {
                    Console.WriteLine($"     {dll}: {File.Exists(dll)}");
                }
                
                // DLL初期化を試行
                Console.WriteLine("   DLL初期化を試行中...");
                
                // Pre-initialization file check
                Console.WriteLine("   - 初期化前ファイル確認:");
                var requiredFiles = new[] { 
                    "models/systolicbloodpressure.onnx", 
                    "models/diastolicbloodpressure.onnx",
                    "models/opencv_face_detector_uint8.pb",
                    "models/opencv_face_detector.pbtxt"
                };
                
                bool allFilesExist = true;
                foreach (var file in requiredFiles)
                {
                    if (File.Exists(file))
                    {
                        var fileInfo = new FileInfo(file);
                        var sizeKB = fileInfo.Length / 1024.0;
                        Console.WriteLine($"     {file}: EXISTS ({sizeKB:F2} KB)");
                        
                        if (fileInfo.Length == 0)
                        {
                            Console.WriteLine($"       WARNING: File is empty (0 bytes)");
                            allFilesExist = false;
                        }
                    }
                    else
                    {
                        Console.WriteLine($"     {file}: NOT FOUND");
                        allFilesExist = false;
                    }
                }
                
                if (!allFilesExist)
                {
                    Console.WriteLine("   ERROR: Required model files are missing or empty");
                    Console.WriteLine("   Cannot proceed with DLL initialization");
                    return;
                }
                
                try
                {
                    int initResult = InitializeBP("models");
                    Console.WriteLine($"   初期化結果: {initResult}");
                    
                    if (initResult == 0)
                    {
                        Console.WriteLine("   初期化失敗 - 戻り値が0");
                        return;
                    }
                    
                    Console.WriteLine("   [SUCCESS] DLL初期化成功");
                }
                catch (DllNotFoundException ex)
                {
                    Console.WriteLine($"   [ERROR] DLLが見つかりません: {ex.Message}");
                    Console.WriteLine($"   詳細: {ex}");
                    return;
                }
                catch (BadImageFormatException ex)
                {
                    Console.WriteLine($"   [ERROR] DLLの形式が不正です: {ex.Message}");
                    Console.WriteLine($"   詳細: {ex}");
                    return;
                }
                catch (EntryPointNotFoundException ex)
                {
                    Console.WriteLine($"   [ERROR] DLLの関数が見つかりません: {ex.Message}");
                    Console.WriteLine($"   詳細: {ex}");
                    return;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"   [ERROR] DLL初期化で予期しないエラー: {ex.Message}");
                    Console.WriteLine($"   例外の種類: {ex.GetType().Name}");
                    Console.WriteLine($"   詳細: {ex}");
                    if (ex.InnerException != null)
                    {
                        Console.WriteLine($"   内部例外: {ex.InnerException.Message}");
                    }
                    return;
                }
                
                // 2. バージョン情報取得テスト
                Console.WriteLine("\n2. バージョン情報取得テスト");
                try
                {
                    string version = GetVersionInfo();
                    Console.WriteLine($"   バージョン: {version}");
                    Console.WriteLine("   [SUCCESS] バージョン情報取得成功");
                }
                catch (DllNotFoundException ex)
                {
                    Console.WriteLine($"   [ERROR] DLLが見つかりません: {ex.Message}");
                    return;
                }
                catch (EntryPointNotFoundException ex)
                {
                    Console.WriteLine($"   [ERROR] 関数が見つかりません: {ex.Message}");
                    return;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"   [ERROR] バージョン情報取得で予期しないエラー: {ex.Message}");
                    Console.WriteLine($"   例外の種類: {ex.GetType().Name}");
                    Console.WriteLine($"   詳細: {ex}");
                    return;
                }
                
                // 3. リクエストID生成テスト
                Console.WriteLine("\n3. リクエストID生成テスト");
                try
                {
                    string requestId = GenerateRequestId();
                    Console.WriteLine($"   生成されたID: {requestId}");
                    Console.WriteLine("   [SUCCESS] リクエストID生成成功");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"   [ERROR] リクエストID生成で予期しないエラー: {ex.Message}");
                    Console.WriteLine($"   例外の種類: {ex.GetType().Name}");
                    return;
                }
                
                // 4. 処理状況取得テスト
                Console.WriteLine("\n4. 処理状況取得テスト");
                try
                {
                    string status = GetProcessingStatus("test_request");
                    Console.WriteLine($"   処理状況: {status}");
                    Console.WriteLine("   [SUCCESS] 処理状況取得成功");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"   [ERROR] 処理状況取得で予期しないエラー: {ex.Message}");
                    Console.WriteLine($"   例外の種類: {ex.GetType().Name}");
                    return;
                }
                
                // 5. 血圧解析テスト（サンプル動画がある場合）
                Console.WriteLine("\n5. 血圧解析テスト");
                try
                {
                    string sampleVideo = "sample_video.webm";
                    if (File.Exists(sampleVideo))
                    {
                        var fileInfo = new FileInfo(sampleVideo);
                        Console.WriteLine($"   サンプル動画: {sampleVideo} ({fileInfo.Length / 1024 / 1024} MB)");
                        
                        // リクエストIDを再生成
                        string requestId = GenerateRequestId();
                        Console.WriteLine($"   リクエストID: {requestId}");
                        
                        // 血圧解析を開始
                        Console.WriteLine("   血圧解析を開始中...");
                        string result = StartBloodPressureAnalysisRequest(
                            requestId, 170, 70, 1, sampleVideo, TestCallback);
                        
                        if (result == "1000") // SUCCESS
                        {
                            Console.WriteLine("   [SUCCESS] 血圧解析リクエスト成功");
                            
                            // 処理完了まで待機
                            Console.WriteLine("   処理完了まで待機中...");
                            Thread.Sleep(5000); // 5秒待機
                            
                            string finalStatus = GetProcessingStatus(requestId);
                            Console.WriteLine($"   最終処理状況: {finalStatus}");
                        }
                        else
                        {
                            Console.WriteLine($"   [ERROR] 血圧解析リクエスト失敗: {result}");
                        }
                    }
                    else
                    {
                        Console.WriteLine("   サンプル動画が見つかりません。スキップします。");
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"   [ERROR] 血圧解析テストで予期しないエラー: {ex.Message}");
                    Console.WriteLine($"   例外の種類: {ex.GetType().Name}");
                }
                
                Console.WriteLine("\n[SUCCESS] すべてのテストが完了しました");
                Console.WriteLine("=== C++ Blood Pressure DLL テスト完了 ===");
                
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
