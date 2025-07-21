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
                        var modelFiles = Directory.GetFiles("models", "*", SearchOption.AllDirectories);
                        foreach (var file in modelFiles)
                        {
                            var fileInfo = new FileInfo(file);
                            var relativePath = file.Replace(Environment.CurrentDirectory, "").TrimStart('\\');
                            Console.WriteLine($"     {relativePath} ({fileInfo.Length / 1024.0:F2} KB)");
                        }
                    }
                    catch (Exception e)
                    {
                        Console.WriteLine($"     [ERROR] Modelsディレクトリ読み取りエラー: {e.Message}");
                    }
                }
                
                // 依存DLLチェック
                Console.WriteLine("   - 依存DLLチェック:");
                string[] requiredDlls = { "opencv_world480.dll", "onnxruntime.dll", "zlib.dll" };
                foreach (var dll in requiredDlls)
                {
                    Console.WriteLine($"     {dll}: {File.Exists(dll)}");
                }
                
                Console.WriteLine("   DLL初期化を試行中...");
                
                // 初期化前ファイル確認
                Console.WriteLine("   - 初期化前ファイル確認:");
                string[] requiredFiles = {
                    "models/systolicbloodpressure.onnx",
                    "models/diastolicbloodpressure.onnx",
                    "models/opencv_face_detector_uint8.pb",
                    "models/opencv_face_detector.pbtxt"
                };
                
                foreach (var file in requiredFiles)
                {
                    if (File.Exists(file))
                    {
                        var fileInfo = new FileInfo(file);
                        Console.WriteLine($"     {file}: EXISTS ({fileInfo.Length / 1024.0:F2} KB)");
                    }
                    else
                    {
                        Console.WriteLine($"     {file}: NOT FOUND");
                    }
                }
                
                int initResult = InitializeBP("models");
                Console.WriteLine($"   初期化結果: {initResult}");
                
                if (initResult == 1)
                {
                    Console.WriteLine("   [SUCCESS] DLL初期化成功");
                }
                else
                {
                    Console.WriteLine("   [ERROR] DLL初期化失敗");
                    return;
                }
                
                // 2. バージョン情報取得テスト
                Console.WriteLine("\n2. バージョン情報取得テスト");
                try
                {
                    Console.WriteLine("   GetVersionInfo関数を呼び出し中...");
                    string version = GetVersionInfo();
                    Console.WriteLine("   GetVersionInfo関数呼び出し完了");
                    
                    if (string.IsNullOrEmpty(version))
                    {
                        Console.WriteLine("   [ERROR] バージョン情報が空です");
                        return;
                    }
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
                catch (AccessViolationException ex)
                {
                    Console.WriteLine($"   [ERROR] メモリアクセス違反: {ex.Message}");
                    Console.WriteLine("   これは文字列のライフタイム問題の可能性があります");
                    Console.WriteLine($"   スタックトレース: {ex.StackTrace}");
                    return;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"   [ERROR] バージョン情報取得で予期しないエラー: {ex.Message}");
                    Console.WriteLine($"   例外の種類: {ex.GetType().Name}");
                    Console.WriteLine($"   スタックトレース: {ex.StackTrace}");
                    return;
                }
                
                // 3. リクエストID生成テスト
                Console.WriteLine("\n3. リクエストID生成テスト");
                try
                {
                    Console.WriteLine("   GenerateRequestId関数を呼び出し中...");
                    string requestId = GenerateRequestId();
                    Console.WriteLine("   GenerateRequestId関数呼び出し完了");
                    
                    if (string.IsNullOrEmpty(requestId))
                    {
                        Console.WriteLine("   [ERROR] リクエストIDが空です");
                        return;
                    }
                    Console.WriteLine($"   生成されたID: {requestId}");
                    Console.WriteLine("   [SUCCESS] リクエストID生成成功");
                }
                catch (AccessViolationException ex)
                {
                    Console.WriteLine($"   [ERROR] メモリアクセス違反: {ex.Message}");
                    Console.WriteLine("   これは文字列のライフタイム問題の可能性があります");
                    Console.WriteLine($"   スタックトレース: {ex.StackTrace}");
                    return;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"   [ERROR] リクエストID生成で予期しないエラー: {ex.Message}");
                    Console.WriteLine($"   例外の種類: {ex.GetType().Name}");
                    Console.WriteLine($"   スタックトレース: {ex.StackTrace}");
                    return;
                }
                
                // 4. 処理状況取得テスト
                Console.WriteLine("\n4. 処理状況取得テスト");
                try
                {
                    Console.WriteLine("   処理状況取得を試行中...");
                    Console.WriteLine("   GetProcessingStatus関数を呼び出し中...");
                    
                    string status = GetProcessingStatus("test_request");
                    Console.WriteLine("   GetProcessingStatus関数呼び出し完了");
                    
                    if (string.IsNullOrEmpty(status))
                    {
                        Console.WriteLine("   [ERROR] 処理状況が空です");
                        return;
                    }
                    
                    Console.WriteLine($"   処理状況: {status}");
                    Console.WriteLine("   [SUCCESS] 処理状況取得成功");
                }
                catch (AccessViolationException ex)
                {
                    Console.WriteLine($"   [ERROR] メモリアクセス違反: {ex.Message}");
                    Console.WriteLine("   これは文字列のライフタイム問題の可能性があります");
                    Console.WriteLine($"   スタックトレース: {ex.StackTrace}");
                    return;
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
                    Console.WriteLine($"   [ERROR] 処理状況取得で予期しないエラー: {ex.Message}");
                    Console.WriteLine($"   例外の種類: {ex.GetType().Name}");
                    Console.WriteLine($"   スタックトレース: {ex.StackTrace}");
                    return;
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

