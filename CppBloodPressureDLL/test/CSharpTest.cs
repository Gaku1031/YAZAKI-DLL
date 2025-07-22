using System;
using System.Runtime.InteropServices;
using System.Threading;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text; // StringBuilderを追加

namespace BloodPressureDllTest
{
    // コールバック関数の型定義（C++ typedef void(*BPCallback)(const char* requestId, int maxBloodPressure, int minBloodPressure, const char* measureRowData, const char* errorsJson); に完全一致）
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate void BPCallback(
        [MarshalAs(UnmanagedType.LPStr)] string requestId,
        int sbp, int dbp,
        [MarshalAs(UnmanagedType.LPStr)] string csv,
        [MarshalAs(UnmanagedType.LPStr)] string errorsJson);

    public class BloodPressureDll
    {
        private const string DllPath = "BloodPressureDLL.dll";

        [DllImport(DllPath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern int InitializeBP([Out] StringBuilder outBuf, int bufSize, [MarshalAs(UnmanagedType.LPStr)] string modelDir);

        [DllImport(DllPath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern int StartBloodPressureAnalysisRequest(
            [MarshalAs(UnmanagedType.LPStr)] string requestId,
            int height, int weight, int sex,
            [MarshalAs(UnmanagedType.LPStr)] string moviePath,
            BPCallback callback);

        [DllImport(DllPath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern int GetProcessingStatus([Out] StringBuilder outBuf, int bufSize, [MarshalAs(UnmanagedType.LPStr)] string requestId);

        [DllImport(DllPath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern int CancelBloodPressureAnalysis([Out] StringBuilder outBuf, int bufSize, [MarshalAs(UnmanagedType.LPStr)] string requestId);

        [DllImport(DllPath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern int GetVersionInfo([Out] StringBuilder outBuf, int bufSize);

        [DllImport(DllPath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern int GenerateRequestId([Out] StringBuilder outBuf, int bufSize);

        [DllImport(DllPath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern int AnalyzeBloodPressureFromImages([Out] StringBuilder outBuf, int bufSize,
            [In] string[] imagePaths, int numImages, int height, int weight, int sex, BPCallback callback);

        [DllImport(DllPath, CallingConvention = CallingConvention.Cdecl)]
        public static extern int EstimateBloodPressure(
            [In] double[] peak_times, int peak_count,
            int height, int weight, int sex,
            out int sbp, out int dbp);

        // IntPtr→string変換時はNULLチェック
        public static string PtrToStringSafe(IntPtr ptr)
        {
            if (ptr == IntPtr.Zero) return null;
            return Marshal.PtrToStringAnsi(ptr);
        }

        // コールバック関数の実装
        public static void TestCallback(IntPtr requestIdPtr, int maxBP, int minBP, IntPtr csvDataPtr, IntPtr errorsJsonPtr)
        {
            string requestId = Marshal.PtrToStringAnsi(requestIdPtr);
            string csvData = Marshal.PtrToStringAnsi(csvDataPtr);
            string errorsJson = Marshal.PtrToStringAnsi(errorsJsonPtr);
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
                
                // Critical: Add safety wrapper for DLL initialization
                Console.WriteLine("   Step 1: Attempting to call InitializeBP");
                Console.Out.Flush();
                
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
                
                Console.WriteLine("   Step 2: Calling InitializeBP function");
                Console.Out.Flush();
                
                int initResult;
                try 
                {
                    var sb = new StringBuilder(256);
                    initResult = InitializeBP(sb, sb.Capacity, "models");
                    Console.WriteLine("   Step 3: InitializeBP call completed successfully");
                    Console.Out.Flush();
                }
                catch (AccessViolationException ex)
                {
                    Console.WriteLine($"   [FATAL] AccessViolationException during InitializeBP: {ex.Message}");
                    Console.WriteLine($"   This indicates memory corruption in the DLL initialization");
                    Console.WriteLine($"   Stack trace: {ex.StackTrace}");
                    return;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"   [ERROR] Exception during InitializeBP: {ex.GetType().Name}: {ex.Message}");
                    Console.WriteLine($"   Stack trace: {ex.StackTrace}");
                    return;
                }
                
                Console.WriteLine($"   初期化結果: {initResult}");
                if (initResult == 1)
                {
                    Console.WriteLine("   [SUCCESS] DLL初期化成功");
                }
                else
                {
                    Console.WriteLine("   [ERROR] DLL初期化失敗");
                    if (File.Exists("dll_error.log"))
                    {
                        Console.WriteLine("--- dll_error.log ---");
                        Console.WriteLine(File.ReadAllText("dll_error.log"));
                        Console.WriteLine("--- end of dll_error.log ---");
                    }
                    if (File.Exists("dll_load.log"))
                    {
                        Console.WriteLine("--- dll_load.log ---");
                        Console.WriteLine(File.ReadAllText("dll_load.log"));
                        Console.WriteLine("--- end of dll_load.log ---");
                    }
                    return;
                }
                
                // 2. バージョン情報取得テスト
                Console.WriteLine("\n2. バージョン情報取得テスト");
                try
                {
                    Console.WriteLine("   GetVersionInfo関数を呼び出し中...");
                    var sb = new StringBuilder(256);
                    int result = GetVersionInfo(sb, sb.Capacity);
                    string version = sb.ToString();
                    Console.WriteLine("   GetVersionInfo関数呼び出し完了");
                    if (result != 0)
                    {
                        Console.WriteLine($"   [ERROR] DLLからエラー返却: {version}");
                        return;
                    }
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
                    var sb = new StringBuilder(256);
                    int result = GenerateRequestId(sb, sb.Capacity);
                    string requestId = sb.ToString();
                    Console.WriteLine("   GenerateRequestId関数呼び出し完了");
                    
                    if (string.IsNullOrEmpty(requestId))
                    {
                        Console.WriteLine("   [ERROR] リクエストIDが空です");
                        return;
                    }
                    Console.WriteLine($"   生成されたID: {requestId}");
                    if (requestId.Contains("failed"))
                    {
                        Console.WriteLine($"   [ERROR] DLLからエラー返却: {requestId}");
                        return;
                    }
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
                    Console.WriteLine("   引数: test_request");
                    
                    // 段階的なテスト
                    Console.WriteLine("   Step 1: 関数呼び出し前の状態確認");
                    Console.WriteLine("   Step 2: GetProcessingStatus関数呼び出し開始");
                    
                    // より詳細なデバッグ情報
                    Console.WriteLine("   Step 2.1: 関数ポインタの確認");
                    Console.WriteLine("   Step 2.2: 引数の準備");
                    string testArg = "test_request";
                    Console.WriteLine($"   Step 2.3: 引数値: {testArg}");
                    
                    var sb = new StringBuilder(256);
                    int result = GetProcessingStatus(sb, sb.Capacity, testArg);
                    string status = sb.ToString();
                    
                    Console.WriteLine("   Step 3: GetProcessingStatus関数呼び出し完了");
                    Console.WriteLine("   Step 4: 戻り値の確認");
                    
                    if (result != 0)
                    {
                        Console.WriteLine($"   [ERROR] DLLからエラー返却: {status}");
                        return;
                    }
                    if (string.IsNullOrEmpty(status))
                    {
                        Console.WriteLine("   [ERROR] 処理状況が空です");
                        return;
                    }
                    
                    Console.WriteLine($"   処理状況: {status}");
                    if (status.Contains("failed"))
                    {
                        Console.WriteLine($"   [ERROR] DLLからエラー返却: {status}");
                        return;
                    }
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
                catch (BadImageFormatException ex)
                {
                    Console.WriteLine($"   [ERROR] DLLの形式が不正です: {ex.Message}");
                    return;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"   [ERROR] 処理状況取得で予期しないエラー: {ex.Message}");
                    Console.WriteLine($"   例外の種類: {ex.GetType().Name}");
                    Console.WriteLine($"   スタックトレース: {ex.StackTrace}");
                    return;
                }
                
                // 5. 血圧推定テスト
                Console.WriteLine("\n5. 血圧推定テスト");
                try
                {
                    string sampleVideo = "sample_video.webm";
                    if (!File.Exists(sampleVideo))
                    {
                        Console.WriteLine($"   [ERROR] サンプル動画が見つかりません: {sampleVideo}");
                        return;
                    }
                    var fileInfo = new FileInfo(sampleVideo);
                    Console.WriteLine($"   サンプル動画: {sampleVideo} ({fileInfo.Length / 1024 / 1024.0:F2} MB)");

                    // ffmpeg.exeのパスを決定
                    string ffmpegExe = "ffmpeg.exe";
                    if (File.Exists("ffmpeg.exe"))
                        ffmpegExe = Path.GetFullPath("ffmpeg.exe");
                    else if (File.Exists("../ffmpeg.exe"))
                        ffmpegExe = Path.GetFullPath("../ffmpeg.exe");
                    // それ以外はPATHに頼る

                    if (!File.Exists(ffmpegExe) && ffmpegExe != "ffmpeg.exe")
                    {
                        Console.WriteLine($"   [ERROR] ffmpeg.exeが見つかりません。配布物に同梱されているか、PATHが通っているか確認してください。");
                        Console.WriteLine("   [DEBUG] testディレクトリのファイル一覧:");
                        foreach (var f in Directory.GetFiles(".", "*", SearchOption.TopDirectoryOnly))
                            Console.WriteLine("   " + f);
                        return;
                    }

                    // webm→画像シーケンス一時変換
                    string tempDir = Path.Combine(Path.GetTempPath(), $"frames_{Guid.NewGuid().ToString().Replace("-", "")}");
                    Directory.CreateDirectory(tempDir);
                    string framePattern = Path.Combine(tempDir, "frame_%05d.jpg");
                    Console.WriteLine($"   ffmpegで画像シーケンスに一時変換: {framePattern}");
                    var ffmpegProc = new System.Diagnostics.Process();
                    ffmpegProc.StartInfo.FileName = ffmpegExe;
                    ffmpegProc.StartInfo.Arguments = $"-y -i \"{sampleVideo}\" -q:v 2 \"{framePattern}\"";
                    ffmpegProc.StartInfo.UseShellExecute = false;
                    ffmpegProc.StartInfo.RedirectStandardOutput = true;
                    ffmpegProc.StartInfo.RedirectStandardError = true;
                    ffmpegProc.Start();
                    Console.WriteLine("   [DEBUG] ffmpegプロセス開始");
                    string ffmpegErr = ffmpegProc.StandardError.ReadToEnd();
                    string ffmpegOut = ffmpegProc.StandardOutput.ReadToEnd();
                    bool exited = ffmpegProc.WaitForExit(60000); // 60秒でタイムアウト
                    if (!exited) {
                        ffmpegProc.Kill();
                        Console.WriteLine("   [ERROR] ffmpeg変換がタイムアウトしました");
                        Console.WriteLine($"   [ffmpeg stderr] {ffmpegErr}");
                        try { Directory.Delete(tempDir, true); } catch { }
                        return;
                    }
                    var frameFiles = Directory.GetFiles(tempDir, "frame_*.jpg").OrderBy(f => f).ToArray();
                    if (frameFiles.Length == 0)
                    {
                        Console.WriteLine($"   [ERROR] ffmpeg画像変換失敗: {ffmpegErr}");
                        try { Directory.Delete(tempDir, true); } catch { }
                        return;
                    }
                    Console.WriteLine($"   ffmpeg変換成功（{frameFiles.Length}フレーム）");

                    // 画像ファイルを1枚ずつimreadして解析 → DLLの新APIで推定
                    int height = 170, weight = 70, sex = 1;
                    bool callbackCalled = false;
                    AutoResetEvent callbackEvent = new AutoResetEvent(false);
                    BPCallback callback = (reqId, sbp, dbp, csv, errorsJson) =>
                    {
                        try {
                            Console.WriteLine($"[CALLBACK] requestId={reqId}, SBP={sbp}, DBP={dbp}");
                            if (!string.IsNullOrEmpty(csv))
                            {
                                Console.WriteLine($"[CALLBACK] CSV length: {csv.Length}");
                            }
                            if (!string.IsNullOrEmpty(errorsJson) && errorsJson != "[]")
                            {
                                Console.WriteLine($"[CALLBACK] Errors: {errorsJson}");
                            }
                            callbackCalled = true;
                            callbackEvent.Set();
                        } catch (Exception ex) {
                            Console.WriteLine($"[FATAL ERROR] コールバック内で例外発生: {ex.Message}");
                            Console.WriteLine($"スタックトレース: {ex.StackTrace}");
                            throw;
                        }
                    };
                    Console.WriteLine("   血圧推定リクエスト送信中（画像配列）...");
                    var sb = new StringBuilder(256);
                    int resultCode = AnalyzeBloodPressureFromImages(sb, sb.Capacity, frameFiles, frameFiles.Length, height, weight, sex, callback);
                    if (resultCode != 0)
                    {
                        Console.WriteLine($"   [ERROR] DLLからエラー返却: {sb.ToString()}");
                        return;
                    }
                    if (!callbackEvent.WaitOne(30000))
                    {
                        Console.WriteLine("   [ERROR] コールバックが30秒以内に呼ばれませんでした");
                    }
                    else if (!callbackCalled)
                    {
                        Console.WriteLine("   [ERROR] コールバックが呼ばれませんでした");
                    }
                    else
                    {
                        Console.WriteLine("   [SUCCESS] 血圧推定テスト完了");
                    }
                    try { Directory.Delete(tempDir, true); } catch { }
                    return;

                }
                catch (Exception ex)
                {
                    Console.WriteLine($"   [ERROR] 血圧推定テストで予期しないエラー: {ex.Message}");
                    Console.WriteLine($"   例外の種類: {ex.GetType().Name}");
                    Console.WriteLine($"   スタックトレース: {ex.StackTrace}");
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

        [DllImport("kernel32.dll", SetLastError = true)]
        static extern bool SetDllDirectory(string lpPathName);

        public static void Main(string[] args)
        {
            SetDllDirectory(System.IO.Directory.GetCurrentDirectory());
            AppDomain.CurrentDomain.UnhandledException += (sender, e) =>
            {
                Console.WriteLine($"[FATAL] UnhandledException: {e.ExceptionObject?.ToString()}");
            };
            System.Threading.Tasks.TaskScheduler.UnobservedTaskException += (sender, e) =>
            {
                Console.WriteLine($"[FATAL] UnobservedTaskException: {e.Exception?.ToString()}");
            };
            Console.WriteLine($"[DEBUG] Is64BitProcess: {Environment.Is64BitProcess}");
            try
            {
                // 1. DLL初期化
                var sb = new StringBuilder(1024);
                int result = InitializeBP(sb, sb.Capacity, "models");
                Console.WriteLine($"InitializeBP result: {result}");
                Console.WriteLine($"InitializeBP buffer: '{sb}'");
                if (result != 0) return;

                // 2. バージョン情報
                sb.Clear();
                result = GetVersionInfo(sb, sb.Capacity);
                Console.WriteLine($"GetVersionInfo result: {result}");
                Console.WriteLine($"GetVersionInfo buffer: '{sb}'");
                if (result != 0) return;

                // 推論テスト
                double[] peakTimes = new double[] { 0.1, 0.5, 1.0, 1.5, 2.0 }; // ダミーデータ
                int height = 170;
                int weight = 65;
                int sex = 1; // 男性=1, 女性=0
                int sbp, dbp;
                int estResult = EstimateBloodPressure(peakTimes, peakTimes.Length, height, weight, sex, out sbp, out dbp);
                Console.WriteLine($"EstimateBloodPressure result: {estResult}");
                Console.WriteLine($"推定SBP: {sbp}, 推定DBP: {dbp}");

                // 仕様準拠の推論テスト
                string requestId = DateTime.Now.ToString("yyyyMMddHHmmssfff") + "_TEST001_000000001";
                int height = 170;
                int weight = 65;
                int sex = 1; // 男性=1, 女性=2
                string moviePath = "sample_video.webm";
                BPCallback callback = (reqId, sbp, dbp, csv, errorsJson) => {
                    Console.WriteLine($"[CALLBACK] requestId={reqId}, SBP={sbp}, DBP={dbp}");
                    if (!string.IsNullOrEmpty(csv))
                    {
                        Console.WriteLine($"[CALLBACK] CSV length: {csv.Length}");
                    }
                    if (!string.IsNullOrEmpty(errorsJson) && errorsJson != "[]")
                    {
                        Console.WriteLine($"[CALLBACK] Errors: {errorsJson}");
                    }
                };
                int ret = StartBloodPressureAnalysisRequest(requestId, height, weight, sex, moviePath, callback);
                Console.WriteLine($"StartBloodPressureAnalysisRequest returned: {ret}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"アプリケーションエラー: {ex.Message}");
                Console.WriteLine($"スタックトレース: {ex.StackTrace}");
                Environment.Exit(1);
            }
        }
    }
}

