using System;
using System.Runtime.InteropServices;
using System.Threading;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text; // StringBuilderを追加
using System.Diagnostics; // Stopwatchを追加

namespace BloodPressureDllTest
{
    // コールバック関数の型定義（C++ typedef void(*BPCallback)(const char* requestId, int maxBloodPressure, int minBloodPressure, const char* measureRowData, const char* errorsJson); に完全一致）
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate void BPCallback(
        [MarshalAs(UnmanagedType.LPStr)] string requestId,
        int maxBloodPressure,
        int minBloodPressure,
        [MarshalAs(UnmanagedType.LPStr)] string measureRowData,
        [MarshalAs(UnmanagedType.LPStr)] string errorsJson);

    public class BloodPressureDll
    {
        private const string DllPath = "BloodPressureDLL.dll";

        [DllImport(DllPath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern int InitializeBP([Out] StringBuilder outBuf, int bufSize, [MarshalAs(UnmanagedType.LPStr)] string modelDir);

        [DllImport(DllPath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern int StartBloodPressureAnalysisRequest(
            [Out] StringBuilder outBuf, int bufSize,
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

        // IntPtr→string変換時はNULLチェック
        public static string PtrToStringSafe(IntPtr ptr)
        {
            if (ptr == IntPtr.Zero) return null;
            return Marshal.PtrToStringAnsi(ptr);
        }

        // 処理時間計測結果を格納するクラス
        public class PerformanceMetrics
        {
            public TimeSpan TotalTime { get; set; }
            public TimeSpan InitializationTime { get; set; }
            public TimeSpan VideoConversionTime { get; set; }
            public TimeSpan BloodPressureAnalysisTime { get; set; }
            public TimeSpan CallbackTime { get; set; }
            public int FrameCount { get; set; }
            public long VideoFileSize { get; set; }
            public bool IsSuccess { get; set; }
            public string ErrorMessage { get; set; }
            public int EstimatedSBP { get; set; }
            public int EstimatedDBP { get; set; }
        }

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

        // 処理時間計測付きの血圧推定テスト
        public static PerformanceMetrics TestBloodPressureEstimationWithTiming()
        {
            var metrics = new PerformanceMetrics();
            var totalStopwatch = Stopwatch.StartNew();
            
            Console.WriteLine("=== 血圧推定処理時間計測テスト ===");
            Console.WriteLine("目標処理時間: 2-3秒");
            Console.WriteLine("計測開始時刻: " + DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff"));
            
            try
            {
                // 1. DLL初期化時間計測
                Console.WriteLine("\n1. DLL初期化時間計測");
                var initStopwatch = Stopwatch.StartNew();
                
                Console.WriteLine("   DLL初期化を試行中...");
                var sb = new StringBuilder(1024); // バッファサイズを大きくする
                int initResult = InitializeBP(sb, sb.Capacity, "models");
                initStopwatch.Stop();
                metrics.InitializationTime = initStopwatch.Elapsed;
                
                Console.WriteLine($"   初期化結果: {initResult}");
                Console.WriteLine($"   初期化時間: {metrics.InitializationTime.TotalMilliseconds:F2} ms");
                Console.WriteLine($"   初期化メッセージ: '{sb.ToString()}'");
                
                if (initResult != 0)
                {
                    metrics.IsSuccess = false;
                    metrics.ErrorMessage = $"DLL初期化失敗: {sb.ToString()}";
                    Console.WriteLine($"   [ERROR] DLL初期化失敗: {sb.ToString()}");
                    
                    // 追加のデバッグ情報
                    Console.WriteLine("   [DEBUG] 追加デバッグ情報:");
                    Console.WriteLine($"   - カレントディレクトリ: {Environment.CurrentDirectory}");
                    Console.WriteLine($"   - Modelsディレクトリ存在: {Directory.Exists("models")}");
                    if (Directory.Exists("models"))
                    {
                        var modelFiles = Directory.GetFiles("models", "*", SearchOption.AllDirectories);
                        Console.WriteLine($"   - Modelsディレクトリ内ファイル数: {modelFiles.Length}");
                        foreach (var file in modelFiles)
                        {
                            var fileInfo = new FileInfo(file);
                            var relativePath = file.Replace(Environment.CurrentDirectory, "").TrimStart('\\');
                            Console.WriteLine($"     {relativePath} ({fileInfo.Length / 1024.0:F2} KB)");
                        }
                    }
                    
                    // 依存DLLのチェック
                    string[] requiredDlls = { "opencv_world480.dll", "onnxruntime.dll", "zlib.dll" };
                    Console.WriteLine("   - 依存DLLチェック:");
                    foreach (var dll in requiredDlls)
                    {
                        bool exists = File.Exists(dll);
                        Console.WriteLine($"     {dll}: {(exists ? "FOUND" : "NOT FOUND")}");
                        if (exists)
                        {
                            var dllInfo = new FileInfo(dll);
                            Console.WriteLine($"       Size: {dllInfo.Length / 1024.0:F2} KB");
                        }
                    }
                    
                    return metrics;
                }
                
                Console.WriteLine("   [SUCCESS] DLL初期化成功");
                
                // 2. 動画変換時間計測
                Console.WriteLine("\n2. 動画変換時間計測");
                var conversionStopwatch = Stopwatch.StartNew();
                
                string sampleVideo = "sample_video.webm";
                if (!File.Exists(sampleVideo))
                {
                    metrics.IsSuccess = false;
                    metrics.ErrorMessage = $"サンプル動画が見つかりません: {sampleVideo}";
                    Console.WriteLine($"   [ERROR] {metrics.ErrorMessage}");
                    return metrics;
                }
                
                var videoFileInfo = new FileInfo(sampleVideo);
                metrics.VideoFileSize = videoFileInfo.Length;
                Console.WriteLine($"   サンプル動画: {sampleVideo} ({videoFileInfo.Length / 1024 / 1024.0:F2} MB)");
                
                // ffmpeg.exeのパスを決定
                string ffmpegExe = "ffmpeg.exe";
                if (File.Exists("ffmpeg.exe"))
                    ffmpegExe = Path.GetFullPath("ffmpeg.exe");
                else if (File.Exists("../ffmpeg.exe"))
                    ffmpegExe = Path.GetFullPath("../ffmpeg.exe");

                if (!File.Exists(ffmpegExe) && ffmpegExe != "ffmpeg.exe")
                {
                    metrics.IsSuccess = false;
                    metrics.ErrorMessage = "ffmpeg.exeが見つかりません";
                    Console.WriteLine($"   [ERROR] {metrics.ErrorMessage}");
                    return metrics;
                }

                // webm→画像シーケンス一時変換
                string tempDir = Path.Combine(Path.GetTempPath(), $"frames_{Guid.NewGuid().ToString().Replace("-", "")}");
                Directory.CreateDirectory(tempDir);
                string framePattern = Path.Combine(tempDir, "frame_%05d.jpg");
                Console.WriteLine($"   ffmpegで画像シーケンスに一時変換: {framePattern}");
                
                var ffmpegProc = new System.Diagnostics.Process();
                ffmpegProc.StartInfo.FileName = ffmpegExe;
                ffmpegProc.StartInfo.Arguments = $"-y -i \"{sampleVideo}\" -ss 20 -t 30 -vf \"scale=640:480\" -r 15 -q:v 2 \"{framePattern}\"";
                ffmpegProc.StartInfo.UseShellExecute = false;
                ffmpegProc.StartInfo.RedirectStandardOutput = true;
                ffmpegProc.StartInfo.RedirectStandardError = true;
                ffmpegProc.Start();
                
                string ffmpegErr = ffmpegProc.StandardError.ReadToEnd();
                string ffmpegOut = ffmpegProc.StandardOutput.ReadToEnd();
                bool exited = ffmpegProc.WaitForExit(60000); // 60秒でタイムアウト
                
                conversionStopwatch.Stop();
                metrics.VideoConversionTime = conversionStopwatch.Elapsed;
                
                Console.WriteLine($"   ffmpeg終了コード: {ffmpegProc.ExitCode}");
                Console.WriteLine($"   動画変換時間: {metrics.VideoConversionTime.TotalMilliseconds:F2} ms");
                
                if (!exited)
                {
                    ffmpegProc.Kill();
                    metrics.IsSuccess = false;
                    metrics.ErrorMessage = "ffmpeg変換がタイムアウトしました";
                    Console.WriteLine($"   [ERROR] {metrics.ErrorMessage}");
                    try { Directory.Delete(tempDir, true); } catch { }
                    return metrics;
                }
                
                if (ffmpegProc.ExitCode != 0)
                {
                    metrics.IsSuccess = false;
                    metrics.ErrorMessage = $"ffmpeg変換が失敗しました (終了コード: {ffmpegProc.ExitCode})";
                    Console.WriteLine($"   [ERROR] {metrics.ErrorMessage}");
                    Console.WriteLine($"   [ffmpeg stderr] {ffmpegErr}");
                    try { Directory.Delete(tempDir, true); } catch { }
                    return metrics;
                }
                
                var frameFiles = Directory.GetFiles(tempDir, "frame_*.jpg").OrderBy(f => f).ToArray();
                if (frameFiles.Length == 0)
                {
                    metrics.IsSuccess = false;
                    metrics.ErrorMessage = "ffmpeg画像変換失敗: フレームファイルが生成されませんでした";
                    Console.WriteLine($"   [ERROR] {metrics.ErrorMessage}");
                    try { Directory.Delete(tempDir, true); } catch { }
                    return metrics;
                }
                
                metrics.FrameCount = frameFiles.Length;
                Console.WriteLine($"   ffmpeg変換成功（{frameFiles.Length}フレーム）");
                
                // 3. 血圧推定時間計測
                Console.WriteLine("\n3. 血圧推定時間計測");
                var analysisStopwatch = Stopwatch.StartNew();
                var callbackStopwatch = Stopwatch.StartNew();
                
                int height = 170, weight = 70, sex = 1;
                bool callbackCalled = false;
                AutoResetEvent callbackEvent = new AutoResetEvent(false);
                BPCallback callback = (reqId, sbp, dbp, csv, errorsJson) =>
                {
                    callbackStopwatch.Stop();
                    metrics.CallbackTime = callbackStopwatch.Elapsed;
                    
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
                        
                        // 血圧推定値を保存
                        metrics.EstimatedSBP = sbp;
                        metrics.EstimatedDBP = dbp;
                        
                        callbackCalled = true;
                        callbackEvent.Set();
                    } catch (Exception ex) {
                        Console.WriteLine($"[FATAL ERROR] コールバック内で例外発生: {ex.Message}");
                        Console.WriteLine($"スタックトレース: {ex.StackTrace}");
                        throw;
                    }
                };
                
                Console.WriteLine("   血圧推定リクエスト送信中（画像配列）...");
                var outBuf = new StringBuilder(256);
                int resultCode = AnalyzeBloodPressureFromImages(outBuf, outBuf.Capacity, frameFiles, frameFiles.Length, height, weight, sex, callback);
                
                if (resultCode != 0)
                {
                    analysisStopwatch.Stop();
                    metrics.BloodPressureAnalysisTime = analysisStopwatch.Elapsed;
                    metrics.IsSuccess = false;
                    metrics.ErrorMessage = $"DLLからエラー返却: {outBuf.ToString()}";
                    Console.WriteLine($"   [ERROR] {metrics.ErrorMessage}");
                    try { Directory.Delete(tempDir, true); } catch { }
                    return metrics;
                }
                
                if (!callbackEvent.WaitOne(30000))
                {
                    analysisStopwatch.Stop();
                    metrics.BloodPressureAnalysisTime = analysisStopwatch.Elapsed;
                    metrics.IsSuccess = false;
                    metrics.ErrorMessage = "コールバックが30秒以内に呼ばれませんでした";
                    Console.WriteLine($"   [ERROR] {metrics.ErrorMessage}");
                    try { Directory.Delete(tempDir, true); } catch { }
                    return metrics;
                }
                
                if (!callbackCalled)
                {
                    analysisStopwatch.Stop();
                    metrics.BloodPressureAnalysisTime = analysisStopwatch.Elapsed;
                    metrics.IsSuccess = false;
                    metrics.ErrorMessage = "コールバックが呼ばれませんでした";
                    Console.WriteLine($"   [ERROR] {metrics.ErrorMessage}");
                    try { Directory.Delete(tempDir, true); } catch { }
                    return metrics;
                }
                
                analysisStopwatch.Stop();
                metrics.BloodPressureAnalysisTime = analysisStopwatch.Elapsed;
                
                Console.WriteLine($"   血圧推定時間: {metrics.BloodPressureAnalysisTime.TotalMilliseconds:F2} ms");
                Console.WriteLine($"   コールバック時間: {metrics.CallbackTime.TotalMilliseconds:F2} ms");
                
                // 精度検証情報
                Console.WriteLine($"\n=== 精度検証情報 ===");
                Console.WriteLine($"フレーム数: {metrics.FrameCount}");
                Console.WriteLine($"動画時間: {metrics.FrameCount / 15.0:F1} 秒 (15fps想定)");
                Console.WriteLine($"推定血圧: SBP={metrics.EstimatedSBP} mmHg, DBP={metrics.EstimatedDBP} mmHg");
                Console.WriteLine($"推定精度: フレーム数{metrics.FrameCount}での推定値");
                
                try { Directory.Delete(tempDir, true); } catch { }
                
                // 4. 総合結果
                totalStopwatch.Stop();
                metrics.TotalTime = totalStopwatch.Elapsed;
                metrics.IsSuccess = true;
                
                Console.WriteLine("\n=== 処理時間計測結果 ===");
                Console.WriteLine($"総処理時間: {metrics.TotalTime.TotalMilliseconds:F2} ms ({metrics.TotalTime.TotalSeconds:F3} 秒)");
                Console.WriteLine($"初期化時間: {metrics.InitializationTime.TotalMilliseconds:F2} ms");
                Console.WriteLine($"動画変換時間: {metrics.VideoConversionTime.TotalMilliseconds:F2} ms");
                Console.WriteLine($"血圧推定時間: {metrics.BloodPressureAnalysisTime.TotalMilliseconds:F2} ms");
                Console.WriteLine($"コールバック時間: {metrics.CallbackTime.TotalMilliseconds:F2} ms");
                Console.WriteLine($"動画ファイルサイズ: {metrics.VideoFileSize / 1024 / 1024.0:F2} MB");
                Console.WriteLine($"フレーム数: {metrics.FrameCount}");
                
                // 目標時間との比較
                double targetTimeMs = 3000; // 3秒
                double actualTimeMs = metrics.TotalTime.TotalMilliseconds;
                
                Console.WriteLine($"\n=== パフォーマンス評価 ===");
                Console.WriteLine($"目標時間: {targetTimeMs:F0} ms (3秒)");
                Console.WriteLine($"実際時間: {actualTimeMs:F2} ms");
                
                if (actualTimeMs <= targetTimeMs)
                {
                    Console.WriteLine($"[SUCCESS] 目標時間内で完了 ({actualTimeMs:F2} ms <= {targetTimeMs:F0} ms)");
                    Console.WriteLine($"[PERFORMANCE] 目標達成率: {(targetTimeMs / actualTimeMs * 100):F1}%");
                }
                else
                {
                    double overTime = actualTimeMs - targetTimeMs;
                    Console.WriteLine($"[WARNING] 目標時間を超過 ({actualTimeMs:F2} ms > {targetTimeMs:F0} ms)");
                    Console.WriteLine($"[PERFORMANCE] 超過時間: {overTime:F2} ms ({(overTime / targetTimeMs * 100):F1}%超過)");
                }
                
                // 各段階の時間分析
                Console.WriteLine($"\n=== 時間分析 ===");
                double initPercent = (metrics.InitializationTime.TotalMilliseconds / actualTimeMs) * 100;
                double conversionPercent = (metrics.VideoConversionTime.TotalMilliseconds / actualTimeMs) * 100;
                double analysisPercent = (metrics.BloodPressureAnalysisTime.TotalMilliseconds / actualTimeMs) * 100;
                double callbackPercent = (metrics.CallbackTime.TotalMilliseconds / actualTimeMs) * 100;
                
                Console.WriteLine($"初期化: {initPercent:F1}% ({metrics.InitializationTime.TotalMilliseconds:F2} ms)");
                Console.WriteLine($"動画変換: {conversionPercent:F1}% ({metrics.VideoConversionTime.TotalMilliseconds:F2} ms)");
                Console.WriteLine($"血圧推定: {analysisPercent:F1}% ({metrics.BloodPressureAnalysisTime.TotalMilliseconds:F2} ms)");
                Console.WriteLine($"コールバック: {callbackPercent:F1}% ({metrics.CallbackTime.TotalMilliseconds:F2} ms)");
                
                // ボトルネック分析
                Console.WriteLine($"\n=== ボトルネック分析 ===");
                var times = new[]
                {
                    ("初期化", metrics.InitializationTime.TotalMilliseconds),
                    ("動画変換", metrics.VideoConversionTime.TotalMilliseconds),
                    ("血圧推定", metrics.BloodPressureAnalysisTime.TotalMilliseconds),
                    ("コールバック", metrics.CallbackTime.TotalMilliseconds)
                };
                
                var maxTime = times.Max(t => t.Item2);
                var bottleneck = times.First(t => t.Item2 == maxTime);
                Console.WriteLine($"最大ボトルネック: {bottleneck.Item1} ({bottleneck.Item2:F2} ms)");
                
                // 改善提案
                Console.WriteLine($"\n=== 改善提案 ===");
                if (conversionPercent > 50)
                {
                    Console.WriteLine("- 動画変換時間が全体の50%以上を占めています");
                    Console.WriteLine("- 動画の事前変換や圧縮率の調整を検討してください");
                }
                if (analysisPercent > 50)
                {
                    Console.WriteLine("- 血圧推定時間が全体の50%以上を占めています");
                    Console.WriteLine("- モデルの最適化やフレーム数の削減を検討してください");
                }
                if (initPercent > 20)
                {
                    Console.WriteLine("- 初期化時間が全体の20%以上を占めています");
                    Console.WriteLine("- モデルの事前読み込みやキャッシュを検討してください");
                }
                
                Console.WriteLine($"\n[SUCCESS] 血圧推定処理時間計測テスト完了");
                Console.WriteLine("=== 血圧推定処理時間計測テスト完了 ===");
                
                return metrics;
            }
            catch (Exception ex)
            {
                totalStopwatch.Stop();
                metrics.TotalTime = totalStopwatch.Elapsed;
                metrics.IsSuccess = false;
                metrics.ErrorMessage = $"予期しないエラー: {ex.Message}";
                Console.WriteLine($"\n[ERROR] {metrics.ErrorMessage}");
                Console.WriteLine($"例外の種類: {ex.GetType().Name}");
                Console.WriteLine($"スタックトレース: {ex.StackTrace}");
                return metrics;
            }
        }

        public static void TestComprehensiveDLL()
        {
            Console.WriteLine("=== C++ Blood Pressure DLL テスト ===");
            
            try
            {
                // 処理時間計測付きの血圧推定テストを実行
                var performanceMetrics = TestBloodPressureEstimationWithTiming();
                
                // 結果をファイルに保存
                string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                string performanceLogFile = $"performance_log_{timestamp}.txt";
                
                using (var writer = new StreamWriter(performanceLogFile, false, Encoding.UTF8))
                {
                    writer.WriteLine("=== Blood Pressure Estimation Performance Log ===");
                    writer.WriteLine($"Test Date: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
                    writer.WriteLine($"Success: {performanceMetrics.IsSuccess}");
                    writer.WriteLine($"Total Time: {performanceMetrics.TotalTime.TotalMilliseconds:F2} ms");
                    writer.WriteLine($"Initialization Time: {performanceMetrics.InitializationTime.TotalMilliseconds:F2} ms");
                    writer.WriteLine($"Video Conversion Time: {performanceMetrics.VideoConversionTime.TotalMilliseconds:F2} ms");
                    writer.WriteLine($"Blood Pressure Analysis Time: {performanceMetrics.BloodPressureAnalysisTime.TotalMilliseconds:F2} ms");
                    writer.WriteLine($"Callback Time: {performanceMetrics.CallbackTime.TotalMilliseconds:F2} ms");
                    writer.WriteLine($"Video File Size: {performanceMetrics.VideoFileSize / 1024 / 1024.0:F2} MB");
                    writer.WriteLine($"Frame Count: {performanceMetrics.FrameCount}");
                    writer.WriteLine($"Error Message: {performanceMetrics.ErrorMessage ?? "None"}");
                }
                
                Console.WriteLine($"パフォーマンスログ保存: {performanceLogFile}");
                
                if (!performanceMetrics.IsSuccess)
                {
                    Console.WriteLine($"\n[ERROR] 血圧推定テスト失敗: {performanceMetrics.ErrorMessage}");
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
            
            // 処理時間計測付きのテストを実行
            TestComprehensiveDLL();
        }
    }
}

