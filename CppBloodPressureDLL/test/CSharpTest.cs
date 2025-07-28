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
            public TimeSpan FrameExtractionTime { get; set; }
            public TimeSpan ModelLoadingTime { get; set; }
            public TimeSpan FaceDetectionTime { get; set; }
            public TimeSpan RPPGProcessingTime { get; set; }
            public TimeSpan BPEstimationTime { get; set; }
            public int FrameCount { get; set; }
            public long VideoFileSize { get; set; }
            public bool IsSuccess { get; set; }
            public string ErrorMessage { get; set; }
            public int EstimatedSBP { get; set; }
            public int EstimatedDBP { get; set; }
            public long PeakMemoryUsage { get; set; }
            public double AverageCpuUsage { get; set; }
            public int DetectedFaces { get; set; }
            public int ProcessedFrames { get; set; }
            public double FramesPerSecond { get; set; }
            public Dictionary<string, TimeSpan> DetailedTimings { get; set; }
            
            public PerformanceMetrics()
            {
                DetailedTimings = new Dictionary<string, TimeSpan>();
            }
        }

        // パフォーマンス監視クラス
        public class PerformanceMonitor
        {
            private System.Diagnostics.PerformanceCounter cpuCounter;
            private long initialMemory;
            private long peakMemory;
            private List<double> cpuReadings;
            private DateTime startTime;

            public PerformanceMonitor()
            {
                try
                {
                    cpuCounter = new System.Diagnostics.PerformanceCounter("Processor", "% Processor Time", "_Total");
                    cpuReadings = new List<double>();
                    startTime = DateTime.Now;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Warning: Performance counter initialization failed: {ex.Message}");
                }
            }

            public void Start()
            {
                initialMemory = GC.GetTotalMemory(false);
                peakMemory = initialMemory;
                cpuReadings.Clear();
                startTime = DateTime.Now;
            }

            public void Update()
            {
                // メモリ使用量の更新
                long currentMemory = GC.GetTotalMemory(false);
                if (currentMemory > peakMemory)
                    peakMemory = currentMemory;

                // CPU使用率の更新
                try
                {
                    if (cpuCounter != null)
                    {
                        double cpuUsage = cpuCounter.NextValue();
                        cpuReadings.Add(cpuUsage);
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Warning: CPU usage reading failed: {ex.Message}");
                }
            }

            public long GetPeakMemoryUsage()
            {
                return peakMemory - initialMemory;
            }

            public double GetAverageCpuUsage()
            {
                if (cpuReadings.Count == 0) return 0.0;
                return cpuReadings.Average();
            }

            public void Dispose()
            {
                cpuCounter?.Dispose();
            }
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
            
            // 詳細タイミング情報を読み取り
            try
            {
                // まずコールバックからタイミング情報を確認
                if (!string.IsNullOrEmpty(errorsJson) && errorsJson.Contains("timing_info"))
                {
                    Console.WriteLine("\n=== DETAILED TIMING ANALYSIS (from callback) ===");
                    // 簡易的なJSON解析（実際のプロジェクトではNewtonsoft.Json等を使用）
                    int startIndex = errorsJson.IndexOf("\"timing_info\":\"") + 15;
                    int endIndex = errorsJson.LastIndexOf("\"");
                    if (startIndex > 14 && endIndex > startIndex)
                    {
                        string timingInfo = errorsJson.Substring(startIndex, endIndex - startIndex);
                        timingInfo = timingInfo.Replace("\\n", "\n").Replace("\\t", "\t");
                        Console.WriteLine(timingInfo);
                    }
                    Console.WriteLine("=== END OF DETAILED TIMING ANALYSIS ===");
                }
                // ファイルからも読み取り（バックアップ）
                else if (File.Exists("detailed_timing.log"))
                {
                    Console.WriteLine("\n=== DETAILED TIMING ANALYSIS (from file) ===");
                    string[] timingLines = File.ReadAllLines("detailed_timing.log");
                    foreach (string line in timingLines)
                    {
                        Console.WriteLine(line);
                    }
                    Console.WriteLine("=== END OF DETAILED TIMING ANALYSIS ===");
                }
                else
                {
                    Console.WriteLine("[INFO] 詳細タイミングログファイルが見つかりません");
                }
            }
            catch (Exception e)
            {
                Console.WriteLine($"[WARNING] タイミング情報読み取りエラー: {e.Message}");
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

        // 詳細なボトルネック分析
        private static void AnalyzeBottlenecks(PerformanceMetrics metrics)
        {
            Console.WriteLine("\n=== 詳細ボトルネック分析 ===");
            
            var timings = new Dictionary<string, double>
            {
                {"初期化", metrics.InitializationTime.TotalMilliseconds},
                {"動画変換", metrics.VideoConversionTime.TotalMilliseconds},
                {"血圧推定", metrics.BloodPressureAnalysisTime.TotalMilliseconds},
                {"コールバック", metrics.CallbackTime.TotalMilliseconds}
            };

            // 詳細タイミングも追加
            foreach (var timing in metrics.DetailedTimings)
            {
                timings[timing.Key] = timing.Value.TotalMilliseconds;
            }

            var sortedTimings = timings.OrderByDescending(x => x.Value).ToList();
            double totalTime = metrics.TotalTime.TotalMilliseconds;

            Console.WriteLine("処理時間ランキング:");
            for (int i = 0; i < sortedTimings.Count; i++)
            {
                var timing = sortedTimings[i];
                double percentage = (timing.Value / totalTime) * 100;
                Console.WriteLine($"  {i + 1}. {timing.Key}: {timing.Value:F2} ms ({percentage:F1}%)");
            }

            // ボトルネック特定
            var primaryBottleneck = sortedTimings[0];
            var secondaryBottleneck = sortedTimings.Count > 1 ? sortedTimings[1] : new KeyValuePair<string, double>("", 0);

            Console.WriteLine($"\n主要ボトルネック: {primaryBottleneck.Key} ({primaryBottleneck.Value:F2} ms)");
            if (secondaryBottleneck.Value > 0)
            {
                Console.WriteLine($"二次ボトルネック: {secondaryBottleneck.Key} ({secondaryBottleneck.Value:F2} ms)");
            }

            // 改善提案
            Console.WriteLine("\n=== 改善提案 ===");
            foreach (var timing in sortedTimings.Take(3))
            {
                double percentage = (timing.Value / totalTime) * 100;
                
                switch (timing.Key)
                {
                    case "初期化":
                        if (percentage > 20)
                        {
                            Console.WriteLine($"- 初期化時間が{percentage:F1}%を占めています");
                            Console.WriteLine(" モデルの事前読み込みやキャッシュ機能の実装を検討");
                            Console.WriteLine(" 軽量モデルへの切り替えを検討");
                        }
                        break;
                    case "動画変換":
                        if (percentage > 30)
                        {
                            Console.WriteLine($"- 動画変換時間が{percentage:F1}%を占めています");
                            Console.WriteLine(" 動画の事前変換や圧縮率の調整を検討");
                            Console.WriteLine(" より高速なコーデックの使用を検討");
                            Console.WriteLine(" フレームレートの削減を検討");
                        }
                        break;
                    case "血圧推定":
                        if (percentage > 40)
                        {
                            Console.WriteLine($"- 血圧推定時間が{percentage:F1}%を占めています");
                            Console.WriteLine(" モデルの最適化や量子化を検討");
                            Console.WriteLine(" フレーム数の削減を検討");
                            Console.WriteLine(" GPUアクセラレーションの実装を検討");
                        }
                        break;
                    case "コールバック":
                        if (percentage > 10)
                        {
                            Console.WriteLine($"- コールバック時間が{percentage:F1}%を占めています");
                            Console.WriteLine(" コールバック処理の最適化を検討");
                        }
                        break;
                }
            }

            // パフォーマンス目標との比較
            double targetTime = 3000; // 3秒
            if (totalTime > targetTime)
            {
                double overTime = totalTime - targetTime;
                double overPercentage = (overTime / targetTime) * 100;
                Console.WriteLine($"\n=== パフォーマンス目標分析 ===");
                Console.WriteLine($"目標時間: {targetTime:F0} ms");
                Console.WriteLine($"実際時間: {totalTime:F2} ms");
                Console.WriteLine($"超過時間: {overTime:F2} ms ({overPercentage:F1}%超過)");
                
                if (overPercentage > 100)
                {
                    Console.WriteLine("重大なパフォーマンス問題: 目標時間の2倍を超過");
                }
                else if (overPercentage > 50)
                {
                    Console.WriteLine("パフォーマンス問題: 目標時間の50%以上超過");
                }
                else
                {
                    Console.WriteLine("軽微なパフォーマンス問題: 目標時間を超過");
                }
            }
            else
            {
                double performanceRatio = (targetTime / totalTime) * 100;
                Console.WriteLine($"\n=== パフォーマンス目標分析 ===");
                Console.WriteLine($"目標達成: {performanceRatio:F1}%の性能");
            }
        }

        // メモリ使用量分析
        private static void AnalyzeMemoryUsage(PerformanceMetrics metrics)
        {
            Console.WriteLine("\n=== メモリ使用量分析 ===");
            double memoryMB = metrics.PeakMemoryUsage / (1024.0 * 1024.0);
            Console.WriteLine($"ピークメモリ使用量: {memoryMB:F2} MB");
            
            if (memoryMB > 1000)
            {
                Console.WriteLine("高メモリ使用量: 1GBを超過");
                Console.WriteLine("メモリリークの可能性を調査");
                Console.WriteLine("画像バッファの適切な解放を確認");
            }
            else if (memoryMB > 500)
            {
                Console.WriteLine("中程度のメモリ使用量: 500MBを超過");
            }
            else
            {
                Console.WriteLine("適切なメモリ使用量");
            }
        }

        // CPU使用率分析
        private static void AnalyzeCpuUsage(PerformanceMetrics metrics)
        {
            Console.WriteLine("\n=== CPU使用率分析 ===");
            Console.WriteLine($"平均CPU使用率: {metrics.AverageCpuUsage:F1}%");
            
            if (metrics.AverageCpuUsage > 80)
            {
                Console.WriteLine("高CPU使用率: 80%を超過");
                Console.WriteLine("  CPU負荷の分散を検討");
                Console.WriteLine("  並列処理の最適化を検討");
            }
            else if (metrics.AverageCpuUsage > 50)
            {
                Console.WriteLine("中程度のCPU使用率: 50%を超過");
            }
            else
            {
                Console.WriteLine("適切なCPU使用率");
            }
        }

        // フレーム処理効率分析
        private static void AnalyzeFrameProcessing(PerformanceMetrics metrics)
        {
            Console.WriteLine("\n=== フレーム処理効率分析 ===");
            Console.WriteLine($"処理フレーム数: {metrics.ProcessedFrames}");
            Console.WriteLine($"フレーム処理速度: {metrics.FramesPerSecond:F1} fps");
            
            if (metrics.FramesPerSecond < 10)
            {
                Console.WriteLine("低フレーム処理速度: 10fps未満");
                Console.WriteLine(" フレーム処理の最適化を検討");
            }
            else if (metrics.FramesPerSecond < 30)
            {
                Console.WriteLine("中程度のフレーム処理速度: 30fps未満");
            }
            else
            {
                Console.WriteLine("良好なフレーム処理速度");
            }
        }

        public static PerformanceMetrics TestBloodPressureEstimationWithTiming()
        {
            var metrics = new PerformanceMetrics();
            var totalStopwatch = Stopwatch.StartNew();
            var performanceMonitor = new PerformanceMonitor();
            
            Console.WriteLine("=== 血圧推定処理時間計測テスト ===");
            Console.WriteLine("目標処理時間: 2-3秒");
            Console.WriteLine("計測開始時刻: " + DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff"));
            
            try
            {
                performanceMonitor.Start();
                
                // 1. DLL初期化時間計測
                Console.WriteLine("\n1. DLL初期化時間計測");
                var initStopwatch = Stopwatch.StartNew();
                
                Console.WriteLine("   DLL初期化を試行中...");
                var sb = new StringBuilder(1024);
                int initResult = InitializeBP(sb, sb.Capacity, "models");
                initStopwatch.Stop();
                metrics.InitializationTime = initStopwatch.Elapsed;
                
                performanceMonitor.Update();
                
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
                ffmpegProc.StartInfo.Arguments = $"-y -i \"{sampleVideo}\" -ss 20 -t 30 -vf \"scale=320:240\" -r 30 -q:v 2 \"{framePattern}\"";
                ffmpegProc.StartInfo.UseShellExecute = false;
                ffmpegProc.StartInfo.RedirectStandardOutput = true;
                ffmpegProc.StartInfo.RedirectStandardError = true;
                ffmpegProc.Start();
                
                string ffmpegErr = ffmpegProc.StandardError.ReadToEnd();
                string ffmpegOut = ffmpegProc.StandardOutput.ReadToEnd();
                bool exited = ffmpegProc.WaitForExit(60000); // 60秒でタイムアウト
                
                conversionStopwatch.Stop();
                metrics.VideoConversionTime = conversionStopwatch.Elapsed;
                
                performanceMonitor.Update();
                
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
                
                performanceMonitor.Update();
                
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
                
                // パフォーマンス監視データを取得
                metrics.PeakMemoryUsage = performanceMonitor.GetPeakMemoryUsage();
                metrics.AverageCpuUsage = performanceMonitor.GetAverageCpuUsage();
                metrics.ProcessedFrames = metrics.FrameCount;
                metrics.FramesPerSecond = metrics.FrameCount / metrics.TotalTime.TotalSeconds;
                
                Console.WriteLine("\n=== 処理時間計測結果 ===");
                Console.WriteLine($"総処理時間: {metrics.TotalTime.TotalMilliseconds:F2} ms ({metrics.TotalTime.TotalSeconds:F3} 秒)");
                Console.WriteLine($"初期化時間: {metrics.InitializationTime.TotalMilliseconds:F2} ms");
                Console.WriteLine($"動画変換時間: {metrics.VideoConversionTime.TotalMilliseconds:F2} ms");
                Console.WriteLine($"血圧推定時間: {metrics.BloodPressureAnalysisTime.TotalMilliseconds:F2} ms");
                Console.WriteLine($"コールバック時間: {metrics.CallbackTime.TotalMilliseconds:F2} ms");
                Console.WriteLine($"動画ファイルサイズ: {metrics.VideoFileSize / 1024 / 1024.0:F2} MB");
                Console.WriteLine($"フレーム数: {metrics.FrameCount}");
                Console.WriteLine($"フレーム処理速度: {metrics.FramesPerSecond:F1} fps");
                Console.WriteLine($"ピークメモリ使用量: {metrics.PeakMemoryUsage / (1024.0 * 1024.0):F2} MB");
                Console.WriteLine($"平均CPU使用率: {metrics.AverageCpuUsage:F1}%");
                
                // 詳細なボトルネック分析を実行
                AnalyzeBottlenecks(metrics);
                AnalyzeMemoryUsage(metrics);
                AnalyzeCpuUsage(metrics);
                AnalyzeFrameProcessing(metrics);
                
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
                
                Console.WriteLine($"\n[SUCCESS] 血圧推定処理時間計測テスト完了");
                Console.WriteLine("=== 血圧推定処理時間計測テスト完了 ===");
                
                performanceMonitor.Dispose();
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
                performanceMonitor?.Dispose();
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
                    writer.WriteLine($"Frames Per Second: {performanceMetrics.FramesPerSecond:F1} fps");
                    writer.WriteLine($"Peak Memory Usage: {performanceMetrics.PeakMemoryUsage / (1024.0 * 1024.0):F2} MB");
                    writer.WriteLine($"Average CPU Usage: {performanceMetrics.AverageCpuUsage:F1}%");
                    writer.WriteLine($"Estimated SBP: {performanceMetrics.EstimatedSBP} mmHg");
                    writer.WriteLine($"Estimated DBP: {performanceMetrics.EstimatedDBP} mmHg");
                    
                    // 詳細タイミング情報
                    writer.WriteLine("\n=== Detailed Timings ===");
                    foreach (var timing in performanceMetrics.DetailedTimings)
                    {
                        writer.WriteLine($"{timing.Key}: {timing.Value.TotalMilliseconds:F2} ms");
                    }
                    
                    // パフォーマンス分析
                    writer.WriteLine("\n=== Performance Analysis ===");
                    double totalTime = performanceMetrics.TotalTime.TotalMilliseconds;
                    double targetTime = 3000.0;
                    
                    writer.WriteLine($"Target Time: {targetTime:F0} ms");
                    writer.WriteLine($"Actual Time: {totalTime:F2} ms");
                    
                    if (totalTime <= targetTime)
                    {
                        double performanceRatio = (targetTime / totalTime) * 100;
                        writer.WriteLine($"Performance: {performanceRatio:F1}% of target (ACHIEVED)");
                    }
                    else
                    {
                        double overTime = totalTime - targetTime;
                        double overPercentage = (overTime / targetTime) * 100;
                        writer.WriteLine($"Performance: {overPercentage:F1}% over target (EXCEEDED)");
                    }
                    
                    // ボトルネック分析
                    writer.WriteLine("\n=== Bottleneck Analysis ===");
                    var timings = new Dictionary<string, double>
                    {
                        {"Initialization", performanceMetrics.InitializationTime.TotalMilliseconds},
                        {"Video Conversion", performanceMetrics.VideoConversionTime.TotalMilliseconds},
                        {"Blood Pressure Analysis", performanceMetrics.BloodPressureAnalysisTime.TotalMilliseconds},
                        {"Callback", performanceMetrics.CallbackTime.TotalMilliseconds}
                    };
                    
                    var sortedTimings = timings.OrderByDescending(x => x.Value).ToList();
                    for (int i = 0; i < sortedTimings.Count; i++)
                    {
                        var timing = sortedTimings[i];
                        double percentage = (timing.Value / totalTime) * 100;
                        writer.WriteLine($"{i + 1}. {timing.Key}: {timing.Value:F2} ms ({percentage:F1}%)");
                    }
                    
                    // 改善提案
                    writer.WriteLine("\n=== Improvement Suggestions ===");
                    foreach (var timing in sortedTimings.Take(3))
                    {
                        double percentage = (timing.Value / totalTime) * 100;
                        
                        switch (timing.Key)
                        {
                            case "Initialization":
                                if (percentage > 20)
                                {
                                    writer.WriteLine("- Initialization time exceeds 20% of total time");
                                    writer.WriteLine(" Consider model pre-loading and caching");
                                    writer.WriteLine(" Consider switching to lighter models");
                                }
                                break;
                            case "Video Conversion":
                                if (percentage > 30)
                                {
                                    writer.WriteLine("- Video conversion time exceeds 30% of total time");
                                    writer.WriteLine(" Consider pre-converting videos or adjusting compression");
                                    writer.WriteLine(" Consider using faster codecs");
                                    writer.WriteLine(" Consider reducing frame rate");
                                }
                                break;
                            case "Blood Pressure Analysis":
                                if (percentage > 40)
                                {
                                    writer.WriteLine("- Blood pressure analysis time exceeds 40% of total time");
                                    writer.WriteLine(" Consider model optimization and quantization");
                                    writer.WriteLine(" Consider reducing frame count");
                                    writer.WriteLine(" Consider implementing GPU acceleration");
                                }
                                break;
                        }
                    }
                    
                    // システム情報
                    writer.WriteLine("\n=== System Information ===");
                    writer.WriteLine($"OS: {Environment.OSVersion}");
                    writer.WriteLine($"Processor Count: {Environment.ProcessorCount}");
                    writer.WriteLine($"Working Set: {Environment.WorkingSet / (1024.0 * 1024.0):F2} MB");
                    writer.WriteLine($"Is 64-bit Process: {Environment.Is64BitProcess}");
                    
                    // エラー情報
                    if (!string.IsNullOrEmpty(performanceMetrics.ErrorMessage))
                    {
                        writer.WriteLine("\n=== Error Information ===");
                        writer.WriteLine($"Error: {performanceMetrics.ErrorMessage}");
                    }
                }
                
                Console.WriteLine($"詳細なパフォーマンスログを保存しました: {performanceLogFile}");
                
                // 簡易結果表示
                Console.WriteLine("\n=== 簡易結果 ===");
                Console.WriteLine($"成功: {performanceMetrics.IsSuccess}");
                Console.WriteLine($"総処理時間: {performanceMetrics.TotalTime.TotalMilliseconds:F2} ms");
                Console.WriteLine($"推定血圧: SBP={performanceMetrics.EstimatedSBP} mmHg, DBP={performanceMetrics.EstimatedDBP} mmHg");
                Console.WriteLine($"フレーム処理速度: {performanceMetrics.FramesPerSecond:F1} fps");
                Console.WriteLine($"ピークメモリ使用量: {performanceMetrics.PeakMemoryUsage / (1024.0 * 1024.0):F2} MB");
                
                if (performanceMetrics.IsSuccess)
                {
                    Console.WriteLine("[SUCCESS] 血圧推定テスト完了");
                }
                else
                {
                    Console.WriteLine($"[ERROR] 血圧推定テスト失敗: {performanceMetrics.ErrorMessage}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[FATAL ERROR] テスト実行中に例外が発生しました: {ex.Message}");
                Console.WriteLine($"例外の種類: {ex.GetType().Name}");
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

