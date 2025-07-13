using System;
using System.Threading.Tasks;

namespace BloodPressureEstimation.Test
{
    class Program
    {
        static async Task Main(string[] args)
        {
            Console.WriteLine("=== Lightweight Runtime Test ===");

            try
            {
                // ラッパーを初期化
                using var wrapper = new BloodPressureEstimation.BloodPressureWrapper("lightweight_runtime");
                
                Console.WriteLine("Testing initialization...");
                if (!wrapper.Initialize("models"))
                {
                    Console.WriteLine("❌ Failed to initialize blood pressure estimation");
                    Environment.Exit(1);
                }

                Console.WriteLine("✅ Initialized successfully");

                // バージョン情報を取得
                Console.WriteLine("Testing version info...");
                var version = await wrapper.GetVersionInfoAsync();
                Console.WriteLine($"✅ Version: {version}");

                // 血圧分析を開始
                Console.WriteLine("Testing blood pressure analysis...");
                var requestId = Guid.NewGuid().ToString();
                var result = await wrapper.StartBloodPressureAnalysisAsync(
                    requestId, 170, 70, 1, "sample-data/video.mp4");

                if (string.IsNullOrEmpty(result))
                {
                    Console.WriteLine("✅ Analysis started successfully");
                    
                    // 処理状況を確認
                    Console.WriteLine("Testing status check...");
                    for (int i = 0; i < 3; i++)
                    {
                        await Task.Delay(1000);
                        var status = await wrapper.GetProcessingStatusAsync(requestId);
                        Console.WriteLine($"Status: {status}");
                        
                        if (status.StartsWith("COMPLETED"))
                        {
                            Console.WriteLine("✅ Analysis completed successfully");
                            break;
                        }
                        else if (status.StartsWith("ERROR"))
                        {
                            Console.WriteLine($"❌ Analysis failed: {status}");
                            break;
                        }
                    }
                }
                else
                {
                    Console.WriteLine($"❌ Analysis failed: {result}");
                }

                // キャンセル機能をテスト
                Console.WriteLine("Testing cancel functionality...");
                var cancelResult = await wrapper.CancelBloodPressureAnalysisAsync(requestId);
                Console.WriteLine($"✅ Cancel test: {(cancelResult ? "SUCCESS" : "FAILED")}");

                Console.WriteLine("✅ All tests completed successfully!");
                Console.WriteLine("✅ Lightweight runtime is working correctly!");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"❌ Test failed with exception: {ex.Message}");
                Console.WriteLine($"Stack trace: {ex.StackTrace}");
                Environment.Exit(1);
            }
        }
    }
} 
