using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;

public class SimpleBPTest
{
    private const string DllPath = "BloodPressureEstimation.dll";

    // DLL関数のインポート
    [DllImport(DllPath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern bool InitializeDLL([MarshalAs(UnmanagedType.LPStr)] string modelDir);

    [DllImport(DllPath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    [return: MarshalAs(UnmanagedType.LPStr)]
    public static extern string StartBloodPressureAnalysisRequest(
        [MarshalAs(UnmanagedType.LPStr)] string requestId,
        int height, int weight, int sex,
        [MarshalAs(UnmanagedType.LPStr)] string moviePath,
        IntPtr callback);

    [DllImport(DllPath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    [return: MarshalAs(UnmanagedType.LPStr)]
    public static extern string GetProcessingStatus([MarshalAs(UnmanagedType.LPStr)] string requestId);

    [DllImport(DllPath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern bool CancelBloodPressureAnalysis([MarshalAs(UnmanagedType.LPStr)] string requestId);

    [DllImport(DllPath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    [return: MarshalAs(UnmanagedType.LPStr)]
    public static extern string GetVersionInfo();

    public static void Main(string[] args)
    {
        Console.WriteLine("=== シンプル血圧推定DLLテスト ===");
        Console.WriteLine();

        try
        {
            // 1. 環境確認
            Console.WriteLine("1. 環境確認");
            if (!CheckFiles())
            {
                Console.WriteLine("必要ファイルが不足しています");
                Console.WriteLine("Enterキーで終了...");
                Console.ReadLine();
                return;
            }

            // 2. DLL初期化
            Console.WriteLine();
            Console.WriteLine("2. DLL初期化");
            bool initResult = InitializeDLL("models");
            Console.WriteLine($"初期化結果: {initResult}");

            if (!initResult)
            {
                Console.WriteLine("DLL初期化に失敗しました");
                Console.WriteLine("Enterキーで終了...");
                Console.ReadLine();
                return;
            }

            // 3. バージョン確認
            Console.WriteLine();
            Console.WriteLine("3. バージョン確認");
            string version = GetVersionInfo();
            Console.WriteLine($"DLLバージョン: {version}");

            // 4. 処理状況テスト
            Console.WriteLine();
            Console.WriteLine("4. 処理状況テスト");
            string status = GetProcessingStatus("test_request");
            Console.WriteLine($"処理状況: {status}");

            // 5. 血圧解析テスト（基本）
            Console.WriteLine();
            Console.WriteLine("5. 血圧解析テスト");
            
            string requestId = GenerateRequestId();
            Console.WriteLine($"リクエストID: {requestId}");

            string videoPath = @"sample-data\100万画素.webm";
            Console.WriteLine($"動画ファイル: {videoPath}");

            if (!File.Exists(videoPath))
            {
                Console.WriteLine($"動画ファイルが見つかりません: {videoPath}");
                Console.WriteLine("Enterキーで終了...");
                Console.ReadLine();
                return;
            }

            // 解析開始（コールバックなし）
            string errorCode = StartBloodPressureAnalysisRequest(
                requestId, 170, 70, 1, videoPath, IntPtr.Zero);

            if (string.IsNullOrEmpty(errorCode))
            {
                Console.WriteLine("解析リクエスト送信成功");
                
                // 処理監視
                Console.WriteLine("処理状況監視中...");
                for (int i = 0; i < 30; i++)
                {
                    Thread.Sleep(1000);
                    status = GetProcessingStatus(requestId);
                    Console.Write($"\r状況: {status} ({i + 1}s)");
                    
                    if (status == "none")
                    {
                        break;
                    }
                }
                Console.WriteLine();
                Console.WriteLine("解析完了（またはタイムアウト）");
            }
            else
            {
                Console.WriteLine($"解析エラー: {errorCode}");
            }

            Console.WriteLine();
            Console.WriteLine("=== テスト完了 ===");
        }
        catch (DllNotFoundException ex)
        {
            Console.WriteLine($"DLLが見つかりません: {ex.Message}");
            Console.WriteLine("BloodPressureEstimation.dll が同じディレクトリにあることを確認してください");
        }
        catch (EntryPointNotFoundException ex)
        {
            Console.WriteLine($"エントリポイントが見つかりません: {ex.Message}");
            Console.WriteLine("DLLのエクスポート関数を確認してください");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"エラー: {ex.Message}");
            Console.WriteLine($"詳細: {ex}");
        }

        Console.WriteLine();
        Console.WriteLine("Enterキーで終了...");
        Console.ReadLine();
    }

    static bool CheckFiles()
    {
        Console.WriteLine("必要ファイル確認中...");

        if (!File.Exists("BloodPressureEstimation.dll"))
        {
            Console.WriteLine("✗ BloodPressureEstimation.dll が見つかりません");
            return false;
        }
        Console.WriteLine("✓ BloodPressureEstimation.dll");

        if (!File.Exists("bp_estimation_simple.py"))
        {
            Console.WriteLine("✗ bp_estimation_simple.py が見つかりません");
            return false;
        }
        Console.WriteLine("✓ bp_estimation_simple.py");

        if (!File.Exists(@"sample-data\100万画素.webm"))
        {
            Console.WriteLine("✗ sample-data\\100万画素.webm が見つかりません");
            return false;
        }
        Console.WriteLine("✓ sample-data\\100万画素.webm");

        return true;
    }

    static string GenerateRequestId()
    {
        string timestamp = DateTime.Now.ToString("yyyyMMddHHmmssfff");
        return $"{timestamp}_9000000001_0000012345";
    }
}