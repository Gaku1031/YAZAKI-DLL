using System;
using System.Runtime.InteropServices;
using System.Text;

class SampleApp
{
    // DLLのパス
    const string DllName = "BloodPressureDLL.dll";

    // コールバックデリゲート
    [UnmanagedFunctionPointer(CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public delegate void BPCallback(
        [MarshalAs(UnmanagedType.LPStr)] string requestId,
        int maxBloodPressure,
        int minBloodPressure,
        [MarshalAs(UnmanagedType.LPStr)] string measureRowData,
        [MarshalAs(UnmanagedType.LPStr)] string errorsJson
    );

    // DLL関数のインポート
    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern int InitializeBP([Out] StringBuilder outBuf, int bufSize, [MarshalAs(UnmanagedType.LPStr)] string modelDir);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern int StartBloodPressureAnalysisRequest(
        [Out] StringBuilder outBuf, int bufSize,
        [MarshalAs(UnmanagedType.LPStr)] string requestId,
        int height, int weight, int sex,
        [MarshalAs(UnmanagedType.LPStr)] string moviePath,
        BPCallback callback);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
    public static extern int GetVersionInfo([Out] StringBuilder outBuf, int bufSize);

    // コールバック関数の実装
    public static void OnBPResult(string requestId, int maxBP, int minBP, string csv, string errorsJson)
    {
        Console.WriteLine("【コールバック受信】");
        Console.WriteLine($"リクエストID: {requestId}");
        Console.WriteLine($"最高血圧: {maxBP} mmHg");
        Console.WriteLine($"最低血圧: {minBP} mmHg");
        Console.WriteLine($"CSVデータ長: {csv?.Length ?? 0} 文字");
        Console.WriteLine($"エラー情報: {errorsJson}");
    }

    static void Main(string[] args)
    {
        // 1. DLL初期化
        var outBuf = new StringBuilder(256);
        int initResult = InitializeBP(outBuf, outBuf.Capacity, "models");
        Console.WriteLine($"DLL初期化: {(initResult == 0 ? "成功" : "失敗")} / メッセージ: {outBuf}");

        if (initResult != 0)
        {
            Console.WriteLine("DLL初期化に失敗しました。");
            return;
        }

        // 2. バージョン情報取得
        outBuf.Clear();
        int verResult = GetVersionInfo(outBuf, outBuf.Capacity);
        Console.WriteLine($"バージョン情報: {outBuf}");

        // 3. 血圧推定リクエスト
        string requestId = DateTime.Now.ToString("yyyyMMddHHmmssfff") + "_9000000001_0000012345";
        string videoPath = @"sample-data\sample_1M.webm"; // フルパスまたは相対パス
        int height = 170;
        int weight = 60;
        int sex = 1; // 1:男性, 2:女性

        outBuf.Clear();
        int reqResult = StartBloodPressureAnalysisRequest(
            outBuf, outBuf.Capacity,
            requestId, height, weight, sex,
            videoPath, OnBPResult);

        if (reqResult != 0)
        {
            Console.WriteLine($"血圧推定リクエスト失敗: {outBuf}");
        }
        else
        {
            Console.WriteLine("血圧推定リクエスト送信成功。コールバックを待機中...");
        }

        // コールバックが非同期で返るため、適宜待機
        Console.WriteLine("Enterキーで終了します。");
        Console.ReadLine();
    }
} 
