using System;
using System.Runtime.InteropServices;
using System.Text;

namespace BloodPressureDllTest
{
    /// <summary>
    /// C#から血圧推定DLLを呼び出すためのサンプルコード
    /// </summary>
    public class BloodPressureDll
    {
        // DLLファイルのパス（同じディレクトリにあることを想定）
        private const string DllPath = "BloodPressureEstimation.dll";

        // コールバック関数の型定義
        [UnmanagedFunctionPointer(CallingConvention.StdCall)]
        public delegate void AnalysisCallback(
            [MarshalAs(UnmanagedType.LPStr)] string requestId,
            int maxBloodPressure,
            int minBloodPressure,
            [MarshalAs(UnmanagedType.LPStr)] string measureRowData,
            IntPtr errors
        );

        // DLL関数のインポート
        [DllImport(DllPath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern bool InitializeDLL([MarshalAs(UnmanagedType.LPStr)] string modelDir);

        [DllImport(DllPath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        [return: MarshalAs(UnmanagedType.LPStr)]
        public static extern string StartBloodPressureAnalysisRequest(
            [MarshalAs(UnmanagedType.LPStr)] string requestId,
            int height,
            int weight,
            int sex,
            [MarshalAs(UnmanagedType.LPStr)] string moviePath,
            AnalysisCallback callback
        );

        [DllImport(DllPath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        [return: MarshalAs(UnmanagedType.LPStr)]
        public static extern string GetProcessingStatus([MarshalAs(UnmanagedType.LPStr)] string requestId);

        [DllImport(DllPath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        public static extern bool CancelBloodPressureAnalysis([MarshalAs(UnmanagedType.LPStr)] string requestId);

        [DllImport(DllPath, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        [return: MarshalAs(UnmanagedType.LPStr)]
        public static extern string GetVersionInfo();

        /// <summary>
        /// DLLテスト実行
        /// </summary>
        public static void TestDLL()
        {
            Console.WriteLine("=== 血圧推定DLL C#テスト開始 ===");

            try
            {
                // 1. DLL初期化
                Console.WriteLine("1. DLL初期化");
                bool initResult = InitializeDLL("models");
                Console.WriteLine($"   初期化結果: {initResult}");

                if (!initResult)
                {
                    Console.WriteLine("DLL初期化に失敗しました");
                    return;
                }

                // 2. バージョン情報取得
                Console.WriteLine("2. バージョン情報取得");
                string version = GetVersionInfo();
                Console.WriteLine($"   バージョン: {version}");

                // 3. 処理状況取得テスト
                Console.WriteLine("3. 処理状況取得テスト");
                string status = GetProcessingStatus("dummy_request");
                Console.WriteLine($"   状況: {status}");

                // 4. リクエストID生成（C#側で実装）
                Console.WriteLine("4. リクエストID生成");
                string requestId = GenerateRequestId("9000000001", "0000012345");
                Console.WriteLine($"   リクエストID: {requestId}");

                // 5. 血圧解析リクエスト（無効パラメータでテスト）
                Console.WriteLine("5. 血圧解析リクエスト（無効パラメータ）");
                
                // コールバック関数定義
                AnalysisCallback callback = (reqId, sbp, dbp, csvData, errors) =>
                {
                    Console.WriteLine($"   コールバック呼び出し:");
                    Console.WriteLine($"     リクエストID: {reqId}");
                    Console.WriteLine($"     最高血圧: {sbp}");
                    Console.WriteLine($"     最低血圧: {dbp}");
                    Console.WriteLine($"     CSVデータサイズ: {csvData?.Length ?? 0}");
                };

                string errorCode = StartBloodPressureAnalysisRequest(
                    "invalid_id", 170, 70, 1, "nonexistent.webm", callback);
                Console.WriteLine($"   エラーコード: {errorCode}");

                // 6. 中断機能テスト
                Console.WriteLine("6. 血圧解析中断テスト");
                bool cancelResult = CancelBloodPressureAnalysis("dummy_request");
                Console.WriteLine($"   中断結果: {cancelResult}");

                Console.WriteLine("=== DLLテスト完了 ===");
            }
            catch (DllNotFoundException ex)
            {
                Console.WriteLine($"DLLが見つかりません: {ex.Message}");
                Console.WriteLine("BloodPressureEstimation.dll が同じディレクトリにあることを確認してください");
            }
            catch (EntryPointNotFoundException ex)
            {
                Console.WriteLine($"エントリポイントが見つかりません: {ex.Message}");
                Console.WriteLine("DLLに必要なエクスポート関数が含まれていない可能性があります");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"エラー: {ex.Message}");
            }
        }

        /// <summary>
        /// README.md準拠のリクエストID生成
        /// </summary>
        /// <param name="customerCode">顧客コード（10桁）</param>
        /// <param name="driverCode">乗務員コード（10桁）</param>
        /// <returns>生成されたリクエストID</returns>
        public static string GenerateRequestId(string customerCode, string driverCode)
        {
            string timestamp = DateTime.Now.ToString("yyyyMMddHHmmssfff");
            return $"{timestamp}_{customerCode}_{driverCode}";
        }

        /// <summary>
        /// メイン関数
        /// </summary>
        public static void Main(string[] args)
        {
            Console.WriteLine("血圧推定DLL C#テストプログラム");
            Console.WriteLine("README.md準拠、64bit対応版");
            Console.WriteLine();

            TestDLL();

            Console.WriteLine();
            Console.WriteLine("Enterキーを押して終了...");
            Console.ReadLine();
        }
    }
}