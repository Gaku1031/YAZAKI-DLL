# C++ Blood Pressure DLL - C#統合ガイド

## 概要

このガイドでは、C++で作成された血圧推定 DLL を C#アプリケーションに統合する方法を説明します。

## 必要なファイル

### 必須ファイル

```
BloodPressureDLL.dll          # メインのDLL
opencv_world480.dll           # OpenCVランタイム
onnxruntime.dll               # ONNX Runtime
zlib.dll                      # ZLIBライブラリ
```

### モデルファイル

```
models/
├── opencv_face_detector_uint8.pb  # OpenCV DNN顔検出モデル
├── opencv_face_detector.pbtxt     # OpenCV DNN設定
├── model_sbp.onnx                 # 収縮期血圧推定モデル
└── model_dbp.onnx                 # 拡張期血圧推定モデル
```

## ディレクトリ構造

### 推奨構造

```
YourApp/
├── YourApp.exe
├── BloodPressureDLL.dll
├── opencv_world480.dll
├── onnxruntime.dll
├── zlib.dll
├── models/
│   ├── opencv_face_detector_uint8.pb
│   ├── opencv_face_detector.pbtxt
│   ├── model_sbp.onnx
│   └── model_dbp.onnx
└── sample-data/
    └── sample_1M.webm
```

## C#コード例

### 基本的な使用方法

```csharp
using System;
using System.Runtime.InteropServices;

namespace YourApp
{
    public class BloodPressureEstimator
    {
        // DLLインポート
        [DllImport("BloodPressureDLL.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int InitializeBP(string modelDir);

        [DllImport("BloodPressureDLL.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int StartBloodPressureAnalysisRequest(
            string requestId, int height, int weight, int age,
            string videoPath, BPCallback callback);

        [DllImport("BloodPressureDLL.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern string GetProcessingStatus(string requestId);

        [DllImport("BloodPressureDLL.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern int CancelBloodPressureAnalysis(string requestId);

        [DllImport("BloodPressureDLL.dll", CallingConvention = CallingConvention.Cdecl)]
        public static extern string GetVersionInfo();

        // コールバック関数のデリゲート
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void BPCallback(string requestId, int maxBP, int minBP,
                                       string csvData, string errorsJson);

        // コールバック実装
        public static void OnBloodPressureResult(string requestId, int maxBP, int minBP,
                                                string csvData, string errorsJson)
        {
            Console.WriteLine($"血圧推定結果 - SBP: {maxBP}, DBP: {minBP}");
            // 結果を処理
        }

        public void EstimateBloodPressure(string videoPath, int height, int weight, int age)
        {
            try
            {
                // 1. 初期化
                string modelDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "models");
                int initResult = InitializeBP(modelDir);

                if (initResult == 0)
                {
                    throw new Exception("DLL初期化に失敗しました");
                }

                // 2. リクエストID生成
                string requestId = Guid.NewGuid().ToString();

                // 3. 血圧解析開始
                BPCallback callback = OnBloodPressureResult;
                int result = StartBloodPressureAnalysisRequest(
                    requestId, height, weight, age, videoPath, callback);

                if (result != 0)
                {
                    throw new Exception("血圧解析の開始に失敗しました");
                }

                // 4. 処理状況の監視
                while (true)
                {
                    string status = GetProcessingStatus(requestId);
                    if (status == "none")
                    {
                        break; // 処理完了
                    }
                    Thread.Sleep(1000);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"エラー: {ex.Message}");
            }
        }
    }
}
```

## トラブルシューティング

### よくある問題

#### 1. DLL が見つからないエラー

```
Unable to load DLL 'BloodPressureDLL.dll' or one of its dependencies
```

**解決方法:**

- すべての依存 DLL が実行ファイルと同じディレクトリにあることを確認
- 32bit/64bit のアーキテクチャが一致していることを確認

#### 2. モデルファイルが見つからないエラー

```
モデルディレクトリが見つかりません
```

**解決方法:**

- `models`ディレクトリが正しい場所にあることを確認
- すべてのモデルファイルが存在することを確認

#### 3. メモリ不足エラー

```
Out of memory
```

**解決方法:**

- 大きな動画ファイルを処理する場合は、十分なメモリを確保
- 動画の解像度を下げることを検討

## パフォーマンス最適化

### 推奨設定

- **動画解像度**: 640x480 または 1280x720
- **フレームレート**: 30fps
- **動画長**: 30 秒〜2 分
- **メモリ**: 最低 4GB 推奨

### 処理時間の目安

- **30 秒動画**: 約 10-30 秒
- **1 分動画**: 約 30-60 秒
- **2 分動画**: 約 60-120 秒

## ライセンスと制限事項

- この DLL は研究・開発目的での使用を想定しています
- 商用利用の場合は、別途ライセンス契約が必要です
- OpenCV、ONNX Runtime、MediaPipe のライセンスも確認してください

## サポート

問題が発生した場合は、以下を確認してください：

1. すべての依存ファイルが正しく配置されているか
2. アーキテクチャ（32bit/64bit）が一致しているか
3. 動画ファイルがサポートされている形式か
4. 十分なメモリが利用可能か

詳細なログを確認するには、テストアプリケーションを実行してください。
