# 血圧推定 DLL

## 1. 同梱物一覧

- `BloodPressureDLL.dll` : 血圧推定 DLL 本体（約 160KB）
- `onnxruntime.dll` : ONNX 推論エンジン（約 12MB）
- `onnxruntime_providers_shared.dll` : ONNX ランタイム依存 DLL（約 22KB）
- `opencv_world480.dll` : OpenCV ランタイム（約 60MB）
- `zlib.dll` : ZLIB ライブラリ（約 83KB）
- `ffmpeg.exe` : 動画フレーム抽出用コマンドラインツール（約 83MB）
- `models/` : 推論モデル・顔検出モデル
  - `systolicbloodpressure.onnx`（約 2.7MB）
  - `diastolicbloodpressure.onnx`（約 2.6MB）
  - `opencv_face_detector_uint8.pb`（約 2.6MB）
  - `opencv_face_detector.pbtxt`（約 36KB）
- `SampleApp.cs` : C#サンプルアプリ（組み込み例）

### ファイルサイズは目安です。実際のバージョンにより多少異なる場合があります。

## ffmpeg.exe について

- `ffmpeg.exe` は、動画ファイル（WebM 形式）から画像フレームを抽出するために DLL 内部で使用されます。
- **Windows 環境で動画解析を行う場合は必須**です。
- ただし、
  - 画像シーケンスのみで推論する場合
  - または ffmpeg を別途システムパスに用意している場合
    には `ffmpeg.exe` を削除しても動作します。
- 通常は package フォルダに同梱したままご利用ください。

## 2. セットアップ手順

- `package`フォルダの内容を、アプリケーション実行ディレクトリに**全てコピー**してください。
- `models`フォルダも必ず同じ階層に配置してください。

## 3. DLL 関数の呼び出し例

`SampleApp.cs` では以下の流れで DLL を利用しています：

1. **DLL 初期化**
   ```csharp
   var outBuf = new StringBuilder(256);
   int result = InitializeBP(outBuf, outBuf.Capacity, "models");
   ```
2. **バージョン情報取得**
   ```csharp
   int verResult = GetVersionInfo(outBuf, outBuf.Capacity);
   ```
3. **血圧推定リクエスト送信**
   ```csharp
   int reqResult = StartBloodPressureAnalysisRequest(
       outBuf, outBuf.Capacity,
       requestId, height, weight, sex,
       videoPath, OnBPResult);
   ```
4. **コールバック受信**
   ```csharp
   public static void OnBPResult(string requestId, int maxBP, int minBP, string csv, string errorsJson)
   {
       // 結果処理
   }
   ```
