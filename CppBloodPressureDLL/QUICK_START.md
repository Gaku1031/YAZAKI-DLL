# Blood Pressure DLL - クイックスタートガイド

## 🚀 ダウンロード後の手順

### 1. ファイルの解凍

GitHub Actions の artifacts からダウンロードした zip ファイルを解凍してください。

### 2. ディレクトリ構造の確認

解凍後、以下のファイルが含まれていることを確認してください：

```
BloodPressureDLL-Complete-Release-xxxxx/
├── BloodPressureDLL.dll          # メインDLL
├── opencv_world480.dll           # OpenCVランタイム
├── onnxruntime.dll               # ONNX Runtime
├── zlib.dll                      # ZLIBライブラリ
├── CSharpTest.exe                # テストアプリケーション
├── README.txt                    # パッケージ説明
├── FILE_LIST.txt                 # ファイル一覧
├── models/                       # モデルファイル
│   ├── opencv_face_detector_uint8.pb
│   ├── opencv_face_detector.pbtxt
│   ├── model_sbp.onnx
│   └── model_dbp.onnx
├── sample-data/                  # サンプルデータ
│   └── sample_1M.webm
└── docs/                         # ドキュメント
    ├── INTEGRATION_GUIDE.md
    └── README.md
```

### 3. テスト実行

1. コマンドプロンプトを開く
2. 解凍したディレクトリに移動
3. テストアプリケーションを実行：
   ```cmd
   CSharpTest.exe
   ```

### 4. 期待される出力

正常に動作する場合、以下のような出力が表示されます：

```
C++ Blood Pressure DLL テスト開始
=== C++ Blood Pressure DLL テスト ===

1. DLL初期化テスト
   初期化結果: 1

2. バージョン情報取得テスト
   バージョン: BloodPressureDLL v1.0.0

3. リクエストID生成テスト
   生成されたID: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

4. 処理状況取得テスト
   処理状況: none

5. 血圧解析テスト
   サンプル動画: sample_video.webm (1.0 MB)
   解析開始結果: 成功
   解析開始成功 - 処理を監視中...
   処理状況 10秒: processing
   処理状況 20秒: processing
   [SUCCESS] 処理完了 (25秒)

=== 血圧解析結果 ===
Request ID: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
最高血圧: 120 mmHg
最低血圧: 80 mmHg
CSVデータサイズ: 2048 文字
[SUCCESS] 血圧推定成功 - 妥当な値が取得されました
[SUCCESS] エラーなし
[SUCCESS] CSVファイル保存: bp_result_xxxxx_20241201_143022.csv

6. 中断テスト
   中断結果: 1

7. テスト結果サマリー
   [SUCCESS] DLL初期化: 成功
   [SUCCESS] バージョン取得: 成功
   [SUCCESS] リクエストID生成: 成功
   [SUCCESS] 処理状況取得: 成功
   [SUCCESS] 血圧解析: 実行済み
   [SUCCESS] 中断機能: テスト済み

=== 全てのテストが完了しました ===
テスト完了
```

## 🔧 C#アプリケーションへの組み込み

### 基本的な統合手順

1. **ファイルの配置**

   - すべての DLL ファイルを C#アプリケーションの実行ディレクトリにコピー
   - `models`ディレクトリも同じ場所に配置

2. **C#コードの追加**

   ```csharp
   using System.Runtime.InteropServices;

   [DllImport("BloodPressureDLL.dll", CallingConvention = CallingConvention.Cdecl)]
   public static extern int InitializeBP(string modelDir);

   [DllImport("BloodPressureDLL.dll", CallingConvention = CallingConvention.Cdecl)]
   public static extern int StartBloodPressureAnalysisRequest(
       string requestId, int height, int weight, int age,
       string videoPath, BPCallback callback);
   ```

3. **詳細な統合方法**
   - `docs/INTEGRATION_GUIDE.md`を参照してください

## ⚠️ トラブルシューティング

### よくある問題

#### DLL が見つからないエラー

```
Unable to load DLL 'BloodPressureDLL.dll' or one of its dependencies
```

**解決方法:**

- すべての DLL ファイルが同じディレクトリにあることを確認
- 32bit/64bit のアーキテクチャが一致していることを確認

#### 初期化エラー

```
初期化結果: 0
```

**解決方法:**

- `models`ディレクトリが存在することを確認
- すべてのモデルファイルが含まれていることを確認

#### 処理が完了しない

```
処理状況 120秒: processing
[WARNING] 処理が120秒を超えました
```

**解決方法:**

- 動画ファイルの形式を確認（WebM、MP4 推奨）
- 動画の解像度を下げる（1280x720 以下推奨）
- 十分なメモリが利用可能か確認

## 📋 システム要件

- **OS**: Windows 10/11 (64bit)
- **.NET**: 6.0 以上
- **メモリ**: 最低 4GB RAM（推奨 8GB 以上）
- **ストレージ**: 500MB 以上の空き容量

## 📞 サポート

問題が解決しない場合は、以下を確認してください：

1. **ログファイルの確認**

   - テスト実行時に生成される CSV ファイル
   - エラーメッセージの詳細

2. **環境の確認**

   - Windows Update が最新か
   - Visual C++ Redistributable がインストールされているか

3. **ファイルの整合性**
   - ダウンロードしたファイルが破損していないか
   - アンチウイルスソフトがファイルをブロックしていないか

詳細なドキュメントは `docs/` ディレクトリを参照してください。
