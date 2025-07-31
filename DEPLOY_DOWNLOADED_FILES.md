# ダウンロードされたファイルの配置手順

このドキュメントは、GitHub Actions でビルドされた血圧推定 DLL ファイルをリポジトリに配置する手順を説明します。

## ダウンロードされたファイル一覧

以下のファイルが GitHub Actions からダウンロードされました：

```
.
├── BloodPressureDLL.dll          # メインの血圧推定DLL
├── INTEGRATION_GUIDE.md          # 統合ガイド
├── README.md                     # READMEファイル
├── abseil_dll.dll               # Abseilライブラリ
├── include
│   └── BloodPressureDLL.h       # C++ヘッダーファイル
├── jpeg62.dll                   # JPEGライブラリ
├── libgcc_s_seh-1.dll          # GCCランタイム
├── libgfortran-5.dll           # Fortranライブラリ
├── liblapack.dll               # LAPACKライブラリ
├── liblzma.dll                 # LZMA圧縮ライブラリ
├── libpng16.dll                # PNGライブラリ
├── libprotobuf.dll             # Protocol Buffersライブラリ
├── libquadmath-0.dll           # Quadmathライブラリ
├── libsharpyuv.dll             # SharpYUVライブラリ
├── libwebp.dll                 # WebPライブラリ
├── libwebpdecoder.dll          # WebPデコーダー
├── libwebpdemux.dll            # WebPデマックス
├── libwebpmux.dll              # WebPマックス
├── libwinpthread-1.dll         # Windows POSIXスレッド
├── models
│   ├── diastolicbloodpressure.onnx    # 拡張期血圧推定モデル
│   ├── opencv_face_detector.pbtxt     # OpenCV顔検出設定
│   ├── opencv_face_detector_uint8.pb  # OpenCV顔検出モデル
│   ├── shape_predictor_68_face_landmarks.dat  # 顔ランドマーク検出
│   └── systolicbloodpressure.onnx     # 収縮期血圧推定モデル
├── onnxruntime.dll             # ONNX Runtime
├── openblas.dll               # OpenBLASライブラリ
├── opencv_core4.dll           # OpenCVコア
├── opencv_dnn4.dll            # OpenCV DNN
├── opencv_imgcodecs4.dll      # OpenCV画像コーデック
├── opencv_imgproc4.dll        # OpenCV画像処理
├── opencv_objdetect4.dll      # OpenCV物体検出
├── tiff.dll                   # TIFFライブラリ
└── zlib1.dll                  # ZLIB圧縮ライブラリ
```

## 配置手順

### 1. 自動配置スクリプトの実行

```bash
# Windowsの場合
deploy_downloaded_files.bat

# PowerShellの場合
.\deploy_downloaded_files.bat
```

### 2. 手動配置（推奨）

以下の手順でファイルを配置してください：

#### 2.1 メイン DLL ファイル（ルートディレクトリ）

```bash
# メインの血圧推定DLL
cp BloodPressureDLL.dll ./
```

#### 2.2 依存ライブラリ DLL（ルートディレクトリ）

```bash
# OpenCV関連
cp opencv_core4.dll ./
cp opencv_dnn4.dll ./
cp opencv_imgcodecs4.dll ./
cp opencv_imgproc4.dll ./
cp opencv_objdetect4.dll ./

# ONNX Runtime
cp onnxruntime.dll ./

# その他の依存ライブラリ
cp abseil_dll.dll ./
cp jpeg62.dll ./
cp libgcc_s_seh-1.dll ./
cp libgfortran-5.dll ./
cp liblapack.dll ./
cp liblzma.dll ./
cp libpng16.dll ./
cp libprotobuf.dll ./
cp libquadmath-0.dll ./
cp libsharpyuv.dll ./
cp libwebp.dll ./
cp libwebpdecoder.dll ./
cp libwebpdemux.dll ./
cp libwebpmux.dll ./
cp libwinpthread-1.dll ./
cp openblas.dll ./
cp tiff.dll ./
cp zlib1.dll ./
```

#### 2.3 モデルファイル（CppBloodPressureDLL/models/）

```bash
# モデルディレクトリの作成
mkdir -p CppBloodPressureDLL/models

# モデルファイルのコピー
cp models/diastolicbloodpressure.onnx CppBloodPressureDLL/models/
cp models/systolicbloodpressure.onnx CppBloodPressureDLL/models/
cp models/opencv_face_detector.pbtxt CppBloodPressureDLL/models/
cp models/opencv_face_detector_uint8.pb CppBloodPressureDLL/models/
cp models/shape_predictor_68_face_landmarks.dat CppBloodPressureDLL/models/
```

#### 2.4 ヘッダーファイル（CppBloodPressureDLL/include/）

```bash
# ヘッダーディレクトリの作成
mkdir -p CppBloodPressureDLL/include

# ヘッダーファイルのコピー
cp include/BloodPressureDLL.h CppBloodPressureDLL/include/
```

#### 2.5 ドキュメントファイル（ルートディレクトリ）

```bash
# ドキュメントファイルのコピー
cp INTEGRATION_GUIDE.md ./
cp README.md ./
```

## ファイル配置後の確認

配置が完了したら、以下のコマンドで確認してください：

```bash
# 主要ファイルの存在確認
ls -la BloodPressureDLL.dll
ls -la opencv_core4.dll
ls -la onnxruntime.dll
ls -la CppBloodPressureDLL/models/
ls -la CppBloodPressureDLL/include/
```

## 血圧推定テストの実行

ファイル配置が完了したら、以下の手順でテストを実行できます：

### 1. GitHub Actions ワークフローの生成

```bash
# PowerShellでワークフロー生成スクリプトを実行
powershell -ExecutionPolicy Bypass -File create_test_workflow.ps1
```

### 2. テストワークフローの実行

1. GitHub リポジトリの Actions タブに移動
2. "Test Blood Pressure Estimation with Real Video"ワークフローを選択
3. "Run workflow"をクリック
4. パラメータを設定：
   - Test video: `sample_1M.webm`
   - Test duration: `30`
   - Performance monitoring: `true`
5. "Run workflow"をクリック

### 3. テスト結果の確認

テスト実行後、以下のアーティファクトが生成されます：

- `blood-pressure-test-results-{commit-hash}`: テスト結果
- `blood-pressure-test-app-{commit-hash}`: テストアプリケーション

## 期待されるテスト結果

### 血圧推定結果

- 収縮期血圧（SBP）: 80-200 mmHg の範囲
- 拡張期血圧（DBP）: 40-120 mmHg の範囲
- 信頼度: 0.7 以上

### パフォーマンス指標

- フレームレート: 25 FPS 以上
- メモリ使用量: 1GB 以下
- CPU 使用率: 80%以下
- 処理時間: 30 秒以内（30 秒の動画）

### ボトルネック分析

- 初期化時間の分析
- 処理効率の評価
- メモリ効率の評価
- 改善提案の提示

## トラブルシューティング

### よくある問題

1. **DLL が見つからないエラー**

   - 依存 DLL が正しく配置されているか確認
   - パスが正しく設定されているか確認

2. **モデルファイルが見つからないエラー**

   - `CppBloodPressureDLL/models/`ディレクトリにモデルファイルが配置されているか確認

3. **OpenCV エラー**

   - OpenCV 関連 DLL が正しく配置されているか確認
   - バージョンの互換性を確認

4. **ONNX Runtime エラー**
   - `onnxruntime.dll`が正しく配置されているか確認
   - モデルファイルが破損していないか確認

### ログの確認

テスト実行時のログを確認して、具体的なエラーメッセージを特定してください。

## 次のステップ

1. テスト結果を分析し、パフォーマンスの改善点を特定
2. 必要に応じて DLL の最適化を実施
3. 異なる動画ファイルでのテストを実行
4. 本番環境での動作確認を実施

## サポート

問題が発生した場合は、以下の情報を収集して報告してください：

- エラーメッセージの詳細
- テスト実行時のログ
- システム環境情報
- 使用した動画ファイルの詳細
