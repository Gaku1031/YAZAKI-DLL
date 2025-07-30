# FFmpeg H.264 対応版の更新手順

## 問題の概要

現在の ffmpeg.exe は h264 デコーダーをサポートしていないため、h264 エンコードされた webm ファイルを読み込めません。

## 解決方法

GitHub Actions で h264 デコーダーを含む ffmpeg をビルドし、Artifacts からダウンロードして使用します。

## 更新手順

### 1. GitHub Actions の実行

1. GitHub リポジトリの「Actions」タブに移動
2. 「Build FFmpeg with H.264 Support」ワークフローを選択
3. 「Run workflow」ボタンをクリック
4. ビルドが完了するまで待機（約 10-15 分）

### 2. Artifacts のダウンロード

1. ビルド完了後、「ffmpeg-ultra-slim」Artifact をダウンロード
2. ダウンロードした zip ファイルをプロジェクトのルートディレクトリに展開
3. `ffmpeg-ultra-slim`フォルダが作成されることを確認

### 3. ffmpeg.exe の更新

```bash
# 更新スクリプトを実行
update_ffmpeg.bat
```

### 4. 動作確認

```bash
# テストスクリプトを実行
test_ffmpeg_h264.bat
```

## 新しい ffmpeg.exe の特徴

### サポートされる機能

- **デコーダー**: h264, vp9
- **デマクサー**: matroska, webm
- **エンコーダー**: mjpeg
- **フィルター**: scale, select
- **プロトコル**: file

### ビルド設定

```bash
--enable-decoder='h264,vp9'  # h264デコーダーを追加
--enable-parser='h264,vp9'   # h264パーサーを追加
```

## トラブルシューティング

### エラー: "ffmpeg-ultra-slim\ffmpeg.exe not found"

- GitHub Actions から Artifacts を正しくダウンロードしたか確認
- zip ファイルを正しく展開したか確認

### エラー: "Decoding requested, but no decoder found for: h264"

- 新しい ffmpeg.exe が正しく配置されているか確認
- `test_ffmpeg_h264.bat`で h264 デコーダーがサポートされているか確認

### テストファイルが見つからない

- `sample-data\sample_1M.webm`が存在するか確認
- または他の webm ファイルでテスト

## ファイル構成

```
YAZAKI-DLL/
├── .github/workflows/build-ffmpeg.yml  # GitHub Actions設定
├── update_ffmpeg.bat                   # 更新スクリプト
├── test_ffmpeg_h264.bat               # テストスクリプト
├── ffmpeg.exe                          # 更新後のffmpeg.exe
└── ffmpeg-ultra-slim/                 # ダウンロードしたArtifacts
    └── ffmpeg.exe
```

## 注意事項

- 新しい ffmpeg.exe は静的リンクされているため、追加の DLL は不要
- ファイルサイズは約 5-10MB 程度
- Windows 64bit 環境で動作
